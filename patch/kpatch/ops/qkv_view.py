"""qkv_view: GDN 层预填充时去掉 torch.cat(q, k, v) 这次拷贝，改传投影输出的零拷贝视图。

原状（10-12 l9_ledger 关图账）：本模型 num_v_heads/num_k_heads = 48/16 = 3，不在厂商融合拆分核的 [1,2,4] 白名单里，
    走 fix_query_key_value_ordering + torch.cat，每层一次 [T,1024]+[T,1024]+[T,3072] → [T,5120]
    （64K 账 CatArrayBatchedCopy 33 次 × 1337 µs = 44.6 ms，约占预填充 1.3–1.5%）。
观察：in_proj_qkvz 的输出每行按 [q | k | v | z] 排列（切分后 1024/1024/3072/3072），q,k,v 本就是每行前 5120 列，
    cat 的结果与 projected_states_qkvz[:, :5120] 逐位相同，只是行跨步从 5120 变成 8192。
改法：仅在预填充（extend 且非验证）走视图；下游唯一读它的是因果卷积（conv1d_own 自写核按 x.stride(1) 读，
    且输出按 (T, dim) 行连续分配）。验证/解码步原样走 cat（view(batch, draft, -1) 要求连续，且 T 小）。
前提：SGLANG_CONV1D_OWN=1（否则厂商核收到跨步输入，conv1d_own 的回退分支会先 .contiguous()）。
第二步（10-12，同一开关）：卷积输出直接写成 q/k/v 三块各自连续的张量（conv1d.hip 分段输出入口），
    省掉 GDN 预填充内核入口 input_guard 对 q/k/v 的三次 .contiguous()（4K 账 10.4 ms/轮 0.87%，64K 18.7 ms/轮 0.63%）。
    实现为 GDNAttnBackend.forward_extend 的非验证分支重写；验证步、有 mamba 跟踪掩码、卷积核不接管的形状一律走原实现。
开关 SGLANG_QKV_VIEW=1。
"""
import os

import torch

_st = {"run": 0, "be": 0, "fb": set()}


def _install_backend(mod):
    G = getattr(mod, "GDNAttnBackend", None)
    if G is None or getattr(G, "_qkv_view_installed", False):
        return
    from kpatch.ops import conv1d_own
    orig = G.forward_extend

    def forward_extend(self, layer, forward_batch, mixed_qkv, a, b, **kw):
        fm = forward_batch.forward_mode
        md = self.forward_metadata
        why = None
        if fm.is_target_verify():
            return orig(self, layer, forward_batch, mixed_qkv, a, b, **kw)
        if not isinstance(mixed_qkv, torch.Tensor) or mixed_qkv.dim() != 2:
            why = "mixed_qkv 非二维张量"
        elif md.has_mamba_track_mask:
            why = "有 mamba 跟踪掩码"
        elif os.path.exists(conv1d_own._OFF):
            why = None if torch.cuda.is_current_stream_capturing() else "卷积开关文件"
        if why is None:
            cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
            conv_states, ssm_states = cache.conv[0], cache.temporal
            x = mixed_qkv.transpose(0, 1)
            why = conv1d_own._why(x, layer.conv_weights, layer.bias, conv_states, layer.activation, md.query_start_loc)
            if why is None and (layer.q_dim % 8 or layer.k_dim % 8 or layer.q_dim + layer.k_dim + layer.v_dim != x.shape[0]):
                why = f"分段 {layer.q_dim}/{layer.k_dim}/{layer.v_dim}"
        if why:
            if why not in _st["fb"]:
                _st["fb"].add(why)
                print(f"[qkv_view] 分段卷积不接管（{why}），走原实现 pid={os.getpid()}", flush=True)
            return orig(self, layer, forward_batch, mixed_qkv, a, b, **kw)
        T = mixed_qkv.shape[0]
        his = forward_batch.extend_prefix_lens > 0
        q, k, v = conv1d_own.own_seg(x, layer.conv_weights, layer.bias, layer.activation, conv_states, his,
                                     md.mamba_cache_indices, md.query_start_loc, layer.q_dim, layer.k_dim)
        query = q.view(1, T, layer.num_q_heads, layer.head_q_dim)
        key = k.view(1, T, layer.num_k_heads, layer.head_k_dim)
        value = v.view(1, T, layer.num_v_heads, layer.head_v_dim)
        g, beta = mod.fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
        core_attn_out, _, h = self.kernel_dispatcher.extend(q=query, k=key, v=value, g=g, beta=beta, ssm_states=ssm_states,
                                                            cache_indices=md.mamba_cache_indices,
                                                            query_start_loc=md.query_start_loc)
        if h is not None:
            self._track_mamba_state_extend(forward_batch, h, ssm_states, md)
        _st["be"] += 1
        if _st["be"] == 1:
            print(f"[qkv_view] ✅ 分段卷积输出连续 q/k/v 首次真正生效：T={T} q{tuple(query.shape)} k{tuple(key.shape)} "
                  f"v{tuple(value.shape)} bs={md.query_start_loc.numel() - 1} mode={fm} "
                  f"capturing={torch.cuda.is_current_stream_capturing()} pid={os.getpid()}", flush=True)
        return core_attn_out

    G.forward_extend = forward_extend
    G._qkv_view_installed = True
    print(f"[qkv_view] 已安装（GDNAttnBackend.forward_extend 预填充分段卷积） pid={os.getpid()}", flush=True)


def install(mod):
    if mod.__name__.endswith("gdn_backend"):
        return _install_backend(mod)
    cls = getattr(mod, "Qwen3_5GatedDeltaNet", None)
    if cls is None or getattr(cls, "_qkv_view_installed", False):
        return
    orig = cls.forward

    def forward(self, hidden_states, forward_batch):
        fm = forward_batch.forward_mode
        if (self.num_v_heads // self.num_k_heads in (1, 2, 4)) or not fm.is_extend() or fm.is_target_verify():
            return orig(self, hidden_states, forward_batch)
        qkvz, ba = self._forward_input_proj(hidden_states)
        k_tp = self.key_dim // self.attn_tp_size
        v_tp = self.value_dim // self.attn_tp_size
        nv_tp = self.num_v_heads // self.attn_tp_size
        mixed_qkv = qkvz[:, : 2 * k_tp + v_tp]
        z = qkvz[:, 2 * k_tp + v_tp:].reshape(qkvz.size(0), -1, self.head_v_dim)
        b, a = ba.split([nv_tp, nv_tp], dim=-1)
        b = b.contiguous()
        a = a.contiguous()
        _st["run"] += 1
        if _st["run"] == 1:
            print(f"[qkv_view] ✅ GDN 预填充 qkv 零拷贝视图首次真正生效：mixed_qkv{tuple(mixed_qkv.shape)} stride{mixed_qkv.stride()} "
                  f"mode={fm} pid={os.getpid()}", flush=True)
        core_attn_out = self.attn(forward_batch, mixed_qkv=mixed_qkv, a=a, b=b)
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        if core_attn_out.shape != z.shape:
            pad = torch.zeros_like(z)
            pad[: core_attn_out.shape[0], :] = core_attn_out
            core_attn_out = pad
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-2], -1)
        output, _ = self.out_proj(core_attn_out)
        return output

    cls.forward = forward
    cls._qkv_view_installed = True
    print(f"[qkv_view] 已安装（Qwen3_5GatedDeltaNet.forward 预填充去 cat） pid={os.getpid()}", flush=True)

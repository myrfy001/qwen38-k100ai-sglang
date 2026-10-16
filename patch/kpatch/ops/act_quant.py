"""act_quant: MLP 的 silu·mul 与 down_proj 输入的逐 token int8 量化合成一个核（厂商 lightop 现成算子，逐位）。

原状（10-10 l4_ledger 关图账，预填充 4096×3）：每层 lightop.fuse_silu_and_mul 写 bf16 [M,8704]，down_proj 的
    W8A8Int8LinearMethod.apply 再调 per_token_quant_int8 把它读回来量化：silu_and_mul 501 µs + _per_token_quant_int8 188 µs，65 次/轮。
厂商其实已留好融合路径：RowParallelLinear.forward(use_fused_silu_mul_quant=True) 对未激活的 gate_up 调
    lightop.activation.lm_fuse_silu_mul_quant，再以 apply(silu_quant_args=[xq, xs]) 下发——但只接了 compressed_tensors 方案
    （SGLANG_USE_FUSED_SILU_MUL_QUANT），本模型的 w8a8_int8 方法不认 silu_quant_args，Qwen2MoeMLP 也从不传这个开关。
改法（三处接线，不设厂商全局变量，以免牵动 compressed_tensors 其他分支）：
    ① sglang.srt.layers.linear 注入 lm_fuse_silu_mul_quant（模块按该变量条件导入，未设时不存在）；
    ② W8A8Int8LinearMethod.apply 接受 silu_quant_args，转成它本来就支持的 input_quant_args；
    ③ Qwen2MoeMLP.forward：down_proj 是 W8A8 时把 gate_up 原样交给 down_proj 并打开融合开关。all-reduce 路径不变。
逐位：probe/actq_test.py（10-12）M=1…16384 × 幅度 1/8，三个厂商融合入口 int8 与 scale 对生产两步全 0 不等。
耗时（图内）：M=4096 496→279 µs（1.78×），M=192 28.4→18.5，M=48 11.5→10.8。
开关 SGLANG_ACT_QUANT=1。取证 SGLANG_ACT_QUANT_CHECK=1（图外调用同时跑原两步，逐位比 int8 与 scale，不碰通信，慢）。
"""
import os

import torch

_st = {"run": 0, "fb": set()}
_CHECK = os.environ.get("SGLANG_ACT_QUANT_CHECK", "0") == "1"


def _install_linear(mod):
    if getattr(mod, "lm_fuse_silu_mul_quant", None) is None:
        from lightop.activation import lm_fuse_silu_mul_quant
        mod.lm_fuse_silu_mul_quant = lm_fuse_silu_mul_quant
    print(f"[act_quant] 已安装（linear 注入 lm_fuse_silu_mul_quant） pid={os.getpid()}", flush=True)


def _install_w8a8(mod):
    cls = getattr(mod, "W8A8Int8LinearMethod", None)
    if cls is None:
        print(f"[act_quant] ❌ 安装失败：{mod.__name__} 无 W8A8Int8LinearMethod pid={os.getpid()}", flush=True)
        return
    if getattr(cls.apply, "_act_quant_wrapped", False):
        return
    orig = cls.apply

    def apply(self, layer, x, bias=None, input_quant_args=None, silu_quant_args=None):
        if silu_quant_args is not None:
            input_quant_args = silu_quant_args
        return orig(self, layer, x, bias=bias, input_quant_args=input_quant_args)

    apply._act_quant_wrapped = True
    cls.apply = apply
    print(f"[act_quant] 已安装（W8A8Int8LinearMethod.apply 接受 silu_quant_args） pid={os.getpid()}", flush=True)


def _install_mlp(mod):
    cls = getattr(mod, "Qwen2MoeMLP", None)
    if cls is None or getattr(cls, "_act_quant_installed", False):
        return
    from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8LinearMethod
    orig = cls.forward

    def forward(self, x, should_allreduce_fusion=False, use_reduce_scatter=False):
        why = None
        if not isinstance(getattr(self.down_proj, "quant_method", None), W8A8Int8LinearMethod):
            why = f"down_proj 量化方法 {type(getattr(self.down_proj, 'quant_method', None)).__name__}"
        elif not self.down_proj.input_is_parallel:
            why = "down_proj 输入未切分"
        if why is None:
            gate_up, _ = self.gate_up_proj(x)
            if gate_up.dim() != 2 or gate_up.dtype != torch.bfloat16 or not gate_up.is_contiguous():
                why = f"gate_up {tuple(gate_up.shape)} {gate_up.dtype}"
            elif gate_up.shape[0] == 0:
                why = "空批"
        if why:
            if why not in _st["fb"]:
                _st["fb"].add(why)
                print(f"[act_quant] 不接管（{why}），走原实现 pid={os.getpid()}", flush=True)
            return orig(self, x, should_allreduce_fusion=should_allreduce_fusion, use_reduce_scatter=use_reduce_scatter)
        skip = should_allreduce_fusion or use_reduce_scatter
        out, _ = self.down_proj(gate_up, skip_all_reduce=skip, use_fused_silu_mul_quant=True)
        _st["run"] += 1
        if _st["run"] == 1:
            print(f"[act_quant] ✅ silu·mul+量化融合首次真正生效：gate_up {tuple(gate_up.shape)} "
                  f"capturing={torch.cuda.is_current_stream_capturing()} pid={os.getpid()}", flush=True)
        if _CHECK and not torch.cuda.is_current_stream_capturing():
            # 只比本卡的量化输入（int8 与 scale），不再调 down_proj：多一次 all-reduce 会打乱成对预填充的段号（10-12 act_chk 教训）
            from sglang.srt.layers import linear as _lin
            from sglang.srt.layers.quantization import w8a8_int8 as _w8
            xq, xs = _lin.lm_fuse_silu_mul_quant(gate_up)
            rq, rs = _w8.per_token_quant_int8(self.act_fn(gate_up))
            xs, rs = xs.reshape(-1).float(), rs.reshape(-1).float()
            # 全零填充行两路 scale 取值不同（int8 全 0，输出不受影响，10-12 落盘已核），不计
            live = (rq != 0).any(dim=-1) | (xq != 0).any(dim=-1)
            nb = int((xq != rq).sum()) + int((~((xs == rs) | (xs.isnan() & rs.isnan())) & live).sum())
            _st["chk"] = _st.get("chk", 0) + 1
            if nb:
                _st["bad"] = _st.get("bad", 0) + 1
                if _st["bad"] <= 3:
                    dq = (xq.int() - rq.int()).abs()
                    print(f"[act_quant] ❌ 取证：量化输入与原两步不等 {nb} 处 gate_up {tuple(gate_up.shape)} "
                          f"int8最大差 {int(dq.max())} 行 {sorted(set((dq > 0).nonzero()[:, 0].tolist()))[:8]} "
                          f"scale不等 {int((xs != rs).sum())} 层 {getattr(self.down_proj, 'prefix', '?')} "
                          f"nan={int(gate_up.isnan().sum())} pid={os.getpid()}", flush=True)
                if _st["bad"] == 1:
                    torch.save({"gate_up": gate_up.cpu(), "xq": xq.cpu(), "xs": xs.cpu(), "rq": rq.cpu(), "rs": rs.cpu()},
                               f"/tmp/actq_bad_{os.getpid()}.pt")
            if _st["chk"] in (100, 1000, 10000):
                print(f"[act_quant] 取证：已比对 {_st['chk']} 次，不等 {_st.get('bad', 0)} 次 pid={os.getpid()}", flush=True)
        return out

    cls.forward = forward
    cls._act_quant_installed = True
    print(f"[act_quant] 已安装（Qwen2MoeMLP.forward 走厂商融合路径） pid={os.getpid()}", flush=True)


def install(mod):
    name = mod.__name__
    if name == "sglang.srt.layers.linear":
        _install_linear(mod)
    elif name.endswith("w8a8_int8"):
        _install_w8a8(mod)
    elif name.endswith("qwen2_moe"):
        _install_mlp(mod)

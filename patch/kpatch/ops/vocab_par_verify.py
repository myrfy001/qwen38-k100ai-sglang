"""EAGLE 验证步的 argmax 改成词表并行：图内只算本卡 logits 分片与全局 argmax，不再收集 [4·bs, 248320] 的整张 logits。

依据:
  - 验证步每轮一次 logits 全收集（图内，M=4·bs），走 PCIe：图内实测 M=16 489 µs、M=64 1821 µs、M=192 5340 µs
  - HIP 上验证恒走贪心分支：torch.argmax(logits) + verify_tree_greedy（eagle_utils.py:661，条件含 `or _is_hip`），
    与温度无关；实验室压测命令本身也发 temperature=0
  - 图内不能按批次条件分支，所以图内一律输出「本卡分片 + 末列全局 argmax」，图外判定：
      干净批（无 logprob / 语法 / 惩罚 / logit_bias / NaN 净化 / 模拟接受长度）→ 直接用末列 argmax；
      否则 → 图外补一次全收集（bf16，与现状逐位相同）再走原 eagle_sample
  - 输出写进 sglang 的共享 logits 缓冲（前 V_local+1 列的视图），不新增显存
同值取首个出现（torch.argmax 的规则）：本地严格大于才换，合并时同值取卡号小的。
只接管 EAGLEWorkerV2 的验证；目标模型的预填充/采样不变。依赖 vocab_par_draft 的内核。
"""
import importlib
import os
import torch, triton, triton.language as tl

_ST = {"eagle": False, "w": None, "full_w": None, "fast": 0, "fb": {}, "first_fast": False, "cur": None}


@triton.jit
def _combine_into(g_ptr, out_ptr, M, V, ostride, W: tl.constexpr):
    """g: all_gather 后的 [W, M, 3] → 把全局 argmax（fp32 精确表示，<2^24）写到 out[m, V]。同值取卡号小的。"""
    m = tl.program_id(0); r = tl.arange(0, W)
    lmax = tl.load(g_ptr + (r * M + m) * 3 + 0); lidx = tl.load(g_ptr + (r * M + m) * 3 + 1)
    gmax = tl.max(lmax, 0)
    win = tl.min(tl.where(lmax == gmax, r, W), 0)
    widx = tl.sum(tl.where(r == win, lidx, 0.0), 0)
    tl.store(out_ptr + m.to(tl.int64) * ostride + V, (win * V).to(tl.float32) + widx)


def _eligible(proc, lm_head):
    ok = getattr(proc, "_kp_vpv_ok", None)
    if ok is not None:
        return ok
    why = None
    try:
        from sglang.srt.server_args import get_global_server_args as get_server_args  # 本镜像无 runtime_context（10-11 vpa_on 因此未生效）
        from sglang.srt.distributed import get_tp_group
        sa = get_server_args()
        si = lm_head.shard_indices
        W, rank = get_tp_group().world_size, get_tp_group().rank_in_group
        width = si.num_elements_padded
        algo = str(sa.speculative_algorithm).upper().split(".")[-1]
        if not _ST["eagle"]:   # 目标模型验证图先于草稿 worker 捕获（10-11 vpa2_on 因此未生效）：主动导入以触发 sitecustomize 安装钩子
            importlib.import_module("sglang.srt.speculative.eagle_worker_v2")
        if not _ST["eagle"]: why = "eagle_worker_v2 的 eagle_sample 钩子未安装"
        elif algo not in ("EAGLE", "NEXTN"): why = f"算法 {algo} 不在白名单（只接管 EAGLEWorkerV2.verify）"
        elif getattr(sa, "enable_multi_layer_eagle", False): why = "multi-layer EAGLE 走另一个 worker"
        elif W & (W - 1) != 0: why = f"TP={W} 不是 2 的幂（合并核 tl.arange 要求）"
        elif not proc.do_tensor_parallel_all_gather or proc.use_attn_tp_group or proc.do_tensor_parallel_all_gather_dp_attn:
            why = "非普通 TP 全收集路径"
        elif proc.logit_scale is not None or proc.final_logit_softcapping: why = "logit_scale/softcapping"
        elif getattr(proc, "use_fp32_lm_head", False): why = "fp32 lm_head"
        elif si.num_org_vocab_padding != 0 or si.org_vocab_start_index != rank * width or si.num_org_elements != width:
            why = f"词表分片有补白或不等分: start={si.org_vocab_start_index} n={si.num_org_elements} width={width}"
        elif W < 2 or W * width != proc.vocab_size: why = f"W={W} width={width} vocab_size={proc.vocab_size}"
        else:
            _ST["w"] = width; _ST["full_w"] = W * width
    except Exception as e:
        why = f"代码异常 {type(e).__name__}: {e}"
    ok = why is None
    proc._kp_vpv_ok = ok
    print(f"[vocab_par_verify] 目标 logits 处理器（验证步）: "
          f"{'启用词表并行 argmax（本地宽度 %d）' % _ST['w'] if ok else '不启用，原因: ' + why} pid={os.getpid()}", flush=True)
    return ok


def _install_logits(mod):
    LP = mod.LogitsProcessor
    if getattr(LP, "_kp_vpv_installed", False):
        return
    orig = LP._get_logits

    def _get_logits(self, hidden_states, lm_head, logits_metadata, embedding_bias=None):
        fm = getattr(logits_metadata, "forward_mode", None)
        if (not getattr(self, "_kp_vpa", False) and fm is not None and fm.is_target_verify()
                and _eligible(self, lm_head)):
            from sglang.srt.distributed import get_tp_group
            from kpatch.ops.vocab_par_draft import local_reduce
            g = get_tp_group()
            local = self._compute_lm_head(hidden_states, lm_head, embedding_bias)       # [M, W] 本卡分片
            if local.stride(-1) != 1 or local.stride(0) != local.shape[1]:
                local = local.contiguous()                                               # 内核按行连续取数
            M, Wd = local.shape
            buf = getattr(logits_metadata, "next_token_logits_buffer", None)
            if buf is not None and buf.dim() == 2 and buf.shape[0] == M and buf.shape[1] >= Wd + 1 and buf.dtype == torch.float32:
                out = buf[:, : Wd + 1]                                                 # 共享缓冲的视图，不新增显存
            else:
                out = torch.empty(M, Wd + 1, device=local.device, dtype=torch.float32)
            out[:, :Wd].copy_(local)
            part = local_reduce(local)
            gathered = torch.empty(g.world_size * M, 3, device=part.device, dtype=part.dtype)
            g.all_gather_into_tensor(gathered, part)                                     # [W*M, 3]，绕开 HCU reshape 分支
            _combine_into[(M,)](gathered, out, M, Wd, out.stride(0), W=g.world_size)
            return out
        return orig(self, hidden_states, lm_head, logits_metadata, embedding_bias)

    LP._get_logits = _get_logits
    LP._kp_vpv_installed = True
    print(f"[vocab_par_verify] 已安装到 {mod.__name__} (pid={os.getpid()})", flush=True)


def _need_full(batch, vocab_mask, eu):
    """返回需要整张 logits 的原因；None 表示干净批。所有判据来自同一个 ScheduleBatch 与环境变量，各卡一致。"""
    si = batch.sampling_info
    if getattr(batch, "return_logprob", False): return "logprob"
    if vocab_mask is not None or getattr(batch, "has_grammar", False): return "grammar"
    if si.acc_additive_penalties is not None or si.acc_scaling_penalties is not None: return "penalty"
    if si.logit_bias is not None: return "logit_bias"
    if os.environ.get("SGLANG_SANITIZE_NAN_LOGITS", "0") not in ("", "0", "false", "False"): return "sanitize_nan"
    if getattr(eu, "_kp_old", False):                 # 旧版快路自己重写了贪心验证，不含模拟接受长度
        from sglang.srt.speculative.spec_utils import SIMULATE_ACC_LEN
        if SIMULATE_ACC_LEN > 0: return "simulate_acc_len"
    if not (si.is_all_greedy or getattr(eu, "_is_hip", False)): return "sampling"
    return None


def _install_eagle_v2(mod):
    """sglang 0.5.12+das185：验证在 eagle_info_v2.EagleVerifyInputV2Mixin.sample，HIP 恒走 torch.argmax 分支。
    干净批：原样调用 sample，只把该模块里的 torch.argmax 对这张 logits 的调用换成取末列的全局 argmax ⇒ 其余逻辑全用原实现。"""
    import sglang.srt.speculative.eagle_info_v2 as ei
    from sglang.srt.distributed import tensor_model_parallel_all_gather
    from kpatch.ops.fused_draft_topk import override_torch
    Mx = ei.EagleVerifyInputV2Mixin
    orig = Mx.sample

    def argmax(x, dim=None, *a, **k):
        c = _ST["cur"]
        if c is not None and x is c and dim == -1 and not a and not k:
            return x[:, _ST["w"]].to(torch.int64)
        return torch.argmax(x, dim, *a, **k)

    def sample(self, batch, logits_output, vocab_mask=None):
        lg = logits_output.next_token_logits
        w = _ST["w"]
        if w is None or batch.forward_mode.is_idle() or not isinstance(lg, torch.Tensor) or lg.dim() != 2:
            return orig(self, batch, logits_output, vocab_mask)
        if lg.shape[-1] != w + 1:
            if lg.shape[-1] != _ST["full_w"]:
                raise RuntimeError(f"[vocab_par_verify] 验证 logits 宽度 {lg.shape[-1]} 既非 {w + 1} 也非全词表 {_ST['full_w']}")
            return orig(self, batch, logits_output, vocab_mask)
        why = _need_full(batch, vocab_mask, ei)
        if why is None:
            _ST["cur"] = lg
            try:
                out = orig(self, batch, logits_output, vocab_mask)
            finally:
                _ST["cur"] = None
            _ST["fast"] += 1
            n = _ST["fast"]
            if n in (100, 10000, 1000000):
                print(f"[vocab_par_verify] 累计: 快路 {n} 次, 回退 {_ST['fb']} pid={os.getpid()}", flush=True)
            if not _ST["first_fast"]:
                _ST["first_fast"] = True
                print(f"[vocab_par_verify] ✅ 首次真正生效（快路）: bs={len(batch.seq_lens)} rows={lg.shape[0]} pid={os.getpid()}", flush=True)
            return out
        full = tensor_model_parallel_all_gather(lg[:, :w].to(torch.bfloat16).contiguous(), dim=-1).float()
        logits_output.next_token_logits = full
        _ST["fb"][why] = _ST["fb"].get(why, 0) + 1
        if _ST["fb"][why] == 1:
            print(f"[vocab_par_verify] 回退全收集: 原因={why}（该原因首次）pid={os.getpid()}", flush=True)
        return orig(self, batch, logits_output, vocab_mask)

    override_torch(ei, "argmax", argmax)
    Mx.sample = sample
    mod._kp_vpv_installed = True
    _ST["eagle"] = True
    print(f"[vocab_par_verify] 已安装到 {ei.__name__}.EagleVerifyInputV2Mixin.sample（经 {mod.__name__}）(pid={os.getpid()})", flush=True)


def _install_eagle(mod):
    if getattr(mod, "_kp_vpv_installed", False):
        return
    if not hasattr(mod, "eagle_sample"):
        return _install_eagle_v2(mod)
    import sglang.srt.speculative.eagle_utils as eu
    eu._kp_old = True
    from sglang.srt.distributed import tensor_model_parallel_all_gather
    orig = mod.eagle_sample

    def eagle_sample(verify_input, batch, logits_output, vocab_mask=None):
        lg = logits_output.next_token_logits
        w = _ST["w"]
        if w is None or batch.forward_mode.is_idle() or not isinstance(lg, torch.Tensor) or lg.dim() != 2:
            return orig(verify_input, batch, logits_output, vocab_mask)
        if lg.shape[-1] != w + 1:
            if lg.shape[-1] != _ST["full_w"]:
                raise RuntimeError(f"[vocab_par_verify] 验证 logits 宽度 {lg.shape[-1]} 既非 {w + 1} 也非全词表 {_ST['full_w']}")
            return orig(verify_input, batch, logits_output, vocab_mask)
        why = _need_full(batch, vocab_mask, eu)
        if why is None:
            bs = len(batch.seq_lens); dtn = verify_input.draft_token_num
            dev = lg.device
            target_predict = lg[:, w].to(torch.int64).reshape(bs, dtn)
            candidates = verify_input.draft_token.reshape(bs, dtn)
            predict = torch.zeros(lg.shape[0], dtype=torch.int32, device=dev)
            accept_index = torch.full((bs, verify_input.max_tree_depth), -1, dtype=torch.int32, device=dev)
            num_correct = torch.empty((bs,), dtype=torch.int32, device=dev)
            predict, accept_index, num_correct = eu.verify_tree_greedy_func(
                predicts=predict, accept_index=accept_index, accept_token_num=num_correct,
                candidates=candidates, retrieve_index=verify_input.retrieve_index,
                retrieve_next_token=verify_input.retrieve_next_token,
                retrieve_next_sibling=verify_input.retrieve_next_sibling,
                target_predict=target_predict, topk=verify_input.tree_topk)
            _ST["fast"] += 1
            n = _ST["fast"]
            if n in (100, 10000, 1000000) and os.environ.get("RANK", "0") in ("0", ""):
                print(f"[vocab_par_verify] 累计: 快路 {n} 次, 回退 {_ST['fb']} pid={os.getpid()}", flush=True)
            if not _ST["first_fast"]:
                _ST["first_fast"] = True
                print(f"[vocab_par_verify] 首次生效（快路）: bs={bs} rows={lg.shape[0]} pid={os.getpid()}", flush=True)
            return predict, num_correct + 1, accept_index
        # 回退：图外补一次全收集（本卡分片 bf16 精确，收集只是拷贝 ⇒ 与现状逐位相同）
        full = tensor_model_parallel_all_gather(lg[:, :w].to(torch.bfloat16).contiguous(), dim=-1).float()
        logits_output.next_token_logits = full
        _ST["fb"][why] = _ST["fb"].get(why, 0) + 1
        if _ST["fb"][why] == 1:                          # 每种原因首次出现都打印
            print(f"[vocab_par_verify] 回退全收集: 原因={why}（该原因首次）pid={os.getpid()}", flush=True)
        return orig(verify_input, batch, logits_output, vocab_mask)

    mod.eagle_sample = eagle_sample
    mod._kp_vpv_installed = True
    _ST["eagle"] = True
    print(f"[vocab_par_verify] 已安装到 {mod.__name__} (pid={os.getpid()})", flush=True)


def install(mod):
    n = mod.__name__
    if n.endswith("logits_processor"):
        _install_logits(mod)
    elif n.endswith("eagle_worker_v2"):
        _install_eagle(mod)
    else:
        raise RuntimeError(f"未知目标模块 {n}")

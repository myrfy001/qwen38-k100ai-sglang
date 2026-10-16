"""draft_prune: 草稿扩展（draft_extend_v2）的 LM-head 只算真正要用的那一行/请求。

原状（10-12 l9_ledger 4K C32 decode 账）：EagleDraftWorker._draft_extend_for_decode 对 bs×(steps+1) 行全部算 LM-head
    （logits 处理器对 draft_extend_v2 不剪枝），之后只取 select_index = arange(bs)·n + accept_lens − 1 这 bs 行。
    C32 时即 M=128 的 lmh2 一次 3.59 ms，外加 [128,124160] fp32 转换 64 MB；真正用到的只有 32 行。
改法：调草稿扩展前按同一公式算好 select_index 暂存；草稿 logits 处理器（_kp_vpa 标记，只草稿模型有）在
    draft_extend_v2、非图捕获、无预分配 logits 缓冲、行数 = bs·n 时，只对这 bs 行算 logits，再按 select_index 填回
    [bs·n, V] 的输出（其余行不写，调用方随后只读 select_index 行）。
不变量：验证步不动 ⇒ 输出由目标模型决定；草稿 logits 行与全算时可能差末位（GEMM 按 M 选解不同），只影响接受长度。
开关 SGLANG_DRAFT_PRUNE=1；需与 vocab_par_draft 同时开（用它的 _kp_vpa 标记识别草稿处理器）。
"""
import os

import torch

_ST = {"sel": None, "run": 0, "fb": set()}


def _install_worker(mod):
    C = getattr(mod, "EagleDraftWorker", None)
    if C is None or getattr(C, "_draft_prune_installed", False):
        return
    orig = C._draft_extend_for_decode

    def _draft_extend_for_decode(self, batch, batch_result):
        bs = len(batch.seq_lens)
        n = self.speculative_num_draft_tokens
        try:
            _ST["sel"] = (torch.arange(bs, device=self.device) * n + batch_result.accept_lens - 1, bs * n)
            return orig(self, batch, batch_result)
        finally:
            _ST["sel"] = None

    C._draft_extend_for_decode = _draft_extend_for_decode
    C._draft_prune_installed = True
    print(f"[draft_prune] 已安装（EagleDraftWorker._draft_extend_for_decode 暂存 select_index） pid={os.getpid()}", flush=True)


def _install_logits(mod):
    LP = mod.LogitsProcessor
    if getattr(LP, "_draft_prune_installed", False):
        return
    orig = LP._get_logits

    def _get_logits(self, hidden_states, lm_head, logits_metadata, embedding_bias=None):
        st = _ST["sel"]
        if st is None or not getattr(self, "_kp_vpa", False):
            return orig(self, hidden_states, lm_head, logits_metadata, embedding_bias)
        sel, n = st
        why = None
        if not logits_metadata.forward_mode.is_draft_extend_v2():
            why = f"模式 {logits_metadata.forward_mode}"
        elif torch.cuda.is_current_stream_capturing():
            why = "图捕获中"
        elif logits_metadata.next_token_logits_buffer is not None:
            why = "有预分配 logits 缓冲"
        elif hidden_states.dim() != 2 or hidden_states.shape[0] != n:
            why = f"行数 {tuple(hidden_states.shape)} ≠ bs·n={n}"
        elif self.do_tensor_parallel_all_gather_dp_attn or self.logit_scale is not None or self.final_logit_softcapping:
            why = "dp-attn/logit_scale/softcap"
        if why:
            if why not in _ST["fb"]:
                _ST["fb"].add(why)
                print(f"[draft_prune] 不接管（{why}），走原实现 pid={os.getpid()}", flush=True)
            return orig(self, hidden_states, lm_head, logits_metadata, embedding_bias)
        lg = orig(self, hidden_states.index_select(0, sel), lm_head, logits_metadata, embedding_bias)
        out = lg.new_empty((n, lg.shape[1]))
        out.index_copy_(0, sel, lg)
        _ST["run"] += 1
        if _ST["run"] == 1:
            print(f"[draft_prune] ✅ 草稿扩展 LM-head 剪枝首次真正生效：{n} 行 → {sel.numel()} 行，logits {tuple(lg.shape)} {lg.dtype} "
                  f"pid={os.getpid()}", flush=True)
        return out

    LP._get_logits = _get_logits
    LP._draft_prune_installed = True
    print(f"[draft_prune] 已安装（LogitsProcessor._get_logits 草稿扩展剪枝） pid={os.getpid()}", flush=True)


def install(mod):
    if mod.__name__.endswith("logits_processor"):
        _install_logits(mod)
    elif mod.__name__.endswith("eagle_worker_v2"):
        _install_worker(mod)

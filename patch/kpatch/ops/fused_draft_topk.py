"""把 EAGLE 草稿选择里的 softmax([M,248320]) + max 融合成一次按行切块的 kernel。

依据:
  - eagle_worker_v2.py 三个调用点在 use_rejection_sampling=False 时, probs 只流向 fast_topk
  - sglang issue #26358: topk_p 必须是真值(硬编码 1.0 会毁掉 MTP), 本实现给出真值
  - 现状瓶颈: PyTorch softmax 每行只用一个 workgroup, M=1..48 时 64 个 CU 严重欠并行
实测(GPU 独占, 图内计时, float32, V=248320): 8.0x ~ 14.6x, index 逐个一致, p 相对误差 <= 4e-7
"""
import torch, triton, triton.language as tl

S_SPLIT = 32          # 每行切成多少块

@triton.jit
def _stage1(l_ptr, pm, ps, pi, Vn, sl, S: tl.constexpr, BLK: tl.constexpr):
    r = tl.program_id(0); c = tl.program_id(1)
    chunk = tl.cdiv(Vn, S)
    lo = c * chunk; hi = tl.minimum(lo + chunk, Vn)
    base = l_ptr + r.to(tl.int64) * sl
    m = -float("inf"); s = 0.0; am = 0
    for start in range(lo, hi, BLK):
        off = start + tl.arange(0, BLK)
        x = tl.load(base + off, mask=off < hi, other=-float("inf")).to(tl.float32)
        bm = tl.max(x, 0); bi = tl.argmax(x, 0) + start
        nm = tl.maximum(m, bm)
        s = s * tl.exp(m - nm) + tl.sum(tl.where(off < hi, tl.exp(x - nm), 0.0), 0)
        am = tl.where(bm > m, bi, am)
        m = nm
    o = r * S + c
    tl.store(pm + o, m); tl.store(ps + o, s); tl.store(pi + o, am)

@triton.jit
def _stage2(pm, ps, pi, p_out, i_out, S: tl.constexpr):
    r = tl.program_id(0)
    off = r * S + tl.arange(0, S)
    m = tl.load(pm + off); s = tl.load(ps + off); i = tl.load(pi + off)
    gm = tl.max(m, 0)
    tot = tl.sum(s * tl.exp(m - gm), 0)
    k = tl.argmax(m, 0)
    tl.store(p_out + r, 1.0 / tot)
    tl.store(i_out + r, tl.sum(tl.where(tl.arange(0, S) == k, i, 0), 0).to(tl.int64))

def fused_max_prob(logits: torch.Tensor):
    """返回 (topk_p[M,1] float32, topk_index[M,1] int64), 与 softmax(logits).max(-1) 等价"""
    M, V = logits.shape
    S = S_SPLIT
    dev = logits.device
    pm = torch.empty(M * S, device=dev, dtype=torch.float32)
    ps = torch.empty(M * S, device=dev, dtype=torch.float32)
    pi = torch.empty(M * S, device=dev, dtype=torch.int32)
    po = torch.empty(M, device=dev, dtype=torch.float32)
    io = torch.empty(M, device=dev, dtype=torch.int64)
    _stage1[(M, S)](logits, pm, ps, pi, V, logits.stride(0), S, 2048, num_warps=8)
    _stage2[(M,)](pm, ps, pi, po, io, S, num_warps=1)
    return po.view(M, 1), io.view(M, 1)

class TorchProxy:
    """模块级 torch 替身：只改写个别函数，其余属性原样转发。只影响被替换的那个模块。"""
    def __init__(self, base):
        object.__setattr__(self, "_b", base); object.__setattr__(self, "_o", {})
    def __getattr__(self, n):
        o = object.__getattribute__(self, "_o")
        return o[n] if n in o else getattr(object.__getattribute__(self, "_b"), n)


def override_torch(mod, name, fn):
    """把 mod 里的 torch.<name> 换成 fn（多个补丁可叠加在同一个替身上）"""
    if not isinstance(mod.torch, TorchProxy):
        mod.torch = TorchProxy(torch)
    object.__getattribute__(mod.torch, "_o")[name] = fn


class _Lazy:
    """占位: 不真的算 softmax, 等 fast_topk 来了一次算完"""
    __slots__ = ("logits",)
    def __init__(self, logits): self.logits = logits

def _install_softmax(mod):
    """sglang 0.5.12+das185：eagle_worker_v2 三处写成 probs = torch.softmax(logits, dim=-1); fast_topk(probs, topk, -1)，
    probs 别无他用（10-11 grep 核实）。softmax 先返回占位，fast_topk 时一次算完。"""
    orig_topk = mod.fast_topk

    def softmax(x, dim=None, *a, **k):
        if (dim == -1 and not a and not k and isinstance(x, torch.Tensor) and x.dim() == 2 and x.is_cuda
                and x.is_floating_point() and x.size(0) > 0):
            return _Lazy(x)
        return torch.softmax(x, dim, *a, **k)

    def ftopk(values, topk, dim):
        if isinstance(values, _Lazy):
            if topk == 1 and values.logits.dtype == torch.float32:
                return fused_max_prob(values.logits)
            return orig_topk(torch.softmax(values.logits, dim=-1), topk, dim)
        return orig_topk(values, topk, dim)

    override_torch(mod, "softmax", softmax)
    mod.fast_topk = ftopk


def install(mod):
    if getattr(mod, "_fused_draft_topk_installed", False):
        return
    import os
    if not hasattr(mod, "renorm_draft_probs"):
        _install_softmax(mod)
        mod._fused_draft_topk_installed = True
        print(f"[fused_draft_topk] 已安装到 {mod.__name__}（接管模块内 torch.softmax + fast_topk）(pid={os.getpid()})", flush=True)
        return
    orig_renorm = mod.renorm_draft_probs
    orig_topk = mod.fast_topk

    def renorm(next_token_logits, sampling_info, use_rejection_sampling):
        if (not use_rejection_sampling and next_token_logits is not None
                and next_token_logits.dim() == 2 and next_token_logits.is_cuda
                and next_token_logits.dtype == torch.float32
                and next_token_logits.size(0) > 0):
            return _Lazy(next_token_logits)
        return orig_renorm(next_token_logits, sampling_info, use_rejection_sampling)

    def ftopk(values, topk, dim):
        if isinstance(values, _Lazy):
            if topk == 1:
                return fused_max_prob(values.logits)
            return orig_topk(orig_renorm(values.logits, None, False), topk, dim)
        return orig_topk(values, topk, dim)

    mod.renorm_draft_probs = renorm
    mod.fast_topk = ftopk
    mod._fused_draft_topk_installed = True
    import os
    print(f"[fused_draft_topk] 已安装到 {mod.__name__} (pid={os.getpid()})", flush=True)

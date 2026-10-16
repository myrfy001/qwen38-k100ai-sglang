"""add + GemmaRMSNorm 与下游 int8 逐 token 量化合成一个核（底层专项 L2，逐位）。

原状（10-09 l3_ledger 关图账）：每层 input_layernorm / post_attention_layernorm 走 lightop.norm.gemma_fused_add_rmsnorm
    （fused_add_rms_kernel_opt<BFloat16,float,8,1024>，bf16 原地），输出随后被下游 W8A8 线性层各自调用一次
    lightop.quant.per_token_quant_int8（_per_token_quant_int8，nearbyint）；GDN 层的 in_proj_qkvz 与 in_proj_ba 吃同一输入，
    同一个 norm 输出被量化两次（账上量化 173 次/轮对 norm 87 次/轮）。
改法：_addrms_q 一个 Triton 核同时写 bf16 归一化输出、残差、int8 与 scale；量化结果按输出张量 data_ptr 登记，
    per_token_quant_int8 的包装先查表命中直接返回，未命中走原实现（同 Hy3 kpatch/ops/norm_quant.py）。
逐位：probe/addrms_q38.py（10-10）在 16 种写法里只有一种四项全 0 不等：
    每线程 8 个连续元素串行 FMA → wave 内 xor 蝶形 → 补 0 到 16 个 wave 的跨 wave 蝶形（5120/8=640 线程，厂商块 1024）；
    方差 = 和 / N（不是乘 1/N：5120 非 2 的幂，差一个 ulp）；rsqrt；输出 (x·inv)·(1+w)。M=1/3/17/48/192/4096 全 0 不等。
    其余写法在 M=4096 有 40–195 个归一化元素不等，说明这里的「逐位」靠的是把厂商的每一步舍入都对上，不是巧合。
耗时（图内，10-10）：M=4096 294 µs 对厂商 239+量化 = 340 µs；M=1–48 约 8 µs 对 9.6–9.8 µs。
只接管：Qwen3_5LinearDecoderLayer / Qwen3_5AttentionDecoderLayer 的两个 norm、带残差、bf16、hidden 5120、连续、无 post_residual_addition。
开关 SGLANG_NORM_QUANT=1。取证 SGLANG_NORM_QUANT_CHECK=1（图外调用同时跑原实现逐位比，慢）。
"""
import os
from collections import OrderedDict

import torch
import triton
import triton.language as tl

_st = {"run": 0, "hit": 0, "fb": set()}
_KQ = OrderedDict()          # data_ptr -> (out 张量, version, numel, q, s)
_KQ_MAX = 8
_CHECK = os.environ.get("SGLANG_NORM_QUANT_CHECK", "0") == "1"
_HID = 5120


@triton.jit
def _cols(v):                        # [T,8] → 8 个 [T] 列，寄存器内拆分
    a, b = tl.split(tl.reshape(v, [v.shape[0], 4, 2]))
    a0, a1 = tl.split(tl.reshape(a, [v.shape[0], 2, 2]))
    b0, b1 = tl.split(tl.reshape(b, [v.shape[0], 2, 2]))
    c0, c4 = tl.split(a0); c2, c6 = tl.split(a1); c1, c5 = tl.split(b0); c3, c7 = tl.split(b1)
    return c0, c1, c2, c3, c4, c5, c6, c7


@triton.jit
def _addrms_q(x_ptr, r_ptr, w_ptr, q_ptr, s_ptr, eps, N: tl.constexpr, T: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    NW: tl.constexpr = T // 64
    t = tl.arange(0, T)[:, None]
    msk = (t < N // 8) & (tl.arange(0, 8)[None, :] < 8)
    off = row * N + t * 8 + tl.arange(0, 8)[None, :]
    sh = (tl.load(x_ptr + off, mask=msk, other=0.0).to(tl.float32)
          + tl.load(r_ptr + off, mask=msk, other=0.0).to(tl.float32)).to(tl.bfloat16)
    tl.store(r_ptr + off, sh, mask=msk)
    sf = sh.to(tl.float32)
    c0, c1, c2, c3, c4, c5, c6, c7 = _cols(sf)
    acc = c0 * c0
    acc = tl.fma(c1, c1, acc); acc = tl.fma(c2, c2, acc); acc = tl.fma(c3, c3, acc)
    acc = tl.fma(c4, c4, acc); acc = tl.fma(c5, c5, acc); acc = tl.fma(c6, c6, acc); acc = tl.fma(c7, c7, acc)
    a2 = tl.reshape(acc, [NW, 64])
    for i in tl.static_range(6):                       # wave 内 xor 蝶形 32..1（reshape 逐级相加复现）
        a2 = tl.sum(tl.reshape(a2, [NW, 2, 32 >> i]), axis=1)
    a1 = tl.reshape(a2, [NW])
    for i in tl.static_range(4):                       # 16 个 wave（后 6 个为 0）：xor 8,4,2,1
        a1 = tl.sum(tl.reshape(a1, [2, (NW // 2) >> i]), axis=0)
    tot = tl.sum(a1, axis=0)
    inv = tl.extra.libdevice.rsqrt(tot / N + eps)
    w = tl.load(w_ptr + t * 8 + tl.arange(0, 8)[None, :], mask=msk, other=0.0).to(tl.float32)
    yh = ((sf * inv) * (1.0 + w)).to(tl.bfloat16)
    tl.store(x_ptr + off, yh, mask=msk)
    yf = yh.to(tl.float32)
    am = tl.maximum(tl.max(tl.max(tl.abs(yf), axis=1), axis=0), 1e-10)
    tl.store(q_ptr + off, tl.extra.libdevice.nearbyint(yf * (127 / am)).to(tl.int8), mask=msk)
    tl.store(s_ptr + row, am / 127)


def addrms_q(x, r, w, eps):
    """x、r 原地更新（同厂商 gemma_fused_add_rmsnorm），返回 (q int8 [M,N], s fp32 [M,1])。"""
    M, N = x.shape
    q = torch.empty((M, N), dtype=torch.int8, device=x.device)
    s = torch.empty((M, 1), dtype=torch.float32, device=x.device)
    if M:
        _addrms_q[(M,)](x, r, w, q, s, eps, N=N, T=1024, num_warps=8 if M <= 256 else 16)
    return q, s


def _lookup(x):
    ent = _KQ.get(x.data_ptr())
    if ent is None:
        return None
    out, ver, numel, q, s = ent
    if out._version != ver or x.numel() != numel or x.dtype != out.dtype or not x.is_contiguous():
        return None
    return q, s


def _wrap_quant(orig):
    def per_token_quant_int8(x, *a, **k):
        if not a and not k:
            hit = _lookup(x)
            if hit is not None:
                _st["hit"] += 1
                if _st["hit"] == 1:
                    print(f"[norm_quant] ✅ 量化复用首次真正生效：x {tuple(x.shape)} "
                          f"capturing={torch.cuda.is_current_stream_capturing()} pid={os.getpid()}", flush=True)
                q, s = hit
                if _CHECK and not torch.cuda.is_current_stream_capturing():
                    q0, s0 = orig(x)
                    nb = int((q0.view(-1) != q.view(-1)).sum()) + int((s0.view(-1) != s.view(-1)).sum())
                    _st["chk_q"] = _st.get("chk_q", 0) + 1
                    if nb:
                        _st["chk_q_bad"] = _st.get("chk_q_bad", 0) + 1
                        if _st["chk_q_bad"] <= 3:
                            print(f"[norm_quant] ❌ 取证：复用的量化与原实现不等 {nb} 处 x {tuple(x.shape)} pid={os.getpid()}", flush=True)
                    if _st["chk_q"] in (100, 1000, 10000):
                        print(f"[norm_quant] 取证：量化已比对 {_st['chk_q']} 次，不等 {_st.get('chk_q_bad', 0)} 次 pid={os.getpid()}", flush=True)
                return q.view(x.shape), s.view(*x.shape[:-1], 1)
        return orig(x, *a, **k)
    per_token_quant_int8._norm_quant_wrapped = True
    return per_token_quant_int8


def _mark(norm):
    orig = norm.forward

    def forward(x, residual=None, *a, **k):
        why = None
        if residual is None:
            why = "无残差"
        elif any(v is not None for v in a) or any(v is not None for v in k.values()):   # communicator 665 行按位置传 post_residual_addition=None
            why = "额外参数"
        elif x.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
            why = f"dtype {x.dtype}"
        elif x.dim() != 2 or x.shape[-1] != _HID or not x.is_contiguous() or not residual.is_contiguous():
            why = f"形状 {tuple(x.shape)}"
        if why:
            if why not in _st["fb"]:
                _st["fb"].add(why)
                print(f"[norm_quant] 不接管（{why}），走原实现 pid={os.getpid()}", flush=True)
            return orig(x, residual, *a, **k)
        chk = _CHECK and not torch.cuda.is_current_stream_capturing()
        if chk:
            x0, r0 = x.clone(), residual.clone()
        q, s = addrms_q(x, residual, norm.weight.data, norm.variance_epsilon)
        if chk:
            x1, r1 = orig(x0, r0)
            ne = lambda a, b: int(((a != b) & ~(torch.isnan(a) & torch.isnan(b))).sum())
            _st["chk_n"] = _st.get("chk_n", 0) + 1
            nb = ne(x1, x) + ne(r1, residual)
            if nb:
                _st["chk_n_bad"] = _st.get("chk_n_bad", 0) + 1
                if _st["chk_n_bad"] <= 3:
                    print(f"[norm_quant] ❌ 取证：融合 norm 与厂商不等 {nb} 处 x {tuple(x.shape)} pid={os.getpid()}", flush=True)
            if _st["chk_n"] in (100, 1000, 10000):
                print(f"[norm_quant] 取证：norm 已比对 {_st['chk_n']} 次，不等 {_st.get('chk_n_bad', 0)} 次 pid={os.getpid()}", flush=True)
        _KQ[x.data_ptr()] = (x, x._version, x.numel(), q, s)
        _KQ.move_to_end(x.data_ptr())
        while len(_KQ) > _KQ_MAX:
            _KQ.popitem(last=False)
        _st["run"] += 1
        if _st["run"] == 1:
            print(f"[norm_quant] ✅ norm 融合量化首次真正生效：x {tuple(x.shape)} "
                  f"capturing={torch.cuda.is_current_stream_capturing()} pid={os.getpid()}", flush=True)
        return x, residual

    norm.forward = forward


def install(mod):
    name = mod.__name__
    if name.endswith("w8a8_int8"):
        if not getattr(mod.per_token_quant_int8, "_norm_quant_wrapped", False):
            mod.per_token_quant_int8 = _wrap_quant(mod.per_token_quant_int8)
            print(f"[norm_quant] 已安装（量化复用，{mod.__name__}） pid={os.getpid()}", flush=True)
        return
    if not name.endswith("qwen3_5"):
        return
    n = 0
    for cname in ("Qwen3_5LinearDecoderLayer", "Qwen3_5AttentionDecoderLayer"):
        cls = getattr(mod, cname, None)
        if cls is None or getattr(cls, "_norm_quant_installed", False):
            continue
        orig_init = cls.__init__

        def __init__(self, *a, __orig=orig_init, **k):
            __orig(self, *a, **k)
            for nm in ("input_layernorm", "post_attention_layernorm"):
                _mark(getattr(self, nm))

        cls.__init__ = __init__
        cls._norm_quant_installed = True
        n += 1
    print(f"[norm_quant] 已安装（norm 融合量化，{n} 类解码层） pid={os.getpid()}", flush=True)

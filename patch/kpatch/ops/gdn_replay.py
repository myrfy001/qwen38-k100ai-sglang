"""GDN 投机验证：不写中间状态，接受长度出来后原地重放（替掉 4 份中间状态写 + 散射拷贝）。

原状（aiter fused_sigmoid_gating_delta_rule_update，target_verify）：每条请求每层读状态 1 次（fp32 HV×V×K，本模型每卡 1.5 MB），
      对 draft_token_num=4 个 token 逐步递推，每步把状态写进 intermediate_ssm（写 4 次）；接受后
      fused_mamba_state_scatter_with_mask 把第 accept_len-1 步的中间状态拷回状态池（再读 1 写 1）。合计 10.5 MB/请求/层。
改法：验证核（aiter 核逐行抄写）不写中间状态、不写回，只额外存下本层 k/v/a/b 原值（每 token 约 8 KB）；
      接受后一个重放核对全部层、每条请求从原状态出发只走前 accept_len 步，写回状态池。合计 4.5 MB/请求/层。
逐位：重放核的单步递推与验证核逐行相同、同 BV/num_warps（同归约次序），第 L 步状态与原方案中间状态逐位一致（探针核对）。"""
import os as _os

import triton
import triton.language as tl

_VS_PF = _os.environ.get("SGLANG_GDN_VS_PF", "0") == "1"


@triton.jit(do_not_specialize=["T"])
def _verify_save_kernel(
    A_log, a, dt_bias, softplus_beta, softplus_threshold,
    q, k, v, b, o, h0_source, h0_indices, cu_seqlens,
    sk, sv, sa, sb,                     # 保存缓冲（本层）：k [tok, H*K]、v [tok, HV*V]、a/b [tok, HV]
    scale, T, stride_q, stride_k, stride_v, stride_b,
    B: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr, SAVE: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    bos, eos = (
        tl.load(cu_seqlens + i_n).to(tl.int64),
        tl.load(cu_seqlens + i_n + 1).to(tl.int64),
    )
    all = T
    T = eos - bos

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_b = b + bos * stride_b + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v
    p_A_log = A_log + i_hv
    p_a = a + bos * HV + i_hv
    p_dt_bias = dt_bias + i_hv
    # 保存指针：k 每个 key 头只由 i_v==0、组内第一个 v 头的程序写；a/b 由 i_v==0 写；v 各写自己的 BV 段
    p_sk = sk + bos * (H * K) + i_h * K + o_k
    p_sv = sv + bos * (HV * V) + i_hv * V + o_v
    p_sa = sa + bos * HV + i_hv
    p_sb = sb + bos * HV + i_hv
    save_k = SAVE & (i_v == 0) & ((i_hv % (HV // H)) == 0)
    save_ab = SAVE & (i_v == 0)

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    idx = tl.load(h0_indices + i_n)
    if idx >= 0:
        p_h0 = h0_source + idx * HV * K * V + i_hv * K * V + o_v[None, :] * K + o_k[:, None]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    b_A = tl.exp(tl.load(p_A_log).to(tl.float32))
    b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

    for _ in range(0, T):
        b_k_raw = tl.load(p_k, mask=mask_k, other=0)
        b_k = b_k_raw.to(tl.float32)
        b_a_raw = tl.load(p_a)
        b_a = b_a_raw.to(tl.float32)
        x = b_a + b_dt_bias
        x_scaled = softplus_beta * x
        x = tl.where(
            x_scaled <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(x_scaled)),
            x,
        )
        b_g = -b_A * x
        if USE_QK_L2NORM_IN_KERNEL:
            b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        b_h *= tl.exp(b_g)
        b_v_raw = tl.load(p_v, mask=mask_v, other=0)
        b_v = b_v_raw.to(tl.float32)
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_b_raw = tl.load(p_b)
        b_v *= tl.sigmoid(b_b_raw.to(tl.float32))
        b_h += b_k[:, None] * b_v[None, :]
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
        b_q = b_q * scale
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        if SAVE:
            tl.store(p_sv, b_v_raw, mask=mask_v)
            if save_k:
                tl.store(p_sk, b_k_raw, mask=mask_k)
            if save_ab:
                tl.store(p_sa, b_a_raw)
                tl.store(p_sb, b_b_raw)
        p_q += stride_q
        p_k += stride_k
        p_v += stride_v
        p_b += stride_b
        p_o += HV * V
        p_a += HV
        p_sk += H * K
        p_sv += HV * V
        p_sa += HV
        p_sb += HV


@triton.jit
def _replay_kernel(
    A_log_all, dt_bias_all,              # 每层参数堆叠：[NL, HV]（与状态池层序一致）
    softplus_beta, softplus_threshold,
    sk, sv, sa, sb,                      # 保存缓冲（全部层）：[NL, TOKMAX, ...]
    st, st_layer_stride,                 # 状态池（全部层）：[NL, slots, HV, V, K]
    h0_indices, last_step,               # [N]：状态槽、最后接受步下标（= 接受长度 − 1，负数跳过；int32/int64 均可）
    tok_layer_stride, T_STEP,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    i_v, i_nh, i_l = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    L = tl.load(last_step + i_n).to(tl.int32) + 1
    idx = tl.load(h0_indices + i_n).to(tl.int64)
    if (L <= 0) | (idx < 0):
        return
    bos = i_n.to(tl.int64) * T_STEP

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    tb = i_l * tok_layer_stride + bos
    p_k = sk + tb * (H * K) + i_h * K + o_k
    p_v = sv + tb * (HV * V) + i_hv * V + o_v
    p_a = sa + tb * HV + i_hv
    p_b = sb + tb * HV + i_hv

    p_h = st + i_l * st_layer_stride + idx * HV * K * V + i_hv * K * V + o_v[None, :] * K + o_k[:, None]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    b_h += tl.load(p_h, mask=mask_h, other=0).to(tl.float32)

    b_A = tl.exp(tl.load(A_log_all + i_l * HV + i_hv).to(tl.float32))
    b_dt_bias = tl.load(dt_bias_all + i_l * HV + i_hv).to(tl.float32)

    for _ in range(0, L):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_a = tl.load(p_a).to(tl.float32)
        x = b_a + b_dt_bias
        x_scaled = softplus_beta * x
        x = tl.where(
            x_scaled <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(x_scaled)),
            x,
        )
        b_g = -b_A * x
        if USE_QK_L2NORM_IN_KERNEL:
            b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        b_h *= tl.exp(b_g)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= tl.sigmoid(tl.load(p_b).to(tl.float32))
        b_h += b_k[:, None] * b_v[None, :]
        p_k += H * K
        p_v += HV * V
        p_a += HV
        p_b += HV

    tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_h)


# 预取版验证核（10-10 晚 L13，probe/gdn_vs_pf.py）：每请求恰 4 个 token 时，4 步的 q/k/v/a/b 在读状态前一次发出，
# 逐步算术与 _verify_save_kernel 逐行相同 → 同 BV/num_warps 逐位（N=48/32/16 探针核对）；N=48 207→193 µs、N=16 92→81。
# 原核每步输入随递推串行等待；拆分实验：只读状态 119 µs（633 GB/s），其余是递推计算与读取未重叠。开关 SGLANG_GDN_VS_PF（默认 0，逐位复核与同实例对照通过前不开）。
@triton.jit
def _step(b_h, k_raw, a_raw, v_raw, b_raw, q_raw, b_A, b_dt_bias, softplus_beta, softplus_threshold, scale):
    b_k = k_raw.to(tl.float32)
    x = a_raw.to(tl.float32) + b_dt_bias
    x_scaled = softplus_beta * x
    x = tl.where(x_scaled <= softplus_threshold, (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(x_scaled)), x)
    b_g = -b_A * x
    b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
    b_h *= tl.exp(b_g)
    b_v = v_raw.to(tl.float32)
    b_v -= tl.sum(b_h * b_k[:, None], 0)
    b_v *= tl.sigmoid(b_raw.to(tl.float32))
    b_h += b_k[:, None] * b_v[None, :]
    b_q = q_raw.to(tl.float32)
    b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
    b_q = b_q * scale
    b_o = tl.sum(b_h * b_q[:, None], 0)
    return b_h, b_o


@triton.jit(do_not_specialize=["T"])
def _vs_pf(A_log, a, dt_bias, softplus_beta, softplus_threshold, q, k, v, b, o, h0_source, h0_indices, cu_seqlens,
           sk, sv, sa, sb, scale, T, stride_q, stride_k, stride_v, stride_b,
           H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
           SAVE: tl.constexpr, STATE_FIRST: tl.constexpr):
    i_v, i_nh = tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    all = T
    L = eos - bos
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_b = b + bos * stride_b + i_hv
    p_a = a + bos * HV + i_hv
    p_o = o + (bos * HV + i_hv) * V + o_v
    idx = tl.load(h0_indices + i_n)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    p_h0 = h0_source + idx.to(tl.int64) * HV * K * V + i_hv * K * V + o_v[None, :] * K + o_k[:, None]
    if STATE_FIRST:
        b_h += tl.load(p_h0, mask=(idx >= 0) & mask_k[:, None] & mask_v[None, :], other=0).to(tl.float32)
    m0 = L > 0
    m1 = L > 1
    m2 = L > 2
    m3 = L > 3
    k0 = tl.load(p_k, mask=mask_k & m0, other=0); k1 = tl.load(p_k + stride_k, mask=mask_k & m1, other=0)
    k2 = tl.load(p_k + 2 * stride_k, mask=mask_k & m2, other=0); k3 = tl.load(p_k + 3 * stride_k, mask=mask_k & m3, other=0)
    q0 = tl.load(p_q, mask=mask_k & m0, other=0); q1 = tl.load(p_q + stride_q, mask=mask_k & m1, other=0)
    q2 = tl.load(p_q + 2 * stride_q, mask=mask_k & m2, other=0); q3 = tl.load(p_q + 3 * stride_q, mask=mask_k & m3, other=0)
    v0 = tl.load(p_v, mask=mask_v & m0, other=0); v1 = tl.load(p_v + stride_v, mask=mask_v & m1, other=0)
    v2 = tl.load(p_v + 2 * stride_v, mask=mask_v & m2, other=0); v3 = tl.load(p_v + 3 * stride_v, mask=mask_v & m3, other=0)
    a0 = tl.load(p_a, mask=m0, other=0); a1 = tl.load(p_a + HV, mask=m1, other=0)
    a2 = tl.load(p_a + 2 * HV, mask=m2, other=0); a3 = tl.load(p_a + 3 * HV, mask=m3, other=0)
    bb0 = tl.load(p_b, mask=m0, other=0); bb1 = tl.load(p_b + stride_b, mask=m1, other=0)
    bb2 = tl.load(p_b + 2 * stride_b, mask=m2, other=0); bb3 = tl.load(p_b + 3 * stride_b, mask=m3, other=0)
    if not STATE_FIRST:
        b_h += tl.load(p_h0, mask=(idx >= 0) & mask_k[:, None] & mask_v[None, :], other=0).to(tl.float32)
    b_A = tl.exp(tl.load(A_log + i_hv).to(tl.float32))
    b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)
    if SAVE:
        save_k = (i_v == 0) & ((i_hv % (HV // H)) == 0)
        save_ab = i_v == 0
        p_sk = sk + bos * (H * K) + i_h * K + o_k
        p_sv = sv + bos * (HV * V) + i_hv * V + o_v
        tl.store(p_sv, v0, mask=mask_v & m0); tl.store(p_sv + HV * V, v1, mask=mask_v & m1)
        tl.store(p_sv + 2 * HV * V, v2, mask=mask_v & m2); tl.store(p_sv + 3 * HV * V, v3, mask=mask_v & m3)
        tl.store(p_sk, k0, mask=mask_k & m0 & save_k); tl.store(p_sk + H * K, k1, mask=mask_k & m1 & save_k)
        tl.store(p_sk + 2 * H * K, k2, mask=mask_k & m2 & save_k); tl.store(p_sk + 3 * H * K, k3, mask=mask_k & m3 & save_k)
        p_sa = sa + bos * HV + i_hv
        p_sb = sb + bos * HV + i_hv
        tl.store(p_sa, a0, mask=m0 & save_ab); tl.store(p_sa + HV, a1, mask=m1 & save_ab)
        tl.store(p_sa + 2 * HV, a2, mask=m2 & save_ab); tl.store(p_sa + 3 * HV, a3, mask=m3 & save_ab)
        tl.store(p_sb, bb0, mask=m0 & save_ab); tl.store(p_sb + HV, bb1, mask=m1 & save_ab)
        tl.store(p_sb + 2 * HV, bb2, mask=m2 & save_ab); tl.store(p_sb + 3 * HV, bb3, mask=m3 & save_ab)
    if L > 0:
        b_h, b_o = _step(b_h, k0, a0, v0, bb0, q0, b_A, b_dt_bias, softplus_beta, softplus_threshold, scale)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
    if L > 1:
        b_h, b_o = _step(b_h, k1, a1, v1, bb1, q1, b_A, b_dt_bias, softplus_beta, softplus_threshold, scale)
        tl.store(p_o + HV * V, b_o.to(p_o.dtype.element_ty), mask=mask_v)
    if L > 2:
        b_h, b_o = _step(b_h, k2, a2, v2, bb2, q2, b_A, b_dt_bias, softplus_beta, softplus_threshold, scale)
        tl.store(p_o + 2 * HV * V, b_o.to(p_o.dtype.element_ty), mask=mask_v)
    if L > 3:
        b_h, b_o = _step(b_h, k3, a3, v3, bb3, q3, b_A, b_dt_bias, softplus_beta, softplus_threshold, scale)
        tl.store(p_o + 3 * HV * V, b_o.to(p_o.dtype.element_ty), mask=mask_v)


def verify_save(A_log, a, dt_bias, q, k, v, b, state, idx, cu, sk, sv, sa, sb, cfg, scale=None, save=True,
                softplus_beta=1.0, softplus_threshold=20.0):
    """aiter target_verify 的等价调用（无中间状态、不写回）；save 时把 k/v/a/b 原值写进 sk/sv/sa/sb 的前 T 行。"""
    import torch
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = cu.numel() - 1
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), int(cfg["BV"]))
    NV = triton.cdiv(V, BV)
    if scale is None:
        scale = K ** -0.5
    o = q.new_empty(1, *v.shape)
    if _VS_PF and T == 4 * N and save:
        _vs_pf[(1, NV, N * HV)](
            A_log, a, dt_bias, softplus_beta, softplus_threshold, q, k, v, b, o, state, idx, cu, sk, sv, sa, sb,
            scale, T, q.stride()[1], k.stride()[1], v.stride()[1], b.stride()[-2],
            H=H, HV=HV, K=K, V=V, BK=BK, BV=BV, SAVE=True, STATE_FIRST=False,
            num_warps=int(cfg["num_warps"]), num_stages=1)
        _st["vs_pf"] = _st.get("vs_pf", 0) + 1
        if _st["vs_pf"] == 1:
            _log(f"✅ 预取版验证核首次真正生效：N={N} T={T} BV={BV} w={cfg['num_warps']}")
        return o.squeeze(0)
    _verify_save_kernel[(1, NV, N * HV)](
        A_log, a, dt_bias, softplus_beta, softplus_threshold,
        q, k, v, b, o, state, idx, cu, sk, sv, sa, sb,
        scale, T, q.stride()[1], k.stride()[1], v.stride()[1], b.stride()[-2],
        B=B, H=H, HV=HV, K=K, V=V, BK=BK, BV=BV, USE_QK_L2NORM_IN_KERNEL=True, SAVE=save,
        num_warps=int(cfg["num_warps"]), num_stages=1)
    return o.squeeze(0)


def replay(A_all, dt_all, sk, sv, sa, sb, st_all, idx, last_step, t_step, cfg, nl=None,
           softplus_beta=1.0, softplus_threshold=20.0):
    """全部层一次重放：st_all[l, idx[i]] ← 对保存的前 last_step[i]+1 个 token 递推。sk 等形状 [NL, TOKMAX, ...]。"""
    NL = st_all.shape[0] if nl is None else nl
    N = last_step.numel()
    HV, V, K = st_all.shape[-3:]
    H = sk.shape[-1] // K
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), int(cfg["BV"]))
    _replay_kernel[(triton.cdiv(V, BV), N * HV, NL)](
        A_all, dt_all, softplus_beta, softplus_threshold, sk, sv, sa, sb,
        st_all, st_all.stride(0), idx, last_step, sk.stride(0) // (H * K), t_step,
        H=H, HV=HV, K=K, V=V, BK=BK, BV=BV, USE_QK_L2NORM_IN_KERNEL=True,
        num_warps=int(cfg["num_warps"]), num_stages=1)


# 配置（10-10 probe/gdn_replay_sweep.py）：逐位与否只取决于 num_warps 是否为 1（K 维归约的线程划分），与 BV 无关。
# 生产 aiter 配置（gdn_cfg 表，T=N×4 取最近档）在 T≤32 是 w=1、更大是 w=4，新核跟随：同为 w=1 或同为 w≥2（BV8/w8 除外）即逐位。
def _prod_w1(n, H=8, HV=24, ts=4):
    try:
        from kpatch.ops import gdn_cfg
        c = gdn_cfg._pick(H, HV, n * ts)
    except Exception:
        c = None
    return int((c or {"num_warps": 1})["num_warps"]) == 1     # 无表时 aiter 默认 BV32/w1


def verify_cfg(n):
    if _prod_w1(n):
        return {"BV": 8 if n <= 2 else 32, "num_warps": 1}
    return {"BV": 64, "num_warps": 4}


def replay_cfg(n):
    if _prod_w1(n):
        return {"BV": 16, "num_warps": 1}
    return {"BV": 16, "num_warps": 4} if n >= 24 else {"BV": 8, "num_warps": 2}


# ---------------------------------------------------------------- 服务接入
import os

_OFF = "/tmp/q38_gdn_replay_off"
_st = {"verify": 0, "replay": 0}


def _log(msg):
    print(f"[gdn_replay] {msg} pid={os.getpid()}", flush=True)


def install(mod):
    """挂在 sglang.srt.layers.attention.linear.gdn_backend 上：
    1) GDNAttnBackend.__init__ 之后分配保存缓冲（图捕获前，常驻）；
    2) forward_extend 记下当前 backend，TritonGDNKernel.target_verify 换成 verify_save（写本层 k/v/a/b）；
    3) HybridLinearAttnBackend.update_mamba_state_after_mtp_verify 的 SSM 散射换成全层重放；卷积窗散射保留。
    任何不满足前提的情况一律报错（不静默回退：图里一旦混用两条路径，状态就错了）。"""
    import torch
    import sys
    if getattr(mod, "_gdn_replay_installed", False):
        return
    if os.path.exists(_OFF):
        _log(f"未安装：{_OFF} 存在")
        return
    from sglang.srt.layers.attention.linear.kernels import gdn_triton
    from sglang.srt.layers.attention import hybrid_linear_attn_backend as hlab
    G = mod.GDNAttnBackend
    cur = {"be": None}

    orig_init = G.__init__

    def __init__(self, model_runner, *a, **kw):
        orig_init(self, model_runner, *a, **kw)
        pool = self.req_to_token_pool
        spec = pool.get_speculative_mamba2_params_all_layers()
        temporal = spec.temporal
        NL, HV, V, K = temporal.shape[0], temporal.shape[2], temporal.shape[3], temporal.shape[4]
        nreq = spec.intermediate_conv_window[0].shape[1]     # intermediate_ssm 可能已被 gdn_replay_mem 换成占位
        ts = model_runner.server_args.speculative_num_draft_tokens
        tok = nreq * ts
        dev = temporal.device
        # H 由 q/k 头数定：本模型每卡 H=8（HV/H=3），首次 verify 时核对
        H = int(os.environ.get("SGLANG_GDN_REPLAY_H", "8"))
        self._rp = dict(
            NL=NL, H=H, HV=HV, K=K, V=V, ts=ts, tok=tok, temporal=temporal,
            sk=torch.zeros(NL, tok, H * K, device=dev, dtype=torch.bfloat16),
            sv=torch.zeros(NL, tok, HV * V, device=dev, dtype=torch.bfloat16),
            sa=torch.zeros(NL, tok, HV, device=dev, dtype=torch.bfloat16),
            sb=torch.zeros(NL, tok, HV, device=dev, dtype=torch.bfloat16),
            params={}, A_all=None, dt_all=None,
        )
        mb = sum(self._rp[x].numel() * 2 for x in ("sk", "sv", "sa", "sb")) / 2**20
        _log(f"保存缓冲已分配：{NL} 层 × {tok} token（{nreq} 请求 × {ts}），共 {mb:.0f} MiB；状态池 {tuple(temporal.shape)} {temporal.dtype}")

    orig_fe = G.forward_extend

    def forward_extend(self, *a, **kw):
        cur["be"] = self
        try:
            return orig_fe(self, *a, **kw)
        finally:
            cur["be"] = None

    def target_verify(self, A_log, dt_bias, q, k, v, a, b, *, ssm_states, cache_indices, query_start_loc,
                      intermediate_states_buffer=None, intermediate_state_indices=None, cache_steps=None,
                      retrieve_parent_token=None, **kwargs):
        be = cur["be"]
        rp = getattr(be, "_rp", None) if be is not None else None
        if rp is None:
            raise RuntimeError("[gdn_replay] target_verify 不在 GDNAttnBackend.forward_extend 内或缓冲未分配")
        if retrieve_parent_token is not None:
            raise RuntimeError("[gdn_replay] 只支持链式投机（topk=1）")
        temporal = rp["temporal"]
        if ssm_states.untyped_storage().data_ptr() != temporal.untyped_storage().data_ptr():
            raise RuntimeError("[gdn_replay] ssm_states 不是状态池的视图")
        li = (ssm_states.storage_offset() - temporal.storage_offset()) // temporal.stride(0)
        T = k.shape[1]
        if k.shape[2] != rp["H"] or v.shape[2] != rp["HV"] or T > rp["tok"] or cache_steps != rp["ts"]:
            raise RuntimeError(f"[gdn_replay] 形状不符 k{tuple(k.shape)} v{tuple(v.shape)} cache_steps={cache_steps} 缓冲 {rp['tok']}")
        for t, name in ((k, "k"), (v, "v"), (a, "a"), (b, "b")):
            if t.dtype != torch.bfloat16:
                raise RuntimeError(f"[gdn_replay] {name} dtype {t.dtype}（缓冲按 bf16 分配）")
        if not (a.is_contiguous() and b.stride(-1) == 1):
            raise RuntimeError(f"[gdn_replay] a/b 布局 a{a.stride()} b{b.stride()}")
        rp["params"].setdefault(li, (A_log, dt_bias))
        n = query_start_loc.numel() - 1
        o = verify_save(A_log, a, dt_bias, q, k, v, b, ssm_states, cache_indices, query_start_loc,
                        rp["sk"][li], rp["sv"][li], rp["sa"][li], rp["sb"][li], verify_cfg(n))
        _st["verify"] += 1
        if _st["verify"] == 1:
            _log(f"✅ 验证核首次真正生效：层 {li}/{rp['NL']} N={n} T={T} cfg={verify_cfg(n)} "
                 f"capturing={torch.cuda.is_current_stream_capturing()}")
        return o

    orig_upd = hlab.HybridLinearAttnBackend.update_mamba_state_after_mtp_verify

    def update_mamba_state_after_mtp_verify(self, last_correct_step_indices, mamba_track_indices, mamba_steps_to_track, model):
        be = self.linear_attn_backend
        rp = getattr(be, "_rp", None)
        if rp is None:
            return orig_upd(self, last_correct_step_indices, mamba_track_indices, mamba_steps_to_track, model)
        if mamba_track_indices is not None:
            raise RuntimeError("[gdn_replay] 前缀缓存跟踪（mamba_track_indices）需要中间状态，须 --disable-radix-cache")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("[gdn_replay] 重放不应在图捕获中")
        if rp["A_all"] is None:
            if len(rp["params"]) != rp["NL"]:
                raise RuntimeError(f"[gdn_replay] 只见到 {len(rp['params'])}/{rp['NL']} 层的参数")
            rp["A_all"] = torch.stack([rp["params"][i][0].reshape(-1) for i in range(rp["NL"])]).contiguous()
            rp["dt_all"] = torch.stack([rp["params"][i][1].reshape(-1) for i in range(rp["NL"])]).contiguous()
            _log(f"参数堆叠：A_log {tuple(rp['A_all'].shape)} {rp['A_all'].dtype}、dt_bias {tuple(rp['dt_all'].shape)} {rp['dt_all'].dtype}")
        n = last_correct_step_indices.shape[0]
        idx = be.forward_metadata.mamba_cache_indices[:n]
        replay(rp["A_all"], rp["dt_all"], rp["sk"], rp["sv"], rp["sa"], rp["sb"], rp["temporal"], idx,
               last_correct_step_indices, rp["ts"], replay_cfg(n))
        spec = be.req_to_token_pool.get_speculative_mamba2_params_all_layers()
        hlab.fused_mamba_state_scatter_with_mask(spec.conv[0], spec.intermediate_conv_window[0], idx, last_correct_step_indices)
        _st["replay"] += 1
        if _st["replay"] == 1:
            _log(f"✅ 重放首次真正生效：N={n} cfg={replay_cfg(n)} last_step dtype={last_correct_step_indices.dtype}")

    G.__init__ = __init__
    G.forward_extend = forward_extend
    gdn_triton.TritonGDNKernel.target_verify = target_verify
    hlab.HybridLinearAttnBackend.update_mamba_state_after_mtp_verify = update_mamba_state_after_mtp_verify
    mod._gdn_replay_installed = True
    _log(f"已安装到 {mod.__name__}（验证核不写中间状态 + 接受后全层重放；散射只留卷积窗）")

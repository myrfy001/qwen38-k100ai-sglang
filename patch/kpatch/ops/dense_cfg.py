"""稠密 int8 线性层按形状查调优表（Hy3 w8a8_int8）。

原路径：sglang W8A8Int8LinearMethod.apply → lightop.gemm_ops.triton_scaled_mm → lightop 汇编内核因 envs.py 白名单在 gfx928 上被关
（LMSLIM_USE_LIGHTOP 只对 gfx936/938 为 1）→ 回落 matmul_int8(config=None)：按 M 五档硬编码 tile，且每次先 torch.zeros 清零输出。
本补丁：按 (N, K) 与 M 查 tools/dense_tune_all.sh 调出的表（取不小于 M 的最近档），
  impl=lightop → lightop.op.gemm_w8_a8_smooth_colmajor（op.so 含 gfx928 代码对象，仅 M≤16）；
  impl=triton  → 同一个 lightop matmul_kernel 换 tile，SPLIT_K=1 时输出用 empty 不清零；
  impl=hip     → 自写 INT8 GEMM（kernels/i8gemm.hip，cfg 为编译宏串），与 matmul_int8 逐位一致（probe/i8test.py 全形状验过）。
另加 split-K 分支（SGLANG_DENSE_SPLITK=表路径，probe/dense_splitk.py 调出）：decode 小 M 时每 program 负责 BN 列 × K/S 段，int32 部分和写工作区，
  最后到达者（原子计数）归约后做与 lightop 相同的收尾 (acc.float()*sa)*sb；int32 加法精确 → 与生产逐位一致（探针逐形状逐 M 验过）。
  qkv M=24/48 2.6–2.8×、共享 gate_up 1.5–2.3×；表里记为 null 的档（生产为 lightop 汇编且无逐位候选）照旧查原表。
表里没有的 (N, K) 或 bias 非空一律走原实现；含 split-K>1 的形状整体拒用（lightop 该内核 split-K 在多流并发时算错，见 probes/dense_stream_probe.py）。开关 SGLANG_DENSE_CFG=表路径（空/0 关闭）。
"""
import bisect
import json
import os

import torch
import triton
import triton.language as tl

_st = {"logged": set(), "ws": {}, "pk": {}}
_BIG_OFF = "/tmp/dense_big_off"   # 存在则 M>32 不用大 M 档（09-27 生产表原只调到 M=32，预填充落到 M=32 小 tile）


@triton.jit
def _splitk_mm(a, sa, b, sb, c, ws, cnt, M, N, K: tl.constexpr, S: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pn = tl.program_id(0); ps = tl.program_id(1)
    rm = tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    KS: tl.constexpr = K // S
    acc = tl.zeros((BM, BN), tl.int32)
    for k0 in range(ps * KS, ps * KS + KS, BK):
        x = tl.load(a + rm[:, None] * K + (k0 + rk)[None, :], mask=rm[:, None] < M, other=0)
        w = tl.load(b + rn[None, :] * K + (k0 + rk)[:, None], mask=rn[None, :] < N, other=0)
        acc += tl.dot(x, w)
    if S > 1:
        tl.store(ws + ps * BM * N + rm[:, None] * N + rn[None, :], acc, mask=rn[None, :] < N)
        t = tl.atomic_add(cnt + pn, 1, sem="acq_rel")
        if t == S - 1:
            acc = tl.zeros((BM, BN), tl.int32)
            for s in range(S):
                acc += tl.load(ws + s * BM * N + rm[:, None] * N + rn[None, :], mask=rn[None, :] < N, other=0, cache_modifier=".cg")
            tl.store(cnt + pn, 0)
            xs = tl.load(sa + rm, mask=rm < M, other=0.); wsc = tl.load(sb + rn, mask=rn < N, other=0.)
            out = (acc.to(tl.float32) * xs[:, None] * wsc[None, :]).to(c.dtype.element_ty)
            tl.store(c + rm[:, None] * N + rn[None, :], out, mask=(rm[:, None] < M) & (rn[None, :] < N))
    else:
        xs = tl.load(sa + rm, mask=rm < M, other=0.); wsc = tl.load(sb + rn, mask=rn < N, other=0.)
        out = (acc.to(tl.float32) * xs[:, None] * wsc[None, :]).to(c.dtype.element_ty)
        tl.store(c + rm[:, None] * N + rn[None, :], out, mask=(rm[:, None] < M) & (rn[None, :] < N))


PBN, PBK = 32, 64


@triton.jit
def _pk_mm(a, sa, b, sb, c, ws, cnt, M, N, K: tl.constexpr, S: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
           PBN_: tl.constexpr, PBK_: tl.constexpr):
    # 与 _splitk_mm 相同，只是权重按 [N/32][K/64][32][64] 连续存放（风冷散列关，跳行读慢）；int32 累加 → 逐位一致（probe/dense_pack_int8.py）
    pn = tl.program_id(0); ps = tl.program_id(1)
    rm = tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    KS: tl.constexpr = K // S
    acc = tl.zeros((BM, BN), tl.int32)
    wn = ((rn // PBN_) * (PBN_ * K) + (rn % PBN_) * PBK_)[None, :]
    for k0 in range(ps * KS, ps * KS + KS, BK):
        kg = k0 + rk
        x = tl.load(a + rm[:, None] * K + kg[None, :], mask=rm[:, None] < M, other=0)
        w = tl.load(b + wn + ((kg // PBK_) * (PBN_ * PBK_) + kg % PBK_)[:, None])
        acc += tl.dot(x, w)
    if S > 1:
        tl.store(ws + ps * BM * N + rm[:, None] * N + rn[None, :], acc)
        t = tl.atomic_add(cnt + pn, 1, sem="acq_rel")
        if t == S - 1:
            acc = tl.zeros((BM, BN), tl.int32)
            for s in range(S):
                acc += tl.load(ws + s * BM * N + rm[:, None] * N + rn[None, :], cache_modifier=".cg")
            tl.store(cnt + pn, 0)
            xs = tl.load(sa + rm, mask=rm < M, other=0.); wsc = tl.load(sb + rn)
            out = (acc.to(tl.float32) * xs[:, None] * wsc[None, :]).to(c.dtype.element_ty)
            tl.store(c + rm[:, None] * N + rn[None, :], out, mask=rm[:, None] < M)
    else:
        xs = tl.load(sa + rm, mask=rm < M, other=0.); wsc = tl.load(sb + rn)
        out = (acc.to(tl.float32) * xs[:, None] * wsc[None, :]).to(c.dtype.element_ty)
        tl.store(c + rm[:, None] * N + rn[None, :], out, mask=rm[:, None] < M)


def _packed(b):
    """b 是 [K,N] 的权重转置视图（原权重 [N,K] 连续）。首次图外调用时另存打包副本；捕获中遇到未打包的返回 None。"""
    key = (b.data_ptr(), tuple(b.shape))
    ent = _st["pk"].get(key)
    if ent is not None:
        return ent[1]
    if torch.cuda.is_current_stream_capturing():
        return None
    K, N = b.shape
    wp = b.t().view(N // PBN, PBN, K // PBK, PBK).permute(0, 2, 1, 3).contiguous().view(-1)
    _st["pk"][key] = (b, wp)             # 持有 b，保证 data_ptr 不被复用
    _st["pk_bytes"] = _st.get("pk_bytes", 0) + wp.numel()
    return wp


def _splitk(a, sa, b, sb, out_dtype, cfg):
    M, K = a.shape; N = b.shape[1]; BN, BK, S, nw = cfg; BM = max(16, triton.next_power_of_2(M))
    key = (a.device.index, torch.cuda.current_stream().cuda_stream)
    buf = _st["ws"].get(key)
    if buf is None:
        if torch.cuda.is_current_stream_capturing():
            return None                                   # 工作区须在捕获前分配；此时退回原表
        buf = _st["ws"][key] = (torch.empty(8 * 64 * 4096, dtype=torch.int32, device=a.device),
                                torch.zeros(1024, dtype=torch.int32, device=a.device))
    assert S * BM * N <= buf[0].numel() and triton.cdiv(N, BN) <= buf[1].numel()
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)
    pcfg = _pk_cfg(N, K, M)
    if pcfg is not None:
        wp = _packed(b)
        if wp is not None:
            BN, BK, S, nw = pcfg
            assert S * BM * N <= buf[0].numel() and N // BN <= buf[1].numel()
            _pk_mm[(N // BN, S)](a, sa, wp, sb, c, buf[0], buf[1], M, N, K, S, BM, BN, BK, PBN, PBK, num_warps=nw)
            k2 = (N, K, "pack")
            if k2 not in _st["logged"]:
                _st["logged"].add(k2)
                print(f"[dense_cfg] ✅ 打包首次真正生效：N={N} K={K} M={M} cfg={pcfg} 累计副本 {_st['pk_bytes']/2**20:.0f} MiB "
                      f"capturing={torch.cuda.is_current_stream_capturing()}", flush=True)
            return c
        if "pk_cap" not in _st["logged"]:
            _st["logged"].add("pk_cap"); print("[dense_cfg] ⚠ 打包副本未就绪且在捕获中，本次走未打包 split-K", flush=True)
    _splitk_mm[(triton.cdiv(N, BN), S)](a, sa, b, sb, c, buf[0], buf[1], M, N, K, S, BM, BN, BK, num_warps=nw)
    return c


def _pk_cfg(N, K, M):
    """SGLANG_DENSE_PACK=表路径：{"N_K": [[maxM, [BN,BK,S,nw] | null], ...]}；只收探针里逐位一致且更快的档。"""
    tab = _st.get("pk_tab")
    if tab is None:
        tab = _st["pk_tab"] = {}
        pth = os.environ.get("SGLANG_DENSE_PACK", "")
        if pth and pth != "0":
            if not os.path.exists(pth):
                print(f"[dense_cfg] ❌ 打包表 {pth} 不存在，不启用打包", flush=True)
            else:
                for nk, ents in json.load(open(pth)).items():
                    n, k = map(int, nk.split("_")); tab[(n, k)] = sorted((int(m), c) for m, c in ents)
                print(f"[dense_cfg] 打包表已载入：{sorted(tab)}", flush=True)
    ents = tab.get((N, K))
    if not ents or M > ents[-1][0]:
        return None
    return next(c for m, c in ents if m >= M)


def install(mod):
    path = os.environ.get("SGLANG_DENSE_CFG", "")
    if getattr(mod, "_dense_cfg_installed", False) or not path:
        return
    if not os.path.exists(path):
        print(f"[dense_cfg] ❌ 未安装：表 {path} 不存在，走原实现", flush=True)
        return
    from lightop import op as lop
    from lightop._lmslim_native.layers.gemm.int8_utils import matmul_kernel
    table = {}
    for nk, rows in json.load(open(path)).items():
        n, k = map(int, nk.split("_"))
        ms = sorted(int(m) for m in rows)
        rs = [rows[str(m)] for m in ms]
        bad = [m for m, r in zip(ms, rs) if r["impl"] != "hip" and (r["cfg"][3] if r["impl"] == "lightop" else r["cfg"]["SPLIT_K"]) != 1]
        if bad:   # split-K>1：lightop 版多流并发会算错、Triton 版原子加不确定，均会拉低投机解码接受长度（09-25）
            print(f"[dense_cfg] ❌ 表中 {nk} 的 M={bad} 为 split-K>1，整个形状不启用，走原实现", flush=True)
            continue
        table[(n, k)] = (ms, rs)
    hip_libs = {}
    hip_defs = sorted({r["cfg"] for ms, rs in table.values() for r in rs if r["impl"] == "hip"})
    if hip_defs:                                   # 启动时就编译/加载，避免在图捕获或首个请求里编译
        from kpatch.ops import i8gemm_lib
        for d in hip_defs:
            try:
                hip_libs[d] = (i8gemm_lib.load(d), i8gemm_lib.tile(d))
            except Exception as e:
                print(f"[dense_cfg] ❌ 自写 INT8 GEMM 变体 [{d}] 编译/加载失败，该档回原实现：{type(e).__name__}: {e}", flush=True)
        print(f"[dense_cfg] 自写 INT8 GEMM 变体已加载 {len(hip_libs)}/{len(hip_defs)}：{hip_defs}", flush=True)
    sk_tab = {}
    sk_path = os.environ.get("SGLANG_DENSE_SPLITK", "")
    if sk_path and sk_path != "0":
        if not os.path.exists(sk_path):
            print(f"[dense_cfg] ❌ split-K 表 {sk_path} 不存在，不启用 split-K", flush=True)
        else:
            for nk, ents in json.load(open(sk_path)).items():     # {"N_K": [[maxM, [BN,BK,S,nw] | null], ...]}
                n, k = map(int, nk.split("_")); sk_tab[(n, k)] = sorted((int(m), c) for m, c in ents)
    orig = mod.quant_ops
    global _ORIG
    _ORIG = orig.triton_scaled_mm          # 原路径（lightop），自检用；注意生产原表路径另见 _Shim

    def _triton(a, sa, b, sb, out_dtype, cfg):
        M, K = a.shape; N = b.shape[1]
        sk = cfg["SPLIT_K"]
        c = torch.empty((M, N), device=a.device, dtype=out_dtype) if sk == 1 else torch.zeros((M, N), device=a.device, dtype=torch.float32)
        grid = (triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]), sk)
        matmul_kernel[grid](a, sa, b, sb, c, M, N, K, a.stride(0), a.stride(1), sa.stride(0), b.stride(0), b.stride(1),
                            sb.stride(0), c.stride(0), c.stride(1), **cfg)
        return c if sk == 1 else c.to(out_dtype)

    class _Shim:
        def __getattr__(self, name):
            return getattr(orig, name)

        @staticmethod
        def triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=None, best_config=None):
            ske = sk_tab.get((b.shape[1], a.shape[1]))
            if ske is not None and bias is None and best_config is None and a.shape[0] <= ske[-1][0] and a.dtype == torch.int8 \
                    and a.is_contiguous() and b.stride(0) == 1 and b.stride(1) == a.shape[1] and scale_a.is_contiguous() \
                    and scale_a.numel() == a.shape[0] and scale_b.is_contiguous() and scale_b.numel() == b.shape[1]:
                cfg = next(c for m, c in ske if m >= a.shape[0])
                if cfg is not None:
                    out = _splitk(a, scale_a, b, scale_b, out_dtype, cfg)
                    if out is not None:
                        key = (b.shape[1], a.shape[1], "splitk")
                        if key not in _st["logged"]:
                            _st["logged"].add(key)
                            print(f"[dense_cfg] ✅ split-K 首次真正生效：N={key[0]} K={key[1]} M={a.shape[0]} cfg={cfg} "
                                  f"capturing={torch.cuda.is_current_stream_capturing()}", flush=True)
                        return out
                    if "sk_cap" not in _st["logged"]:
                        _st["logged"].add("sk_cap"); print("[dense_cfg] split-K 工作区未就绪且在捕获中，退回原表", flush=True)
            ent = table.get((b.shape[1], a.shape[1]))
            if ent is None or bias is not None or best_config is not None:
                return orig.triton_scaled_mm(a, b, scale_a, scale_b, out_dtype=out_dtype, bias=bias, best_config=best_config)
            M = a.shape[0]; ms, rows = ent
            if M > 32 and os.path.exists(_BIG_OFF):   # 09-27 运行时对照：大 M 退回旧行为（落到 ≤32 的最后一档）
                ms = [m for m in ms if m <= 32]
            i = min(bisect.bisect_left(ms, M), len(ms) - 1)
            r = rows[i]
            if ms[i] > 32 and ("big", b.shape[1]) not in _st["logged"]:
                _st["logged"].add(("big", b.shape[1]))
                print(f"[dense_cfg] ✅ 大 M 档首次真正生效：N={b.shape[1]} K={a.shape[1]} M={M} → 表档 {ms[i]} {r['cfg']}", flush=True)
            if r["impl"] == "hip":
                N, K = b.shape[1], a.shape[1]
                hl = hip_libs.get(r["cfg"])
                ok = (hl is not None and out_dtype == torch.bfloat16 and a.dtype == torch.int8 and a.is_contiguous()
                      and b.stride(0) == 1 and b.stride(1) == K and N % hl[1][1] == 0 and K % 128 == 0
                      and scale_a.dtype == torch.float32 and scale_a.is_contiguous() and scale_a.numel() == M
                      and scale_b.dtype == torch.float32 and scale_b.is_contiguous() and scale_b.numel() == N)
                if not ok:
                    if ("hipbad", N, K) not in _st["logged"]:
                        _st["logged"].add(("hipbad", N, K))
                        print(f"[dense_cfg] ⚠ 自写 GEMM 前提不满足，回原实现：N={N} K={K} M={M} a{tuple(a.stride())}{a.dtype} "
                              f"b{tuple(b.stride())} sa{scale_a.dtype}{tuple(scale_a.shape)} sb{scale_b.dtype}{tuple(scale_b.shape)} out{out_dtype}", flush=True)
                    return orig.triton_scaled_mm(a, b, scale_a, scale_b, out_dtype=out_dtype, bias=bias)
                from kpatch.ops import i8gemm_lib
                out = i8gemm_lib.gemm(hl[0], a, b.data_ptr(), scale_a, scale_b,
                                      torch.empty((M, N), device=a.device, dtype=out_dtype), M, N, K)
            elif r["impl"] == "lightop" and M <= 16:
                out = lop.gemm_w8_a8_smooth_colmajor(a, b, scale_a, scale_b, None, out_dtype, *r["cfg"])
            elif r["impl"] == "triton":
                out = _triton(a, scale_a, b, scale_b, out_dtype, r["cfg"])
            else:                                  # lightop 档只调在 M≤16；M 超出表的最大 lightop 档时回原实现
                return orig.triton_scaled_mm(a, b, scale_a, scale_b, out_dtype=out_dtype, bias=bias)
            if _st.get("check", 0) > 0 and not torch.cuda.is_current_stream_capturing():
                _st["check"] -= 1
                ref = orig.triton_scaled_mm(a, b, scale_a, scale_b, out_dtype=out_dtype, bias=None)
                rel = ((out.float() - ref.float()).norm() / (ref.float().norm() + 1e-12)).item()
                print(f"[dense_cfg] 自检 N={b.shape[1]} K={a.shape[1]} M={M} {r['impl']} 相对差={rel:.2e} 最大差={(out.float()-ref.float()).abs().max().item():.2e} "
                      f"a{tuple(a.stride())}{a.dtype} b{tuple(b.stride())}{b.dtype} sa{tuple(scale_a.shape)}{tuple(scale_a.stride())}{scale_a.dtype} "
                      f"sb{tuple(scale_b.shape)}{tuple(scale_b.stride())}{scale_b.dtype} out{out.dtype}", flush=True)
            key = (b.shape[1], a.shape[1], r["impl"])
            if key not in _st["logged"]:
                _st["logged"].add(key)
                print(f"[dense_cfg] ✅ 首次真正生效：N={key[0]} K={key[1]} M={M} → {r['impl']} {r['cfg']}", flush=True)
            return out

    _st["check"] = int(os.environ.get("SGLANG_DENSE_CHECK", "0"))   # 前 N 次图外调用同时跑原实现并打印差异（取证用）
    mod.quant_ops = _Shim()
    mod._dense_cfg_installed = True
    print(f"[dense_cfg] 已安装：{len(table)} 个 (N,K) 形状，表 {path}；split-K {sorted(sk_tab)} 表 {sk_path or '关'}", flush=True)

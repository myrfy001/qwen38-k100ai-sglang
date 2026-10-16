"""LM-head 逐 M 选 rocBLAS 解（直调 rocblas_gemm_ex，solution_index），替代 TunableOp。

依据（10-09，tools/lmh2.sh 两步法，[M,5120]×[124160,5120]ᵀ bf16，图内计时、分进程对照）：
  默认启发式 M=1–16 约 3.5–4.0 ms、M=34–64 8.6–12.6 ms、M≥100 22–32 ms；
  逐 M 最快解 1.5–1.8 / 3.7 / 4.0–4.5 ms（2.2–7.3×），argmax 一致率与默认相同。
为什么不用 TunableOp：它是全局开关，开了以后其余 bf16 GEMM（GDN 的 N=48 投影等）也改走它的默认路径，
  b5_lmh 实测 c1 TPOT 14.1→15.0 ms。本补丁只接管 LM-head 一处。
表：SGLANG_LMH_TABLE=<CSV>（TunableOp 格式，取「tn_N_M_K…,Gemm_Rocblas_<解号>」），只接管表内 (N,K,M)；其余原路径。
.so：SGLANG_LMH_SO（默认 /tmp/q38_lmh.so，不存在则用 kernels/lmh_rocblas.cpp 现编）。
  文件名不能含 "rocblas"：rocBLAS 初始化时在已加载库里按路径找自己来定位 Tensile 库，
  先于它加载的 /tmp/lmh_rocblas.so 会被当成它，去读 /tmp/rocblas/library_gpu4/ 而崩（10-09 b7/b8）。
开关：SGLANG_LMH_TABLE 非空即开。
自写核（底层专项 L1，kernels/lmh_gemm.hip v2/v3）：SGLANG_LMH_OWN_TABLE=<CSV：M,ver,kc,pad,…>（probe/lmh_own_tune.py 产出，
  只收「图内比 rocBLAS 解快 ≥5%、误差不大于 torch 默认、argmax 一致」的 M），表内 M 先走自写核，其余仍走 rocBLAS 解。
  .so：SGLANG_LMH_OWN_SO（默认 /tmp/q38_lmh_own2.so，不存在则现编）。"""
import ctypes
import os
import re
import subprocess

import torch

_TAB = {}      # (N, K) -> {M: sol}
_tp = os.environ.get("SGLANG_LMH_TABLE", "")
if _tp and os.path.exists(_tp):
    for line in open(_tp):
        m = re.match(r"GemmTunableOp_BFloat16_TN,tn_(\d+)_(\d+)_(\d+)_ld_\d+_\d+_\d+,Gemm_Rocblas_(\d+)", line)
        if m:
            N, M, K, s = map(int, m.groups())
            _TAB.setdefault((N, K), {})[M] = s
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "kernels", "lmh_rocblas.cpp")
_SO = os.environ.get("SGLANG_LMH_SO", "/tmp/q38_lmh.so")
_lib = None
_seen = set()
_OWN = {}      # M -> (ver, kc, pad)
_otp = os.environ.get("SGLANG_LMH_OWN_TABLE", "")
if _otp and os.path.exists(_otp):
    for line in open(_otp):
        f = line.strip().split(",")
        if len(f) >= 4 and f[0].isdigit():
            _OWN[int(f[0])] = (int(f[1]), int(f[2]), int(f[3]))
_OSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "kernels", "lmh_gemm.hip")
_OSO = os.environ.get("SGLANG_LMH_OWN_SO", "/tmp/q38_lmh_own2.so")
_olib = None
_oseen = set()


def _load_own():
    global _olib
    if _olib is None:
        if not os.path.exists(_OSO):
            tmp = f"{_OSO}.{os.getpid()}.tmp"
            subprocess.run(["hipcc", "-O3", "-shared", "-fPIC", "--offload-arch=gfx928", "-o", tmp, _OSRC], check=True)
            os.replace(tmp, _OSO)
        _olib = ctypes.CDLL(_OSO)
        for f in ("lmh_gemm_own2", "lmh_gemm_own3"):
            getattr(_olib, f).argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
            getattr(_olib, f).restype = ctypes.c_int
        _olib.lmh_set_pad.argtypes = [ctypes.c_int]
    return _olib


def _load():
    global _lib
    if _lib is None:
        if not os.path.exists(_SO):
            tmp = f"{_SO}.{os.getpid()}.tmp"
            subprocess.run(["hipcc", "-O2", "-shared", "-fPIC", "-o", tmp, _SRC, "-lrocblas"], check=True)
            os.replace(tmp, _SO)
        _lib = ctypes.CDLL(_SO)
        _lib.lmh_gemm.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        _lib.lmh_gemm.restype = ctypes.c_int
    return _lib


def lm_head_mm(x, w):
    """x [M,K] bf16 连续，w [N,K] bf16 连续；命中表返回 [M,N]，否则 None。"""
    N, K = w.shape
    tab = _TAB.get((N, K))
    if tab is None or x.dim() != 2 or x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        return None
    M = x.shape[0]
    own = _OWN.get(M) if N == 124160 and K == 5120 else None
    if own is not None:
        ver, kc, pad = own
        x = x.contiguous()
        out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
        lib = _load_own()
        lib.lmh_set_pad(pad)
        fn = lib.lmh_gemm_own3 if ver == 3 else lib.lmh_gemm_own2
        r = fn(x.data_ptr(), w.data_ptr(), out.data_ptr(), M, N, K, kc, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        if r != 0:
            raise RuntimeError(f"[lmh_rocblas] 自写核返回 {r}（M={M} v{ver} KC={kc}）")
        if M not in _oseen:
            _oseen.add(M)
            if len(_oseen) <= 8:
                print(f"[lmh_rocblas] ✅ 自写核首次真正生效 M={M} v{ver} KC={kc} pad={pad} "
                      f"capturing={torch.cuda.is_current_stream_capturing()} pid={os.getpid()}", flush=True)
        return out
    sol = tab.get(M)
    if sol is None:
        return None
    x = x.contiguous()
    out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    r = _load().lmh_gemm(x.data_ptr(), w.data_ptr(), out.data_ptr(), M, N, K, sol,
                         ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if r != 0:
        raise RuntimeError(f"[lmh_rocblas] rocblas_gemm_ex 返回 {r}（M={M} N={N} K={K} sol={sol}）")
    if M not in _seen:
        _seen.add(M)
        if len(_seen) <= 8:
            print(f"[lmh_rocblas] ✅ 首次真正生效 M={M} N={N} K={K} sol={sol} pid={os.getpid()}", flush=True)
    return out


def install(mod):
    if getattr(mod, "_lmh_rocblas_installed", False) or not _TAB:
        if not _TAB:
            print(f"[lmh_rocblas] ⚠ 表为空（SGLANG_LMH_TABLE={_tp!r}），不安装", flush=True)
        return
    LP = mod.LogitsProcessor
    orig = LP._compute_lm_head

    def _compute_lm_head(self, hidden_states, lm_head, embedding_bias=None):
        w = getattr(lm_head, "weight", None)
        if (w is not None and not hasattr(lm_head, "set_lora") and not self.use_fp32_lm_head
                and w.is_contiguous()):
            y = lm_head_mm(hidden_states.to(w.dtype), w)
            if y is not None:
                return y
        return orig(self, hidden_states, lm_head, embedding_bias)

    LP._compute_lm_head = _compute_lm_head
    _load()
    if _OWN:
        _load_own()
        print(f"[lmh_rocblas] 自写核表 {_otp}：{len(_OWN)} 个 M（{min(_OWN)}–{max(_OWN)}）pid={os.getpid()}", flush=True)
    mod._lmh_rocblas_installed = True
    print(f"[lmh_rocblas] 已安装到 {mod.__name__} pid={os.getpid()}，表 {_tp}：{ {k: len(v) for k, v in _TAB.items()} }", flush=True)

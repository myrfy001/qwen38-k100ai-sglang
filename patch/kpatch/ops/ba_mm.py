"""ba_mm: GDN 的 in_proj_ba（bf16 不量化，[M,5120]×[48,5120]ᵀ，每卡 b/a 各 24 头）按 M 段直调 rocBLAS 解。

发现（10-12 l9 账）：预填充 4K 每轮 11.6 ms（约 1%），Tensile MT64x64x32，读 x 只到 170–290 GB/s；
    decode 每轮 48 层各一次（M=4·bs），默认启发式 22–31 µs，远高于读 0.5 MB 权重的下限。
选解（probe/ba_tune.py，TunableOp 在 M=128/2432/4096/8192/16384 出候选 11394–11398，再逐 M 直调图内计时，卡 0）：
    M≤384 用 11394：M=1–96 22.2–25.9 → 11.0–11.2 µs（2.0–2.3×），M=192 30.6→12.7；
    M≤2560 用 11395：M=512 34.7→18.6，M=1024 54.2→29.7，M=2176 130.6→60.6；
    其余用 11397：M=4096 143.6→111.0，M=4352 237.3→112.3，M=8192 305.7→198.3，M=16384 720.9→354.3（2.0×）。
不逐位：每个解与默认约 0.03% 元素差 1 ulp（M=4096 共 196608 个输出中 57–74 个），对 fp32 的相对误差与默认相同（2.2–2.4e-3）。
接入：Qwen3_5GatedDeltaNet 构造后把 in_proj_ba.quant_method 包一层（只这一个层，其余不量化层不动）；
    复用 lmh_rocblas 的 rocblas_gemm_ex 直调 .so（SGLANG_LMH_SO）。条件不满足（有 bias、非 bf16、形状不符）走原实现并打原因日志。
开关 SGLANG_BA_MM=<规则>，形如 "384:11394,2560:11395,0:11397"（M≤上限用该解，0 表示其余）；空或 0 关闭。
"""
import ctypes
import os

import torch

_RAW = os.environ.get("SGLANG_BA_MM", "")
_RULE = []
for _p in _RAW.split(","):
    if ":" in _p:
        _a, _b = _p.split(":")
        _RULE.append((int(_a) or 1 << 30, int(_b)))
_RULE.sort()
_st = {"seen": set(), "fb": set()}


def _pick(M):
    for lim, sol in _RULE:
        if M <= lim:
            return sol
    return None


class _BaQM:
    """包住原不量化方法（UnquantizedEmbeddingMethod 或 UnquantizedLinearMethod）；只改 apply，其余属性（process_weights_after_loading 等）原样转发。"""

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, k):
        return getattr(self.__dict__["inner"], k)

    def apply(self, layer, x, bias=None):
        w = layer.weight
        why = None
        if bias is not None:
            why = "有 bias"
        elif x.dim() != 2 or x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
            why = f"x {tuple(x.shape)} {x.dtype} w {w.dtype}"
        elif w.dim() != 2 or w.shape[1] != x.shape[1] or not w.is_contiguous():
            why = f"w {tuple(w.shape)} 与 x {tuple(x.shape)} 不符或非连续"
        elif x.shape[0] == 0:
            why = "空批"
        sol = None if why else _pick(x.shape[0])
        if sol is None:
            why = why or f"M={x.shape[0]} 无规则"
            if why not in _st["fb"] and len(_st["fb"]) < 8:
                _st["fb"].add(why)
                print(f"[ba_mm] 不接管（{why}），走原实现 pid={os.getpid()}", flush=True)
            return self.inner.apply(layer, x, bias)
        from kpatch.ops import lmh_rocblas as _L
        M, N, K = x.shape[0], w.shape[0], w.shape[1]
        x = x.contiguous()
        out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
        r = _L._load().lmh_gemm(x.data_ptr(), w.data_ptr(), out.data_ptr(), M, N, K, sol,
                                ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        if r != 0:
            raise RuntimeError(f"[ba_mm] rocblas_gemm_ex 返回 {r}（M={M} N={N} K={K} sol={sol}）")
        if sol not in _st["seen"]:
            _st["seen"].add(sol)
            print(f"[ba_mm] ✅ 首次真正生效 M={M} N={N} K={K} sol={sol} "
                  f"capturing={torch.cuda.is_current_stream_capturing()} pid={os.getpid()}", flush=True)
        return out


def install(mod):
    cls = getattr(mod, "Qwen3_5GatedDeltaNet", None)
    if cls is None or getattr(cls, "_ba_mm_installed", False):
        return
    if not _RULE:
        print(f"[ba_mm] ⚠ 规则为空（SGLANG_BA_MM={_RAW!r}），不安装", flush=True)
        return
    orig = cls.__init__

    def __init__(self, *a, **k):
        orig(self, *a, **k)
        lin = getattr(self, "in_proj_ba", None)
        qm = getattr(lin, "quant_method", None)
        # w8a8_int8 对不量化层返回 UnquantizedEmbeddingMethod（apply 就是 F.linear），10-12 首臂因只认 Linear 未接管
        if type(qm).__name__ in ("UnquantizedLinearMethod", "UnquantizedEmbeddingMethod"):
            lin.quant_method = _BaQM(qm)
        elif not isinstance(qm, _BaQM) and "qm" not in _st["fb"]:
            _st["fb"].add("qm")
            print(f"[ba_mm] 不接管（in_proj_ba 量化方法 {type(qm).__name__}） pid={os.getpid()}", flush=True)

    cls.__init__ = __init__
    cls._ba_mm_installed = True
    print(f"[ba_mm] 已安装（Qwen3_5GatedDeltaNet.in_proj_ba 按 M 选解 {_RULE}） pid={os.getpid()}", flush=True)

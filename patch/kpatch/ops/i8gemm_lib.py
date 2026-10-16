"""自写 INT8 GEMM（kernels/i8gemm.hip）的编译与加载。dense_cfg（服务）与 probe/i8test.py（离线）共用。

变体由编译宏串决定（如 "DBUF=0 WN=4 LBOUND=512 PAD=8"）；.so 按「源码 + 宏」的 md5 缓存在 SGLANG_I8GEMM_DIR（默认 /tmp/i8gemm），
多个 TP 进程同时编译时各写临时文件再原子改名，不会互相覆盖。发射只做异步内核调用（ctypes → 当前流），可被 CUDA graph 捕获。"""
import ctypes
import hashlib
import os
import subprocess

import torch

SRC = os.environ.get("SGLANG_I8GEMM_SRC",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "kernels", "i8gemm.hip"))
OUT = os.environ.get("SGLANG_I8GEMM_DIR", "/tmp/i8gemm")
_libs = {}


def so_path(defs):
    flags = [f"-D{d}" for d in defs.split()]
    tag = hashlib.md5((open(SRC).read() + " ".join(flags)).encode()).hexdigest()[:12]
    return os.path.join(OUT, f"i8gemm_{tag}.so"), flags


def load(defs):
    lib = _libs.get(defs)
    if lib is not None:
        return lib
    so, flags = so_path(defs)
    if not os.path.exists(so):
        os.makedirs(OUT, exist_ok=True)
        tmp = f"{so}.{os.getpid()}.tmp"
        subprocess.run(["hipcc", "-O3", "-shared", "-fPIC", "--offload-arch=gfx928", *flags, "-o", tmp, SRC], check=True)
        os.replace(tmp, so)
    lib = ctypes.CDLL(so)
    lib.i8gemm_launch.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    lib.i8gemm_launch.restype = ctypes.c_int
    lib.i8gemm_tile.argtypes = [ctypes.POINTER(ctypes.c_int)] * 2
    _libs[defs] = lib
    return lib


def tile(defs):
    bm, bn = ctypes.c_int(), ctypes.c_int()
    load(defs).i8gemm_tile(ctypes.byref(bm), ctypes.byref(bn))
    return bm.value, bn.value


def gemm(lib, a, w_nk_ptr, sa, sb, out, M, N, K):
    """a [M,K] int8 连续；w_nk_ptr 指向 [N,K] 行主序 int8；sa [M] / sb [N] float32 连续；out [M,N] bf16 连续。"""
    r = lib.i8gemm_launch(a.data_ptr(), w_nk_ptr, sa.data_ptr(), sb.data_ptr(), out.data_ptr(), M, N, K,
                          ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if r != 0:
        raise RuntimeError(f"i8gemm_launch 返回 {r}（M={M} N={N} K={K}）")
    return out

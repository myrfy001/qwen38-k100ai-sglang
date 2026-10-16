"""GDN 预填充因果卷积（sglang causal_conv1d_triton._causal_conv1d_fwd_kernel）按块长选启动参数。

为什么：sglang 把启动参数写死为 BLOCK_M=8、BLOCK_N=256（num_warps 默认 4）：每个程序只做 8 个 token、每个 token 读 512 B，
16384 token 的块要起上万个程序、每个读 8 行彼此相隔一整行的小段。风冷 K100_AI（DF 地址散列未开）实测只有 29–39 GB/s：
T=16384 每次 4.27 ms（预填充总账里 32768 输入该核 4.35 ms/次，对得上），按数据量应约 0.26 ms。
实测（离线探针，每卡维度 2048、核宽 4，与生产同布局；全部配置与默认逐位相同——每个输出元素的累加顺序与分块无关）：
    T=2432 274→206 µs（1.33×）  T=4096 521→251（2.08×）  T=8192 1724→368（4.7×）  T=16384 4269→691（6.2×）  T=32768 9303→1284（7.2×）
    T≤1024 无差别（Python 开销主导）
做法：不改 sglang 源码、不复制其函数，把模块里的内核对象换成代理：kernel[grid](...) 时按总 token 数（参数 cu_seqlen）改写
BLOCK_M / BLOCK_N / num_warps 三个关键字参数再交给原内核；sglang 的 grid 函数按这组参数算网格，自动一致。
开关：SGLANG_CONV1D_CFG=1 开（默认关）。"""
import os

_enabled = os.environ.get("SGLANG_CONV1D_CFG", "0") != "0"
# (总 token 数下限, BLOCK_M, BLOCK_N, num_warps)，从大到小匹配；低于最小档不改
_TABLE = [(24576, 128, 1024, 2), (6144, 64, 1024, 2), (2048, 16, 512, 4)]
_seen = {"run": False}


class _KernelProxy:
    def __init__(self, k):
        self._k = k

    def __getitem__(self, grid):
        launch = self._k[grid]

        def run(*args, **kw):
            # 位置参数顺序：x, weight, bias, conv_states, cache_indices, has_initial_state, query_start_loc, out, dim, cu_seqlen, ...
            try:
                T = int(args[9]) if len(args) > 9 else int(kw.get("cu_seqlen", 0))
                dim = int(args[8]) if len(args) > 8 else int(kw.get("dim", 0))
            except Exception:
                T, dim = 0, 0
            for lo, bm, bn, nw in _TABLE:
                if T >= lo and bn <= dim:
                    kw.update(BLOCK_M=bm, BLOCK_N=bn, num_warps=nw)
                    if not _seen["run"]:
                        _seen["run"] = True
                        print(f"[conv1d_cfg] ✅ 首次真正生效：T={T} dim={dim} → BLOCK_M={bm} BLOCK_N={bn} num_warps={nw} pid={os.getpid()}", flush=True)
                    break
            return launch(*args, **kw)
        return run

    def __getattr__(self, name):
        return getattr(self._k, name)


def install(mod):
    if not _enabled or getattr(mod, "_conv1d_cfg_installed", False) or not hasattr(mod, "_causal_conv1d_fwd_kernel"):
        return
    mod._conv1d_cfg_installed = True
    mod._causal_conv1d_fwd_kernel = _KernelProxy(mod._causal_conv1d_fwd_kernel)
    print(f"[conv1d_cfg] 已安装：分档 {_TABLE}（总 token 数下限, BLOCK_M, BLOCK_N, num_warps） pid={os.getpid()}", flush=True)

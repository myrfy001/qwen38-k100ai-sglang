"""GDN 预填充两个内核补上 gfx928 启动参数（只改 warps/stages/分块，逐位一致）。

10-08 卡 6 扫参（27B 每卡 Hg=8 H=24 K=V=128 BT=64，64K 单请求剖析里两者合计约 3.4 s / 26.8 s）：
  recompute_w_u_fwd_kernel（sglang wy_fast，写死 BK64/BV64/w4/s3）：
      T=8192 2326→1172 us，T=16384 9069→4001 us，取 BK32/BV32/w1/s1，逐位一致
  chunk_fwd_kernel_o（aiter，只带 gfx936/938 配置，gfx928 回落 BK128/BV64/w4/s2）：
      T=16384 6763→4473 us，取 BK128/BV32/w8/s2，逐位一致（BK<128 会改累加顺序，不用）
开关 SGLANG_GDN_PF_CFG=1。
"""
import json
import os

_WU = {"BK": 32, "BV": 32, "num_warps": 1, "num_stages": 1}
_CO = {"BK": 128, "BV": 32, "num_warps": 8, "num_stages": 2}
# 10-09：SGLANG_GDN_TABLE（probe/gdn_aiter_sweep.py 的 JSON）有 "o" 段时，取最大 T 档的逐位最优覆盖 _CO
_tp = os.environ.get("SGLANG_GDN_TABLE", "")
if _tp and os.path.exists(_tp):
    _o = json.load(open(_tp)).get("o", {}).get("K=128,V=128,BT=64")
    if _o:
        _CO = _o[max(_o, key=int)]


class _WUShim:
    """替换 wy_fast 模块里的 recompute_w_u_fwd_kernel 全局名：包装函数按全局名取内核，替换即生效。"""

    _seen = False

    def __init__(self, k):
        self._k = k

    def __getitem__(self, grid):
        launch = self._k[grid]

        def run(*a, **kw):
            if kw.get("K") == 128 and kw.get("V") == 128:
                kw.update(_WU)
                if not _WUShim._seen:
                    _WUShim._seen = True
                    print(f"[gdn_pf_cfg] ✅ 首次真正生效 recompute_w_u T={kw.get('T')} H={kw.get('H')} {_WU} pid={os.getpid()}", flush=True)
            return launch(*a, **kw)
        return run

    def __getattr__(self, n):
        return getattr(self._k, n)


def install(mod):
    if getattr(mod, "_gdn_pf_cfg_installed", False):
        return
    name = mod.__name__
    if name.endswith("fla.wy_fast") and hasattr(mod, "recompute_w_u_fwd_kernel"):
        mod.recompute_w_u_fwd_kernel = _WUShim(mod.recompute_w_u_fwd_kernel)
    elif name.endswith("sglang.chunk_o") and hasattr(mod, "_load_chunk_o_configs"):
        orig = mod._load_chunk_o_configs

        def _load():
            c = orig()
            if c:
                return c
            print(f"[gdn_pf_cfg] ✅ 首次真正生效 chunk_o 补 gfx928 配置 K=128,V=128,BT=64 {_CO} pid={os.getpid()}", flush=True)
            return {"K=128,V=128,BT=64": dict(_CO)}   # 只在本架构没有自带配置时补上
        mod._load_chunk_o_configs = _load
        mod._get_chunk_o_config.cache_clear()
    else:
        return
    mod._gdn_pf_cfg_installed = True
    print(f"[gdn_pf_cfg] 已安装到 {name} pid={os.getpid()}", flush=True)

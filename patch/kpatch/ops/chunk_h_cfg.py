"""给 aiter 的 prefill GDN chunk_delta_h kernel 补上 gfx928 的调优配置。

背景与 gdn_cfg 完全同构:
  aiter 按架构名加载 configs/chunk_gated_delta_rule_fwd_h/chunk_gated_delta_rule_fwd_h-{arch}.json
  目录里只有 gfx936 / gfx938, **没有 gfx928** -> 回退到写死的默认 {BV:32, num_warps:8, num_stages:2}。
  而 SGLang 两处调用点都写死 kernel_cfg=None (fla/chunk_delta_h.py aiter 分支),
  所以生产上 100% 吃这个默认值。

★ 关键证据 —— 厂商自己的 gfx938 调优表里就有我们这个形状那一条:
    "default"               {"BV": 32, "num_warps": 8, "num_stages": 2}
    "K=128,V=128,BT=64,H=8" {"BV": 16, "num_warps": 4, "num_stages": 2}   <- 就是我们的形状
  即**厂商实测认定默认值对该形状是错的**, 只是 gfx928 拿不到这张表。

形状核实 (每条都查过源码/配置, 未采信任何转述):
  K = linear_key_head_dim   = 128        config.json
  V = linear_value_head_dim = 128        config.json
  H = linear_num_value_heads 32 / TP4    = 8   (kernel 内 H = u.shape[-2])
  BT = CHUNK_SIZE = 64                   fla/chunk_delta_h.py:27 与 fla/chunk.py:35

网格 = (cdiv(V, BV), N * H), **不含 chunk 维**(递推在核内串行) ->
并行度与序列长度无关, 短长 prefill 一样饥饿:
  BV=32(当前默认): (4, 8) =  32 个 workgroup, 120 个 CU 上仅 27%
  BV=16(厂商值)  : (8, 8) =  64                              53%
  BV= 8          : (16,8) = 128                              铺满
gfx928 的 CU 数多于 gfx938, 所以 BV=8 有可能优于厂商的 16 —— **必须实测, 不要照抄**。

默认取厂商实测过的 {BV:16, num_warps:4}(保守、有据), 用环境变量扫 BV/warps:
  SGLANG_CHUNK_H_BV=8 SGLANG_CHUNK_H_WARPS=4
开关: SGLANG_CHUNK_H_CFG=0 关闭。

Qwen3.8-27B（10-09）：每卡 H=24，上面的形状判断不命中。SGLANG_GDN_TABLE=<probe/gdn_aiter_sweep.py 的 JSON> 时，
读 "h" 段 {"K=128,V=128,BT=64,H=24": {T: cfg}}；网格不含 chunk 维、与 T 无关，取最大 T 档的最优（长预填充占大头）。

★ 未做的部分 (有意留白, 附证据):
  chunk_fwd_o 不动。厂商 gfx938 表里 K=128,V=128,BT=64 的调优值 **等于默认值**,
  且其网格 (cdiv(V,BV), NT, B*H) 含 NT, 2112 token 时 528 个 workgroup, 不饥饿。
"""
import json
import os

_TAB = {}   # (K, V, BT, H) -> cfg
_tp = os.environ.get("SGLANG_GDN_TABLE", "")
if _tp and not os.path.exists(_tp):
    print(f"[%s] ⚠ SGLANG_GDN_TABLE={_tp} 不存在，不用表" % __name__.split(".")[-1], flush=True)
    _tp = ""
if _tp:
    for _k, _v in json.load(open(_tp)).get("h", {}).items():
        _p = {x.split("=")[0]: int(x.split("=")[1]) for x in _k.split(",")}
        _TAB[(_p["K"], _p["V"], _p["BT"], _p["H"])] = _v[max(_v, key=int)]
_seen = set()

_enabled = os.environ.get("SGLANG_CHUNK_H_CFG", "1") != "0"

def _cfg_for(K, V, BT, H):
    if (K, V, BT, H) in _TAB:
        return _TAB[(K, V, BT, H)]
    # 只覆盖已核实的形状, 其余一律返回 None 走原逻辑
    if (K, V, BT, H) != (128, 128, 64, 8):
        return None
    bv = int(os.environ.get("SGLANG_CHUNK_H_BV", "16"))
    nw = int(os.environ.get("SGLANG_CHUNK_H_WARPS", "4"))
    ns = int(os.environ.get("SGLANG_CHUNK_H_STAGES", "2"))
    return {"BV": bv, "num_warps": nw, "num_stages": ns}

def install(mod):
    if getattr(mod, "_chunk_h_cfg_installed", False) or not _enabled:
        return
    name = "launch_chunk_gated_delta_rule_fwd_kernel_h_blockdim64"
    orig = getattr(mod, name, None)
    if orig is None:
        print(f"[chunk_h_cfg] 未找到 {name}, 跳过", flush=True)
        return

    def wrapped(*args, **kw):
        try:
            if kw.get("kernel_cfg") is None:
                cfg = _cfg_for(kw.get("K"), kw.get("V"), kw.get("BT"), kw.get("H"))
                if cfg is not None:
                    kw["kernel_cfg"] = cfg
                    if not _seen:
                        _seen.add(1)
                        print(f"[chunk_h_cfg] ✅ 首次真正生效 K={kw.get('K')} V={kw.get('V')} BT={kw.get('BT')} H={kw.get('H')} cfg={cfg} pid={os.getpid()}", flush=True)
        except Exception as e:
            print(f"[chunk_h_cfg] 选配置异常, 走原逻辑: {type(e).__name__}: {e}", flush=True)
        return orig(*args, **kw)

    setattr(mod, name, wrapped)
    mod._chunk_h_cfg_installed = True
    print("[chunk_h_cfg] 已安装到 %s (pid=%d, BV=%s warps=%s)" % (
        mod.__name__, os.getpid(),
        os.environ.get("SGLANG_CHUNK_H_BV", "16"),
        os.environ.get("SGLANG_CHUNK_H_WARPS", "4")), flush=True)

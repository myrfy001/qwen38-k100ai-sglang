"""给 aiter 的 GatedDeltaNet 递归更新 kernel 补上 gfx928 的调优配置。

背景: aiter 按架构名加载配置
  configs/fused_sigmoid_gating_delta_rule_update_recurrent/...-{arch}.json
目录里只有 gfx936 / gfx938, **没有 gfx928** -> 回退到写死的默认 {BV:32, num_warps:1}。
而该 kernel 是 decode 画像里最大的单个非 GEMM 算子(每轮 30 次, 占约 5.9%)。

实测(生产 target_verify 参数, 输出与中间状态缓冲逐位一致):
  bs   现状BV32/w1   最优BV8/w4    加速
   1      34.41us      17.75us    1.94x
   2      35.53        21.23*     1.67x   (*w2 略优, w4 差 7%)
   4      43.45        30.88*     1.41x   (*w2 略优, w4 差 4%)
   8      71.53        48.25      1.48x
  16     163.46        79.18      2.06x
  32     375.83       144.35      2.60x
  48     571.65       208.99      2.74x
带宽利用率 254.7 -> 533.0 GB/s (天花板 777 的 69%)。
根因: 默认配置只有 512 个 workgroup 且每个仅 1 个 wave, 占用率严重不足。

作用范围: 只覆盖 target_verify 路径(intermediate_states_buffer 非空)且 H=4/HV=8
(即本模型每卡的线性注意力形状), 其余一律走原逻辑。
开关: SGLANG_GDN_CFG=0 关闭。

Qwen3.8-27B（10-09）：每卡 H=8/HV=24，上表不命中。SGLANG_GDN_TABLE=<probe/gdn_aiter_sweep.py 产出的 JSON> 时，
读其中 "verify" 段 {"T=..,H=8,HV=24": cfg}（逐位一致的最优），按 T 取最近档（与 aiter 自带的近邻规则相同）。
"""
import json
import os

_enabled = os.environ.get("SGLANG_GDN_CFG", "1") != "0"
# 键: (H, HV) -> 配置。只放实测验证过的形状。
_CFG = {(4, 8): {"BV": 8, "num_warps": 4}}
_TAB = {}   # (H, HV) -> [(T, cfg), ...]
_tp = os.environ.get("SGLANG_GDN_TABLE", "")
if _tp and not os.path.exists(_tp):
    print(f"[%s] ⚠ SGLANG_GDN_TABLE={_tp} 不存在，不用表" % __name__.split(".")[-1], flush=True)
    _tp = ""
if _tp:
    for _k, _v in json.load(open(_tp)).get("verify", {}).items():
        _p = {x.split("=")[0]: int(x.split("=")[1]) for x in _k.split(",")}
        _TAB.setdefault((_p["H"], _p["HV"]), []).append((_p["T"], _v))
_seen = set()


def _pick(H, HV, T):
    rows = _TAB.get((H, HV))
    if rows:
        return min(rows, key=lambda r: abs(r[0] - T))[1]
    return _CFG.get((H, HV))

def install(mod):
    if getattr(mod, "_gdn_cfg_installed", False) or not _enabled:
        return
    orig = mod.fused_sigmoid_gating_delta_rule_update

    def wrapped(*args, **kw):
        try:
            if kw.get("kernel_cfg") is None and kw.get("intermediate_states_buffer") is not None:
                k = kw.get("k"); v = kw.get("v")
                if k is not None and v is not None and k.dim() == 4 and v.dim() == 4:
                    key = (k.shape[2], v.shape[2], k.shape[1])
                    cfg = _pick(*key)
                    if cfg is not None:
                        kw["kernel_cfg"] = cfg
                        if key not in _seen:
                            _seen.add(key)
                            print(f"[gdn_cfg] ✅ 首次真正生效 H={key[0]} HV={key[1]} T={key[2]} cfg={cfg} pid={os.getpid()}", flush=True)
        except Exception as e:
            print(f"[gdn_cfg] 选配置异常, 走原逻辑: {type(e).__name__}: {e}", flush=True)
        return orig(*args, **kw)

    mod.fused_sigmoid_gating_delta_rule_update = wrapped
    mod._gdn_cfg_installed = True
    print(f"[gdn_cfg] 已安装到 {mod.__name__} (pid={os.getpid()}, 表 {_tp or '无'} {sum(len(v) for v in _TAB.values())} 档)", flush=True)

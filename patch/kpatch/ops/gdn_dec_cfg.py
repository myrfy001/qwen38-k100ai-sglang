"""aiter GDN decode 内核（fused_recurrent_gated_delta_rule_packed_decode）补上 gfx928 配置。

aiter 只带 gfx936/938 的配置，gfx928 回落默认 BV32/w1/s1。10-08 卡 4 扫参（27B 每卡 H=8 HV=24，输出一致）：
  B=1 13.67→6.39 us（2.14×）  B=8 48.4→41.2  B=16 98.7→83.8（1.18×）  B=32 →160
配置文件：SGLANG_GDN_DEC_CFG_FILE（默认 /workspace/out/gdn_dec_gfx928.json，由 gdn_dec_sweep.py 生成）。开关 SGLANG_GDN_DEC_CFG=1。
只改启动参数（BV/warps/stages），数学不变。"""
import os, json

_FILE = os.environ.get("SGLANG_GDN_DEC_CFG_FILE", "/workspace/out/gdn_dec_gfx928.json")


def install(mod):
    if getattr(mod, "_gdn_dec_cfg_installed", False) or not hasattr(mod, "_load_fused_recurrent_packed_decode_configs"):
        return
    try:
        cfgs = json.load(open(_FILE))["config"]
    except Exception as e:
        print(f"[gdn_dec_cfg] ⚠ 读不到配置 {_FILE}: {e}，不安装", flush=True); return
    orig = mod._load_fused_recurrent_packed_decode_configs

    def _load():
        c = orig()
        return c if c else cfgs        # 只在本架构没有自带配置时补上

    mod._load_fused_recurrent_packed_decode_configs = _load   # _get_… 按全局名调用，替换即生效
    mod._get_fused_recurrent_packed_decode_config.cache_clear()
    mod._gdn_dec_cfg_installed = True
    print(f"[gdn_dec_cfg] 已安装：{len(cfgs)} 档（{_FILE}） pid={os.getpid()}", flush=True)

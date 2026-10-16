"""配合 gdn_replay：中间 SSM 状态缓冲（intermediate_ssm）不再预留、不再分配。

原状：开投机后 handle_max_mamba_cache 按 max_running × draft_token_num 份「每请求 mamba 状态」预留显存，
      MambaPool 再分配 intermediate_ssm [层, 槽, draft, HV, V, K]（本模型每卡 48×49×4×1.5 MB = 13.78 GB）。
      gdn_replay 开启后验证核不写它、散射不读它，它整块闲置；而且正因为它按 max_running 成比例预留，
      调大 max_running 会挤小 KV 池（10-10 sg14_mr64：605696→417792 token）。
改法：预留只扣卷积窗那部分（SSM 部分加回给 KV 池）；MambaPool 构造后把 intermediate_ssm 换成每层 1 个元素的占位张量并释放。
开关：与 gdn_replay 同为 SGLANG_GDN_REPLAY（只开一个会算错状态或白占显存）。"""
import os

import torch

_done = set()


def _log(msg):
    print(f"[gdn_replay_mem] {msg} pid={os.getpid()}", flush=True)


def _patch_mixin(mod):
    C = mod.ModelRunnerKVCacheMixin
    orig = C.handle_max_mamba_cache

    def handle_max_mamba_cache(self, total_rest_memory):
        rest = orig(self, total_rest_memory)
        sa = self.server_args
        if self.spec_algorithm.is_none() or sa.max_running_requests is None:
            return rest
        p = self.mambaish_config.mamba2_cache_params
        ssm = int(torch.Size(p.shape.temporal).numel()) * p.dtype.temporal.itemsize * len(p.layers)
        mr = sa.max_running_requests // (self.dp_size if sa.enable_dp_attention else 1)
        back = ssm * mr * sa.speculative_num_draft_tokens / (1 << 30)
        _log(f"预留加回 {back:.2f} GB（SSM 中间状态 {mr} 请求 × {sa.speculative_num_draft_tokens} 步）：剩余 {rest:.2f} → {rest + back:.2f} GB")
        return rest + back

    C.handle_max_mamba_cache = handle_max_mamba_cache


def _patch_pool(mod):
    P = mod.MambaPool
    orig = P.__init__

    def __init__(self, *a, **kw):
        orig(self, *a, **kw)
        mc = self.mamba_cache
        if not isinstance(mc, P.SpeculativeState):
            return
        big = mc.intermediate_ssm
        gb = big.numel() * big.element_size() / (1 << 30)
        tiny = torch.zeros((big.shape[0], 1, 1, 1, 1, 1), dtype=big.dtype, device=big.device)
        object.__setattr__(mc, "intermediate_ssm", tiny)
        del big
        torch.cuda.empty_cache()
        self.mem_usage = mc.mem_usage_bytes() / (1 << 30)
        _log(f"✅ intermediate_ssm 已换占位并释放 {gb:.2f} GB；mamba 池现占 {self.mem_usage:.2f} GB")

    P.__init__ = __init__


def install(mod):
    if mod.__name__ in _done:
        return
    if mod.__name__.endswith("model_runner_kv_cache_mixin"):
        _patch_mixin(mod)
    elif mod.__name__.endswith("memory_pool"):
        _patch_pool(mod)
    else:
        return
    _done.add(mod.__name__)
    _log(f"已安装到 {mod.__name__}")

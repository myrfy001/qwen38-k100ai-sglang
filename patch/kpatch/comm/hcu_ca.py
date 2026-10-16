"""decode 小消息 all-reduce 改走厂商 custom all-reduce（vllm_hcu CustomAllreduce，标准内核）。

依据（离线探针，8 卡，图内 80 次背靠背，fp16 [M,4096]）：
    M      RCCL     CA      比
    3      49.1    10.7    4.57×
    6      42.1    11.7    3.59×
    12     42.5    13.1    3.25×
    24     51.6    15.8    3.28×
    48     98.2    36.0    2.73×
    64    100.8    43.6    2.31×
  与 RCCL 最大差 ≤0.016（求和顺序不同，非逐位）；图外 10 轮变输入 0 错。
  decode 每轮约 170 次 all-reduce，按表每轮约省 5 ms。
原状：启动参数 --disable-custom-all-reduce（K100 上 aiter CA 段错误），ca_comm 恒为 None。
改法：GroupCoordinator 初始化后，world_size≥2 且 ca_comm 为空时，建 vllm_hcu 的 CustomAllreduce（is_fully_connected 强制真，
  走标准内核；_pcie 变体默认派发有竞态，见 Qwen 09-14 记录）。框架已有的 ca 分发/图捕获注册（graph_capture 里 ca_comm.capture()）原样生效；
  只接管字节数 < SGLANG_HCU_CA_MAX（默认 512 KiB）的张量，更大的仍走 RCCL / quant_ar。
注意：直接调 torch.ops.hcu_ops 的写法会 Invalid address（原因未查），必须用厂商类。
开关 SGLANG_HCU_CA=1。
"""
import os

import torch

_MAX = int(os.environ.get("SGLANG_HCU_CA_MAX", str(512 * 1024)))
# 09-27：厂商默认在小尺寸选一段式，紧跟计算时 M≥12 反而比 RCCL 慢（0.72–0.87×）；两段式全尺寸 1.08–1.25×。
os.environ.setdefault("VLLM_CUSTOM_ALLREDUCE_ALGO", "2stage")
_st = {"n": 0}


def install(mod):
    GC = mod.GroupCoordinator
    if getattr(GC, "_hcu_ca_installed", False):
        return
    orig_init = GC.__init__

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        if self.world_size < 2 or self.ca_comm is not None:
            return
        try:
            from vllm.platforms import current_platform
            current_platform.__class__.is_fully_connected = classmethod(lambda cls, ids: True)
            from vllm_hcu.distributed.device_communicators.custom_all_reduce import CustomAllreduce
            ca = CustomAllreduce(self.cpu_group, self.device, max_size=_MAX)
            if ca.disabled:
                print(f"[hcu_ca] ❌ {self.unique_name} 厂商 CA 自行禁用，走原路径", flush=True)
                return
            orig_should = ca.should_custom_ar

            def should_custom_ar(inp, _o=orig_should, _ca=ca, _name=self.unique_name, _r=self.rank_in_group):
                ok = _o(inp)
                if ok and _st["n"] == 0:
                    print(f"[hcu_ca] ✅ 首次真正生效：{_name} {tuple(inp.shape)} {inp.dtype} "
                          f"capturing={torch.cuda.is_current_stream_capturing()} rank {_r} pid={os.getpid()}", flush=True)
                if ok:
                    _st["n"] += 1
                return ok

            ca.should_custom_ar = should_custom_ar
            self.ca_comm = ca
            print(f"[hcu_ca] 已安装：{self.unique_name} world {self.world_size} max {_MAX} B rank {self.rank_in_group}", flush=True)
        except Exception as e:
            print(f"[hcu_ca] ❌ {self.unique_name} 建 CA 失败，走原路径：{type(e).__name__}: {e}", flush=True)

    GC.__init__ = __init__
    GC._hcu_ca_installed = True

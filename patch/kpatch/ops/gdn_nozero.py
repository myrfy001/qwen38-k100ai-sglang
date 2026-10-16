"""gdn_nozero: 去掉 GDN 预填充链里 chunk_fwd_o 的无效清零。

## 事实（2026-09-13 在我们自己的模型上实测，非来自任何文档）
`fla/chunk_o.py:153` 有 `o = torch.zeros_like(v)`；随后内核把 o 每个元素都写一遍：
  - aiter 路径 `launch_chunk_fwd_kernel_o`，grid = (cdiv(V,BV), NT, B*H) 覆盖全部 (V块, chunk, 头)
  - 内核里对 o 只有一句无条件 `tl.store(p_o, ...)`（chunk_o.py:130 同形）
⇒ 那次清零是纯白干。

投毒实验判据（zpoison4.py，四道自证门全过）：
  A 基线可复现：每次调用给全新 initial_state 后，连跑 3 次逐位全同
    （踩过的坑：复用同一个 initial_state 张量会被内核原地更新，第 2 次起状态被污染，
      相对差 0.76，会把一切逐位判据变成假的）
  B 代理无害：只把 zeros_like 换成 empty_like().zero_() 时，输出逐位不变、残 NaN=0
  C 探针灵敏：对已知必须保留的 chunk_fwd.py:390 的 A 投毒，被抓出（差异张量 3，残 NaN 1.06e8）
  D 探针必须命中：带命中计数，未执行的行一律判「无结论」
    （踩过的坑：chunk_intra 的两处 zeros 在我们路径上命中 0 次，
      不加计数就会被误读成「填零是死的」）
  正式结论：chunk_o 的 o 投 NaN 后 **输出逐位相同、残 NaN=0** ⇒ 可删。
            chunk_fwd 的 A 必须保留（下三角矩阵，上三角靠清零）。

## 代价与收益（T=8192，生产形状 1×8192×32×128 bf16 = 67.1 MB）
  torch.zeros_like  80.8 µs  |  torch.empty_like  6.5 µs  ⇒ 省 74.3 µs
  ×30 个 GDN 层 = 2.23 ms / 每个 8192-token 块
  分母：8192 块 TTFT ≈ 711 ms（16384 行 1423 ms ÷ 2 块）⇒ 0.31%
        短行 1088：≈0.4 ms / 157.4 ms ⇒ 0.25%
  折成分数 ≈ O +0.26。**小，但逐位已证、零算法风险**。

## 为什么打在 chunk 而不是 chunk_o
`fla/chunk.py` 用 `from ...chunk_o import chunk_fwd_o` 在导入期就绑定了符号，
改 chunk_o 模块属性对它无效。所以 TARGETS 挂 `...fla.chunk`，换掉它的全局名字。

## 守卫（任一不满足就走原实现）
  1) chunk_o._use_prefill_aiter_linear_attn is True —— 只验证过 aiter 路径
  2) cu_seqlens is not None —— 只验证过 varlen 形态（预填充）
  3) 图捕获中不改行为（本函数只做分配，不含 .item()，但仍保持一致）
"""
import os
import hashlib

import torch as _real_torch

_AUDIT = os.environ.get("SGLANG_GDN_NOZERO_AUDIT", "1") != "0"


class _TorchProxy:
    """代理 chunk_o 模块的全局 torch：受控地把 zeros_like 降级为 empty_like。
    其余一切属性透传给真 torch，所以 torch.Tensor / isinstance 等都不受影响。"""

    __slots__ = ("armed", "hits")

    def __init__(self):
        self.armed = False
        self.hits = 0

    def __getattr__(self, name):
        return getattr(_real_torch, name)

    def zeros_like(self, *a, **kw):
        if self.armed:
            self.hits += 1
            return _real_torch.empty_like(*a, **kw)
        return _real_torch.zeros_like(*a, **kw)


def install(module):
    """module 是 sglang.srt.layers.attention.fla.chunk"""
    orig = getattr(module, "chunk_fwd_o", None)
    if orig is None:
        print("[gdn_nozero] 未找到 chunk_fwd_o，跳过", flush=True)
        return
    if getattr(orig, "_gdn_nozero", False):
        return

    import sglang.srt.layers.attention.fla.chunk_o as CO

    if not getattr(CO, "_use_prefill_aiter_linear_attn", False):
        print(
            "[gdn_nozero] 未安装: chunk_o._use_prefill_aiter_linear_attn 为假，"
            "本补丁只在已验证的 aiter 路径上生效",
            flush=True,
        )
        return

    proxy = _TorchProxy()
    CO.torch = proxy  # 只改 chunk_o 自己的全局绑定，不动共享的 torch 模块属性

    state = {"fired": 0, "fallback": 0, "logged": False, "warned": False}

    def dispatched(*args, **kwargs):
        cu = kwargs.get("cu_seqlens", None)
        if cu is None and len(args) >= 7:
            cu = args[6]
        if cu is None:
            state["fallback"] += 1
            if _AUDIT and not state["warned"] and state["fallback"] >= 200 and state["fired"] == 0:
                state["warned"] = True
                print(
                    f"[gdn_nozero] 警告: 已回退 {state['fallback']} 次仍未生效一次 "
                    f"pid={os.getpid()}",
                    flush=True,
                )
            return orig(*args, **kwargs)

        proxy.armed = True
        before = proxy.hits
        try:
            out = orig(*args, **kwargs)
        finally:
            proxy.armed = False

        if proxy.hits == before:
            # 谓词满足但代理没被调用 ⇒ 上游实现变了，必须知道
            state["fallback"] += 1
            if _AUDIT and not state["warned"]:
                state["warned"] = True
                print(
                    "[gdn_nozero] 警告: 谓词满足但 zeros_like 未被调用，"
                    "chunk_o 实现可能已变更，本补丁当次未生效",
                    flush=True,
                )
            return out

        state["fired"] += 1
        if _AUDIT and not state["logged"]:
            state["logged"] = True
            print(
                f"[gdn_nozero] 首次真正生效: o.shape={tuple(out.shape)} "
                f"省{out.numel()*out.element_size()/1e6:.1f}MB清零 pid={os.getpid()}",
                flush=True,
            )
        if _AUDIT and state["fired"] % 2000 == 0:
            print(
                f"[gdn_nozero] 累计 生效={state['fired']} 回退={state['fallback']} "
                f"pid={os.getpid()}",
                flush=True,
            )
        return out

    dispatched._gdn_nozero = True
    module.chunk_fwd_o = dispatched

    try:
        h = hashlib.sha1(open(__file__.replace(".pyc", ".py"), "rb").read()).hexdigest()[:8]
    except Exception:
        h = "?"
    print(
        f"[gdn_nozero] 已安装: 目标={module.__name__}.chunk_fwd_o aiter路径=True "
        f"pid={os.getpid()} sha1={h}",
        flush=True,
    )

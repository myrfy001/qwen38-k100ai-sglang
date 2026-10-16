"""按请求长度选预填充块大小：长请求用大块，其余保持启动参数给定的块大小。

依据（2026-09-19，成对预填充开启，c=1 粗测）：启动参数改成 --chunked-prefill-size 16384 时
    32768 档 2584 → 2460 ms（−4.8%）、65536 档约 −6%（大块的 GEMM 形状效率更高、块数减半后每块的固定开销减半）；
    但 16384 档 1191 → 1320 ms（+10.8%）：整条请求只剩一个图外块，成对发射无块可配。
    且启动参数给 16384 会让框架自动把 mem_fraction_static 从 0.720 降到 0.547，KV 池 231 万 → 122 万 token，
    131072 档高并发行会排队。
做法：启动参数不动（块大小 8192，显存规划不变），只在规划某一批预填充时，若当前请求（进行中的分块请求，
    否则等待队列队首）的输入长度 ≥ 阈值，临时把调度器的 chunked_prefill_size 提到大块。
    图外执行器的静态缓冲按 max_total_num_tokens 定尺寸，不受块大小限制；max_prefill_tokens 默认 16384。
结果（同一实例 关—开—关，KV 池仍为 231 万 token）：32768 档 2576 → 2444 ms（−5.1%）、65536 档 6460 → 6115（−5.3%）、
    131072 档 18188 → 16991（−6.6%）；16384 档不变（1175 对 1185）。
    再加一档「输入 ≥ 65536 用 32768 块」：65536 档 → 5960（再 −2.5%）、131072 档 → 16589（再 −2.4%）；需要 --max-prefill-tokens 32768。
不涉及任何 GPU 并发，数值上只是改了分块边界。成对预填充关闭时块数不要求为偶数。
    相对 R14（固定 8192 块）的 c1 实测：16384 档 1381 → 1309 ms、32768 档 2982 → 2710、65536 档 7284 → 6507、131072 档 20086 → 17940。
    试过「16384 档整条一块」：c1 1217 ms 更快，但 c4 2924 对 2830 更慢（多块时块间的 CPU 准备被重叠调度藏住了），净收益约为零，未采用。
开关 SGLANG_ADAPTIVE_CHUNK=1；分档 SGLANG_ADAPTIVE_CHUNK_TIERS="输入下限:块大小,…"（默认 "32768:16384,65536:32768"）。
均分（SGLANG_ADAPTIVE_CHUNK_EVEN，默认 1）：见 _pick 的说明；运行时对照 /tmp/achunk_noeven。
运行时对照：存在 /tmp/achunk_off 则一律用启动参数的块大小。
"""
import os

_TIERS = sorted((int(a), int(b)) for a, b in (t.split(":") for t in
                os.environ.get("SGLANG_ADAPTIVE_CHUNK_TIERS", "32768:16384,65536:32768").split(",") if t.strip()))
_OFF_FILE = "/tmp/achunk_off"
_st = {}


_EVEN = os.environ.get("SGLANG_ADAPTIVE_CHUNK_EVEN", "1") != "0"
_PAIR_ON = os.environ.get("SGLANG_PAIR_PREFILL", "0") != "0"
_EVEN_OFF_FILE = "/tmp/achunk_noeven"
_SLACK = 1.15          # 均分后的块最多比档位块大 15%


def _pick(n, base, page):
    """返回本请求的块大小。先按输入长度取档位块 C；再（均分开启时）把请求均分成偶数块：
    压测输入是 8192 的整倍数，但经文本往返后实际多出约 3%（16384 → 16947），按固定块切会剩一个几百 token 的尾块，
    它要单独跑一次前向（约 90 ms）。均分成偶数块既消掉尾块，又保证每块都有同伴可成对发射。"""
    C = base
    for lo, chunk in _TIERS:
        if n >= lo:
            C = max(C, chunk)
    if not _EVEN or n <= C or os.path.exists(_EVEN_OFF_FILE):
        return C
    if n <= _SLACK * C:
        return -(-n // page) * page               # 只比一个块多一点：整条一块做完，不留尾块
    step = 2 if _PAIR_ON else 1                 # 成对预填充开启时要偶数块（每块都有同伴）；否则块数越少越好
    k = step
    while -(-n // k) > _SLACK * C:
        k += step
    per = -(-n // k)
    return -(-per // page) * page


def install(mod):
    S = getattr(mod, "Scheduler", None)
    if S is None or getattr(S, "_achunk_installed", False):
        return
    orig = S.get_new_batch_prefill

    def get_new_batch_prefill(self, *a, **kw):
        base = self.chunked_prefill_size
        if base is None or base <= 0:
            return orig(self, *a, **kw)
        req = self.chunked_req
        if req is None and self.waiting_queue:
            req = self.waiting_queue[0]
        if req is None:
            return orig(self, *a, **kw)
        try:
            n = len(req.origin_input_ids)
        except Exception:
            n = 0
        use = base if os.path.exists(_OFF_FILE) else _pick(n, base, max(1, int(getattr(self, "page_size", 1) or 1)))
        mp = getattr(self, "max_prefill_tokens", None)
        if mp is not None and use > mp:
            use = max(base, mp)                       # 块不超过每批预填充 token 上限（--max-prefill-tokens）
        if n > base and len(_st) < 8 and use not in _st:
            _st[use] = True
            print(f"[adaptive_chunk] ✅ 首次真正生效：输入 {n} token 的请求用块大小 {use}（启动参数 {base}）", flush=True)
        if use == base:
            return orig(self, *a, **kw)
        self.chunked_prefill_size = use
        try:
            return orig(self, *a, **kw)
        finally:
            self.chunked_prefill_size = base

    S.get_new_batch_prefill = get_new_batch_prefill
    S._achunk_installed = True
    print(f"[adaptive_chunk] 已安装：分档 {_TIERS}（输入下限, 块大小） (pid={os.getpid()})", flush=True)

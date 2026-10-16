"""同一请求相邻两个预填充块的「成对交错发射」：让一块的跨卡通信与另一块的计算重叠。

依据（2026-09-19 四卡探针 lane_probe.py，生产形状 T=8192、GPU_MAX_HW_QUEUES=2、int8 量化通信）：
    两批串行 310.6 ms → 同一通信器下逐层交错发射 239.7 ms（−23%）；纯 GEMM 负载 198 → 140 ms（−29%）。
    反例：两批各用一个 RCCL 通信器并发（不交错）251 ms，且与计算内容无关——两个通信器抢同一条 PCIe，
    单次通信由 2.5 ms 变 6.3 ms；GPU_MAX_HW_QUEUES=4 时退化为完全串行。故线上同一时刻只能有一个集合通信，
    且两批的通信必须按 A1,B1,A2,B2… 交替入队，这只能在 CPU 侧逐层交错发射。
    不切半块：图外预填充每次前向的 CPU 发射约 280 ms，切半后 CPU 发射翻倍会反超 GPU 时间。
做法：
    调度器规划出块 k（非末块、单请求、图外）后，用它的副本作 last_batch 预先规划块 k+1；两块各占一个协程（greenlet，
    同一操作系统线程）与一条计算流，共用同一个通信器，在目标模型前向的每次 all_reduce 处交接控制权。
    不用线程：实测两个线程交替向同一 HIP 上下文提交内核会在主机侧段错误（SIGSEGV），而单线程内切换没有这个问题。
    块 k+1 每段计算开始前等待块 k 同一段的完成事件（KV 与 GDN 状态的依赖）；草稿层扩展不交接，独占执行。
    交接时保存/恢复：当前 CUDA 流、两个进程级前向上下文（见 _globals）、注意力后端的 forward_metadata
    （全注意力与线性注意力各一份）、调度器的 forward_stream 属性。
    图外执行器的静态输入缓冲在成对期间不使用（两批各持副本），否则后一批会覆盖前一批。
MoE 与通信不可真并发（2026-09-19 定位，见 docs/技术报告 §7.4）：
    本平台上 Triton fused_moe 内核与 RCCL 集合通信在两个硬件队列上同时运行时，fused_moe 的结果会偶发出错
    （独立探针 docs/上限验证/09_通信与计算重叠/重叠的正确性/race_probe2.py：T=8192、本项目 MoE 分块配置，每臂四卡 600 次，
    act_flat/fast_moe_align 开/开（交付组合）3 次、关/开 6 次、开/关 39 次、关/关 30 次，单次最多 24 行，误差与输出同量级；
    两条流落在同一硬件队列时四臂均为 0；GEMM、逐点、归一化、SDPA、Triton 量化核同样并发 0 次；all_reduce 自身 0 次）。
    现象是长上下文下成对发射的输出不可复现。二分：两块共用一条流、两条流但 GPU 全串行、仅 MoE 段串行 —— 三者都与不成对逐字相同；
    仅注意力/GDN 段串行仍不可复现。因此 MoE 段（偶数段）开始前等待另一块的通信结束，只让注意力/GDN 段与通信重叠。
适用范围（v1）：同一请求的相邻块。不同请求之间不成对——规划下一批需要上一批运行后才产生的 spec_info。
状态：**2026-09-23 起默认开启**（SGLANG_PAIR_PREFILL=1、SGLANG_PAIR_PREFILL_MOE_OVERLAP=1）。下文「MoE 与通信不可真并发」的根因已查明是
    lightop 的 moe_align_block_size（不是 fused_moe 的 GEMM），成对期间改用 kpatch/ops/safe_align.py 后逐位检测 0/12（lightop 对照 7/12），
    详见 docs/上限验证/09_通信与计算重叠/说明.md。以下为 09-19 时的原始记录。
结果（2026-09-19，同一实例 关—开—关；两臂都已启用 adaptive_chunk）：MoE 段不重叠的安全版 16384 c1 −6.0%、32768 c1 −5.2%、
    65536 c1 约 −3%、131072 c1 −2.7%。早先「MoE 段也重叠」的版本是 −10%~−14%，但结果会偶发出错（见上），R15 全表（O 126.29）因此作废。
验证（安全版）：逐 token 检测（返回全部输入 token 的 logprob，与不成对逐位置比较）16384 输入 12 次 0 差异（旧版 8 次里 4 次出错，最大差约 1.0）；
    首 token logprob 检测 16K/34K/72K/125K 共 24 次 0 差异（旧版 12 次里 8 次不一致）。
    残余风险：注意力/GDN 段仍与通信重叠，上述约 6700 个重叠段里未见出错（每段出错率 95% 上界约 0.045%），但机理未明，故默认关闭。
    验证请求带 return_logprob，需 SGLANG_PAIR_PREFILL_ALLOW_LOGPROB=1 与 SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1；脚本 det_probe.py / det_probe2.py。
    ⚠ 2026-09-23 起（R19 链路）det_probe2.py（返回全部输入 token 的 logprob）会使调度进程断言退出
      （logprob_result_processor.py:248），复测改用 det_probe.py（首 token 逐位比较）。
上限：逐段 GPU 计时显示重叠时计算段 3.4→5.3~6.4 ms、通信段 2.6→3.3~5.4 ms，通信只有约 27% 真正藏进了另一块的计算。
    四进程探针里同样的交替对纯 GEMM 计算几乎贴着下界、对访存受限的逐点计算只有 43%——线上算子多为访存受限，这是硬件特性。
    已试并否决：用 IPC 映射 + 跨卡拷贝（拷贝引擎）替换 RCCL——结果逐位相同，但单独延迟 3.8 对 2.6 ms，交替时也不占优。
开关 SGLANG_PAIR_PREFILL=1（默认 0）。运行时对照：存在 /tmp/pair_off 则不成对（first_token_first.py 里判定）。
"""
import dataclasses
import os
import time

import torch

ENABLED = os.environ.get("SGLANG_PAIR_PREFILL", "0") != "0"
if ENABLED:
    try:
        from greenlet import greenlet as _greenlet_check  # noqa: F401  协程库缺失时整体停用，而不是在第一对块上抛错
    except Exception as _e:
        ENABLED = False
        print(f"[pair_prefill] ⚠ 未找到 greenlet（{type(_e).__name__}），成对预填充停用；pip install greenlet 后重启即可", flush=True)
# 调试用：1 = 两批共用同一条计算流（只验证交错逻辑，GPU 上无并发、无收益）；2 = 各用一条流（正常模式）
_NSTREAMS = int(os.environ.get("SGLANG_PAIR_PREFILL_STREAMS", "2"))
# 调试用二分：seq = 只预先规划、两批仍先后执行；其余值 = 正常交错
_DEBUG_MODE = os.environ.get("SGLANG_PAIR_PREFILL_DEBUG", "")
# 诊断用：前 N 对打印逐段 CPU 发射耗时与 GPU 事件计时（末尾会同步一次，只用于定位，不用于测性能）
_TIMING = int(os.environ.get("SGLANG_PAIR_PREFILL_TIMING", "0"))
# 严格交替：块 A 的下一段计算等块 B 本段计算结束，两块的计算不同时上 GPU（两路计算并行在这块卡上没有收益，只会互相拉长）。
# 同一实例 A-B-A-B：32768 c1 自由交错 2643/2634 ms，严格交替 2573/2589 ms。设 0 可退回自由交错。
_STRICT = os.environ.get("SGLANG_PAIR_PREFILL_STRICT", "1") != "0"
_NO_COMM_ORDER = os.environ.get("SGLANG_PAIR_PREFILL_NO_COMM_ORDER", "0") != "0"   # 仅供复现上述问题的对照
# 哪些注意力后端的预填充内核要独占 GPU（逗号分隔的类名；设为 none 可关闭，仅供复现问题）
_EXCL_BACKENDS = set(x for x in os.environ.get("SGLANG_PAIR_PREFILL_EXCL", "FlashAttentionBackend").split(",") if x and x != "none")
# 仅供复现问题：1 = 允许 MoE 段与另一块的通信重叠（结果会偶发出错，切勿用于服务）
_MOE_OVERLAP = os.environ.get("SGLANG_PAIR_PREFILL_MOE_OVERLAP", "0") != "0"
# 成对期间 MoE 的对齐排序改走并发安全实现（2026-09-23 定位：「MoE 与通信不可真并发」的根因是 lightop 的
# moe_align_block_size 在真并发下偶发丢 token 对，GEMM 本身无误；见 kpatch/ops/safe_align.py）。
# /tmp/pair_lightop 存在时成对期间仍用 lightop（阳性对照用）。设 0 关闭。
_SAFE_ALIGN = os.environ.get("SGLANG_PAIR_PREFILL_SAFE_ALIGN", "1") != "0"
_LIGHTOP_FILE = "/tmp/pair_lightop"
_SEQ_FILE = "/tmp/pair_seq"
# 实验用运行时开关（每对成对开始时读取，不需重启）：存在则关闭注意力独占 / 关闭严格交替
_NOEXCL_FILE = "/tmp/pair_noexcl"
_NOSTRICT_FILE = "/tmp/pair_nostrict"      # 仅诊断模式下有效：文件存在则成对的两块不交接，作为逐段计时的串行对照


class _State:
    active = False
    role = None                  # 当前正在执行的块："A" / "B" / None
    handoff = {"A": False, "B": False}
    main = None                  # 驱动协程（主流程）
    ev_A = None                  # 块 A 最近一段计算的结束事件
    ev_B = None                  # 块 B 最近一段计算的结束事件（仅严格交替时用）
    ev_comm = {}                 # 各块最近一次通信的结束事件
    force_serial = False
    n_layers = 0
    excl_logged = False
    strict = False
    backends = []
    stream_B = None
    stream_A = None
    n_pairs = 0
    n_switch = 0
    logged = False
    mm_logged = False
    n_safe = 0
    safe_align = False
    noexcl = False
    toggles_logged = None
    evlog = {"A": [], "B": []}
    ev0 = None
    indep = False                # 两块来自不同请求（Qwen3.8 v2）：块 B 不依赖块 A 的 KV/状态


S = _State()


def _globals():
    """每次前向会改写、且被层内代码读取的进程级全局量：(模块, 属性名)。交接时逐块保存/恢复。
    Qwen3.6（das 0.5.15）是 forward_context._current 与 tc_piecewise 上下文；Qwen3.8 镜像（sglang 0.5.12）只有
    compilation.piecewise_context_manager._forward_context。按版本取存在的那些。"""
    import importlib
    out = []
    for mn, a in (("sglang.srt.model_executor.forward_context", "_current"),
                  ("sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager", "_tc_piecewise_forward_context"),
                  ("sglang.srt.compilation.piecewise_context_manager", "_forward_context")):
        try:
            m = importlib.import_module(mn)
        except Exception:
            continue
        if hasattr(m, a):
            out.append((m, a))
    return out


def in_handoff():
    return S.active and S.role is not None and S.handoff[S.role]


def before_ar():
    """all_reduce 入队之前：记录本块本段计算的结束点（不含通信）；并让本次通信排在另一块上一次通信之后。
    两块共用一个通信器、却各在自己的流上发集合通信。RCCL 不保证同一通信器在两条流上交叠的操作互不干扰：
    另一块的通信尚未结束时本块的通信就开始，各卡上交叠的次序又不同，收发会串。现象是超长上下文下输出不可复现
    （2026-09-19：12.5 万 token 的同一提示词，不成对两次输出一致，成对三次三个样；131072 档接受长度 3.98 → 2.95）。"""
    other = S.ev_comm.get("B" if S.role == "A" else "A")
    if other is not None and not _NO_COMM_ORDER:
        torch.cuda.current_stream().wait_event(other)
    if S.role == "A" or S.strict or _TIMING:
        ev = torch.cuda.Event(enable_timing=bool(_TIMING))
        ev.record()
        if S.role == "A":
            S.ev_A = ev
        else:
            S.ev_B = ev
        if _TIMING:
            S.evlog[S.role].append(ev)


def after_ar():
    """all_reduce 入队之后：回到驱动协程，由它切到另一块。"""
    S.n_switch += 1
    ev = torch.cuda.Event(enable_timing=bool(_TIMING))
    ev.record()
    S.ev_comm[S.role] = ev                          # 本块这次通信的结束点，另一块的下一次通信要等它
    if _TIMING:
        S.evlog[S.role].append(ev)
        if os.path.exists(_SEQ_FILE):               # 诊断对照：同样打点，但不交接（块 A 整块跑完再跑块 B）
            return
    S.main.switch()


def disable_handoff_in_current():
    if S.role is not None:
        S.handoff[S.role] = False


def _tiny_latency(F, G, x, w, tiny):
    """F 上排约 40 ms 计算后，立刻在 G 上排一个极小内核，返回它的完成延迟（ms）。
    同一硬件队列：要等 F 的计算做完（≈40 ms）；不同队列：立即完成（≈0）。
    2026-09-19 用跨卡拷贝重叠率作真值逐流核对过，14 个组合判别全部一致；主机↔设备拷贝不能作判别（与队列无关，恒可重叠）。"""
    best = 1e9
    for _ in range(2):
        torch.cuda.synchronize()
        with torch.cuda.stream(F):
            for _ in range(60):
                x @ w
        with torch.cuda.stream(G):
            tiny.add_(1)
            e = torch.cuda.Event()
            e.record()
        t0 = time.perf_counter()
        e.synchronize()
        best = min(best, time.perf_counter() - t0)
        torch.cuda.synchronize()
    return best * 1e3


def _pick_stream_B(F):
    """GPU_MAX_HW_QUEUES=2 时进程内所有流只映射到 2 个硬件队列，且与创建顺序的关系不固定。
    两块的流若撞在同一队列，GPU 上没有任何并发，成对发射收益为零（实测 32768 c1：不同队列 2573 ms，同一队列 2955 ms = 不成对）。
    这里实测挑一条与前向流不同队列的流。"""
    x = torch.randn(4096, 2048, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(2048, 4096, dtype=torch.bfloat16, device="cuda")
    tiny = torch.zeros(16, device="cuda")
    _tiny_latency(F, torch.cuda.Stream(), x, w, tiny)                   # 预热
    cands, lats, pick = [], [], None
    for _ in range(6):
        G = torch.cuda.Stream()
        lat = _tiny_latency(F, G, x, w, tiny)
        cands.append(G)
        lats.append(round(lat, 1))
        if pick is None and lat < 5.0:
            pick = G
    print(f"[pair_prefill] 流→硬件队列自检：候选流上小内核的完成延迟 {lats} ms（≈0 与前向流不同队列，≈40 同一队列）→ "
          f"{'选第 %d 条' % cands.index(pick) if pick is not None else '⚠ 没有不同队列的流，成对发射将没有收益'}", flush=True)
    return pick if pick is not None else cands[0]


def _exclusive(fn):
    """本块的注意力内核启动前，等另一块已入队的通信全部做完；配合严格交替（另一块的下一段计算要等本段结束），
    该内核运行期间另一条流上没有任何工作。
    原因（2026-09-19 二分）：12 万 token 以上的上下文里，成对发射的输出不可复现（同一提示词三次三个样），
    而「两块共用一条流」「两条流但 GPU 全串行」两种对照都与不成对逐字相同；通信换回 bf16 后仍不可复现。
    即：一块的 RCCL 通信与另一块的计算在 GPU 上真并发时，计算结果会偶发出错；只在全注意力内核很长（上下文很长）时出现。"""
    def w(*a, **kw):
        if S.active and S.role is not None and _NSTREAMS >= 2 and not S.noexcl:
            oc = S.ev_comm.get("B" if S.role == "A" else "A")
            if oc is not None:
                torch.cuda.current_stream().wait_event(oc)
            if not S.excl_logged:
                S.excl_logged = True
                print("[pair_prefill] ✅ 首次真正生效：注意力内核启动前等待另一块的通信结束", flush=True)
        return fn(*a, **kw)
    return w


def _install_safe_align():
    if not _SAFE_ALIGN:
        return
    try:
        import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as FM
        from ..ops.safe_align import safe_moe_align_block_size
    except Exception as e:
        print(f"[pair_prefill] ⚠ 并发安全对齐不可用（{type(e).__name__}: {e}），成对期间仍用原实现", flush=True)
        return
    orig = FM.moe_align_block_size
    if getattr(orig, "_pp_safe", False):
        return

    def dispatch(topk_ids, block_size, num_experts, *a, **kw):
        if S.active and not os.path.exists(_LIGHTOP_FILE):
            S.n_safe += 1
            if S.n_safe == 1:
                print(f"[pair_prefill] ✅ 首次真正生效（并发安全对齐）：topk_ids {tuple(topk_ids.shape)} 块长 {block_size}", flush=True)
            return safe_moe_align_block_size(topk_ids, block_size, num_experts, *a, **kw)
        return orig(topk_ids, block_size, num_experts, *a, **kw)
    dispatch._pp_safe = True
    FM.moe_align_block_size = dispatch
    S.safe_align = True
    print(f"[pair_prefill] 已安装：成对期间 MoE 对齐排序走并发安全实现（原实现 {getattr(orig, '__qualname__', orig)}）；"
          f"MoE 段与通信重叠={'开' if _MOE_OVERLAP else '关'}", flush=True)


def _setup_once(sched):
    _install_safe_align()
    S.stream_B = _pick_stream_B(sched.forward_stream)
    try:
        mw0 = sched.model_worker
        S.n_layers = int(getattr(mw0, "target_worker", mw0).model_runner.model_config.num_hidden_layers)
    except Exception as e:
        S.n_layers = 0
        S.force_serial = True
        print(f"[pair_prefill] ⚠ 取不到层数（{type(e).__name__}），所有段都不与通信重叠", flush=True)
    mw = sched.model_worker
    tgt = getattr(mw, "target_worker", mw)
    ab = tgt.model_runner.attn_backend
    S.backends = [ab] + list(getattr(ab, "attn_backend_list", []))
    # 图外执行器的 load_batch 把每批输入拷进固定的静态缓冲并返回其视图；两批交错时后一批会覆盖前一批仍在用的缓冲
    # （KV 写入位置、长度等）。成对期间各持副本，等价于框架的 SGLANG_EAGER_INPUT_NO_COPY 路径。
    er = getattr(tgt.model_runner, "eager_runner", None)
    if er is not None and not getattr(er, "_pp_wrapped", False):
        orig_lb = er.load_batch

        def _load_batch(forward_batch, *a, **kw):
            if S.active and S.role is not None:
                return dataclasses.replace(forward_batch)
            return orig_lb(forward_batch, *a, **kw)

        er.load_batch = _load_batch
        er._pp_wrapped = True
    # 全注意力内核独占 GPU：见 _exclusive 的说明
    n_ex = 0
    for b in S.backends:
        if type(b).__name__ in _EXCL_BACKENDS and not getattr(b, "_pp_excl", False):
            b.forward_extend = _exclusive(b.forward_extend)
            b._pp_excl = True
            n_ex += 1
    print(f"[pair_prefill] 挂钩：{n_ex} 个注意力后端的预填充内核独占 GPU（{sorted(_EXCL_BACKENDS)}）", flush=True)
    print(f"[pair_prefill] 挂钩：后端 {[type(b).__name__ for b in S.backends]}；"
          f"静态输入缓冲绕过={'已装' if er is not None else '未找到 eager_runner'}；"
          f"worker={type(mw).__name__} target={type(tgt).__name__}", flush=True)
    dw = getattr(mw, "draft_worker", None)
    if dw is not None and hasattr(dw, "_draft_extend_for_prefill") and not getattr(dw, "_pp_wrapped", False):
        orig = dw._draft_extend_for_prefill

        def _wrapped(*a, **kw):
            disable_handoff_in_current()            # 草稿层扩展独占执行，不在其 all_reduce 处交接
            if S.active and S.role == "A" and _NSTREAMS >= 2 and S.ev_comm.get("B") is not None:
                torch.cuda.current_stream().wait_event(S.ev_comm["B"])     # 草稿层里也有 MoE：避开块 B 最后一次通信
            if S.active and S.role == "B" and S.stream_A is not None:
                # 块 B 的草稿层要读块 A 的草稿 KV，而块 A 的草稿层扩展排在它的目标前向之后、不在逐段完成事件的覆盖范围内。
                # 此时块 A 的协程已结束（全部工作已入队），等它的流即可。
                torch.cuda.current_stream().wait_stream(S.stream_A)
            return orig(*a, **kw)

        dw._draft_extend_for_prefill = _wrapped
        dw._pp_wrapped = True
    print(f"[pair_prefill] 挂钩：草稿层扩展不交接={'已装' if dw is not None and getattr(dw, '_pp_wrapped', False) else '未装'}", flush=True)


def _report_timing(tl, t_start):
    """诊断输出。CPU 侧：某一段远大于中位 = 该段里有主机同步点。GPU 侧：事件成对出现（计算结束, 通信结束），
    计算段 = 上一次通信结束 → 本次计算结束（含等待另一块的时间），通信段 = 计算结束 → 通信结束。"""
    t_l = time.perf_counter()
    torch.cuda.synchronize()
    t_g = time.perf_counter()
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    mode = "串行对照" if os.path.exists(_SEQ_FILE) else ("严格交替" if S.strict else "交错")
    for r in ("A", "B"):
        d = [x for t, x in tl if t == r]
        if not d:
            continue
        top = sorted(((x, i) for i, (t, x) in enumerate(tl) if t == r), reverse=True)[:4]
        print(f"[pair_prefill] CPU计时 第{S.n_pairs}对 块{r}（{mode}）: {len(d)} 段, 合计 {1e3*sum(d):.0f} ms, "
              f"中位 {1e3*sorted(d)[len(d)//2]:.2f} ms, 最长 {[(i, round(1e3*x, 1)) for x, i in top]}", flush=True)
    print(f"[pair_prefill] CPU计时 第{S.n_pairs}对: CPU 发射完 {1e3*(t_l-t_start):.0f} ms, GPU 做完 {1e3*(t_g-t_start):.0f} ms", flush=True)
    tt = {r: [S.ev0.elapsed_time(e) for e in S.evlog[r]] for r in ("A", "B")}

    def med(v):
        return sorted(v)[len(v) // 2] if v else float("nan")

    for r in ("A", "B"):
        t = tt[r]
        n = len(t) // 2
        if n < 4:
            continue
        comp = [t[2*i] - (t[2*i-1] if i else 0.0) for i in range(n)]
        comm = [t[2*i+1] - t[2*i] for i in range(n)]
        print(f"[pair_prefill] GPU计时 第{S.n_pairs}对 块{r}（{mode}）: {n} 段, 计算段合计 {sum(comp[1:]):.0f} ms, 通信段合计 {sum(comm):.0f} ms, "
              f"首段结束 {t[0]:.0f} ms, 末次通信结束 {t[-1]:.0f} ms; 中位 计算/通信: 奇数段 {med(comp[1::2]):.2f}/{med(comm[1::2]):.2f}, "
              f"偶数段 {med(comp[2::2]):.2f}/{med(comm[2::2]):.2f}", flush=True)
    k = 80
    ra = " ".join(f"{x:.1f}" for x in tt["A"][k:k+12])
    rb = " ".join(f"{x:.1f}" for x in tt["B"][k:k+12])
    print(f"[pair_prefill] GPU计时 时间线(ms) 事件 {k}..{k+11}（偶=计算结束 奇=通信结束）\n    A: {ra}\n    B: {rb}", flush=True)


def run_pair(sched, batch_a, batch_b, indep=False, serial=False):
    """交错执行两个预填充批的 run_batch；返回 (result_a, result_b)。indep=True：两块是不同请求、互不依赖。"""
    S.indep = indep
    if _DEBUG_MODE == "seq":
        ra = sched.run_batch(batch_a)
        rb = sched.run_batch(batch_b)
        if not S.logged:
            S.logged = True
            print("[pair_prefill] ✅（调试 seq）预先规划两批后先后执行完成", flush=True)
        return ra, rb

    from greenlet import greenlet
    if S.stream_B is None:
        _setup_once(sched)

    gl = _globals()
    gl_before = [getattr(m, a) for m, a in gl]
    cur_before = torch.cuda.current_stream()
    F, Fctx = sched.forward_stream, sched.forward_stream_ctx
    G = S.stream_B if _NSTREAMS >= 2 else F
    lane = {"A": (F, Fctx), "B": (G, torch.cuda.stream(G))}
    out = {}
    g = {"A": greenlet(lambda: out.__setitem__("A", sched.run_batch(batch_a))),
         "B": greenlet(lambda: out.__setitem__("B", sched.run_batch(batch_b)))}
    # 每块的可恢复状态：当前流、前向上下文、各后端 forward_metadata
    saved = {"A": None, "B": None}
    S.main, S.ev_A, S.ev_B = greenlet.getcurrent(), None, None
    S.ev_comm = {}
    S.stream_A = F if _NSTREAMS >= 2 else None
    S.strict = _STRICT and not os.path.exists(_NOSTRICT_FILE)
    S.noexcl = os.path.exists(_NOEXCL_FILE)
    if S.toggles_logged != (S.strict, S.noexcl):
        S.toggles_logged = (S.strict, S.noexcl)
        print(f"[pair_prefill] 本对起生效：严格交替={'开' if S.strict else '关'}  注意力独占={'关' if S.noexcl else '开'}", flush=True)
    S.handoff = {"A": True, "B": True}
    S.active = True
    tl, t_start = [], time.perf_counter()
    seg_no = {"A": -1, "B": -1}
    if _TIMING:
        S.evlog = {"A": [], "B": []}
        S.ev0 = torch.cuda.Event(enable_timing=True)
        S.ev0.record()
    try:
        turn = "A"
        while not (g["A"].dead and g["B"].dead):
            if g[turn].dead:
                turn = "B" if turn == "A" else "A"
                continue
            # ---- 切入
            S.role = turn
            sched.forward_stream, sched.forward_stream_ctx = lane[turn]
            sv = saved[turn]
            if sv is None:
                for m, a in gl:
                    setattr(m, a, None)            # 每块从「无前向上下文」开始，由它自己的前向去设置
                torch.cuda.set_stream(cur_before)
            else:
                torch.cuda.set_stream(sv[0])
                for (m, a), v in zip(gl, sv[1]):
                    setattr(m, a, v)
                for b, md in sv[2]:
                    b.forward_metadata = md
            if turn == "B" and S.ev_A is not None and _NSTREAMS >= 2 and not S.indep:
                # 块 B 的这一段依赖块 A 同一段写入的 KV 与 GDN 状态：只等 A 的计算，不等 A 的通信
                (sv[0] if sv is not None else G).wait_event(S.ev_A)
            seg_no[turn] += 1                       # 本块即将执行的段号：0=嵌入，奇数=注意力/GDN，偶数(≥2)=MoE
            is_moe = 2 <= seg_no[turn] <= 2 * S.n_layers and seg_no[turn] % 2 == 0
            if (_DEBUG_MODE == "serial2" or S.force_serial or serial or (is_moe and not _MOE_OVERLAP)
                    or (_DEBUG_MODE == "serial_odd" and seg_no[turn] % 2 == 1)) and _NSTREAMS >= 2:
                # MoE 段不与另一块的通信重叠（见文件头「MoE 与通信不可真并发」）；serial2 / force_serial：所有段都不重叠
                oc = S.ev_comm.get("A" if turn == "B" else "B")
                if oc is not None:
                    (sv[0] if sv is not None else lane[turn][0]).wait_event(oc)
            if turn == "A" and S.strict and S.ev_B is not None and sv is not None:
                sv[0].wait_event(S.ev_B)           # 严格交替：块 A 的下一段计算等块 B 本段计算结束，两块的计算不同时在 GPU 上
            t_in = time.perf_counter()
            g[turn].switch()
            if _TIMING:
                tl.append((turn, time.perf_counter() - t_in))
            # ---- 切出
            saved[turn] = (torch.cuda.current_stream(), [getattr(m, a) for m, a in gl],
                           [(b, getattr(b, "forward_metadata", None)) for b in S.backends])
            S.role = None
            turn = "B" if turn == "A" else "A"
    finally:
        S.active, S.role = False, None
        sched.forward_stream, sched.forward_stream_ctx = F, Fctx
        torch.cuda.set_stream(cur_before)
        for (m, a), v in zip(gl, gl_before):
            setattr(m, a, v)
        if _NSTREAMS >= 2:
            F.wait_stream(G)                       # 之后发到 F 上的工作都排在 B 之后
            try:
                sched.schedule_stream.wait_stream(G)
            except Exception:
                pass
    S.n_pairs += 1
    if serial and not S.mm_logged:
        S.mm_logged = True
        print(f"[pair_prefill] 含图请求的一对按全程不重叠执行（段数 A={seg_no['A'] + 1} B={seg_no['B'] + 1}），之后的纯文本对照常重叠", flush=True)
    if S.n_layers and not S.force_serial and not serial and (seg_no["A"] + 1 != 2 * S.n_layers + 2 or seg_no["B"] + 1 != 2 * S.n_layers + 2):
        # 「偶数段(≥2) = MoE」依赖每块恰好 2L+1 次 all_reduce（嵌入一次、每层注意力后与 MoE 后各一次），即 2L+2 段
        # （段 0 嵌入，段 2L+1 是末次通信之后的收尾）；结构不符就不再区分，全部不重叠
        S.force_serial = True
        print(f"[pair_prefill] ⚠ 每块段数 A={seg_no['A'] + 1} B={seg_no['B'] + 1}，与 2×{S.n_layers}+2 不符：无法按段号识别 MoE 段，"
              "此后所有段都不与通信重叠（成对发射不再有收益，但结果安全）", flush=True)
    if _TIMING and S.n_pairs <= _TIMING:
        try:
            _report_timing(tl, t_start)
        except Exception as e:
            print(f"[pair_prefill] 计时输出失败: {type(e).__name__}: {e}", flush=True)
    if not S.logged:
        S.logged = True
        print(f"[pair_prefill] ✅ 首次真正生效：两个预填充块交错发射完成，交接 {S.n_switch} 次"
              f"（块 A {batch_a.extend_num_tokens} token，块 B {batch_b.extend_num_tokens} token，{_NSTREAMS} 条流）", flush=True)
    return out["A"], out["B"]

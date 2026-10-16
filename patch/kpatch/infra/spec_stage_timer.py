"""投机 decode 每轮分段计时：draft / verify / draft_extend 三段各记 CPU 起止与 GPU event，外加 resolve_seq_lens_cpu 的阻塞时长。
不插同步：event 留在队列里，4 轮后 query() 已完成才读 elapsed_time。只记 TP0、只记 decode 轮、图捕获期跳过。默认关。

为什么需要：09-25 抽测 c1 每轮 52 ms 而 hy-smi 利用率仅 33–38%；py-spy 指向 HIP 分支的 publish_ready.synchronize()
（overlap_utils.resolve_seq_lens_cpu，注释「Event.wait() regresses TPOT on AMD MI355」）——CPU 等到上一轮 verify 结束才开始备下一轮，
能与 GPU 重叠的只剩 draft_extend。profiler trace 会把每轮拉长 3.5 倍、不能量空闲，故用 event。
每轮一行 JSON（SGLANG_SPEC_TIMER_FILE，默认 /tmp/spec_timer.jsonl）：
  bs；cpu_* = 各段 CPU 耗时 ms；gpu_* = 同段首尾 event 间 GPU 时长 ms（含段内因 CPU 派发慢造成的空转）；
  gap_prev = 上一轮末段结束 → 本轮首段开始的 GPU 空档；gap_A__B = 段间空档；resolve = 本轮开始前 resolve 阻塞 ms。
开关 SGLANG_SPEC_TIMER=1。
"""
import collections
import contextlib
import json
import os
import time

import torch

_st = {"f": None, "cur": None, "q": collections.deque(), "prev_end": None, "resolve": 0.0, "logged": False, "n": 0}


def _rank0():
    try:
        return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    except Exception:
        return True


def _ev():
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def _flush_ready(path):
    q = _st["q"]
    while len(q) > 4:
        r = q[0]
        if not r["e"][-1][1].query():
            return
        q.popleft()
        out = {"bs": r["bs"], "resolve": round(r["resolve"], 3)}
        for name, e0, e1, c0, c1 in r["s"]:
            out["cpu_" + name] = round((c1 - c0) * 1e3, 3)
            out["gpu_" + name] = round(e0.elapsed_time(e1), 3)
        names = [s[0] for s in r["s"]]
        first = {}
        for st_name, g0, g1, gc0, gc1 in r["g"]:
            out["graph_" + st_name] = round(out.get("graph_" + st_name, 0) + g0.elapsed_time(g1), 3)
            out["cpu_launch_" + st_name] = round(out.get("cpu_launch_" + st_name, 0) + (gc1 - gc0) * 1e3, 3)
            first.setdefault(st_name, (g0, g1))
        for name, e0, e1, c0, c1 in r["s"]:
            if name in first:
                out["pre_" + name] = round(e0.elapsed_time(first[name][0]), 3)   # 段开始 → 首次回放开始
                out["cpu_pre_" + name] = round(([g[3] for g in r["g"] if g[0] == name][0] - c0) * 1e3, 3)
        for (a, _, a1, _, _), (b, b0, _, _, _) in zip(r["s"], r["s"][1:]):
            out[f"gap_{a}__{b}"] = round(a1.elapsed_time(b0), 3)
        if r["prev_end"] is not None and r["s"]:
            out["gap_prev"] = round(r["prev_end"].elapsed_time(r["s"][0][1]), 3)
        out["cpu_round"] = round((r["t1"] - r["t0"]) * 1e3, 3)
        out["gpu_round"] = round(r["s"][0][1].elapsed_time(r["s"][-1][2]), 3)   # 本轮首段开始 → 末段结束
        out["stages"] = ",".join(names)
        if _st["f"] is None:
            _st["f"] = open(path, "a", buffering=1)
        _st["f"].write(json.dumps(out, separators=(",", ":")) + "\n")
        if not _st["logged"]:
            _st["logged"] = True
            print(f"[spec_timer] ✅ 首次真正生效：{out}", flush=True)


def install(mod):
    if getattr(mod, "_spec_timer_installed", False):
        return
    path = os.environ.get("SGLANG_SPEC_TIMER_FILE", "/tmp/spec_timer.jsonl")
    from sglang.srt.managers import overlap_utils as ou
    hooks = []   # 本镜像（Qwen3.8，sglang 0.5.12）没有 resolve_seq_lens_cpu / spec_stage_span / 图后端 replay，各挂点有才装

    orig_resolve = getattr(ou.FutureMap, "resolve_seq_lens_cpu", None)

    def resolve_seq_lens_cpu(self, batch):
        t0 = time.perf_counter()
        try:
            return orig_resolve(self, batch)
        finally:
            _st["resolve"] += (time.perf_counter() - t0) * 1e3

    if orig_resolve is not None:
        ou.FutureMap.resolve_seq_lens_cpu = resolve_seq_lens_cpu
        hooks.append("resolve")

    orig_span = getattr(mod, "spec_stage_span", None)

    @contextlib.contextmanager
    def _stage(name):
        r = _st["cur"]
        if r is None or torch.cuda.is_current_stream_capturing():
            yield
            return
        c0, e0 = time.perf_counter(), _ev()
        _st["stage"] = name
        try:
            yield
        finally:
            _st["stage"] = None
            r["s"].append((name, e0, _ev(), c0, time.perf_counter()))

    @contextlib.contextmanager
    def spec_stage_span(name):
        with orig_span(name), _stage(name):
            yield

    if orig_span is not None:
        mod.spec_stage_span = spec_stage_span
        hooks.append("spec_stage_span")
    else:   # 旧版：直接包草稿 / 草稿扩展方法（verify 下面统一包）
        dw = mod.EagleDraftWorker
        for meth, name in (("draft", "draft"), ("_draft_extend_for_decode", "draft_extend")):
            om = getattr(dw, meth, None)
            if om is None:
                continue

            def wrapped(self, *a, __om=om, __n=name, **kw):
                with _stage(__n):
                    return __om(self, *a, **kw)

            setattr(dw, meth, wrapped)
            hooks.append(meth)

    # 图回放前后各一个 event：把段内 GPU 跨度拆成「回放前空等（CPU 还在备输入）+ 图执行 + 回放后」
    try:
        from sglang.srt.model_executor.runner_backend import full_cuda_graph_backend as fgb
        orig_replay = fgb.FullCudaGraphBackend.replay
    except Exception:
        fgb = orig_replay = None

    def replay(self, *a, **kw):
        r, stage = _st["cur"], _st.get("stage")
        if r is None or stage is None:
            return orig_replay(self, *a, **kw)
        c0, e0 = time.perf_counter(), _ev()
        out = orig_replay(self, *a, **kw)
        r["g"].append((stage, e0, _ev(), c0, time.perf_counter()))
        return out

    if orig_replay is not None:
        fgb.FullCudaGraphBackend.replay = replay
        hooks.append("replay")

    cls = mod.EAGLEWorkerV2
    orig_verify, orig_fbg = cls.verify, cls.forward_batch_generation

    def verify(self, *a, **kw):
        with _stage("verify"):
            return orig_verify(self, *a, **kw)

    def forward_batch_generation(self, batch, *a, **kw):
        if (batch.forward_mode.is_extend() or getattr(batch, "is_extend_in_batch", False) or not _rank0()
                or torch.cuda.is_current_stream_capturing()):
            return orig_fbg(self, batch, *a, **kw)
        r = {"bs": int(batch.seq_lens.shape[0]), "s": [], "g": [], "resolve": _st["resolve"], "prev_end": _st["prev_end"],
             "t0": time.perf_counter()}
        _st["resolve"] = 0.0
        _st["cur"] = r
        try:
            return orig_fbg(self, batch, *a, **kw)
        finally:
            _st["cur"] = None
            r["t1"] = time.perf_counter()
            if r["s"]:
                r["e"] = [(s[0], s[2]) for s in r["s"]]
                _st["prev_end"] = r["s"][-1][2]
                _st["q"].append(r)
                try:
                    _flush_ready(path)
                except Exception as e:
                    if not _st.get("err"):
                        _st["err"] = True
                        print(f"[spec_timer] ⚠ 记录失败（{e!r}），之后静默", flush=True)

    cls.verify, cls.forward_batch_generation = verify, forward_batch_generation
    mod._spec_timer_installed = True
    print(f"[spec_timer] 已安装 (pid={os.getpid()}) → {path}，挂点 {hooks + ['verify']}", flush=True)

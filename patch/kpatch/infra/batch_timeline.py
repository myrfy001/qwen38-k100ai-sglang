"""调度时间线：每个批次发射（L）与结果取回（D）各记一行毫秒时间戳。纯记录，不插同步、不改行为，默认关。

为什么需要：服务日志的 Prefill/Decode 行只有秒级时间戳、预填充批没有耗时，长请求「谁在等谁」还原不出来
（Qwen 项目靠逐请求 ITL + 秒级日志推，分辨率不够）。有了这个就能直接画出每条请求的
排队 → 分块预填充 → 首 token → decode 时间线，并算出 decode 被预填充挡住的时长。
口径：event_loop_overlap 里 L 是 CPU 发射时刻；D 是 process_batch_result 入口（已同步取回结果），
重叠调度下 D 比 GPU 真正完成晚「下一批的发射耗时」，量级 ms。
开关 SGLANG_BATCH_TIMELINE=1；输出 SGLANG_BATCH_TIMELINE_FILE（默认 /tmp/batch_timeline.jsonl），只记 tp_rank 0。
每行 JSON：t（epoch 秒，微秒精度）、ev（L/D）、it（批序号）、mode、bs、ntok、以及预填充批的逐请求 [rid 前 8 位, 本块 token, 已有前缀, 输入总长]。
"""
import json
import os
import time

_st = {"f": None, "logged": False}


def _desc(batch):
    d = {"mode": getattr(batch.forward_mode, "name", str(batch.forward_mode)), "bs": len(batch.reqs)}
    if batch.forward_mode.is_extend():
        el, pl = getattr(batch, "extend_lens", None), getattr(batch, "prefix_lens", None)   # copy() 也保留这两项快照
        d["ntok"] = int(sum(el)) if el else int(getattr(batch, "extend_num_tokens", 0) or 0)
        d["reqs"] = [[r.rid[:8], int(el[i]) if el else -1, int(pl[i]) if pl else -1, len(r.origin_input_ids)]
                     for i, r in enumerate(batch.reqs)]
    else:
        d["ntok"] = len(batch.reqs)
        d["rids"] = [r.rid[:8] for r in batch.reqs]
    return d


def install(mod):
    if os.environ.get("SGLANG_BATCH_TIMELINE", "0") != "1" or getattr(mod, "_batch_timeline_installed", False):
        return
    cls = mod.Scheduler
    orig_run, orig_proc = cls.run_batch, cls.process_batch_result
    path = os.environ.get("SGLANG_BATCH_TIMELINE_FILE", "/tmp/batch_timeline.jsonl")

    def _emit(self, ev, batch):
        ps = getattr(self, "ps", None)   # 此版 sglang 把并行秩放在 self.ps（09-25 首版读 self.tp_rank 恒缺省，8 个进程全写）
        if getattr(ps, "tp_rank", getattr(self, "tp_rank", 0)) != 0:
            return
        try:
            if _st["f"] is None:
                _st["f"] = open(path, "a", buffering=1)
            rec = {"t": round(time.time(), 6), "ev": ev, "it": getattr(batch, "forward_iter", -1)}   # D 侧 copy() 不带 forward_iter，按 FIFO 与 L 配对
            rec.update(_desc(batch))
            _st["f"].write(json.dumps(rec, separators=(",", ":")) + "\n")
            if not _st["logged"]:
                _st["logged"] = True
                print(f"[batch_timeline] ✅ 首次真正生效：{ev} {rec['mode']} bs={rec['bs']} → {path}", flush=True)
        except Exception as e:   # 记录失败绝不影响服务
            if not _st.get("err"):
                _st["err"] = True
                print(f"[batch_timeline] ⚠ 记录失败（{e!r}），之后静默", flush=True)

    def run_batch(self, batch, *a, **kw):
        out = orig_run(self, batch, *a, **kw)
        _emit(self, "L", batch)   # 发射后记：forward_iter 在 run_batch 里才赋值
        return out

    def process_batch_result(self, batch, result, *a, **kw):
        _emit(self, "D", batch)
        return orig_proc(self, batch, result, *a, **kw)

    cls.run_batch, cls.process_batch_result = run_batch, process_batch_result
    mod._batch_timeline_installed = True
    print(f"[batch_timeline] 已安装 (pid={os.getpid()}) → {path}", flush=True)

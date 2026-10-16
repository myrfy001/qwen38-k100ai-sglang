"""让先完成预填充的请求尽快送出首个「可见」token。

现象（2026-09-19，逐请求 TTFT 明细 + 并发流式探针）：
 ① 重叠调度是「先规划并发射下一批，再处理上一批的结果」。图外预填充的 CPU 发射要 250 ms 以上，
    先完成的请求的首 token 被压到后一条请求发射完才送出：4096 c2 逐请求 TTFT [627, 684]，串行理论值 [348, 696]。
    关掉整个重叠调度 TTFT −7%~−21%，但 TPOT +16%，不可取。
 ② 压测端只在流式分片的 text 非空时才记 TTFT。首 token 若是不完整的 UTF-8 字节（约 1/8~1/4 的请求），
    首分片文本为空，要等下一轮 decode 才成字；而 decode 被排在所有待做的预填充之后，
    该请求的 TTFT 就从「自己那次预填充结束」变成「全部预填充结束」。
做法：
 模式 1：连续两个预填充批之间先处理前一批结果（等价于框架内置 SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP）。
 模式 2：上一批是预填充时，先处理其结果、再规划下一批（顺序与非重叠事件循环一致）；decode→decode 仍保持重叠，TPOT 不受影响。
 模式 3：在模式 2 基础上，若刚完成预填充的请求首 token 解码为不完整字符，先插一轮 decode（约 20 ms）再继续预填充；
         仅在没有进行中的分块请求时插入（框架在分块请求进行中从不跳过预填充，保持这一不变式）。
模式 2/3 需替换 event_loop_overlap；安装前校验原函数源码指纹，不符则退回模式 1 的做法并告警。
"""
import hashlib
import inspect
import os
from collections import deque

_MODE = int(os.environ.get("SGLANG_FIRST_TOKEN_FIRST", "0"))
try:
    from . import pair_prefill as _PP   # 同一请求相邻块成对交错发射（SGLANG_PAIR_PREFILL=1），见该文件头
except Exception:
    _PP = None
_LOOP_MD5 = "8c61826770145363a0e3f11e1cf7e494"   # inspect.getsource(Scheduler.event_loop_overlap) @ sglang-das 93b80182
_st = {"n_early": 0, "n_dec": 0, "log_early": False, "log_dec": False, "tok_cache": {}}


def _incomplete(self, tid):
    c = _st["tok_cache"]
    if tid not in c:
        try:
            t = self.tokenizer.decode([tid])
            c[tid] = (t == "") or ("�" in t)
        except Exception:
            c[tid] = False
    return c[tid]


def _pp_first_ok(self, batch):
    """块 k：图外的预填充批、只含一个请求、且该请求还有后续块。"""
    try:
        return (batch is not None and batch.forward_mode.is_extend() and len(batch.reqs) == 1
                and self.chunked_req is not None and batch.reqs[0] is self.chunked_req
                and (not batch.return_logprob or _PP_ALLOW_LOGPROB) and batch.extend_num_tokens > _PP_MIN_TOKENS
                and getattr(self, "_pp_carry", None) is None
                and not os.path.exists(_PP_OFF_FILE))
    except Exception:
        return False


def _pp_second_ok(a, b):
    """块 k+1：同一请求的下一块，同样在图外。"""
    try:
        return (b is not None and b.forward_mode.is_extend() and len(b.reqs) == 1 and b.reqs[0] is a.reqs[0]
                and (not b.return_logprob or _PP_ALLOW_LOGPROB) and b.extend_num_tokens > _PP_MIN_TOKENS)
    except Exception:
        return False


_PP_OFF_FILE = os.environ.get("SGLANG_PAIR_PREFILL_OFF_FILE", "/tmp/pair_off")   # 存在则不成对（同实例 A/B 用；只在空闲时切换）
# 仅供验证：允许带 return_logprob 的请求成对（用首个输出 token 的 logprob 浮点值作逐位检测）
_PP_ALLOW_LOGPROB = os.environ.get("SGLANG_PAIR_PREFILL_ALLOW_LOGPROB", "0") != "0"
_PP_MIN_TOKENS = int(os.environ.get("SGLANG_PAIR_PREFILL_MIN_TOKENS", "2432"))   # 不大于最大预填充图桶的批走图，不成对


def install(mod):
    S = getattr(mod, "Scheduler", None)
    if _MODE == 0 or S is None or getattr(S, "_ftf_installed", False):
        return
    mode = _MODE
    if mode >= 2:
        try:
            src = inspect.getsource(S.event_loop_overlap)
            md5 = hashlib.md5(src.encode()).hexdigest()
        except Exception as e:
            md5 = f"取源码失败:{e}"
        if md5 != _LOOP_MD5:
            print(f"[first_token_first] ⚠ event_loop_overlap 源码指纹不符（{md5}），退回模式 1。sglang-das 版本与本仓库对齐的提交 93b80182 不同：并发测点的首响优化只部分生效，成绩会低于 BASELINE", flush=True)
            mode = 1

    # ---- 模式 1：沿用框架的判定函数，只放宽到「连续预填充」
    orig_dis = S.is_disable_overlap_for_batch

    def is_disable_overlap_for_batch(self, batch, last_batch):
        if orig_dis(self, batch, last_batch):
            return True
        return (mode == 1 and batch is not None and last_batch is not None and len(self.result_queue) > 0
                and batch.forward_mode.is_extend() and last_batch.forward_mode.is_extend())

    S.is_disable_overlap_for_batch = is_disable_overlap_for_batch

    if mode >= 2:
        envs = mod.envs
        NextBatchPlan = mod.NextBatchPlan

        def event_loop_overlap(self):
            """与原实现逐句一致，改动两处：①上一批为预填充时，把「处理上一批结果」提前到规划下一批之前；
            ②（SGLANG_PAIR_PREFILL=1）同一请求的相邻两个图外预填充块成对交错发射。"""
            self.result_queue = deque()
            pair_on = bool(_PP is not None and _PP.ENABLED)
            if pair_on:
                print(f"[pair_prefill] 已启用：同一请求相邻块成对发射（块 token 数 > {_PP_MIN_TOKENS}）", flush=True)

            def pop_and_process():
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)

            while True:
                if self.gracefully_exit:
                    break
                recv_reqs = self.request_receiver.recv_requests()
                self.process_input_requests(recv_reqs)
                if self._engine_paused:
                    continue
                self._apply_war_barrier()

                early = False
                lb = self.last_batch
                flush_dec = getattr(self, "_ftf_flush_next", False)     # 上一批是为「首 token 不成字」插入的 decode：其结果同样要立刻处理
                if lb is not None and (lb.forward_mode.is_extend() or flush_dec) and len(self.result_queue) > 0:
                    self._ftf_flush_next = False
                    while self.result_queue:
                        pop_and_process()
                    early = True
                    _st["n_early"] += 1
                    if mode >= 3 and self.tokenizer is not None and lb.forward_mode.is_extend():
                        for req in lb.reqs:
                            out = getattr(req, "output_ids", None)
                            if out and len(out) == 1 and not req.finished() and _incomplete(self, out[0]):
                                self._ftf_need_decode = True
                                break
                    if not _st["log_early"]:
                        _st["log_early"] = True
                        print(f"[first_token_first] ✅ 首次真正生效：预填充结果先于下一批规划处理（模式 {mode}）", flush=True)

                carry = getattr(self, "_pp_carry", None)
                if carry is not None:                  # 上一轮为成对而预先规划、但没能成对的那一批
                    plan, self._pp_carry = carry, None
                else:
                    plan = self.get_next_batch_to_run(running_batch=self.running_batch, last_batch=self.last_batch)
                self.running_batch = plan.running_batch
                batch = plan.batch_to_run
                self.cur_batch_for_debug = batch

                pair_b = None
                if pair_on and _pp_first_ok(self, batch):
                    # 用副本作 last_batch：规划会把分块请求从 last_batch 里过滤掉，而块 k 此时尚未发射
                    plan2 = self.get_next_batch_to_run(running_batch=self.running_batch, last_batch=batch.copy())
                    self.running_batch = plan2.running_batch
                    b2 = plan2.batch_to_run
                    if _pp_second_ok(batch, b2):
                        pair_b = b2
                    else:
                        self._pp_carry = plan2
                disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch, last_batch=self.last_batch)

                if disable_overlap_for_batch and not early:
                    pop_and_process()
                    if self.enable_unified_memory:
                        try:
                            self.token_to_kv_pool_allocator.flush_opportunistic()
                        except Exception:
                            pass

                n_before = len(self.result_queue)
                if pair_b is not None:
                    ra, rb = _PP.run_pair(self, batch, pair_b)
                    self.result_queue.append((batch.copy(), ra))
                    self.result_queue.append((pair_b.copy(), rb))
                    if self.is_generation:
                        self.launch_batch_sample_if_needed(ra, batch)
                    batch, batch_result = pair_b, rb
                elif batch:
                    batch_result = self.run_batch(batch)
                    self.result_queue.append((batch.copy(), batch_result))
                else:
                    batch_result = None

                if self.last_batch:
                    if not disable_overlap_for_batch and not early:
                        for _ in range(n_before):      # 处理本轮发射之前已在队列里的结果
                            pop_and_process()
                elif batch is None:
                    self.on_idle()

                if self.is_generation:
                    self.launch_batch_sample_if_needed(batch_result, batch)

                self.last_batch = batch

                if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                    self.invariant_checker.self_check_during_busy()

        S.event_loop_overlap = event_loop_overlap

        if mode >= 3:
            orig_prefill = S.get_new_batch_prefill

            def get_new_batch_prefill(self, running_batch):
                if (getattr(self, "_ftf_need_decode", False) and self.chunked_req is None
                        and not running_batch.is_empty() and not running_batch.is_prefill_only):
                    self._ftf_need_decode = False
                    _st["n_dec"] += 1
                    if not _st["log_dec"]:
                        _st["log_dec"] = True
                        print("[first_token_first] ✅ 首次真正生效：首 token 为不完整字符，先插一轮 decode 再继续预填充", flush=True)
                    self._ftf_flush_next = True
                    return NextBatchPlan(batch_to_run=None, running_batch=running_batch)
                self._ftf_need_decode = False
                return orig_prefill(self, running_batch)

            S.get_new_batch_prefill = get_new_batch_prefill

    S._ftf_installed = True
    print(f"[first_token_first] 已安装 模式={mode} (pid={os.getpid()})", flush=True)

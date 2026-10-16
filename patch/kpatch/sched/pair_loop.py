"""Qwen3.8（sglang 0.5.12 镜像）的调度循环：不同请求的两个预填充批成对交错发射，让一批的跨卡通信与另一批的计算重叠。

为什么（10-09 l3_ledger，T8 配置关图 4096×3）：预填充 26% 是 quant_ar 的 int8 通信（PCIe 9.4 GB/s，已到可达 76%），与计算零重叠。
    4K c32 的 SLA 卡在 TTFT 均值 10.4–11.0 s（线 10 s）：第一波 32 条同时到达，等长请求排队预填充，均值≈16.5×单条预填充时间；
    过线要单条预填充快约 5%。Qwen3.6 的 pair_prefill（同一请求相邻块）实测通信约 27% 藏进另一块计算（MoE 段不重叠的安全版 −3~−6%）；
    本模型是稠密的，没有 MoE 段的限制（那条的根因是 lightop moe_align，见 pair_prefill.py 文件头），所有段都可重叠。
与 Qwen3.6 版的差别：
    ① 本镜像的 event_loop_overlap 与 das 0.5.15 指纹不同（27B 项目因此没能用成对），这里按本镜像源码逐句照抄，只在发射处加成对分支；
    ② 成对的是**不同请求**的两批（v2）：4K 请求只有一块，同请求相邻块的配对（v1）用不上。两批互不依赖，块 B 不等块 A 的同段计算。
    ③ 长请求（单请求、还有后续块）走 v1：同请求相邻两块，块 B 每段等块 A 同段完成（KV 与 GDN 状态依赖）；
       规划方式与 Qwen3.6 first_token_first 相同（以块 k 副本作 last_batch 再规划一次），第二块不合格则留到下一轮直接跑。
       开关 SGLANG_PAIR_CHUNKS（默认 1）。
做法：
    get_next_batch_to_run() 规划出预填充批 A 后，若 A 不含分块请求、本调度器也没有在飞的分块请求，就直接调 get_new_batch_prefill()
    从等待队列再规划一批 B（不经 last_batch 合并：A 尚未运行，没有 spec_info，不能合进运行批）。两批交给 pair_prefill.run_pair(indep=True)。
    发射后：先处理 A 的结果（CPU 等 A 做完，此时 B 仍在 GPU 上跑，不产生空泡），按 get_next_batch_to_run 里同样的规则把 A 过滤、合进运行批；
    B 作为 last_batch 走原流程（下一轮规划时合并、结果在下一批发射后处理）。
    规划 B 时 A 的请求还不在运行批里：请求数上限会少算 A 的条数，但 KV/mamba 槽位是真实分配的，不会超额。
开关：SGLANG_PAIR_PREFILL=1（与 pair_prefill 同一开关）；SGLANG_PAIR_MIN_TOKENS（A 的 token 数下限，默认 2048）。
    运行时对照：存在 /tmp/pair_off 则不成对（同实例 A/B）。带 return_logprob 的批默认不成对，SGLANG_PAIR_PREFILL_ALLOW_LOGPROB=1 放开（逐位检测用）。
"""
import inspect
import os
from collections import deque

_MIN = int(os.environ.get("SGLANG_PAIR_MIN_TOKENS", "2048"))
_ALLOW_LP = os.environ.get("SGLANG_PAIR_PREFILL_ALLOW_LOGPROB", "0") != "0"
_OFF = "/tmp/pair_off"
_V1 = os.environ.get("SGLANG_PAIR_CHUNKS", "1") != "0"   # 同一请求相邻块成对（v1，长请求）
_st = {"pairs": 0, "solo_none": 0, "logged": False, "merge_logged": False, "v1": 0, "v1_logged": False, "carry": 0}

# 本镜像原实现里必须出现的语句（指纹）：不符就不安装，免得照抄的循环与框架脱节
_FP = ["batch = self.get_next_batch_to_run()", "disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch)",
       "self.launch_batch_sample_if_needed(batch_result)", "self.result_queue.append((batch.copy(), batch_result))",
       "self.self_check_during_busy()"]


def _merge_prefill_into_running(self, b):
    """与 get_next_batch_to_run 里「Merge the prefill batch into the running batch」一段相同的规则。"""
    excl = []
    if self.chunked_req is not None:
        excl.append(self.chunked_req)
    if b.chunked_req is not None:
        excl.append(b.chunked_req)
    bs0 = b.batch_size()
    b.filter_batch(chunked_req_to_exclude=excl)
    if b.batch_size() < bs0:
        self.running_batch.batch_is_full = False
    if not b.is_empty():
        if self.running_batch.is_empty():
            self.running_batch = b
        else:
            self.running_batch.merge_batch(b)
    if not _st["merge_logged"]:
        _st["merge_logged"] = True
        print(f"[pair_loop] ✅ 首次真正生效：批 A 结果已处理并合入运行批（{bs0}→{b.batch_size()} 条，运行批 {self.running_batch.batch_size()} 条）", flush=True)


def _has_mm(batch):
    """带图/视频的请求：视觉编码器在 TP 下有额外 all_reduce，每块段数 ≠ 2L+2，按段号认 MLP 段会认错（10-11 精度评测 MathVision 实见 A=242 B=298）。"""
    try:
        return any(getattr(r, "multimodal_inputs", None) is not None for r in batch.reqs)
    except Exception:
        return True


def _v1_first_ok(self, batch):
    """块 k：只含一个请求、该请求还有后续块（Qwen3.6 v1 的条件）。"""
    try:
        return (_V1 and batch is not None and batch.forward_mode.is_extend() and not batch.forward_mode.is_mixed()
                and len(batch.reqs) == 1 and self.chunked_req is not None and batch.reqs[0] is self.chunked_req
                and (not batch.return_logprob or _ALLOW_LP) and (batch.extend_num_tokens or 0) >= _MIN
                and getattr(self, "_pl_carry", None) is None and not os.path.exists(_OFF) and not _has_mm(batch))
    except Exception:
        return False


def _v1_second_ok(a, b):
    try:
        return (b is not None and b.forward_mode.is_extend() and not b.forward_mode.is_mixed() and len(b.reqs) == 1
                and b.reqs[0] is a.reqs[0] and (not b.return_logprob or _ALLOW_LP) and (b.extend_num_tokens or 0) >= _MIN
                and not _has_mm(b))
    except Exception:
        return False


def _pair_ok(self, batch):
    if batch is None or not batch.forward_mode.is_extend() or batch.forward_mode.is_mixed():
        return False
    if os.path.exists(_OFF):
        return False
    if batch.chunked_req is not None or self.chunked_req is not None:
        return False
    if batch.return_logprob and not _ALLOW_LP:
        return False
    if (batch.extend_num_tokens or 0) < _MIN:
        return False
    if _has_mm(batch):
        return False
    return True


def install(mod):
    from . import pair_prefill as PP
    S = mod.Scheduler
    if getattr(S, "_pair_loop_installed", False):
        return
    if not PP.ENABLED:
        print("[pair_loop] ⚠ pair_prefill 未启用（缺 greenlet？），不安装", flush=True)
        return
    try:
        src = inspect.getsource(S.event_loop_overlap)
    except Exception as e:
        print(f"[pair_loop] ⚠ 取不到原 event_loop_overlap 源码（{type(e).__name__}），不安装", flush=True)
        return
    miss = [x for x in _FP if x not in src]
    if miss:
        print(f"[pair_loop] ⚠ event_loop_overlap 指纹不符（缺 {miss}），不安装", flush=True)
        return
    envs = mod.envs

    def event_loop_overlap(self):
        """与本镜像原实现逐句一致，只在「Launch the current batch」处加成对分支。"""
        self.result_queue = deque()
        print(f"[pair_loop] 已启用：不同请求的两个预填充批成对交错发射（A ≥ {_MIN} token；/tmp/pair_off 关闭）", flush=True)

        def pop_and_process():
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            carry = getattr(self, "_pl_carry", None)
            if carry is not None:          # 上一轮为 v1 预先规划、但没能成对的那一批：直接跑，不重新规划
                batch, self._pl_carry = carry, None
            else:
                batch = self.get_next_batch_to_run()
            self.cur_batch = batch
            disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch)

            pair_b = None
            v1 = False
            if _v1_first_ok(self, batch):
                # 再规划一次：规划开头会暂存块 k（_chunked_req_scheduled_last_iter 已置位），然后规划块 k+1。
                # last_batch 临时置空：Qwen3.6 用块 k 的副本，本镜像 filter_batch 要等 spec_info.verify_done，
                # 未运行的块 k 没有 spec_info 会抛错（10-10 p3_v1）；块 k 是分块请求，本来就不合并进运行批。
                saved = self.last_batch
                self.last_batch = None
                b2 = self.get_next_batch_to_run()
                self.last_batch = saved
                if _v1_second_ok(batch, b2):
                    pair_b, v1 = b2, True
                else:
                    self._pl_carry = b2
                    _st["carry"] += 1
            elif _pair_ok(self, batch):
                pair_b = self.get_new_batch_prefill()
                if pair_b is None:
                    _st["solo_none"] += 1

            if disable_overlap_for_batch:
                pop_and_process()

            if pair_b is not None:
                n_before = len(self.result_queue)
                # B 批是从队列新规划的、退不回去：带图就只让这一对全程不重叠（不再像旧版那样永久关掉重叠）
                ra, rb = PP.run_pair(self, batch, pair_b, indep=not v1, serial=_has_mm(pair_b))
                self.result_queue.append((batch.copy(), ra))
                self.result_queue.append((pair_b.copy(), rb))
                if self.is_generation:
                    self.cur_batch = batch
                    self.launch_batch_sample_if_needed(ra)
                # 先处理本轮之前的结果，再处理 A（CPU 等 A 做完；B 仍在 GPU 上），然后把 A 合进运行批
                while len(self.result_queue) > 1:
                    pop_and_process()
                _merge_prefill_into_running(self, batch)
                _st["pairs"] += 1
                if v1:
                    _st["v1"] += 1
                    if not _st["v1_logged"]:
                        _st["v1_logged"] = True
                        print(f"[pair_loop] ✅ 首次真正生效（v1 同请求相邻块）：块 A {batch.extend_num_tokens} + 块 B {pair_b.extend_num_tokens} token，"
                              f"请求输入 {len(pair_b.reqs[0].origin_input_ids) if pair_b.reqs else -1}", flush=True)
                if not _st["logged"]:
                    _st["logged"] = True
                    print(f"[pair_loop] ✅ 首次真正生效：成对发射 A {batch.batch_size()} 条/{batch.extend_num_tokens} token + "
                          f"B {pair_b.batch_size()} 条/{pair_b.extend_num_tokens} token（之前队列 {n_before}）", flush=True)
                elif _st["pairs"] in (10, 100, 1000):
                    print(f"[pair_loop] 已成对 {_st['pairs']} 次（其中同请求相邻块 {_st['v1']} 次；规划不出 B 而单发 {_st['solo_none']} 次；"
                          f"v1 第二块不合格转下轮 {_st['carry']} 次）", flush=True)
                batch, batch_result = pair_b, rb
                self.cur_batch = batch
                # 上一批（成对之前的）结果已在上面处理完，下面的「Process the last batch」不再弹出
                disable_overlap_for_batch = True
            elif batch:
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            if self.last_batch:
                if not disable_overlap_for_batch:
                    pop_and_process()
            elif batch is None:
                self.on_idle()

            if self.is_generation:
                self.launch_batch_sample_if_needed(batch_result)

            self.last_batch = batch

            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.self_check_during_busy()

    S.event_loop_overlap = event_loop_overlap
    S._pair_loop_installed = True
    print(f"[pair_loop] 已安装到 {mod.__name__}（pid={os.getpid()}）", flush=True)

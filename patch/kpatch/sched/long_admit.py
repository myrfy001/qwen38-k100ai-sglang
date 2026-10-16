"""长请求准入：限制同时在场（decode 中 + 本轮预填充中）的长请求数 ≤ K，新的长请求留在等待队列。

依据（2026-09-21 逐请求实证 + R14 离线推演，记忆 long-row-tpot-is-prefill-stall）：
  压测里 c 条长请求同时到达，预填充串行且期间不给 decode 机会 ⇒ 已在 decode 的请求被后到者的预填充堵住，
  131072 c16 的 TPOT 里 91% 是等待、c4 72%；65536 c16 74%、c4 56%。
  离线推演（只对输入 ≥65536 的行）：K=4 → P +13 / O −1；K=2 → P +42 / O −2。代价是后到请求的 TTFT 变长。
  短输入行不能用（decode 段远长于预填充，TTFT 会崩），故只对 ≥ 阈值的请求生效。

改动点【核 schedule_policy.py:996】`PrefillAdder.add_one_req` 入口：
  待入请求是长请求且在场长请求数 ≥ K ⇒ 返回 AddReqResult.OTHER。
  scheduler.py:3088 对非 CONTINUE 一律 break 并释放已分配的 mamba 槽（3097-3110），本轮无可跑预填充即转 decode。
  续块走 add_chunked_req（不经本函数），不受影响。

开关：SGLANG_LONG_ADMIT=1；SGLANG_LONG_ADMIT_THRESH（默认 65536，按 origin_input_ids 长度）；SGLANG_LONG_ADMIT_K（默认 2）。
运行时覆盖（免重启 A/B）：文件 /tmp/long_admit_k、/tmp/long_admit_thresh 存在则每轮读取；K ≥ 1000 视为关闭。
token 预算（10-08 27B）：SGLANG_LONG_ADMIT_TOK（运行时 /tmp/long_admit_tok，0=不用）——在场长请求输入 token 之和 + 待入 > 预算即留队。
  依据：16K c16 用 K=12（约 21 万 token 在场）O156/P137，远好于不限 170/74 与 K=8 118/121；按 token 计可让 64K/128K 行自动得到 3/1–2 条。
"""
import os

_ON = os.environ.get("SGLANG_LONG_ADMIT", "0") != "0"
_THRESH = int(os.environ.get("SGLANG_LONG_ADMIT_THRESH", "65536"))
_K = int(os.environ.get("SGLANG_LONG_ADMIT_K", "2"))
_TOK = int(os.environ.get("SGLANG_LONG_ADMIT_TOK", "0"))
_st = {"tok": _TOK, "held": 0, "passed": 0, "logged": False, "err": False,
       "k": _K, "th": _THRESH, "next_check": 0.0, "sig": None}

# 运行时覆盖文件的检查间隔（秒）。调度热路径上每轮都 stat 两个文件没有必要，
# 而做 A/B 时只需要"改完一两秒内生效"。设为 0 则每轮都查（仅调试用）。
_POLL = float(os.environ.get("SGLANG_LONG_ADMIT_POLL", "1.0"))
_FILES = ("/tmp/long_admit_k", "/tmp/long_admit_thresh", "/tmp/long_admit_tok")


def _knobs():
    """返回当前生效的 (K, 阈值)。运行时覆盖文件按 _POLL 间隔轮询，不在每轮调度上做系统调用。"""
    import time
    now = time.monotonic()
    if now < _st["next_check"]:
        return _st["k"], _st["th"], _st["tok"]
    _st["next_check"] = now + _POLL
    try:
        sig = tuple(os.stat(f).st_mtime_ns if os.path.exists(f) else None for f in _FILES)
    except OSError:
        sig = _st["sig"]
    if sig == _st["sig"]:                      # 文件未变，沿用缓存值
        return _st["k"], _st["th"], _st["tok"]
    _st["sig"] = sig
    k, th, tok = _K, _THRESH, _TOK
    try:
        if os.path.exists(_FILES[0]):
            k = int(open(_FILES[0]).read().strip() or _K)
        if os.path.exists(_FILES[1]):
            th = int(open(_FILES[1]).read().strip() or _THRESH)
        if os.path.exists(_FILES[2]):
            tok = int(open(_FILES[2]).read().strip() or _TOK)
    except Exception:
        pass
    if (k, th, tok) != (_st["k"], _st["th"], _st["tok"]) or not _st["logged"]:
        print(f"[long_admit] 生效参数 K={k} 阈值={th} token 预算={tok}{'（K≥1000 即关闭）' if k >= 1000 else ''} pid={os.getpid()}", flush=True)
    _st["k"], _st["th"], _st["tok"] = k, th, tok
    return k, th, tok


def install(mod):
    if not _ON or getattr(mod, "_long_admit_installed", False):
        return
    PA = getattr(mod, "PrefillAdder", None)
    ARR = getattr(mod, "AddReqResult", None)
    if PA is None or ARR is None or not hasattr(PA, "add_one_req"):
        print("[long_admit] 未找到 PrefillAdder.add_one_req / AddReqResult，不安装", flush=True)
        return
    orig = PA.add_one_req

    def add_one_req(self, req, *a, **kw):
        try:
            k, th, tok = _knobs()
            n = len(req.origin_input_ids)
            if (k < 1000 or tok > 0) and n >= th:
                # 在场长请求 = decode 中的 + 本轮已收进预填充的（含续块，add_chunked_req 先于本调用入列）
                if getattr(self, "_la_run", None) is None:
                    rb = getattr(self, "running_batch", None)
                    self._la_run = sum(1 for r in rb.reqs if len(r.origin_input_ids) >= th) if rb is not None else 0
                    self._la_tok = sum(len(r.origin_input_ids) for r in rb.reqs if len(r.origin_input_ids) >= th) if rb is not None else 0
                n_cur = self._la_run + sum(1 for r in self.can_run_list if len(r.origin_input_ids) >= th)
                t_cur = self._la_tok + sum(len(r.origin_input_ids) for r in self.can_run_list if len(r.origin_input_ids) >= th)
                # 本 sglang 版本：本轮无可跑请求且 running 为空时，非 CONTINUE 会被当作"超出 KV 容量"直接 abort → 此时一律放行
                rb = getattr(self, "running_batch", None)
                idle = (rb is None or len(rb.reqs) == 0) and len(self.can_run_list) == 0
                if (n_cur >= k or (tok > 0 and t_cur + n > tok)) and not idle:
                    _st["held"] += 1
                    if not _st["logged"]:
                        _st["logged"] = True
                        print(f"[long_admit] ✅ 首次真正生效：待入请求 {n} token，在场长请求 {n_cur}"
                              f"（decode 中 {self._la_run}，本轮预填充 {n_cur - self._la_run}）≥ K={k}，留队 pid={os.getpid()}", flush=True)
                    elif _st["held"] % 200 == 0:
                        print(f"[long_admit] 累计留队 {_st['held']} 次 / 放行 {_st['passed']} 次", flush=True)
                    return ARR.OTHER
                _st["passed"] += 1
        except Exception as e:
            if not _st["err"]:
                _st["err"] = True
                print(f"[long_admit] ⚠ 判定失败，本请求按原逻辑放行: {type(e).__name__}: {e}", flush=True)
        return orig(self, req, *a, **kw)

    PA.add_one_req = add_one_req
    mod._long_admit_installed = True
    print(f"[long_admit] 已安装 K={_K} 阈值={_THRESH}（运行时可用 /tmp/long_admit_k、/tmp/long_admit_thresh 覆盖） pid={os.getpid()}", flush=True)

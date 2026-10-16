"""分块预填充的块间插 decode：长请求分块预填充进行中，每做完一块先跑 N 轮 decode 再续下一块。

问题（09-26 源码核实 scheduler.py:2885 `if new_batch is not None: ret = new_batch  # Run prefill first if possible`）：
  只要有进行中的分块请求（chunked_req），每轮都先组预填充，decode 一轮都插不进；投机解码下 mixed-chunk 又被禁
  （server_args.py:7424）。于是 131072 的请求 16 块 × 约 2.7 s，其他已在 decode 的请求被整段卡住约 43 s。
  长行并发的 TPOT 率只有 0.14–0.6，长请求准入（long_admit）是用 TTFT 换这部分，本补丁改为不排队、只让路。
做法：包 Scheduler.get_new_batch_prefill。条件全满足时返回空预填充计划 → 本轮走 decode：
  ① 有进行中的分块请求（self.chunked_req 非空）；② 上一轮做的是预填充（计数 < N 时继续插）；
  ③ running_batch 非空且不是纯预填充批。
  续块状态：get_next_batch_to_run 开头已按 extend_range 把已算部分 stash（scheduler.py:2818），
  跳过本轮不会重复 stash，下一轮 _get_new_batch_prefill_raw → add_chunked_req 照常续块。
  与 first_token_first 模式 3 的「插一轮 decode」同一机制，只是去掉了它的 `chunked_req is None` 条件。
开关：SGLANG_CHUNK_INTERLEAVE=N（每块之后插 N 轮 decode，0 关闭）；
  SGLANG_CHUNK_INTERLEAVE_MIN（分块请求输入长度下限，默认 0）。
运行时覆盖：文件 /tmp/chunk_interleave 存在则每秒读取其中的 N。
"""
import os
import time

_N = int(os.environ.get("SGLANG_CHUNK_INTERLEAVE", "0"))
_MIN = int(os.environ.get("SGLANG_CHUNK_INTERLEAVE_MIN", "0"))
_FILE = "/tmp/chunk_interleave"
_st = {"ins": 0, "logged": False, "n": _N, "next": 0.0, "err": False}


def _n():
    now = time.monotonic()
    if now >= _st["next"]:
        _st["next"] = now + 1.0
        try:
            if os.path.exists(_FILE):
                n = int(open(_FILE).read().strip() or _N)
                if n != _st["n"]:
                    print(f"[chunk_interleave] 运行时覆盖 N={n} pid={os.getpid()}", flush=True)
                _st["n"] = n
        except Exception:
            pass
    return _st["n"]


def install(mod):
    # 27B 用的 sglang 0.5.12+das185：get_new_batch_prefill(self) 直接返回 Optional[ScheduleBatch]，在跑批次是 self.running_batch
    if getattr(mod, "_chunk_interleave_installed", False):
        return
    S = mod.Scheduler
    orig = S.get_new_batch_prefill

    def get_new_batch_prefill(self):
        try:
            n = _n(); cr = self.chunked_req; rb = self.running_batch
            if (n > 0 and cr is not None and len(cr.origin_input_ids) >= _MIN
                    and getattr(self, "_ci_dec", n) < n and rb is not None and not rb.is_empty()):
                self._ci_dec = getattr(self, "_ci_dec", n) + 1
                _st["ins"] += 1
                if not _st["logged"]:
                    _st["logged"] = True
                    print(f"[chunk_interleave] ✅ 首次真正生效：分块请求 {len(cr.origin_input_ids)} token 进行中，"
                          f"插 decode（在跑 {rb.batch_size()} 条）pid={os.getpid()}", flush=True)
                elif _st["ins"] in (100, 1000, 10000):
                    print(f"[chunk_interleave] 累计插入 {_st['ins']} 轮 decode", flush=True)
                return None
        except Exception as e:
            if not _st["err"]:
                _st["err"] = True
                print(f"[chunk_interleave] ⚠ 判定失败，按原逻辑: {type(e).__name__}: {e}", flush=True)
        b = orig(self)
        if b is not None:
            self._ci_dec = 0
        return b

    S.get_new_batch_prefill = get_new_batch_prefill
    mod._chunk_interleave_installed = True
    print(f"[chunk_interleave] 已安装 N={_N} MIN={_MIN}（运行时可用 {_FILE} 覆盖） pid={os.getpid()}", flush=True)

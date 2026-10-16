#!/usr/bin/env python3
"""按主办方 10-04 口径汇总官方 benchmark.py 的输出目录。

口径：SLA 取均值（4K: TTFT<10s 且 TPOT<80ms；64K: TTFT<30s 且 TPOT<100ms），
均值按正式轮全部请求逐请求平均（官方 summary 只给 p50/p95，这里从 round-*/requests.jsonl 重算）。
吞吐取各正式轮 output_throughput_tps 的均值（与官方 aggregate 一致）。

用法：python3 sla.py <结果目录>...      每个目录是一次 benchmark.py 的 --output-dir
输出一行一个测点：in c 吞吐 TTFT均值/p95 TPOT均值/p95 SLA 轮数 状态
"""
import glob
import json
import os
import sys

SLA = {4096: (10.0, 80.0), 65536: (30.0, 100.0)}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def p95(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return float("nan")
    pos = (len(xs) - 1) * 0.95
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def one(d):
    cfg = json.load(open(os.path.join(d, "config.json")))
    rounds = sorted(glob.glob(os.path.join(d, "round-*")))
    reqs, tps, status = [], [], []
    for r in rounds:
        s = json.load(open(os.path.join(r, "summary.json")))
        status.append(s["status"])
        tps.append(s["output_throughput_tps"])
        for line in open(os.path.join(r, "requests.jsonl")):
            x = json.loads(line)
            if x["error"] is None and x["done"]:
                reqs.append(x)
    ttft = [x["ttft_s"] for x in reqs]
    tpot = [x["tpot_ms"] for x in reqs]
    lim = SLA.get(cfg["input_len"])
    tm, pm = mean(ttft), mean(tpot)
    ok = all(st == "PASS" for st in status) and len(status) > 0
    sla = "-" if lim is None else ("过" if ok and tm < lim[0] and pm < lim[1] else "不过")
    return (f"in={cfg['input_len']:<6} c={cfg['concurrency']:<4} 吞吐={mean(tps):8.2f} "
            f"TTFT均值={tm:7.3f}s p95={p95(ttft):7.3f}s TPOT均值={pm:7.2f}ms p95={p95(tpot):7.2f}ms "
            f"SLA={sla} 轮={len(status)} 请求={len(reqs)} 状态={','.join(status) or '无'}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        try:
            print(one(d))
        except Exception as e:  # 目录不完整（中途崩溃）也要给出一行
            print(f"{d}: 无法汇总 {type(e).__name__}: {e}")

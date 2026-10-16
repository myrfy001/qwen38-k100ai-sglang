#!/usr/bin/env python3
"""Qwen3.8 冒烟（约 3 分钟）：两个代表测点（官方 benchmark.py，1 预热 + 1 正式，输出 256）+ 中文生成 + gsm8k 10 题。
用法（容器内）：python3 smoke.py [标签]      PORT/SERVED 可覆盖
gsm8k 走 /v1/chat/completions，关思考（chat_template_kwargs.enable_thinking=false）：Qwen 系思考模式会把 1024 token 用完，假性掉分。
"""
import json, os, re, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
TAG = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%m%d-%H%M")
PORT = os.environ.get("PORT", "30000"); SERVED = os.environ.get("SERVED", "qwen38")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", f"/workspace/q38/runs/smoke_{TAG}")
PTS = [(4096, 1), (4096, 16)]


def chat(msg, max_tokens):
    body = {"model": SERVED, "messages": [{"role": "user", "content": msg}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=900))
    c = r["choices"][0]
    return (c["message"].get("content") or ""), c.get("finish_reason")


t0 = time.time()
print("=" * 78); print(f"  Qwen3.8 冒烟 [{TAG}]"); print("=" * 78)
fail = []
for i, c in PTS:
    d = f"{OUT}/{i}_c{c}"
    subprocess.run([sys.executable, f"{HERE}/benchmark.py", "--base-url", f"http://127.0.0.1:{PORT}", "--model", SERVED,
                    "--output-dir", d, "--input-len", str(i), "--output-len", "256", "--concurrency", str(c),
                    "--warmup-rounds", "1", "--measure-rounds", "1", "--timeout", "1800"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    line = subprocess.run([sys.executable, f"{HERE}/sla.py", d], capture_output=True, text=True).stdout.strip()
    print("  " + line)
    if "PASS" not in line:
        fail.append(f"{i}/c{c} 未全部成功")

zh, fr = chat("用一句话说明什么是快速排序。", 128)
zh_ok = len(set(zh)) > 12 and "排序" in zh
print(f"\n  中文生成: {'✅' if zh_ok else '❌'} {zh[:60]!r} ({fr})")

rows = [json.loads(l) for l in open(os.environ.get("GSM8K_PATH", "/workspace/q38/datasets/gsm8k_test.jsonl"))]
gold = lambda a: float(a.split("####")[-1].strip().replace(",", ""))


def ask(i):
    t, f = chat(rows[i]["question"] + "\nPut the final numeric answer after '####'.", 1024)
    m = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", t) or re.findall(r"-?\d[\d,]*\.?\d*", t)
    try:
        return float(m[-1].replace(",", "")), f
    except Exception:
        return None, f


Q = list(range(10))
with ThreadPoolExecutor(10) as ex:
    got = list(ex.map(ask, Q))
ok = sum(1 for i, (p, _) in zip(Q, got) if p is not None and abs(p - gold(rows[i]["answer"])) < 1e-6)
trunc = sum(1 for _, f in got if f == "length")
print(f"  gsm8k 10 题: {'✅' if ok >= 9 else '⚠️' if ok >= 8 else '❌'} {ok}/10  截断 {trunc}")
corr = zh_ok and ok >= 9 and not fail
print("\n  【对不对】" + ("✅ 通过" if corr else "❌ " + "；".join(fail + ([] if zh_ok else ["中文异常"]) + ([] if ok >= 9 else [f"gsm8k {ok}/10"]))))
print(f"  用时 {time.time()-t0:.0f} 秒"); print("=" * 78)
sys.exit(0 if corr else 1)

"""一臂实验结束后的日志体检（arm.sh 末尾自动跑；也可单独跑）。
用法：python3 logaudit.py <本臂目录> [基线臂目录]      输出写 <本臂目录>/audit.txt 并打印摘要。
检查项均来自开发中实际遇到的问题：
  1 补丁：装上但从未「首次真正生效」、任何 ❌ 行（静默回退）
  2 配置表回落默认（Using default MoE kernel config / sub-optimal）
  3 崩溃与异常（Traceback、VMFault、显存不足、retract、NCCL WARN）
  4 压测行：请求失败（ok<并发）、接受长度塌缩（acc<2.0）
  5 与基线比：KV 池容量变化、逐行 TTFT/TPOT 变化超过 ±10%
注意：日志行常被 tqdm 进度条接在前面，标记一律不锚行首。"""
import os, re, sys
from collections import Counter, defaultdict

D = sys.argv[1]; B = sys.argv[2] if len(sys.argv) > 2 else ""
NO_EFFECT_MARK = {"fast_import", "aiter", "fused_draft_topk", "vocab_par_verify", "vocab_par_draft", "chunk_interleave"}  # 本身不打「首次生效」或另有标记
BENIGN = re.compile(r"server_args=|NUMA balancing|torchaudio|Ignore import error|overrides --custom-all-reduce")
BAD = [("崩溃", re.compile(r"Traceback|VMFault|Memory access fault|core dumped|Segmentation fault")),
       ("显存", re.compile(r"out of memory|OutOfMemory|KV cache pool is full")),
       ("调度退让", re.compile(r"[Rr]etract")),
       ("通信", re.compile(r"NCCL WARN|RCCL WARN")),
       ("配置回落", re.compile(r"Using default MoE kernel config|sub-optimal|Config file not found|config not found|not found, using default"))]
ROW = re.compile(r"行(\d+) in=\s*(\d+)\s+out=\s*\d+\s+c=\s*(\d+)\s*\|\s*TTFT=\s*([\d.]+)\s+TPOT=\s*([\d.]+)\s+ok=(\d+)/(\d+)\s+acc=([\d.]+)")
KV = re.compile(r"max_total_num_tokens=(\d+)")


def read(p):
    try:
        return open(p, errors="replace").read()
    except OSError:
        return ""


SPEC = {}
try:
    import json
    SPEC = {x["excel_row"]: x for x in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench", "spec45_hy3.json")))}
except Exception:
    pass


def rows(txt, d=None):
    """优先读 <臂>/out/formal_results.json（run45 逐行增量写、含逐请求接受长度，cmd 模式下行输出不进 arm.log）；没有再从文本解析。"""
    try:
        fr = json.load(open(f"{d}/out/formal_results.json")) if d else None
    except Exception:
        fr = None
    if fr and SPEC:
        res = {}
        for r, ms in fr.items():
            r = int(r); sp = SPEC.get(r, {}); n = len(ms)
            res[r] = dict(inp=sp.get("inp", 0), c=sp.get("conc", 0), ttft=sum(m["mean_ttft"] for m in ms) / n,
                          tpot=sum(m["mean_tpot"] for m in ms) / n, ok=min(m["successful"] for m in ms), n=sp.get("conc", 0),
                          acc=sum(m["accept"] for m in ms) / n)
        return res
    return {int(m[0]): dict(inp=int(m[1]), c=int(m[2]), ttft=float(m[3]), tpot=float(m[4]), ok=int(m[5]), n=int(m[6]), acc=float(m[7]))
            for m in ROW.findall(txt)}


srv, arm = read(f"{D}/server.log"), read(f"{D}/arm.log")
out, bad, warn = [], 0, 0
def say(level, msg):
    global bad, warn
    bad += level == "❌"; warn += level == "⚠"; out.append(f"{level} {msg}")

# 1 补丁
inst, eff = Counter(), Counter()
for tag, rest in re.findall(r"\[([a-z_0-9]+)\] ([^\n\[]{0,80})", srv):
    if "已安装" in rest or "已载入" in rest: inst[tag] += 1
    if re.search(r"首次|已捕获|✅|守卫判定", rest): eff[tag] += 1
for tag in sorted(inst):
    if tag not in NO_EFFECT_MARK and eff[tag] == 0:
        say("⚠", f"补丁 [{tag}] 已安装 {inst[tag]} 次但从未「首次真正生效」")
fails = Counter(re.sub(r"\d{3,}", "#", m) for m in re.findall(r"(\[[a-z_0-9]+\] ❌[^\n\[]{0,120})", srv))
for m, k in fails.most_common(10):
    say("❌", f"{m}（×{k}）")
warns = Counter(re.sub(r"\d{3,}", "#", m) for m in re.findall(r"(\[[a-z_0-9]+\] ⚠[^\n\[]{0,120})", srv))
for m, k in warns.most_common(10):
    say("⚠", f"{m}（×{k}）")
# 2/3 异常与配置回落
for name, pat in BAD:
    hits = [l for l in srv.splitlines() if pat.search(l) and not BENIGN.search(l)]
    if hits:
        say("❌" if name in ("崩溃", "显存", "配置回落") else "⚠", f"{name} {len(hits)} 行，首条：{hits[0][:160]}")
# 4 压测行
R = rows(arm, D)
for r, x in sorted(R.items()):
    if x["ok"] < x["n"]:
        say("❌", f"行{r} 请求失败 {x['ok']}/{x['n']}（该行记 0 分）")
    if x["acc"] < 2.0:
        say("⚠", f"行{r} in={x['inp']} c={x['c']} 接受长度 {x['acc']:.2f} < 2.0")
# 5 基线
if B:
    kv0, kv1 = KV.findall(read(f"{B}/server.log")), KV.findall(srv)
    if kv0 and kv1 and kv0[0] != kv1[0]:
        say("⚠", f"KV 池容量 {kv0[0]} → {kv1[0]} token（{(int(kv1[0]) / int(kv0[0]) - 1) * 100:+.1f}%），长行并发可能受影响")
    R0 = rows(read(f"{B}/arm.log"), B)
    moved = []
    for r in sorted(set(R) & set(R0)):
        a, b = R0[r], R[r]
        dt, dp = b["ttft"] / a["ttft"] - 1, b["tpot"] / a["tpot"] - 1
        if abs(dt) > 0.10 or abs(dp) > 0.10:
            moved.append(f"行{r}(in={b['inp']} c={b['c']}) TTFT {dt * 100:+.0f}% TPOT {dp * 100:+.0f}% acc {a['acc']:.2f}→{b['acc']:.2f}")
    if moved:
        say("⚠", f"相对基线 {os.path.basename(B.rstrip('/'))} 变化超 ±10% 的行 {len(moved)}/{len(set(R) & set(R0))}：\n    " + "\n    ".join(moved))
head = f"【日志体检】{os.path.basename(D.rstrip('/'))}：❌ {bad} 项 ⚠ {warn} 项；补丁已安装 {len(inst)} 个、有生效标记 {sum(1 for t in inst if eff[t])} 个；压测行 {len(R)}"
txt = head + ("\n" + "\n".join(out) if out else "\n✅ 无异常")
open(f"{D}/audit.txt", "w").write(txt + "\n")
print(txt)

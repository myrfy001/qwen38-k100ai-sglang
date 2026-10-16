#!/usr/bin/env bash
# 截断题加长重跑：从 <源目录>/acc_<任务>.jsonl 去掉 finish=length 的题，写到 <输出目录>/acc_<任务>.jsonl，
# 再用 acc_eval.py 按题号续跑（只补被去掉的题），生成上限 MAXTOK（默认 81920）。其余题沿用原记录。
# 用法（容器内 bench/ 下，服务已起）：bash acc_long.sh <源目录> <输出目录> [任务...]
# 可续跑：输出文件已存在时不重写，acc_eval.py 按题号只补缺的题（被杀后重开同一输出目录即可）。
set -u
SRC=$1; OUT=$2; shift 2; TASKS=${*:-gsm8k rwqa mathvision lcb}; MAXTOK=${MAXTOK:-81920}; mkdir -p "$OUT"
for t in $TASKS; do
  [ -s "$OUT/acc_$t.jsonl" ] || python3 - "$SRC/acc_$t.jsonl" "$OUT/acc_$t.jsonl" <<'PY'
import json, sys
src, dst = sys.argv[1:3]; keep = drop = 0
with open(dst, "w") as f:
    for l in open(src):
        if json.loads(l).get("finish") == "length": drop += 1; continue
        f.write(l); keep += 1
print(f"[acc_long] {src}: 保留 {keep}，待加长重跑 {drop}")
PY
  python3 acc_eval.py $t "$OUT/acc_$t.jsonl" --c 64 --max-tokens $MAXTOK 2>&1 | tail -3
done
python3 acc_eval.py sum "$OUT"/acc_*.jsonl

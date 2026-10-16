#!/usr/bin/env bash
# 用官方 benchmark.py 跑若干测点并按主办方口径汇总。
# 用法：PTS="4096:1 4096:16 65536:1" [WR=1 MR=1] [TAG=名字] [OUT=目录] [SEED=official] bash pts.sh
#   官方 prompt 只由 (seed,轮次,序号) 决定、与并发无关，同一服务连测多档会命中前缀缓存（不关前缀缓存时 c8 的 TTFT 会偏低）。
#   默认每个测点用不同种子；SEED=official 时用官方默认种子（定稿口径，须每档重启或关 radix cache）。
#   粗测默认 1 轮预热 + 1 轮正式（WR/MR）；定稿用 WR=2 MR=3（官方口径）。
#   每个测点结果在 $OUT/<in>_c<c>/，汇总追加到 $OUT/sla.txt
set -u
B=$(cd "$(dirname "$0")" && pwd)
PORT=${PORT:-30000}; MODEL=${SERVED:-qwen38}
OUT=${OUT:-/workspace/q38/runs/${TAG:-pts}_$(date +%m%d_%H%M)}; mkdir -p "$OUT"
for p in $PTS; do
  I=${p%:*}; C=${p#*:}; d=$OUT/${I}_c$C
  timeout ${PT_TIMEOUT:-3600} python3 -u "$B/benchmark.py" --base-url http://127.0.0.1:$PORT --model "$MODEL" \
    --output-dir "$d" --input-len "$I" --output-len ${OUTLEN:-1024} --concurrency "$C" \
    $( [ "${SEED:-}" = official ] || echo --seed $((20261009 + I * 131 + C * 7919 + RANDOM)) ) --warmup-rounds ${WR:-1} --measure-rounds ${MR:-1} --timeout 3600 > "$d.log" 2>&1
  python3 "$B/sla.py" "$d" | tee -a "$OUT/sla.txt"
done

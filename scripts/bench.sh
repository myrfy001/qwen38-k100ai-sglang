#!/usr/bin/env bash
# 官方口径测量（服务已由 scripts/run.sh 起好并打印 "The server is fired up and ready to roll"）。
# 用法（容器内）：bash scripts/bench.sh <输出目录> [fixed|sharegpt|all]     默认 all；PORT=30000
#   fixed   ：固定长度 4K C1/8/16/32/64/128 + 64K C1/8/16/32，输出 1024，官方默认种子，2 轮预热 + 3 轮正式（官方 benchmark.py）
#             每档原始输出 <目录>/<in>_c<c>/（config.json、round-*/requests.jsonl、progress.json），按主办方口径汇总到 <目录>/sla.txt
#   sharegpt：ShareGPT 1000 条 × 3 轮，C1/4/16/32/64（官方 4.2 命令，sglang.bench_serving），汇总 <目录>/sharegpt.txt
#             数据集路径 SG_DATASET（默认 /workspace/q38/datasets/ShareGPT_V3_unfiltered_cleaned_split.json）
# 注意：官方 prompt 只由 (seed,轮次,序号) 决定，服务已关 radix cache，连测多档不会命中前缀缓存；无须每档重启。
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd); OUT=${1:?输出目录}; WHAT=${2:-all}; mkdir -p "$OUT"
export PORT=${PORT:-30000} SERVED=qwen38
echo "[bench] 开始 $(date '+%F %T')  目录 $OUT  项目 $WHAT" | tee -a "$OUT/timeline.txt"
if [ "$WHAT" = fixed ] || [ "$WHAT" = all ]; then
  OUT=$OUT SEED=official WR=2 MR=3 PT_TIMEOUT=14400 \
    PTS="4096:1 4096:8 4096:16 4096:32 4096:64 4096:128 65536:1 65536:8 65536:16 65536:32" bash "$ROOT/bench/pts.sh"
  echo "[bench] 固定长度完成 $(date '+%F %T')" | tee -a "$OUT/timeline.txt"
fi
if [ "$WHAT" = sharegpt ] || [ "$WHAT" = all ]; then
  CS="1 4 16 32 64" N=1000 R=3 OUT=$OUT SG_TIMEOUT=7200 bash "$ROOT/bench/sharegpt.sh"
  echo "[bench] ShareGPT 完成 $(date '+%F %T')" | tee -a "$OUT/timeline.txt"
fi
echo "[bench] 结束 $(date '+%F %T')" | tee -a "$OUT/timeline.txt"

#!/usr/bin/env bash
# ShareGPT 测点（官方 4.2 命令原样，只改地址/模型名/条数）。用法：CS="1 16 64" N=200 R=1 OUT=<目录> bash sharegpt.sh
# 官方：每档 1000 条、预热 64、3 轮、--seed 20260912（三轮同一批 prompt → 开 radix cache 时后两轮命中前缀）。
# 粗测默认 N=200、R=1；定稿 N=1000 R=3。汇总追加到 $OUT/sharegpt.txt
set -u
CS=${CS:-"1 4 16 32 64"}; N=${N:-200}; R=${R:-1}; OUT=${OUT:?}; PORT=${PORT:-30000}
mkdir -p $OUT
for C in $CS; do for r in $(seq 1 $R); do
  F=$OUT/sharegpt-c$C/round-0$r.jsonl; mkdir -p $(dirname $F)
  timeout ${SG_TIMEOUT:-3600} python3 -u -m sglang.bench_serving --backend sglang-oai --base-url http://127.0.0.1:$PORT \
    --dataset-name sharegpt --dataset-path ${SG_DATASET:-/workspace/q38/datasets/ShareGPT_V3_unfiltered_cleaned_split.json} \
    --model ${MODEL:-/models/q38} --served-model-name ${SERVED:-qwen38} --sharegpt-context-len 131072 --num-prompts $N \
    --max-concurrency $C --request-rate inf --warmup-requests 64 --seed 20260912 --disable-tqdm \
    --output-details --output-file $F > $OUT/sharegpt-c$C/round-0$r.log 2>&1
  L=$OUT/sharegpt-c$C/round-0$r.log
  echo "sharegpt c=$C 轮$r N=$N 输出吞吐=$(grep -oP 'Output token throughput \(tok/s\):\s+\K[\d.]+' $L) TTFT均值=$(grep -oP 'Mean TTFT \(ms\):\s+\K[\d.]+' $L)ms TPOT均值=$(grep -oP 'Mean TPOT \(ms\):\s+\K[\d.]+' $L)ms 成功=$(grep -oP 'Successful requests:\s+\K\d+' $L) 接受长度=$(grep -oP 'Accept length:\s+\K[\d.]+' $L)" | tee -a $OUT/sharegpt.txt
done; done

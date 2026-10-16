#!/usr/bin/env bash
# 下载压测与精度评测数据（HuggingFace 数据集；国内用 hf-mirror.com，可用 HF_ENDPOINT 覆盖）。
# 用法（容器内）：bash scripts/prepare_data.sh [目标目录]      默认 /workspace/q38/datasets（= bench 脚本默认读取位置）
# 换目录时：ShareGPT 用 SG_DATASET=<目录>/ShareGPT_V3_unfiltered_cleaned_split.json，精度评测用 Q38_DATASETS=<目录>
set -eu
D=${1:-/workspace/q38/datasets}; H=${HF_ENDPOINT:-https://hf-mirror.com}
mkdir -p "$D/eval"
get() {   # 已存在则跳过；失败即退出（set -e）
  mkdir -p "$(dirname "$2")"
  if [ ! -s "$2" ]; then curl -fL --retry 5 -o "$2.part" "$H/datasets/$1"; mv -f "$2.part" "$2"; fi
  ls -la "$2"
}
get anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json "$D/ShareGPT_V3_unfiltered_cleaned_split.json"
get openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet "$D/eval/openai_gsm8k/test-00000-of-00001.parquet"
get xai-org/RealworldQA/resolve/main/data/test-00000-of-00002.parquet "$D/eval/xai-org_RealworldQA/test-00000-of-00002.parquet"
get xai-org/RealworldQA/resolve/main/data/test-00001-of-00002.parquet "$D/eval/xai-org_RealworldQA/test-00001-of-00002.parquet"
get MathLLMs/MathVision/resolve/main/data/testmini-00000-of-00001-f8ff70fcb2f29b1d.parquet "$D/eval/MathLLMs_MathVision/testmini-00000-of-00001-f8ff70fcb2f29b1d.parquet"
get livecodebench/code_generation_lite/resolve/main/test6.jsonl "$D/eval/livecodebench_code_generation_lite/test6.jsonl"
# GSM8K 转成 bench 脚本读的 jsonl（question / answer 两字段，1319 条）
python3 - "$D" <<'EOF'
import json, sys, pyarrow.parquet as pq
d = sys.argv[1]
rows = pq.read_table(f"{d}/eval/openai_gsm8k/test-00000-of-00001.parquet").to_pylist()
with open(f"{d}/gsm8k_test.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps({"question": r["question"], "answer": r["answer"]}, ensure_ascii=False) + "\n")
print("gsm8k_test.jsonl", len(rows))
EOF
echo "[prepare_data] 完成：$D"

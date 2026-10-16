#!/usr/bin/env bash
# Qwen3.8-27B-Channel-INT8-w8a8 · K100-AI ×2（TP2）· SGLang 0.5.12+das185 一键启动（镜像 sglang:0.5.12-ubuntu22.04-dtk2604-py3.10）。
# 用法（容器内）：bash scripts/run.sh [追加 sglang 参数...]
#   MODEL=权重目录（默认 /models/q38）  PORT=30000  HIP_VISIBLE_DEVICES=0,1
#   全部优化项默认打开（= 报告定稿配置）。任一开关设 0（表类设空串）即关闭该项做对照：SGLANG_VERIFY_ATTN=0 bash scripts/run.sh
#   Q38_BASELINE=1：自测参照配置（镜像原生 sglang + 平台必需项 + 投机 3/1/4，不加载补丁、extra_buffer、块 16384），报告 §2 与精度配对用它作对照。
# 补丁不改镜像源码：patch/sitecustomize.py 经 PYTHONPATH 自动加载，按开关在模块导入时安装；日志里每项有「已安装」与「✅ 首次真正生效」。
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${MODEL:-/models/q38}; PORT=${PORT:-30000}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1}

# ---- 平台必需项（K100-AI 两个前序项目沿用；缺任一项会挂起或崩溃）----
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-2}            # 硬件队列限 2：同机其他卡有进程时队列超额会挂起
export HSA_FORCE_FINE_GRAIN_PCIE=${HSA_FORCE_FINE_GRAIN_PCIE:-1}
export SGLANG_USE_LIGHTOP=1 SGLANG_USE_AITER_LINEAR_ATTN=1 SGLANG_USE_CAUSAL_CONV1D=1 SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ROCM_USE_AITER_MOE=False SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

COMMON=(--model-path "$MODEL" --served-model-name qwen38 --host 0.0.0.0 --port "$PORT"
  --tp-size 2 --dtype bfloat16 --quantization w8a8_int8 --kv-cache-dtype auto
  --attention-backend fa3 --mm-attention-backend fa3
  --page-size 64 --disable-custom-all-reduce                 # 必需：页 64（HCU 分页注意力）；自定义 all-reduce 在 gfx928 段错误
  --mem-fraction-static 0.80 --cuda-graph-max-bs 64 --context-length 131072 --reasoning-parser qwen3
  --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)

if [ "${Q38_BASELINE:-0}" = 1 ]; then
  echo "[run.sh] 自测参照配置：无补丁、extra_buffer、chunked-prefill 16384"
  exec python3 -m sglang.launch_server "${COMMON[@]}" --mamba-scheduler-strategy extra_buffer --chunked-prefill-size 16384 "$@"
fi

# ---- 补丁层与自写内核 ----
export PYTHONPATH=$ROOT/patch${PYTHONPATH:+:$PYTHONPATH}
B=$ROOT/build; mkdir -p "$B/i8gemm"                          # .so 由 scripts/build_kernels.sh 预编译；缺失时各补丁首次加载现编到同一路径
export SGLANG_I8GEMM_DIR=$B/i8gemm SGLANG_LMH_SO=$B/q38_lmh.so SGLANG_LMH_OWN_SO=$B/q38_lmh_own.so
export SGLANG_PF_ATTN_SO=$B/q38_pf_attn.so SGLANG_VERIFY_ATTN_SO=$B/q38_va_attn.so SGLANG_CONV1D_OWN_SO=$B/q38_conv1d.so
C=$ROOT/config

# ---- 优化开关（默认 = 定稿；编号对应报告 §3）----
: "${SGLANG_DENSE_CFG=$C/dense_v2.json}"                      # 1  稠密 INT8 GEMM 逐 (N,K,M) 选解：lightop / triton tile / 自写 HIP 核（kernels/i8gemm.hip）
: "${SGLANG_NORM_QUANT=1}"                                    # 2  add+RMSNorm+int8 量化融合为一个核，量化结果复用（逐位）
: "${SGLANG_ACT_QUANT=1}"                                     # 2b MLP 的 silu·mul 与 down_proj 输入量化合为一个核（厂商 lm_fuse_silu_mul_quant，逐位；M=4096 496→279 µs）
: "${SGLANG_LMH_TABLE=$C/lmh_final_0.csv}"                    # 3  LM-head 逐 M 直调 rocBLAS 最优解
: "${SGLANG_LMH_OWN_TABLE=$C/lmh_own_0.csv}"                  # 3b LM-head 小 M 自写核（kernels/lmh_gemm.hip）
: "${SGLANG_GDN_CFG=1}" "${SGLANG_GDN_PF_CFG=1}" "${SGLANG_CHUNK_H_CFG=1}"   # 4 GDN decode/预填充 Triton 核配置（aiter 无 gfx928 条目）
: "${SGLANG_GDN_TABLE=$C/gdn_aiter_nb.json}"                  # 4  同上：按形状查表
: "${SGLANG_CONV1D_OWN=1}"                                    # 5  GDN 预填充因果卷积自写核（kernels/conv1d.hip）
: "${SGLANG_QKV_VIEW=1}"                                      # 5b GDN 预填充去 cat(q,k,v) + 卷积分段输出连续 q/k/v（逐位；64K TTFT −1.6%）
: "${SGLANG_PF_ATTN=1}"                                       # 6  预填充分页注意力自写核（GQA6 展平，kernels/pf_attn.hip）
: "${SGLANG_VERIFY_ATTN=1}"                                   # 7  投机验证步注意力：bs≥12 沿 KV 切分 + lse 合并（kernels/va_attn.hip）
: "${SGLANG_GDN_REPLAY=1}"                                    # 8  投机验证不写 GDN 中间状态、接受后重放（逐位；省 13.78 GB/卡）
: "${SGLANG_GDN_VS_PF=1}"                                    # 8b 重放路径的预取版验证核（每请求 4 token 时一次发出 4 步输入读；逐位；内核 N=48 207→193 µs）
: "${SGLANG_QUANT_AR=1}" "${SGLANG_QAR_WORLDS=2}"             # 9  预填充 all-reduce int8 分组量化通信（分组 32）
: "${SGLANG_PAIR_PREFILL=1}" "${SGLANG_PAIR_PREFILL_MOE_OVERLAP=1}" "${SGLANG_PAIR_PREFILL_ALLOW_LOGPROB=1}"   # 10 两个预填充批成对交错发射（通信/计算重叠）
: "${SGLANG_ADAPTIVE_CHUNK=1}" "${SGLANG_ADAPTIVE_CHUNK_TIERS=8192:16384}"   # 11 按请求长度选 chunk 并均分
: "${SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP=1}"           # 12 sglang 原生开关：关连续预填充重叠（本栈下更快）
: "${SGLANG_HCU_CA=1}" "${SGLANG_HCU_CA_MAX=409600}"           # 14 decode 小消息 all-reduce 走 vllm_hcu 的 custom all-reduce（≤400 KB：TP2 扫描 M≤32 快 17–52%、M≥48 反慢；4K C1 TPOT −3.1%）
: "${SGLANG_FUSED_DRAFT_TOPK=1}" "${SGLANG_VPA_DRAFT=1}" "${SGLANG_VPA_VERIFY=1}"   # 15 草稿/验证 LM-head 词表并行：各卡只交换 [M,3] 的局部最大值，不收集整张 logits（4K C32 TPOT −4.8%，c1 逐请求一致）
: "${SGLANG_DRAFT_PRUNE=1}"                                   # 15b 草稿扩展的 LM-head 只算每请求被采纳的那一行（[4·bs,·]→[bs,·]；c1 逐请求一致，4K C32 TPOT −0.7%）
export SGLANG_DENSE_CFG SGLANG_NORM_QUANT SGLANG_ACT_QUANT SGLANG_LMH_TABLE SGLANG_LMH_OWN_TABLE SGLANG_GDN_CFG SGLANG_GDN_PF_CFG SGLANG_CHUNK_H_CFG \
  SGLANG_GDN_TABLE SGLANG_CONV1D_OWN SGLANG_QKV_VIEW SGLANG_PF_ATTN SGLANG_VERIFY_ATTN SGLANG_GDN_REPLAY SGLANG_GDN_VS_PF SGLANG_QUANT_AR SGLANG_QAR_WORLDS \
  SGLANG_PAIR_PREFILL SGLANG_PAIR_PREFILL_MOE_OVERLAP SGLANG_PAIR_PREFILL_ALLOW_LOGPROB SGLANG_ADAPTIVE_CHUNK SGLANG_ADAPTIVE_CHUNK_TIERS \
  SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP SGLANG_HCU_CA SGLANG_HCU_CA_MAX \
  SGLANG_FUSED_DRAFT_TOPK SGLANG_VPA_DRAFT SGLANG_VPA_VERIFY SGLANG_DRAFT_PRUNE

echo "[run.sh] 开关：$(env | grep -E '^(SGLANG_|GPU_MAX|HSA_)' | sort | tr '\n' ' ')"
# 13 调度：mamba no_buffer（不为投机回滚另备缓冲）+ 关 radix cache（no_buffer 必须关，否则页大小被改 1 → VMFault）+ chunk 4096
CMD=(python3 -m sglang.launch_server "${COMMON[@]}" --mamba-scheduler-strategy no_buffer --disable-radix-cache --chunked-prefill-size 4096 "$@")
echo "[run.sh] 启动命令：${CMD[*]}"
[ "${Q38_DRYRUN:-0}" = 1 ] && exit 0          # Q38_DRYRUN=1：只打印开关与命令，不起服务
exec "${CMD[@]}"

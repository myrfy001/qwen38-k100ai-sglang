#!/usr/bin/env bash
# 预编译自写内核（gfx928）到 build/，路径与 scripts/run.sh 导出的 SGLANG_*_SO 一致；约 2–3 分钟（容器内，需 /opt/dtk/bin/hipcc）。
# 不跑也能起服务：各补丁首次加载时会现编到同一路径（两个 TP 进程各写临时文件再原子改名，不冲突）。
set -eu
export ROOT=$(cd "$(dirname "$0")/.." && pwd); K=$ROOT/kernels; B=$ROOT/build; mkdir -p "$B/i8gemm"
F="-O3 --offload-arch=gfx928 -shared -fPIC"
hipcc $F -DPF_NW=4 -DPF_MT=2 -o "$B/q38_pf_attn.so" "$K/pf_attn.hip"       # 预填充分页注意力（GQA6 展平，128 行/块）
hipcc $F -DPF_NW=4 -DPF_MT=1 -o "$B/q38_va_attn.so" "$K/va_attn.hip"       # 投机验证步注意力（沿 KV 切分 + lse 合并）
hipcc $F -o "$B/q38_conv1d.so" "$K/conv1d.hip"                              # GDN 预填充因果卷积
hipcc $F -o "$B/q38_lmh_own.so" "$K/lmh_gemm.hip"                           # LM-head 小 M GEMM
hipcc -O2 -shared -fPIC -o "$B/q38_lmh.so" "$K/lmh_rocblas.cpp" -lrocblas   # LM-head 直调 rocBLAS 指定解
# 稠密 INT8 GEMM：变体由 config/dense_v2.json 中 impl=hip 条目的 cfg 宏串决定，.so 文件名 = 「源码+宏」md5（与 patch/kpatch/ops/i8gemm_lib.py 同一规则）
SGLANG_I8GEMM_DIR=$B/i8gemm SGLANG_I8GEMM_SRC=$K/i8gemm.hip PYTHONPATH=$ROOT/patch python3 - <<'EOF'
import json, os
from kpatch.ops import i8gemm_lib
t = json.load(open(os.path.join(os.environ["ROOT"], "config", "dense_v2.json")))
for d in sorted({r["cfg"] for ms in t.values() for r in ms.values() if r["impl"] == "hip"}):
    i8gemm_lib.load(d); print("i8gemm", repr(d), "->", i8gemm_lib.so_path(d)[0])
EOF
ls -la "$B" "$B/i8gemm"
echo "[build_kernels] 完成 $(date)"

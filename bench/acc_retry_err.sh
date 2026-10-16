#!/usr/bin/env bash
# 把某个 jsonl 里客户端出错（finish 以 error 开头，如读超时）的题去掉，再按题号续跑这些题。
# 用法（容器内 bench/ 下，服务已起）：ACC_TIMEOUT=21600 bash acc_retry_err.sh <任务> <jsonl> [并发] [生成上限]
set -u
T=$1; F=$2; CC=${3:-16}; MAXTOK=${4:-81920}
[ -f "$F.bak_err" ] || cp "$F" "$F.bak_err"
python3 - "$F" <<'PY'
import json, sys
f = sys.argv[1]; rows = [json.loads(l) for l in open(f)]
keep = [r for r in rows if not str(r.get("finish")).startswith("error")]
open(f, "w").writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in keep)
print(f"[acc_retry_err] {f}: 保留 {len(keep)}，重跑出错题 {len(rows) - len(keep)}")
PY
python3 acc_eval.py $T "$F" --c $CC --max-tokens $MAXTOK 2>&1 | tail -3
python3 acc_eval.py sum "$F"

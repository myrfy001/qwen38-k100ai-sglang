# Qwen3.8-27B INT8 · K100-AI ×2 · SGLang 赛道 — 复现说明

本包在**未改动镜像源码**的前提下，用 PYTHONPATH 覆盖层补丁（`patch/`）+ 自写 HIP/Triton 内核（`kernels/`）+ 调优表（`config/`）
把 Qwen3.8-27B-Channel-INT8-w8a8 在两张 K100-AI（TP2）上的吞吐与时延做到报告中的数字。所有优化项由 `scripts/run.sh` 默认打开，
每项都能单独关闭做对照；`Q38_BASELINE=1` 即自测参照配置（镜像原生 sglang，不加载补丁）。

```
submit/
├─ README.md               本文
├─ env_declare.yaml        环境声明（赛题 5.2 字段）
├─ requirements.txt        依赖（镜像自带为主；精度评测额外两包）
├─ config/                 调优表（稠密 GEMM 选解、LM-head 选解、GDN 核配置）+ 模型 config.json 副本
├─ scripts/run.sh          一键启动（完整启动参数、全部开关）
├─ scripts/build_kernels.sh 预编译自写内核 → build/
├─ scripts/bench.sh        官方口径测量（固定长度 + ShareGPT）
├─ scripts/prepare_data.sh 下载 ShareGPT 与四项评测集
├─ patch/                  覆盖层补丁：sitecustomize.py + kpatch/{ops,sched,comm,infra,kv}
├─ kernels/                自写内核源码（*.hip / *.cpp）
├─ bench/                  官方 benchmark.py（未改）+ 汇总脚本 sla.py + pts.sh / sharegpt.sh / acc_eval.py
├─ tools/logaudit.py      启动日志体检
├─ logs/                   4 类原始日志（01 硬件驱动 / 02 启动 / 03 benchmark 原始输出 / 04 结束统计）+ 05_timestamped_text（宿主现场执行的命令及原样输出，每条带执行时间，按主办方 10-04 答复代替截图）+ 06_accuracy（四项精度逐题记录，本方案 / 镜像默认 / 关量化通信三臂）
└─ LICENSE / NOTICE        Apache-2.0；第三方部分说明
```

## 0. 目录约定

容器内固定两处路径：本包在 `/workspace/q38/submit/`，数据集在 `/workspace/q38/datasets/`。做法是把**本包的上一级目录**挂到 `/workspace/q38`（下文 `<工作目录>`，其下有 `submit/`）。脚本里的其他路径都相对本包，或者可以用环境变量覆盖。

## 1. 环境准备

1. 镜像：`harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10`（光合官方镜像，自带 sglang 0.5.12+das185、torch 2.10、DTK 26.04）。
2. 宿主：驱动 hycu 6.3.30-V1.4.1a，`/opt/hyhal` 存在。其余见 `env_declare.yaml`。
3. 起容器（只用卡 0、1；`--network host` 便于本机压测）：
   ```bash
   docker run -d --name q38 --network host --ipc host --shm-size 64g --privileged \
     --device /dev/kfd --device /dev/mkfd --device /dev/dri --group-add video \
     -v /opt/hyhal:/opt/hyhal:ro \
     -v <权重目录>/Qwen3.8-27B-Channel-INT8-w8a8:/models/q38:ro \
     -v <工作目录>:/workspace/q38 \
     harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10 sleep infinity
   docker exec -it q38 bash
   cd /workspace/q38/submit
   ```
4. 前置检查：
   - 在**宿主**上运行 `hy-smi --showpids`，确认卡 0、1 上没有其他进程（容器内读不到宿主进程目录，会报 `Unable to open process directory`）。同机其他卡有进程不影响，`GPU_MAX_HW_QUEUES=2` 就是为这种情况设的。
   - CPU 调频：我们的数据在 `schedutil`（系统默认）下测得，复现时保持一致。改成 `performance` 时短输入的 TTFT 会更低。
5. 权重：光合官方发布的 `Qwen3.8-27B-Channel-INT8-w8a8`（含 EAGLE 草稿头）。校验命令：
   `(cd /models/q38 && sha256sum -c /workspace/q38/submit/logs/01_hardware_driver/weights.sha256)`
6. 依赖：服务与压测只用镜像自带的包。精度评测另需 `pip install math-verify==0.9.0`（镜像已有 pyarrow），见 `requirements.txt`。
7. 数据：`bash scripts/prepare_data.sh`，从 hf-mirror.com 下载 ShareGPT 和四项评测集到 `/workspace/q38/datasets/`，约 1 GB。可用 `HF_ENDPOINT=` 换源。

## 2. 构建自写内核（约 2–3 分钟）

```bash
bash scripts/build_kernels.sh && ls build/*.so build/i8gemm/*.so | wc -l     # 应为 7
```
产物在 `build/`（gfx928）。如果漏了这一步，各补丁会在首次加载时现编到同一路径，结果相同，但首次启动要多等几分钟。建议先构建。

## 3. 启动与冒烟

```bash
mkdir -p logs/02_server_launch
bash scripts/run.sh 2>&1 | tee logs/02_server_launch/server_repro.log    # 前台；或 nohup … & 后台。不要写到 server.log，那是本作品提交的证据日志
```
看到 `The server is fired up and ready to roll` 即就绪（首次启动含 CUDA graph 捕获，约 3–4 分钟）。确认三件事：
- 第一行 `[run.sh] 开关：…` 列出全部开关，第二行 `[run.sh] 启动命令：…` 是完整启动参数；
- 下列 20 个补丁模块各有 `已安装` 日志（每个 TP 进程各打一次）：dense_cfg、norm_quant、act_quant、qkv_view、lmh_rocblas、gdn_pf_cfg、chunk_h_cfg、conv1d_own、pf_attn（含验证步分支）、gdn_replay、gdn_replay_mem、quant_ar、pair_loop、pair_prefill、adaptive_chunk、hcu_ca、fused_draft_topk、vocab_par_draft、vocab_par_verify、draft_prune。其中 vocab_par_draft、vocab_par_verify 另有「✅ 首次真正生效」日志，验证侧每 100/10000 次打印快路与回退计数。首批请求后应出现对应的 `✅ … 首次真正生效`（验证步注意力的标记是 `[verify_attn]`）。`SGLANG_GDN_CFG` 和 `SGLANG_GDN_TABLE` 是 GDN 验证核的配置表，开启重放后由 gdn_replay 读取，不会单独打 `已安装`；
- 日志里没有 `安装失败`、`Traceback`、`VMFault`。（`chunk_fwd_o config not found, using default` 这一条不影响结果：厂商 gfx936/938 表在本形状上的取值就等于默认值。）

另开一个终端做两分钟冒烟（4K C1/C16 各一轮，再加 gsm8k 10 题），退出码 0 即通过：
```bash
cd /workspace/q38/submit/bench && OUT=/workspace/q38/submit/logs/smoke_repro python3 smoke.py
```
体检脚本：`python3 tools/logaudit.py logs/02_server_launch`。它读取该目录下的日志，列出补丁生效情况、配置回落和异常。

对照方式：
- 自测参照配置（不加载补丁）：`Q38_BASELINE=1 bash scripts/run.sh`
- 关闭单项：`SGLANG_VERIFY_ATTN=0 bash scripts/run.sh`。表类开关设成空串即关闭，例如 `SGLANG_DENSE_CFG= bash scripts/run.sh`
- 只打印开关和启动命令、不起服务：`Q38_DRYRUN=1 bash scripts/run.sh`

## 4. 测量

```bash
bash scripts/bench.sh /workspace/q38/submit/logs/03_benchmark_raw/ours all      # 固定长度 10 档 + ShareGPT 5 档 × 3 轮，约 10 小时
bash scripts/bench.sh <目录> fixed                                                  # 只跑固定长度，约 7 小时
```
- 固定长度：输入 4096 / 65536，输出 1024，官方默认种子，2 轮预热 + 3 轮正式。每档的逐请求原始记录在 `<目录>/<in>_c<c>/round-0k/requests.jsonl`。`sla.txt` 按主办方口径汇总：吞吐取三轮均值，TTFT/TPOT 取正式轮逐请求平均，并给出 SLA 判定。
- ShareGPT：用 `sglang.bench_serving` 跑，每档 1000 条 × 3 轮，seed 20260912，预热 64 条。汇总在 `<目录>/sharegpt.txt`。
- 精度（四项，思考开启）：同一套脚本分别在自测参照配置和本方案两个服务上各跑一遍，再配对比较。
  ```bash
  cd /workspace/q38/submit/bench
  for t in gsm8k rwqa mathvision lcb; do python3 acc_eval.py $t ../logs/acc_repro_ours/acc_$t.jsonl --c 48; done      # 服务：bash scripts/run.sh
  for t in gsm8k rwqa mathvision lcb; do python3 acc_eval.py $t ../logs/acc_repro_vendor/acc_$t.jsonl --c 48; done    # 服务：Q38_BASELINE=1 bash scripts/run.sh
  python3 acc_eval.py sum ../logs/acc_repro_*/*.jsonl
  ```
  中断后重跑会按题号续跑。
  第二轮（截断题加长重跑，报告 §4.3 主表）：在对应服务上执行 `bash acc_long.sh ../logs/acc_repro_ours ../logs/accL_repro_ours`（参照组同理），生成上限 81920 token、可续跑；客户端超时的题用 `ACC_TIMEOUT=21600 bash acc_retry_err.sh lcb <jsonl> 16` 补跑。配对：`python3 acc_eval.py pair <本作品 jsonl> <参照 jsonl>`。四项合计约 4–6 小时（每个服务）。

## 5. 预期结果（本机实测，官方口径三正式轮均值；详见报告 §4 与 logs/03_benchmark_raw/final/）

| 测点 | 输出吞吐 tok/s | TTFT 均值 | TPOT 均值 | SLA |
|---|--:|--:|--:|---|
| 4K C1 | 77.45 | 1.13 s | 11.83 ms | 过 |
| 4K C8 | 287.24 | 2.88 s | 23.15 ms | 过 |
| 4K C16 | 404.05 | 4.87 s | 32.16 ms | 过 |
| 4K C32 | 502.76 | 8.44 s | 51.62 ms | 过 |
| 4K C64 | 550.85 | 33.56 s | 72.67 ms | 不过 |
| 4K C128 | 555.28 | 120.13 s | 78.27 ms | 不过 |
| 64K C1 | 31.17 | 17.93 s | 14.58 ms | 过 |
| 64K C8 | 45.76 | 52.28 s | 121.45 ms | 不过 |
| 64K C16 | 49.05 | 99.48 s | 220.02 ms | 不过 |
| 64K C32 | 49.69 | 343.23 s | 249.07 ms | 不过 |
| ShareGPT C1 / C4 / C16 / C32 / C64 | 64.73 / 170.33 / 326.35 / 398.72 / 474.47 | 0.36 / 0.37 / 0.41 / 0.50 / 7.04 s | 15.4 / 24.9 / 53.3 / 89.0 / 114.4 ms | — |

允许 ±5% 环境波动；同机对自测参照配置（`Q38_BASELINE=1`）的增量见报告 §4.4。

## 6. 已知边界与代价

- 仅验证 TP=2、K100-AI（gfx928）、本镜像版本；自写内核以 `--offload-arch=gfx928` 编译，其他卡型需重编并重新对拍。
- 与原实现不逐位一致的有五项（预填充量化通信为唯一有损项；其余为求和次序不同，argmax 或误差量级与原实现一致），清单见报告 §5「数值一致性」；四项精度对自测参照配置的配对结果见报告 §4.3。
- `--disable-radix-cache` 为 no_buffer 策略所必需，多轮对话场景不享受前缀缓存。
- 自写内核在 `bs<12` 的验证步、`q<阈值` 的预填充保留厂商核，由补丁按形状自动选择。
- 修改自写内核时的两个编译器陷阱：`__launch_bounds__` 写得比实际线程数小，编译器会把依赖线程号的分支折叠掉（结果错而非变慢）；`__syncthreads()` 前会被插入全量 `vmcnt(0)`，等掉已发出的预取。

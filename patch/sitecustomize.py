"""Qwen3.8 覆盖层补丁加载器（每个 Python 进程自动加载，父进程与所有 TP 子进程都覆盖到，不改动镜像源码树）。
结构照搬 Hy3 的 patch/sitecustomize.py：不带任何默认环境变量，补丁一律默认关闭，靠环境变量逐项打开做 A/B。
补丁实现在 kpatch/：来自 Qwen3.6-27B（patch36，同一镜像 das185）与 Hy3 两个项目，按本模型形状逐项核对后再开。

镜像自带 /usr/lib/python3.10/sitecustomize.py（只装 apport 异常钩子）。本目录排在 PYTHONPATH 前面时会把它遮住，
所以先按原样执行它，再注册我们的查找器。"""
import sys, os, importlib, importlib.abc

_SYS_SC = "/usr/lib/python3.10/sitecustomize.py"
if os.path.exists(_SYS_SC) and os.path.abspath(_SYS_SC) != os.path.abspath(__file__):
    try:
        exec(compile(open(_SYS_SC).read(), _SYS_SC, "exec"), {"__name__": "_sys_sitecustomize"})
    except Exception:
        pass

# 模块名 -> [(补丁模块, 开关环境变量, 开关默认值), ...]；同一模块按列表顺序安装
TARGETS = {
    "sglang.srt.layers.quantization.w8a8_int8": [
        ("kpatch.ops.dense_cfg", "SGLANG_DENSE_CFG", ""),               # INT8 GEMM 按 (N,K,M) 查调优表（值为表路径）
        ("kpatch.ops.norm_quant", "SGLANG_NORM_QUANT", "0"),           # 量化复用：命中 norm 融合核已算好的 int8
        ("kpatch.ops.act_quant", "SGLANG_ACT_QUANT", "0"),             # apply 接受 silu_quant_args（厂商融合 silu·mul+量化）
    ],
    "sglang.srt.layers.linear": [
        ("kpatch.ops.act_quant", "SGLANG_ACT_QUANT", "0"),             # 注入 lm_fuse_silu_mul_quant
    ],
    "sglang.srt.models.qwen2_moe": [
        ("kpatch.ops.act_quant", "SGLANG_ACT_QUANT", "0"),             # Qwen2MoeMLP 走 down_proj 的融合路径（逐位）
    ],
    "sglang.srt.models.qwen3_5": [
        ("kpatch.ops.norm_quant", "SGLANG_NORM_QUANT", "0"),           # add+GemmaRMSNorm+int8 量化一个核（逐位）
        ("kpatch.ops.qkv_view", "SGLANG_QKV_VIEW", "0"),               # GDN 预填充去掉 cat(q,k,v)，传零拷贝视图给自写卷积
        ("kpatch.ops.ba_mm", "SGLANG_BA_MM", ""),                       # in_proj_ba（bf16 N=48）按 M 段直调 rocBLAS 解（值为规则）
    ],
    "sglang.srt.layers.attention.flashattention_backend": [
        ("kpatch.ops.pf_attn_own", "SGLANG_PF_ATTN", "0"),             # 自写分页预填充注意力（GQA6 展平行，128 行/块）
        # SGLANG_VERIFY_ATTN=1：同一补丁内验证步（q≤10/请求、bs≥12）走 kernels/va_attn.hip 切分核
    ],
    "sglang.srt.layers.attention.fla.wy_fast": [
        ("kpatch.ops.gdn_pf_cfg", "SGLANG_GDN_PF_CFG", "0"),
    ],
    "aiter.ops.triton.fla.sglang.chunk_o": [
        ("kpatch.ops.gdn_pf_cfg", "SGLANG_GDN_PF_CFG", "0"),
    ],
    "aiter.ops.triton.fla.sglang.chunk_delta_h": [
        ("kpatch.ops.chunk_h_cfg", "SGLANG_CHUNK_H_CFG", "0"),
    ],
    "aiter.ops.triton.fla.fused_recurrent": [
        ("kpatch.ops.gdn_dec_cfg", "SGLANG_GDN_DEC_CFG", "0"),
    ],
    "aiter.ops.triton.fla.fused_sigmoid_gating_recurrent": [
        ("kpatch.ops.gdn_cfg", "SGLANG_GDN_CFG", "0"),
    ],
    "sglang.srt.layers.attention.mamba.causal_conv1d_triton": [
        ("kpatch.ops.conv1d_cfg", "SGLANG_CONV1D_CFG", "0"),
    ],
    "sglang.srt.layers.attention.linear.gdn_backend": [
        ("kpatch.ops.conv1d_own", "SGLANG_CONV1D_OWN", "0"),           # 预填充因果卷积换自写核（厂商 causal_conv1d_fn_hcu 的 1.4–1.96×）
        ("kpatch.ops.gdn_replay", "SGLANG_GDN_REPLAY", "0"),           # 投机验证不写中间状态 + 接受后全层重放（逐位）
        ("kpatch.ops.qkv_view", "SGLANG_QKV_VIEW", "0"),               # 预填充卷积直接写出连续 q/k/v（省三次 .contiguous()）
    ],
    "sglang.srt.mem_cache.memory_pool": [
        ("kpatch.ops.gdn_replay_mem", "SGLANG_GDN_REPLAY", "0"),       # 开重放后中间 SSM 状态缓冲不分配（每卡 13.78 GB）
    ],
    "sglang.srt.model_executor.model_runner_kv_cache_mixin": [
        ("kpatch.ops.gdn_replay_mem", "SGLANG_GDN_REPLAY", "0"),       # 且不按 max_running 预留
    ],
    "sglang.srt.layers.attention.fla.chunk": [
        ("kpatch.ops.gdn_nozero", "SGLANG_GDN_NOZERO", "0"),
    ],
    "sglang.srt.speculative.eagle_worker_v2": [
        ("kpatch.ops.fused_draft_topk", "SGLANG_FUSED_DRAFT_TOPK", "0"),   # 顺序固定：topk → vpa_draft → vpa_verify
        ("kpatch.ops.vocab_par_draft", "SGLANG_VPA_DRAFT", "0"),
        ("kpatch.ops.vocab_par_verify", "SGLANG_VPA_VERIFY", "0"),
        ("kpatch.ops.draft_prune", "SGLANG_DRAFT_PRUNE", "0"),         # 草稿扩展 LM-head 只算 select_index 那 bs 行
        ("kpatch.infra.spec_stage_timer", "SGLANG_SPEC_TIMER", "0"),
        ("kpatch.ops.draft_extend_graph", "SGLANG_DRAFT_EXTEND_GRAPH", "0"),
    ],
    "sglang.srt.layers.logits_processor": [
        ("kpatch.ops.vocab_par_draft", "SGLANG_VPA_DRAFT", "0"),
        ("kpatch.ops.vocab_par_verify", "SGLANG_VPA_VERIFY", "0"),
        ("kpatch.ops.draft_prune", "SGLANG_DRAFT_PRUNE", "0"),
        ("kpatch.ops.lmh_rocblas", "SGLANG_LMH_TABLE", ""),            # LM-head 逐 M 直调 rocBLAS 解（值为 TunableOp 格式 CSV）
    ],
    "sglang.srt.models.qwen3_5_mtp": [
        ("kpatch.ops.vocab_par_draft", "SGLANG_VPA_DRAFT", "0"),
    ],
    "sglang.srt.managers.scheduler": [
        ("kpatch.infra.batch_timeline", "SGLANG_BATCH_TIMELINE", "0"),
        ("kpatch.sched.first_token_first", "SGLANG_FIRST_TOKEN_FIRST", "0"),
        ("kpatch.sched.chunk_interleave", "SGLANG_CHUNK_INTERLEAVE", "0"),
        ("kpatch.sched.adaptive_chunk", "SGLANG_ADAPTIVE_CHUNK", "0"),
        ("kpatch.sched.pair_loop", "SGLANG_PAIR_PREFILL", "0"),          # 不同请求的两个预填充批成对交错发射（通信与计算重叠）
    ],
    "sglang.srt.managers.schedule_policy": [
        ("kpatch.sched.long_admit", "SGLANG_LONG_ADMIT", "0"),
    ],
    "sglang.srt.distributed.parallel_state": [
        ("kpatch.comm.quant_ar", "SGLANG_QUANT_AR", "0"),
        ("kpatch.comm.hcu_ca", "SGLANG_HCU_CA", "0"),
    ],
    "sglang.srt.entrypoints.warmup": [
        ("kpatch.infra.warm_batch", "SGLANG_WARM_BATCH", "0"),
    ],
    "vllm_hcu.patch_utils": [
        ("kpatch.infra.fast_import", "SGLANG_FAST_IMPORT", "0"),
    ],
}


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, real, plist):
        self.real, self.plist = real, plist
    def create_module(self, spec): return self.real.create_module(spec)
    def exec_module(self, module):
        self.real.exec_module(module)
        for pm, env, dflt in self.plist:
            if os.environ.get(env, dflt) in ("0", ""):
                continue
            try:
                importlib.import_module(pm).install(module)
            except Exception as e:
                print(f"[{pm.rsplit(chr(46), 1)[-1]}] 安装失败, 回退原实现: {type(e).__name__}: {e}", flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        plist = TARGETS.get(fullname)
        if not plist: return None
        for f in sys.meta_path:
            if f is self: continue
            spec = f.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _PatchLoader(spec.loader, plist)
                return spec
        return None


if not any(isinstance(f, _Finder) for f in sys.meta_path):
    sys.meta_path.insert(0, _Finder())

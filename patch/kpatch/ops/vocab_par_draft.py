"""EAGLE 草稿的 top1 改成词表并行：各卡在本地 logits 分片上求 (最大值, 首个最大值下标, sumexp)，
只收集 [M,3]，不再收集 [M,248320] 的整张 logits。

依据:
  - 草稿每轮 3 次 logits 全收集（MTP 两步在图内 M=bs，草稿扩展 eager M=4·bs），走 PCIe 约 13 GB/s：
    图内实测 M=16 489 µs、M=64 1821 µs，并发 16 每轮约 2.8 ms
  - HIP 上草稿选词恒为 renorm_draft_probs → fast_topk（eagle_worker_v2.py 三处），topk=1 时只需要 argmax 与它的概率；
    链式 topk=1 下 topk_p 不参与选词，所以 argmax 逐位一致 ⇒ 草稿 token 逐位一致 ⇒ 整条生成逐位一致
  - 同值取首个出现（与 fused_draft_topk 一致）：本地严格大于才换，合并时同值取卡号小的
只动草稿模型（Qwen3_5ForCausalLMMTP）的 logits 处理器；目标模型/验证步不变。请求要 logprob 的那次调用回退全收集。
依赖 fused_draft_topk（草稿选词入口），它没装上则本补丁不生效。

草稿高频词表子集（SGLANG_DRAFT_HOT=<tools/hot_vocab.py 产出的 json>，SGLANG_DRAFT_HOT_N=32768）：
  草稿 LM-head 只对全局前 N 个高频词（ShareGPT 词频）里落在本卡分片的那些行做 GEMM，top1 只在这些词里取（FR-Spec 思路）。
  验证步不变 ⇒ 输出逐位不变，只影响接受长度。SGLang 原生 speculative_token_map 在 TP 下用全局下标索引本卡分片、取错行，不能用。
  热词几乎都在卡 0 分片（前 32768 个里卡 0 占 32626）：卡 0 算 3.3 万行（原 12.4 万行的 26%），卡 1 只算一百多行。
  热词行在 set_embed_and_head（草稿图捕获之前）一次性拷出；本卡热词按全局词号升序，同值取首个 = 全词表上取最小词号。
"""
import importlib
import os
import torch, triton, triton.language as tl

_STATE = {"eagle_hooked": False, "local_w": None, "full_w": None, "first": False,
          "hot_w": None, "hot_ids": None}   # 高频词子集：本卡热词行 [Hl,K] 与它们的全局词号（fp32）


@triton.jit
def _stage1(x_ptr, pmax_ptr, pidx_ptr, psum_ptr, V, stride, S: tl.constexpr, BLOCK: tl.constexpr):
    m = tl.program_id(0); s = tl.program_id(1)
    chunk = tl.cdiv(V, S); start = s * chunk
    rmax = tl.full((), float("-inf"), tl.float32); ridx = tl.full((), 2147483647, tl.int32); rsum = tl.zeros((), tl.float32)
    for off in range(0, chunk, BLOCK):
        c = start + off + tl.arange(0, BLOCK)
        msk = (c < V) & (off + tl.arange(0, BLOCK) < chunk)
        x = tl.load(x_ptr + m.to(tl.int64) * stride + c, mask=msk, other=float("-inf")).to(tl.float32)
        x = tl.where(msk & (x != x), float("inf"), x)      # NaN 当最大（与 torch.argmax 一致）
        bmax = tl.max(x, 0)
        bidx = tl.min(tl.where((x == bmax) & msk, c, 2147483647), 0)
        nmax = tl.maximum(rmax, bmax)
        rsum = rsum * tl.exp(rmax - nmax) + tl.sum(tl.where(msk, tl.exp(x - nmax), 0.0), 0)
        ridx = tl.where((bmax > rmax) | (ridx == 2147483647), bidx, ridx)   # 严格大于才换（取首个）；首块无条件取
        rmax = nmax
    tl.store(pmax_ptr + m * S + s, rmax); tl.store(pidx_ptr + m * S + s, ridx); tl.store(psum_ptr + m * S + s, rsum)


@triton.jit
def _stage2(pmax_ptr, pidx_ptr, psum_ptr, out_ptr, S: tl.constexpr):
    m = tl.program_id(0); i = tl.arange(0, S)
    pm = tl.load(pmax_ptr + m * S + i); pi = tl.load(pidx_ptr + m * S + i); ps = tl.load(psum_ptr + m * S + i)
    gmax = tl.max(pm, 0)
    gidx = tl.min(tl.where(pm == gmax, pi, 2147483647), 0)
    gsum = tl.sum(ps * tl.exp(pm - gmax), 0)
    tl.store(out_ptr + m * 3 + 0, gmax); tl.store(out_ptr + m * 3 + 1, gidx.to(tl.float32)); tl.store(out_ptr + m * 3 + 2, gsum)


@triton.jit
def _combine(g_ptr, idx_ptr, prob_ptr, M, V, W: tl.constexpr):
    """g: all_gather 后的 [W, M, 3]（卡号为最外维）→ 全局下标 int64 与 top1 概率。同值取卡号小的。"""
    m = tl.program_id(0); r = tl.arange(0, W)
    lmax = tl.load(g_ptr + (r * M + m) * 3 + 0); lidx = tl.load(g_ptr + (r * M + m) * 3 + 1); lsum = tl.load(g_ptr + (r * M + m) * 3 + 2)
    gmax = tl.max(lmax, 0)
    win = tl.min(tl.where(lmax == gmax, r, W), 0)
    widx = tl.sum(tl.where(r == win, lidx, 0.0), 0)
    gsum = tl.sum(lsum * tl.exp(lmax - gmax), 0)
    tl.store(idx_ptr + m, win.to(tl.int64) * V + widx.to(tl.int64))
    tl.store(prob_ptr + m, 1.0 / gsum)


def local_reduce(x, S=64, BLOCK=1024):
    """x: [M, V] 本地分片（行连续）→ [M, 3] fp32：本地最大值、本地下标（fp32 存，<2^24 精确）、sumexp"""
    M, V = x.shape
    pmax = torch.empty(M, S, device=x.device, dtype=torch.float32)
    pidx = torch.empty(M, S, device=x.device, dtype=torch.int32)
    psum = torch.empty(M, S, device=x.device, dtype=torch.float32)
    out = torch.empty(M, 3, device=x.device, dtype=torch.float32)
    _stage1[(M, S)](x, pmax, pidx, psum, V, x.stride(0), S=S, BLOCK=BLOCK)
    _stage2[(M,)](pmax, pidx, psum, out, S=S)
    return out


def vocab_par_top1(logits):
    """logits: 本卡分片 [M, V_local] → (topk_p [M,1] fp32, topk_index [M,1] int64)，与整张 logits 上 softmax().max() 等价"""
    from sglang.srt.distributed import get_tp_group
    g = get_tp_group()
    M, V = logits.shape
    if not logits.is_contiguous():
        logits = logits.contiguous()
    part = local_reduce(logits)
    hot = _STATE["hot_ids"] is not None and V == _STATE["local_w"]
    if hot:                                                 # 本卡热词下标 → 全局词号；合并时 V=0 即直接取它
        part[:, 1] = _STATE["hot_ids"][part[:, 1].long()]
    gathered = torch.empty(g.world_size * M, 3, device=part.device, dtype=part.dtype)
    g.all_gather_into_tensor(gathered, part)                # [W*M, 3]，卡号为最外维
    idx = torch.empty(M, device=logits.device, dtype=torch.int64)
    prob = torch.empty(M, device=logits.device, dtype=torch.float32)
    _combine[(M,)](gathered, idx, prob, M, 0 if hot else V, W=g.world_size)
    if not _STATE["first"]:
        _STATE["first"] = True
        cap = torch.cuda.is_current_stream_capturing()
        print(f"[vocab_par_draft] ✅ 首次真正生效: M={M} V_local={V} 高频词子集={hot} W={g.world_size} capturing={cap} pid={os.getpid()}", flush=True)
    return prob.view(M, 1), idx.view(M, 1)


class _Local:
    """占位：本卡分片 logits，等 fast_topk 来了走词表并行 top1"""
    __slots__ = ("logits",)
    def __init__(self, logits): self.logits = logits


def _is_local(x):
    w = _STATE["local_w"]
    return w is not None and isinstance(x, torch.Tensor) and x.dim() == 2 and x.shape[-1] == w


def _eligible(proc, lm_head):
    """每个草稿 logits 处理器只判一次。任一条件不满足就永远走原路（全收集）。"""
    ok = getattr(proc, "_kp_vpa_ok", None)
    if ok is not None:
        return ok
    why = None
    try:
        from sglang.srt.server_args import get_global_server_args as get_server_args  # 本镜像无 runtime_context（10-11 vpa_on 因此未生效）
        from sglang.srt.distributed import get_tp_group
        sa = get_server_args()
        si = lm_head.shard_indices
        W, rank = get_tp_group().world_size, get_tp_group().rank_in_group
        width = si.num_elements_padded
        algo = str(sa.speculative_algorithm).upper().split(".")[-1]
        if not _STATE["eagle_hooked"]:   # 目标模型验证图先于草稿 worker 捕获：主动导入以触发 sitecustomize 安装钩子
            importlib.import_module("sglang.srt.speculative.eagle_worker_v2")
        if not _STATE["eagle_hooked"]: why = "eagle_worker_v2 的草稿选词钩子未安装（需 SGLANG_FUSED_DRAFT_TOPK=1）"
        elif torch.version.hip is None: why = "非 HIP（CUDA 路径在本卡分片上直接 argmax）"
        elif algo not in ("EAGLE", "NEXTN"): why = f"算法 {algo} 不在白名单"
        elif getattr(sa, "enable_multi_layer_eagle", False): why = "multi-layer EAGLE"
        elif W & (W - 1) != 0: why = f"TP={W} 不是 2 的幂（合并核 tl.arange 要求）"
        elif not proc.do_tensor_parallel_all_gather or proc.use_attn_tp_group or proc.do_tensor_parallel_all_gather_dp_attn:
            why = "非普通 TP 全收集路径"
        elif sa.speculative_eagle_topk != 1: why = f"speculative_eagle_topk={sa.speculative_eagle_topk}"
        elif getattr(sa, "speculative_use_rejection_sampling", False): why = "拒绝采样"
        elif sa.speculative_token_map is not None: why = "speculative_token_map"
        elif os.environ.get("SGLANG_ENABLE_ASYNC_ASSERT", "0") not in ("", "0", "false", "False"):
            why = "SGLANG_ENABLE_ASYNC_ASSERT 会按本地宽度判越界"
        elif si.num_org_vocab_padding != 0 or si.org_vocab_start_index != rank * width or si.num_org_elements != width:
            why = f"词表分片有补白或不等分: start={si.org_vocab_start_index} n={si.num_org_elements} width={width}"
        elif W * width == proc.vocab_size and width == proc.vocab_size:
            why = "本地宽度与全词表宽度相同，无法按宽度区分"
        elif W * width != proc.vocab_size:
            why = f"W*width={W * width} != vocab_size={proc.vocab_size}"
        else:
            _STATE["local_w"] = width; _STATE["full_w"] = W * width
            if _STATE["hot_w"] is not None:
                hw = _STATE["hot_w"].shape[0]
                if hw in (width, W * width):
                    why = f"本卡热词数 {hw} 与分片/全词表宽度相同，无法按宽度区分"
                else:
                    _STATE["local_w"] = hw
    except Exception as e:
        why = f"代码异常 {type(e).__name__}: {e}"
    ok = why is None
    proc._kp_vpa_ok = ok
    print(f"[vocab_par_draft] 草稿 logits 处理器: {'启用词表并行 top1（本地宽度 %d）' % _STATE['local_w'] if ok else '不启用，原因: ' + why}"
          f" pid={os.getpid()}", flush=True)
    return ok


class _HotHead:
    """只带 weight 的代理头：LogitsProcessor._compute_lm_head 走 matmul(x, weight.T)；lmh_rocblas 按 (N,K) 查表不中即回落原路"""
    __slots__ = ("weight",)
    def __init__(self, w): self.weight = w


def _build_hot(model):
    """在 set_embed_and_head 之后（图捕获之前）拷出本卡热词行。失败只打日志、不启用。"""
    path = os.environ.get("SGLANG_DRAFT_HOT", "")
    if not path:
        return
    try:
        import json
        from sglang.srt.distributed import get_tp_group
        n = int(os.environ.get("SGLANG_DRAFT_HOT_N", "32768"))
        si = model.lm_head.shard_indices
        start, num = si.org_vocab_start_index, si.num_org_elements
        order = json.load(open(path))["order"][:n]
        w = model.lm_head.weight
        g = get_tp_group(); W, rank = g.world_size, g.rank_in_group
        if os.environ.get("SGLANG_DRAFT_HOT_BAL", "1") != "0" and W > 1:
            # 两卡均衡（10-11 vpa2_hot：热词集中在小词号，卡 0 有 32626 行、卡 1 只有 142 行，卡 0 的草稿 LM-head 成短板）。
            # 热词升序后按连续区间均分给各卡；所需行由各卡把自己分片里的热词行 all_gather 一次拷齐（初始化时一次，临时显存 n×K×2）。
            # 按升序连续区间分配 ⇒ 卡号小者词号小，_combine 同值取卡号小、卡内同值取下标小 ⇒ 与全热词集上 argmax 取最小词号一致。
            # 直接用 device_group 的 all_gather：避开 quant_ar 等对 GroupCoordinator 的接管。
            H = sorted(order)
            per = (len(H) + W - 1) // W
            own = [t for t in H if start <= t < start + num]
            cnt = torch.tensor([len(own)], device=w.device, dtype=torch.long)
            cnts = [torch.zeros_like(cnt) for _ in range(W)]
            torch.distributed.all_gather(cnts, cnt, group=g.device_group)
            cmax = int(max(int(c) for c in cnts))
            src = torch.zeros(cmax, w.shape[1], device=w.device, dtype=w.dtype)
            if own:
                src[:len(own)] = w.data.index_select(0, torch.tensor([t - start for t in own], device=w.device, dtype=torch.long))
            allr = torch.empty(W * cmax, w.shape[1], device=w.device, dtype=w.dtype)
            torch.distributed.all_gather_into_tensor(allr, src, group=g.device_group)
            # allr 第 r 段是卡 r 分片内的热词行（升序）；各卡分片按词号升序排列 ⇒ 拼起来去掉补白即 H 的顺序
            keep = torch.cat([torch.arange(r * cmax, r * cmax + int(cnts[r]), device=w.device) for r in range(W)])
            loc = H[rank * per:(rank + 1) * per]
            sel = keep[rank * per:rank * per + len(loc)]
            _STATE["hot_w"] = allr.index_select(0, sel).contiguous()
            _STATE["hot_ids"] = torch.tensor(loc, device=w.device, dtype=torch.float32)
            del allr, src
            mine = [i for i, t in enumerate(loc) if start <= t < start + num]   # 自检：属于本卡分片的行须与原权重逐位相同
            if mine:
                ii = torch.tensor(mine, device=w.device); tt = torch.tensor([loc[i] - start for i in mine], device=w.device)
                if not torch.equal(_STATE["hot_w"].index_select(0, ii), w.data.index_select(0, tt)):
                    raise RuntimeError("均衡拷贝后本卡分片内的热词行与原权重不一致")
            torch.cuda.empty_cache()
        else:
            loc = sorted(t for t in order if start <= t < start + num)
            if not loc:
                print(f"[vocab_par_draft] ⚠ 本卡分片 [{start},{start + num}) 无热词，高频词子集不启用 pid={os.getpid()}", flush=True)
                return
            idx = torch.tensor([t - start for t in loc], device=w.device, dtype=torch.long)
            _STATE["hot_w"] = w.data.index_select(0, idx).contiguous()
            _STATE["hot_ids"] = torch.tensor(loc, device=w.device, dtype=torch.float32)
        print(f"[vocab_par_draft] 已安装高频词子集：全局前 {n}（{path}），本卡 rank={get_tp_group().rank_in_group} "
              f"分片 [{start},{start + num}) 取 {len(loc)} 行（词号 {loc[0]}–{loc[-1]}，均衡={os.environ.get('SGLANG_DRAFT_HOT_BAL', '1') != '0'}），额外显存 {_STATE['hot_w'].numel() * 2 / 2**20:.0f} MiB pid={os.getpid()}", flush=True)
    except Exception as e:
        _STATE["hot_w"] = _STATE["hot_ids"] = None
        print(f"[vocab_par_draft] ❌ 高频词子集安装失败 {type(e).__name__}: {e}，走全分片 pid={os.getpid()}", flush=True)


def _install_logits(mod):
    LP = mod.LogitsProcessor
    if getattr(LP, "_kp_vpa_installed", False):
        return
    orig = LP._get_logits

    def _get_logits(self, hidden_states, lm_head, logits_metadata, embedding_bias=None):
        if (getattr(self, "_kp_vpa", False) and not getattr(logits_metadata, "extend_return_logprob", False)
                and _eligible(self, lm_head)):
            self.do_tensor_parallel_all_gather = False
            head = _HotHead(_STATE["hot_w"]) if _STATE["hot_w"] is not None else lm_head
            try:
                return orig(self, hidden_states, head, logits_metadata, embedding_bias)
            finally:
                self.do_tensor_parallel_all_gather = True
        return orig(self, hidden_states, lm_head, logits_metadata, embedding_bias)

    LP._get_logits = _get_logits
    LP._kp_vpa_installed = True
    print(f"[vocab_par_draft] 已安装到 {mod.__name__} (pid={os.getpid()})", flush=True)


def _install_mtp(mod):
    C = mod.Qwen3_5ForCausalLMMTP
    if getattr(C, "_kp_vpa_installed", False):
        return
    orig_init = C.__init__

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        self.logits_processor._kp_vpa = True

    orig_set = C.set_embed_and_head

    def set_embed_and_head(self, embed, head):
        orig_set(self, embed, head)
        _build_hot(self)

    C.__init__ = __init__
    C.set_embed_and_head = set_embed_and_head
    C._kp_vpa_installed = True
    print(f"[vocab_par_draft] 已安装到 {mod.__name__} (pid={os.getpid()})", flush=True)


def _install_eagle(mod):
    if getattr(mod, "_kp_vpa_installed", False):
        return
    if not getattr(mod, "_fused_draft_topk_installed", False):
        print(f"[vocab_par_draft] {mod.__name__} 上 fused_draft_topk 未安装，本补丁不生效", flush=True)
        return
    prev_topk = mod.fast_topk
    if not hasattr(mod, "renorm_draft_probs"):          # 新版：softmax 占位由 fused_draft_topk 产生
        from kpatch.ops.fused_draft_topk import _Lazy

        def ftopk_new(values, topk, dim):
            if isinstance(values, _Lazy) and _is_local(values.logits):
                assert topk == 1 and dim == -1
                return vocab_par_top1(values.logits)
            if isinstance(values, _Lazy) and _STATE["local_w"] is not None and values.logits.shape[-1] != _STATE["full_w"]:
                raise RuntimeError(f"[vocab_par_draft] 草稿 logits 宽度 {values.logits.shape[-1]} 既非本卡 {_STATE['local_w']} 也非全词表 {_STATE['full_w']}")
            return prev_topk(values, topk, dim)

        mod.fast_topk = ftopk_new
        mod._kp_vpa_installed = True
        _STATE["eagle_hooked"] = True
        print(f"[vocab_par_draft] 已安装到 {mod.__name__}（包 fast_topk）(pid={os.getpid()})", flush=True)
        return
    prev_renorm = mod.renorm_draft_probs

    def renorm(next_token_logits, sampling_info, use_rejection_sampling):
        w = _STATE["local_w"]
        if (w is not None and isinstance(next_token_logits, torch.Tensor) and next_token_logits.dim() == 2
                and next_token_logits.shape[-1] not in (w, _STATE["full_w"])):
            raise RuntimeError(f"[vocab_par_draft] 草稿 logits 宽度 {next_token_logits.shape[-1]} 既非本卡 {w} 也非全词表 {_STATE['full_w']}")
        if _is_local(next_token_logits):
            assert not use_rejection_sampling
            return _Local(next_token_logits)
        return prev_renorm(next_token_logits, sampling_info, use_rejection_sampling)

    def ftopk(values, topk, dim):
        if isinstance(values, _Local):
            assert topk == 1 and dim == -1
            if values.logits.size(0) == 0:
                return (torch.empty(0, 1, device=values.logits.device, dtype=torch.float32),
                        torch.empty(0, 1, device=values.logits.device, dtype=torch.int64))
            return vocab_par_top1(values.logits)
        return prev_topk(values, topk, dim)

    mod.renorm_draft_probs = renorm
    mod.fast_topk = ftopk
    mod._kp_vpa_installed = True
    _STATE["eagle_hooked"] = True
    print(f"[vocab_par_draft] 已安装到 {mod.__name__} (pid={os.getpid()})", flush=True)


def install(mod):
    n = mod.__name__
    if n.endswith("logits_processor"):
        _install_logits(mod)
    elif n.endswith("qwen3_5_mtp"):
        _install_mtp(mod)
    elif n.endswith("eagle_worker_v2"):
        _install_eagle(mod)
    else:
        raise RuntimeError(f"未知目标模块 {n}")

"""并发安全的 moe_align_block_size：只用原子加与内核边界保证顺序，线程块之间不传任何标志。
lightop 的 moe_align_block_size_kernel_opt_cub 在另一硬件队列有真并发时偶发丢/错 token 对（race5 实测 37 次首错级全在此）。
输出语义与 sglang fused_moe 使用的一致：sorted_ids（专家段内顺序任意，各段补齐到 block_size 的倍数，补位值 = 有效对数）、
expert_ids（每块所属专家）、num_tokens_post_pad（补齐后总长）。"""
import torch, triton, triton.language as tl

@triton.jit
def _count(ids_ptr, cnt_ptr, n, NE: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    e = tl.load(ids_ptr + o, mask=m, other=0)
    ok = m & (e >= 0) & (e < NE)
    tl.atomic_add(cnt_ptr + e, 1, mask=ok)

@triton.jit
def _scan(cnt_ptr, cur_ptr, cum_ptr, npad_ptr, NE: tl.constexpr, BM: tl.constexpr, EP2: tl.constexpr):
    e = tl.arange(0, EP2)
    m = e < NE
    c = tl.load(cnt_ptr + e, mask=m, other=0)
    p = (c + BM - 1) // BM * BM
    cum = tl.cumsum(p, 0)
    tl.store(cur_ptr + e, cum - p, mask=m)          # 散射游标 = 各专家段起点
    tl.store(cum_ptr + e, cum, mask=m)
    tl.store(npad_ptr, tl.sum(p, 0))

@triton.jit
def _blocks(cum_ptr, eid_ptr, nblk, NE: tl.constexpr, BM: tl.constexpr, EP2: tl.constexpr, BB: tl.constexpr):
    b = tl.program_id(0) * BB + tl.arange(0, BB)
    e = tl.arange(0, EP2)
    cum = tl.load(cum_ptr + e, mask=e < NE, other=2147483647)
    start = b * BM
    idx = tl.sum((cum[None, :] <= start[:, None]).to(tl.int32), 1)   # 起点落在哪个专家段
    idx = tl.minimum(idx, NE - 1)
    tl.store(eid_ptr + b, idx, mask=b < nblk)

@triton.jit
def _scatter(ids_ptr, cur_ptr, out_ptr, n, NE: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    e = tl.load(ids_ptr + o, mask=m, other=0)
    ok = m & (e >= 0) & (e < NE)
    slot = tl.atomic_add(cur_ptr + e, 1, mask=ok)
    tl.store(out_ptr + slot, o, mask=ok)

def safe_moe_align_block_size(topk_ids, block_size, num_experts, ignore_invalid_expert=False):
    ids = topk_ids.reshape(-1)
    if ids.dtype != torch.int32: ids = ids.to(torch.int32)
    ids = ids.contiguous(); n = ids.numel(); dev = ids.device
    NE = num_experts + 1                             # 与快路径一致：多留一个专家位
    mx = n * block_size if n < NE else n + NE * (block_size - 1)
    nblk = triton.cdiv(mx, block_size)
    sorted_ids = torch.full((mx,), n, dtype=torch.int32, device=dev)
    expert_ids = torch.empty((nblk,), dtype=torch.int32, device=dev)
    npad = torch.empty((1,), dtype=torch.int32, device=dev)
    cnt = torch.zeros((NE,), dtype=torch.int32, device=dev)
    cur = torch.empty((NE,), dtype=torch.int32, device=dev)
    cum = torch.empty((NE,), dtype=torch.int32, device=dev)
    EP2 = triton.next_power_of_2(NE); B = 1024
    _count[(triton.cdiv(n, B),)](ids, cnt, n, NE=NE, BLOCK=B)
    _scan[(1,)](cnt, cur, cum, npad, NE=NE, BM=block_size, EP2=EP2)
    _blocks[(triton.cdiv(nblk, 16),)](cum, expert_ids, nblk, NE=NE, BM=block_size, EP2=EP2, BB=16)
    _scatter[(triton.cdiv(n, B),)](ids, cur, sorted_ids, n, NE=NE, BLOCK=B)
    return sorted_ids, expert_ids, npad

"""预填充大张量的 all-reduce 改走 int8 分组量化（RCCL all_to_all + all_gather）。

依据（2026-09-18，四卡独占探针，生产同款 NCCL 环境）：
    T       bf16 all_reduce   int8 量化路径   比值
    2048       1.096 ms          0.862 ms      0.787
    4096       2.027 ms          1.432 ms      0.707
    8192       3.888 ms          2.514 ms      0.647
    16384      7.614 ms          4.681 ms      0.615
    1024       0.650 ms          0.794 ms      1.221   ← 小张量更慢，故设阈值
机制：卡到 CPU 的链路是 PCIe Gen3 x16（根端口上限 8 GT/s），RCCL 已达链路 97.8%，
      字节数是唯一变量。bf16 环形 all-reduce 每卡过链路 1.5 S；
      本路径 = 分片量化 → all_to_all（0.375 S）→ 反量化求和再量化 → all_gather（0.375 S），合计 0.75 S。
      本卡自己那一片全程不量化。

为什么不用框架自带的 quick all-reduce：其白名单只有 gfx936/938，且走卡间内核直访，
本平台该路径仅约 5 GB/s，低于 RCCL 的 13 GB/s。

作用范围：仅 [T, H] bf16、T×H ≥ 阈值、非图捕获期、world_size==4。decode 与图内预填充不受影响。
运行时开关：存在文件 $SGLANG_QAR_OFF_FILE（默认 /tmp/qar_off）时走原路径，供同实例 A/B。
             只能在服务空闲时切换（各 rank 必须同时看到同一状态）。
"""
import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl

_G = int(os.environ.get("SGLANG_QAR_GROUP", "32"))   # 缺省 32：128 在超长上下文下接受长度下降 10%–32%
_MIN_NUMEL = int(os.environ.get("SGLANG_QAR_MIN_NUMEL", str(2048 * 2048)))
_OFF_FILE = os.environ.get("SGLANG_QAR_OFF_FILE", "/tmp/qar_off")
_IN_GRAPH = int(os.environ.get("SGLANG_QAR_IN_GRAPH", "1"))     # 0 = 图内与 pynccl 上下文保持原路径
_CFG_FILE = os.environ.get("SGLANG_QAR_CFG_FILE", "/tmp/qar_cfg.json")
# 底层专项 L3（10-10，probe/qar_kern.py，图内计时、三核逐位 0 不等）：本卡那一片在 _deq_sum_requant 里不读，
# 只量化外发行；三核 num_warps 4→8。TP2 合计 T=4096 297→224 µs、T=16384 1106→763 µs（−25~31%）。SGLANG_QAR_NW=4 + SGLANG_QAR_ALL_ROWS=1 复原。
_NW = int(os.environ.get("SGLANG_QAR_NW", "8"))
_ALL_ROWS = os.environ.get("SGLANG_QAR_ALL_ROWS", "0") == "1"
_st = {"logged": False, "n_q": 0, "n_orig": 0, "cfg_mtime": None, "dbg_n": 0}
try:
    from ..sched import pair_prefill as _PP
except Exception:          # 补丁目录不在 sys.path 时不影响量化路径本身
    _PP = None


def _refresh_cfg():
    """运行时配置（各 rank 读同一文件；只能在服务空闲时改）。目前支持 G（分组大小）与 debug（误差记录条数）。"""
    global _G
    try:
        m = os.stat(_CFG_FILE).st_mtime
    except OSError:
        return
    if m == _st["cfg_mtime"]:
        return
    _st["cfg_mtime"] = m
    import json
    c = json.load(open(_CFG_FILE))
    _G = int(c.get("G", _G))
    _st["dbg_left"] = int(c.get("debug", 0))
    _st["dbg_tag"] = c.get("tag", "")
    print(f"[quant_ar] 运行时配置已更新：G={_G} debug={_st['dbg_left']} tag={_st['dbg_tag']}", flush=True)


def _debug_compare(self, x, orig):
    """同一输入分别走精确路径与量化路径，记录真实激活上的量化误差（仅 rank0 落盘）。"""
    ref = orig(self, x.clone())     # 必须取返回值：目标模型这条路径是「返回新张量」而非原位修改
    part_amax = x.float().abs().amax().item()
    kurt_in = x.float()
    g = kurt_in.reshape(x.shape[0], -1, _G)
    ratio = (g.abs().amax(-1) / (g.pow(2).mean(-1).sqrt() + 1e-12)).mean().item()   # 组内 峰值/均方根：高斯≈2.8，离群主导时远大于此
    x_in = x.clone() if _st["dbg_n"] + 1 in (2, 3, 83) else None
    y = quant_all_reduce_(x, self.device_group, self.rank_in_group, self.world_size)
    if x_in is not None:
        torch.save({"x": x_in.cpu(), "ref": ref.cpu(), "y": y.cpu()},
                   f"/workspace/sx/qar/dump_n{_st['dbg_n'] + 1}_r{self.rank_in_group}.pt")
    d = (y.float() - ref.float())
    rel = (d.norm() / ref.float().norm()).item()
    rowrel = (d.norm(dim=1) / (ref.float().norm(dim=1) + 1e-12))
    _st["dbg_n"] += 1
    if self.rank_in_group == 0:
        with open(os.environ.get("SGLANG_QAR_ERR_LOG", "/workspace/sx/qar/err.log"), "a") as f:
            f.write(f"{_st['dbg_tag']} n={_st['dbg_n']} T={x.shape[0]} G={_G} rel={rel:.4e} row_med={rowrel.median().item():.4e} "
                    f"row_max={rowrel.max().item():.4e} peak_over_rms={ratio:.2f} part_amax={part_amax:.3e}\n")
    return y


# 27B 移植：H=5120、G=32 → NG=160 不是 2 的幂（tl.arange 的要求）。改为二维偏移，组数补到 NGP=2 的幂，越界组加 mask。
@triton.jit
def _quant_rows(x_ptr, q_ptr, s_ptr, x_rs, q_rs, s_rs,
                H: tl.constexpr, G: tl.constexpr, NG: tl.constexpr, NGP: tl.constexpr):
    r = tl.program_id(0)
    gi = tl.arange(0, NGP)
    offs = gi[:, None] * G + tl.arange(0, G)[None, :]
    m = gi[:, None] < NG
    xg = tl.load(x_ptr + r * x_rs + offs, mask=m, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(xg), axis=1)
    s16 = (amax / 127.0).to(tl.bfloat16)
    s = s16.to(tl.float32)
    s = tl.where(s == 0.0, 1.0, s)
    q = tl.floor(xg / s[:, None] + 0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(q_ptr + r * q_rs + offs, q.to(tl.int8), mask=m)
    tl.store(s_ptr + r * s_rs + gi, s16, mask=gi < NG)


@triton.jit
def _deq_sum_requant(own_ptr, rq_ptr, rs_ptr, oq_ptr, os_ptr,
                     own_rs, rq_ps, rq_rs, rs_ps, rs_rs, oq_rs, os_rs,
                     RANK: tl.constexpr, WORLD: tl.constexpr,
                     H: tl.constexpr, G: tl.constexpr, NG: tl.constexpr, NGP: tl.constexpr):
    r = tl.program_id(0)
    gi = tl.arange(0, NGP)
    offs = gi[:, None] * G + tl.arange(0, G)[None, :]
    m = gi[:, None] < NG
    acc = tl.load(own_ptr + r * own_rs + offs, mask=m, other=0.0).to(tl.float32)
    for p in tl.static_range(WORLD):
        if p != RANK:
            q = tl.load(rq_ptr + p * rq_ps + r * rq_rs + offs, mask=m, other=0).to(tl.float32)
            s = tl.load(rs_ptr + p * rs_ps + r * rs_rs + gi, mask=gi < NG, other=0.0).to(tl.float32)
            acc += q * s[:, None]
    # 本卡这一片直接原位写回全精度结果
    tl.store(own_ptr + r * own_rs + offs, acc.to(tl.bfloat16), mask=m)
    amax = tl.max(tl.abs(acc), axis=1)
    s16 = (amax / 127.0).to(tl.bfloat16)
    s2 = s16.to(tl.float32)
    s2 = tl.where(s2 == 0.0, 1.0, s2)
    q2 = tl.floor(acc / s2[:, None] + 0.5)
    q2 = tl.minimum(tl.maximum(q2, -127.0), 127.0)
    tl.store(oq_ptr + r * oq_rs + offs, q2.to(tl.int8), mask=m)
    tl.store(os_ptr + r * os_rs + gi, s16, mask=gi < NG)


@triton.jit
def _deq_rows(q_ptr, s_ptr, o_ptr, q_rs, s_rs, o_rs,
              H: tl.constexpr, G: tl.constexpr, NG: tl.constexpr, NGP: tl.constexpr):
    r = tl.program_id(0)
    gi = tl.arange(0, NGP)
    offs = gi[:, None] * G + tl.arange(0, G)[None, :]
    m = gi[:, None] < NG
    q = tl.load(q_ptr + r * q_rs + offs, mask=m, other=0).to(tl.float32)
    s = tl.load(s_ptr + r * s_rs + gi, mask=gi < NG, other=0.0).to(tl.float32)
    tl.store(o_ptr + r * o_rs + offs, (q * s[:, None]).to(tl.bfloat16), mask=m)


def quant_all_reduce_(x, group, rank, world, pc=None):
    """原位：x[T,H] ← 各卡 x 之和。T 不必是 world 的整倍数。
    pc 非空时用 pynccl 通信器（图捕获期必须用它：框架在图内的集合通信全部走 pynccl）。"""
    T, H = x.shape
    G = _G
    NG = H // G
    W = H + 2 * NG
    Ts = (T + world - 1) // world
    Tp = Ts * world
    dev = x.device
    send = torch.empty(Tp, W, dtype=torch.int8, device=dev)
    recv = torch.empty(Tp, W, dtype=torch.int8, device=dev)
    red = torch.empty(Ts, W, dtype=torch.int8, device=dev)
    gath = torch.empty(Tp, W, dtype=torch.int8, device=dev)
    if Tp != T:
        send[T:].zero_()
    send_s = send[:, H:].view(torch.bfloat16)
    recv_s = recv[:, H:].view(torch.bfloat16)
    red_s = red[:, H:].view(torch.bfloat16)
    gath_s = gath[:, H:].view(torch.bfloat16)

    lo = rank * Ts
    n_own = max(0, min(T, lo + Ts) - lo)          # 末片可能不满，甚至为空
    NGP = triton.next_power_of_2(NG)
    # 外发行 = [0, lo) ∪ [lo+n_own, T)；send 里本卡那一片 all_to_all 只回给自己且不被读
    for a, b in (((0, T),) if _ALL_ROWS else ((0, lo), (lo + n_own, T))):
        if b > a:
            _quant_rows[(b - a,)](x[a:b], send[a:b], send_s[a:b], x.stride(0), send.stride(0), send_s.stride(0),
                                  H=H, G=G, NG=NG, NGP=NGP, num_warps=_NW)
    if not _st.get("logged_l3"):
        _st["logged_l3"] = True
        print(f"[quant_ar] ✅ L3 首次真正生效：只量化外发行={not _ALL_ROWS} num_warps={_NW} T={T} world={world}", flush=True)
    if pc is not None:
        # 不用 pc.all_to_all_single：厂商封装里 _resolve_stream 多传了一个参数，调用即 TypeError。直接调底层。
        from sglang.srt.distributed.device_communicators.pynccl_wrapper import (
            buffer_type, cudaStream_t, ncclDataTypeEnum)
        pc.nccl.ncclAllToAll(buffer_type(send.data_ptr()), buffer_type(recv.data_ptr()),
                             send.numel() // world, ncclDataTypeEnum.from_torch(send.dtype),
                             pc.comm, cudaStream_t(pc._resolve_stream().cuda_stream))
    else:
        dist.all_to_all_single(recv, send, group=group)

    if n_own < Ts:
        red[n_own:].zero_()
    if n_own > 0:
        own = x[lo:lo + n_own]
        _deq_sum_requant[(n_own,)](own, recv, recv_s, red, red_s,
                                   own.stride(0), Ts * recv.stride(0), recv.stride(0),
                                   Ts * recv_s.stride(0), recv_s.stride(0),
                                   red.stride(0), red_s.stride(0),
                                   RANK=rank, WORLD=world, H=H, G=G, NG=NG, NGP=NGP, num_warps=_NW)
    if pc is not None:
        pc.all_gather(gath, red)
    else:
        dist.all_gather_into_tensor(gath, red, group=group)
    for p in range(world):
        if p == rank:
            continue
        a = p * Ts
        n = max(0, min(T, a + Ts) - a)
        if n == 0:
            continue
        q, s, o = gath[a:a + n], gath_s[a:a + n], x[a:a + n]
        _deq_rows[(n,)](q, s, o, q.stride(0), s.stride(0), o.stride(0), H=H, G=G, NG=NG, NGP=NGP, num_warps=_NW)
    return x


_WORLDS = {int(w) for w in os.environ.get("SGLANG_QAR_WORLDS", "4").split(",") if w}   # 27B 移植：TP2 设 "2"


def _eligible(self, t):
    return (self.world_size in _WORLDS and t.is_cuda and t.dtype == torch.bfloat16
            and t.dim() == 2 and t.is_contiguous()
            and t.shape[1] % _G == 0 and t.numel() >= _MIN_NUMEL)


def install(mod):
    GC = getattr(mod, "GroupCoordinator", None)
    if GC is None or getattr(GC, "_quant_ar_installed", False):
        return
    orig = GC.all_reduce

    def all_reduce(self, input_):
        # 成对预填充（pair_prefill）在目标模型前向的每次 all_reduce 处交接控制权
        if _PP is not None and _PP.ENABLED and _PP.in_handoff():
            _PP.before_ar()
            out = _ar_impl(self, input_)
            _PP.after_ar()
            return out
        return _ar_impl(self, input_)

    def _ar_impl(self, input_):
        _refresh_cfg()
        pc = getattr(self, "pynccl_comm", None)
        pc = pc if (pc is not None and not pc.disabled) else None      # 图捕获上下文里框架会把它置为可用
        in_graph = torch.cuda.is_current_stream_capturing()
        if _IN_GRAPH == 0 and (in_graph or pc is not None):
            return orig(self, input_)
        if in_graph and pc is None:
            return orig(self, input_)
        if _eligible(self, input_) and not os.path.exists(_OFF_FILE):
            if pc is not None:
                if not _st.get("logged_graph"):
                    _st["logged_graph"] = True
                    print(f"[quant_ar] ✅ 图内路径首次真正生效：{tuple(input_.shape)}（pynccl，捕获中={in_graph}，rank {self.rank_in_group}）", flush=True)
                return quant_all_reduce_(input_, self.device_group, self.rank_in_group, self.world_size, pc=pc)
            if _st.get("dbg_left", 0) > 0:
                _st["dbg_left"] -= 1
                return _debug_compare(self, input_, orig)
            _st["n_q"] += 1
            if not _st["logged"]:
                _st["logged"] = True
                print(f"[quant_ar] ✅ 首次真正生效：{tuple(input_.shape)} bf16 → int8 分组 {_G} 量化路径"
                      f"（组 {self.unique_name}，rank {self.rank_in_group}）", flush=True)
            return quant_all_reduce_(input_, self.device_group, self.rank_in_group, self.world_size)
        return orig(self, input_)

    GC.all_reduce = all_reduce
    GC._quant_ar_installed = True
    print(f"[quant_ar] 已安装：阈值 numel≥{_MIN_NUMEL}，分组 {_G}，关闭文件 {_OFF_FILE}", flush=True)

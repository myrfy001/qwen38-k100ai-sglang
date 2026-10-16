"""GDN 预填充因果卷积换自写核（kernels/conv1d.hip，核宽 4）。

原状：SGLANG_USE_CAUSAL_CONV1D=1 时 gdn_backend 预填充走厂商 causal_conv1d_fn_hcu；sglang Triton 版与 conv1d_cfg 补丁都不在生产路径上
      （conv1d_cfg 从未打出「首次真正生效」即因此）。厂商核实测只有 225–283 GB/s（probe/conv1d_hcu_vs_triton.py）。
改法：每线程 8 个通道（16 B 向量读写）、沿 token 连走 TC=4 个、前 3 个 token 的历史放寄存器；块宽 64、__launch_bounds__(64)、无溢出。
      终态由每条序列首块的线程写（读旧状态 → 写末 3 个 token，长度 <3 时用旧状态左移补齐）；pad 槽整条跳过。
实测（10-10，每卡维度 5120，单卡）：T=4096 300→215 µs（1.40×）、5 条不等长 317→213（1.49×）、3×4096 1068→573（1.86×）、
      T=16384 1476→752（1.96×）、T=32768 2978→1527（1.95×）、T=2432 176→164（1.07×）。
数值：NV=5（每个乘积先舍入 bf16 再累加、__expf SiLU）与 sglang Triton 版逐位一致、与厂商差 1 ulp 量级；终态与厂商逐位一致。
网格只依赖形状（chunk 维取上界 ceil(T/TC)+B，多余块在核内返回），不读 seq_lens_cpu 的值，图捕获安全。
只接管：bf16、核宽 4、通道步长 1 且 16 B 对齐、dim % 8 == 0；其余走原实现并记一次原因。
开关 SGLANG_CONV1D_OWN=1；取证 SGLANG_CONV1D_OWN_CHECK=N（前 N 次图外调用同时跑厂商核，打印最大差与状态是否一致）。
.so：SGLANG_CONV1D_OWN_SO（默认 /tmp/q38_conv1d_seg.so，不存在则现编）。"""
import ctypes
import os
import subprocess

import torch

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "kernels", "conv1d.hip")
_SO = os.environ.get("SGLANG_CONV1D_OWN_SO", "/tmp/q38_conv1d_seg.so")   # 10-12 加分段输出入口后改名，旧 .so 无该符号
_CHECK = int(os.environ.get("SGLANG_CONV1D_OWN_CHECK", "0"))
_OFF = "/tmp/q38_conv1d_own_off"     # 同实例 A-B-A：该文件存在时走原实现（图外每次调用查一次）
_NV, _TC, _PAD = 5, 4, -1
_st = {"run": 0, "fb": set(), "chk": 0}
_lib = None


def _load():
    global _lib
    if _lib is None:
        if not os.path.exists(_SO):
            tmp = f"{_SO}.{os.getpid()}.tmp"
            subprocess.run(["hipcc", "-O3", "--offload-arch=gfx928", "-shared", "-fPIC", "-o", tmp, _SRC], check=True)
            os.replace(tmp, _SO)
        _lib = ctypes.CDLL(_SO)
        _lib.conv1d_launch.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 3 + [ctypes.c_long] * 5 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
        _lib.conv1d_launch.restype = ctypes.c_int
        _lib.conv1d_launch_seg.argtypes = ([ctypes.c_void_p] * 8 + [ctypes.c_int] * 3 + [ctypes.c_long] * 5 + [ctypes.c_int] * 6
                                           + [ctypes.c_int] * 2 + [ctypes.c_long] * 4 + [ctypes.c_void_p])
        _lib.conv1d_launch_seg.restype = ctypes.c_int
    return _lib


def _why(x, w, bias, st, activation, qsl):
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16 or st.dtype != torch.bfloat16:
        return f"dtype {x.dtype}/{w.dtype}/{st.dtype}"
    if bias is not None and (bias.dtype != torch.bfloat16 or bias.stride(-1) != 1):
        return "bias"
    if x.dim() != 2 or w.dim() != 2 or w.shape[1] != 4 or w.shape[0] != x.shape[0]:
        return f"形状 x{tuple(x.shape)} w{tuple(w.shape)}"
    if st.dim() != 3 or st.shape[1] != x.shape[0] or st.shape[2] != 3:
        return f"状态形状 {tuple(st.shape)}"
    if x.shape[0] % 8 or x.stride(0) != 1 or x.stride(1) % 8 or x.data_ptr() % 16:
        return f"布局 stride{x.stride()}"
    if activation not in (None, "silu", "swish"):
        return f"激活 {activation}"
    if qsl is None or qsl.dtype != torch.int32:
        return "query_start_loc"
    return None


def _own(x, w, bias, activation, st, his, ci, qsl):
    dim, T = x.shape
    B = qsl.numel() - 1
    out = torch.empty((T, dim), dtype=x.dtype, device=x.device).t()  # (dim, T) 视图、(T, dim) 行连续内存：调用方 .transpose(0, 1) 后即连续（x 可为带行跨步的视图）
    ci = ci.to(torch.int32) if ci is not None and ci.dtype != torch.int32 else ci
    his = his.to(torch.bool) if his is not None and his.dtype != torch.bool else his
    nch = (T + _TC - 1) // _TC + B                 # 上界：Σ ceil(len_b/TC) ≤ ceil(T/TC) + B；只依赖形状
    r = _load().conv1d_launch(x.data_ptr(), w.data_ptr(), bias.data_ptr() if bias is not None else 0, st.data_ptr(),
                              qsl.data_ptr(), ci.data_ptr() if ci is not None else 0, his.data_ptr() if his is not None else 0,
                              out.data_ptr(), dim, B, nch, x.stride(1), out.stride(1), st.stride(0), st.stride(1), st.stride(2),
                              w.stride(0), w.stride(1), _PAD, 1 if activation in ("silu", "swish") else 0, _NV, _TC,
                              ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if r != 0:
        raise RuntimeError(f"[conv1d_own] 自写核返回 {r}")
    return out


def own_seg(x, w, bias, activation, st, his, ci, qsl, d0, d1):
    """同 _own，但输出直接写成三块各自行连续的张量：通道 [0,d0)、[d0,d0+d1)、[d0+d1,dim)（供 qkv_view 省掉 q/k/v 的 .contiguous()）。"""
    dim, T = x.shape
    B = qsl.numel() - 1
    d2 = dim - d0 - d1
    buf = torch.empty(T * dim, dtype=x.dtype, device=x.device)
    ci = ci.to(torch.int32) if ci is not None and ci.dtype != torch.int32 else ci
    his = his.to(torch.bool) if his is not None and his.dtype != torch.bool else his
    nch = (T + _TC - 1) // _TC + B
    r = _load().conv1d_launch_seg(x.data_ptr(), w.data_ptr(), bias.data_ptr() if bias is not None else 0, st.data_ptr(),
                                  qsl.data_ptr(), ci.data_ptr() if ci is not None else 0, his.data_ptr() if his is not None else 0,
                                  buf.data_ptr(), dim, B, nch, x.stride(1), d0, st.stride(0), st.stride(1), st.stride(2),
                                  w.stride(0), w.stride(1), _PAD, 1 if activation in ("silu", "swish") else 0, _NV, _TC,
                                  d0, d0 + d1, T * d0, T * (d0 + d1), d1, d2,
                                  ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if r != 0:
        raise RuntimeError(f"[conv1d_own] 分段输出核返回 {r}")
    _st["seg"] = _st.get("seg", 0) + 1
    if _st["seg"] == 1:
        print(f"[conv1d_own] ✅ 自写因果卷积首次真正生效（分段输出 {d0}/{d1}/{d2}）：x{tuple(x.shape)} stride{x.stride()} bs={B} "
              f"capturing={torch.cuda.is_current_stream_capturing()} pid={os.getpid()}", flush=True)
    return buf[: T * d0].view(T, d0), buf[T * d0: T * (d0 + d1)].view(T, d1), buf[T * (d0 + d1):].view(T, d2)


def install(mod):
    if getattr(mod, "_conv1d_own_installed", False):
        return
    if not hasattr(mod, "causal_conv1d_fn_hcu"):
        print(f"[conv1d_own] 未安装：{mod.__name__} 没有 causal_conv1d_fn_hcu（SGLANG_USE_CAUSAL_CONV1D 未开？） pid={os.getpid()}", flush=True)
        return
    orig = mod.causal_conv1d_fn_hcu

    def causal_conv1d_fn_hcu(x, weight, bias=None, activation=None, initial_states=None, has_initial_state=None,
                             cache_indices=None, query_start_loc=None, seq_lens_cpu=None, **kw):
        if x.dim() == 2 and x.stride(0) == 1 and x.stride(1) != x.shape[0]:
            xc = x.transpose(0, 1).contiguous().transpose(0, 1)    # qkv_view 传来的跨步视图：厂商回退路径先补成连续
        else:
            xc = x
        call = lambda st: orig(xc, weight, bias, activation=activation, initial_states=st, has_initial_state=has_initial_state,
                               cache_indices=cache_indices, query_start_loc=query_start_loc, seq_lens_cpu=seq_lens_cpu, **kw)
        if os.path.exists(_OFF) and not torch.cuda.is_current_stream_capturing():
            return call(initial_states)
        why = "额外参数" if kw else ("无状态" if initial_states is None else
                                    _why(x, weight, bias, initial_states, activation, query_start_loc))
        if why:
            if why not in _st["fb"]:
                _st["fb"].add(why)
                print(f"[conv1d_own] 不接管（{why}），走原实现 pid={os.getpid()}", flush=True)
            return call(initial_states)
        chk = _st["chk"] < _CHECK and not torch.cuda.is_current_stream_capturing()
        if chk:
            st_ref = initial_states.clone()
            ref = call(st_ref)
        out = _own(x, weight, bias, activation, initial_states, has_initial_state, cache_indices, query_start_loc)
        _st["run"] += 1
        if _st["run"] == 1:
            print(f"[conv1d_own] ✅ 自写因果卷积首次真正生效：x{tuple(x.shape)} stride{x.stride()} bs={query_start_loc.numel() - 1} "
                  f"state{tuple(initial_states.shape)} act={activation} bias={bias is not None} "
                  f"capturing={torch.cuda.is_current_stream_capturing()} pid={os.getpid()}", flush=True)
        if chk:
            _st["chk"] += 1
            d = (out.float() - ref.float()).abs().max().item()
            print(f"[conv1d_own] 取证 {_st['chk']}/{_CHECK}：对厂商输出最大差 {d:.3e}、终态逐位一致={torch.equal(st_ref, initial_states)} "
                  f"x{tuple(x.shape)} bs={query_start_loc.numel() - 1} pid={os.getpid()}", flush=True)
        return out

    mod.causal_conv1d_fn_hcu = causal_conv1d_fn_hcu
    mod._conv1d_own_installed = True
    _load()
    print(f"[conv1d_own] 已安装到 {mod.__name__}（NV={_NV} TC={_TC}，.so {_SO}） pid={os.getpid()}", flush=True)

"""自写分页预填充注意力（底层专项 L4，kernels/pf_attn.hip，PF_NW=4 PF_MT=2：每块 4 wave × 每 wave 32 行 = 128 行 Q）。

原状：fa3 后端 HCU 分支 forward_extend → vllm_flash_attn_varlen_func（分页，K [页,2,64,256]、V 转置 [页,2,256,64]）。
      probe/pf_attn_test.py（10-10，64K 按块 16384 四段）：厂商 45–46 TFLOPS，16 层合计 9.20 s，占 64K TTFT 22.5 s 的约 41%。
改法：骨架移植自 Qwen3.6 pp_v5；GQA6 的 6 个 q 头与 token 展平成行（零浪费）、bf16 直接装 LDS、每块 128 行共享一份 K/V 半页
      （每读 1 KB K/V 做 128 K FLOP，把 L2 受限的上限抬高一倍）。自写 77–80 TFLOPS，1.71–1.73×，16 层合计 5.33 s。
正确性：fp32 参考对拍误差 2.1–2.3e-3（厂商 2.0–2.15e-3），多请求不等长/前缀覆盖 mod 64 各余数类/页表乱序；
        对抗性掩码（页尾与未来 token 写 1e4）输出逐位不变。
只接管：bf16 Q/KV、因果、无滑窗、descale 为 None 或恒 1 的单元素张量（bf16 KV 时后端固定传 [1.0]）、12 q 头/2 KV 头/D256/页 64、max_seqlen_q ≥ SGLANG_PF_ATTN_MINQ（默认 64，
        验证步 q=4、草稿扩展不受影响 → 不动接受长度）。其余走原实现并记一次原因。
开关 SGLANG_PF_ATTN=1；取证 SGLANG_PF_ATTN_CHECK=N（前 N 次图外调用同时跑厂商核，打印相对差）。
.so：SGLANG_PF_ATTN_SO（默认 /tmp/q38_pf_attn_nw42.so，不存在则现编）。"""
import ctypes
import os
import subprocess

import torch

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "kernels", "pf_attn.hip")
_SO = os.environ.get("SGLANG_PF_ATTN_SO", "/tmp/q38_pf_attn_nw42.so")
_MINQ = int(os.environ.get("SGLANG_PF_ATTN_MINQ", "64"))
_CHECK = int(os.environ.get("SGLANG_PF_ATTN_CHECK", "0"))
_st = {"run": 0, "fb": set(), "chk": 0, "va": 0, "vchk": 0}
_lib = None
# 验证步分支（10-10，kernels/va_attn.hip）：q ≤ 10 token/请求、bs ≥ MINBS 时沿 KV 切 S 段 + lse 合并。
#   离线（probe/verify_attn_test.py，ctx≈4K，图内）：bs16 401→253 µs、bs32 823→427、bs48 1445→698（1.6–2.1×），误差 2.34e-3 低于厂商 2.90e-3；
#   bs≤8 厂商更快（bs1 42 µs），保留厂商。S 取「总块数 bs·2·S ≈ 192」（每 CU ~1.6 块）实测最好。
_VA = os.environ.get("SGLANG_VERIFY_ATTN", "0") == "1"
_VA_MINBS = int(os.environ.get("SGLANG_VERIFY_ATTN_MINBS", "12"))
_VA_SO = os.environ.get("SGLANG_VERIFY_ATTN_SO", "/tmp/q38_va_mt1.so")
_VA_CAP = 192                      # S·bs 上限（工作区按此一次分配，图捕获安全）
_va = {"lib": None, "po": None, "pl": None}


def _va_split(bs):
    return max(1, min(6, round(96 / bs)))   # S8 在 bs16·64K 撞通道（0.90×），封顶 6


def _va_load():
    if _va["lib"] is None:
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "kernels", "va_attn.hip")
        if not os.path.exists(_VA_SO):
            tmp = f"{_VA_SO}.{os.getpid()}.tmp"
            subprocess.run(["hipcc", "-O3", "--offload-arch=gfx928", "-shared", "-fPIC", "-DPF_NW=4", "-DPF_MT=1",
                            "-o", tmp, src], check=True)
            os.replace(tmp, _VA_SO)
        lib = ctypes.CDLL(_VA_SO)
        lib.va_attn_run.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int] * 5 + [ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
        lib.va_attn_run.restype = ctypes.c_int
        _va["lib"] = lib
    return _va["lib"]


def _load():
    global _lib
    if _lib is None:
        if not os.path.exists(_SO):
            tmp = f"{_SO}.{os.getpid()}.tmp"
            subprocess.run(["hipcc", "-O3", "--offload-arch=gfx928", "-shared", "-fPIC", "-DPF_NW=4", "-DPF_MT=2",
                            "-o", tmp, _SRC], check=True)
            os.replace(tmp, _SO)
        _lib = ctypes.CDLL(_SO)
        _lib.pf_attn_run.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 5 + [ctypes.c_float, ctypes.c_void_p]
        _lib.pf_attn_run.restype = ctypes.c_int
    return _lib


_UNIT = {}     # data_ptr -> 是否恒为 1（bf16 KV 时后端传 self.k_scale = tensor([1.0])，flashattention_backend.py:191/730）


def _unit(d, bf16_kv=False):
    if d.numel() != 1:
        return False
    key = (d.data_ptr(), d.dtype)
    if key not in _UNIT:
        if torch.cuda.is_current_stream_capturing():
            # 捕获期不能读值；bf16 KV 时后端固定传 [1.0]（且非 1 本身就是错配），验证步分支按 1 处理
            return bf16_kv
        _UNIT[key] = float(d.reshape(-1)[0]) == 1.0
    return _UNIT[key]


def _why(q, k, v, cu_seqlens_q, max_seqlen_q, seqused_k, causal, window_size, block_table, q_descale, k_descale, v_descale,
         bf16_kv_unit=False):
    if not causal:
        return "非因果"
    if block_table is None or seqused_k is None:
        return "非分页"
    if window_size is not None and tuple(window_size) != (-1, -1):
        return "滑窗"
    for d in (q_descale, k_descale, v_descale):
        if d is not None and not _unit(d, bf16_kv_unit and k.dtype == torch.bfloat16):
            return "descale≠1"
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        return f"dtype {q.dtype}/{k.dtype}"
    if q.dim() != 3 or q.shape[2] != 256 or k.dim() != 4 or tuple(k.shape[1:]) != (2, 64, 256) or tuple(v.shape[1:]) != (2, 256, 64):
        return f"形状 q{tuple(q.shape)} k{tuple(k.shape)} v{tuple(v.shape)}"
    if q.shape[1] != 12:
        return f"q 头 {q.shape[1]}"
    if block_table.dtype != torch.int32 or seqused_k.dtype != torch.int32 or cu_seqlens_q.dtype != torch.int32:
        return "索引 dtype"
    return None


def _verify(call, q, k, v, cu_seqlens_q, mq, seqused_k, max_seqlen_k, softmax_scale, causal, window_size, block_table,
            q_descale, k_descale, v_descale):
    bs = cu_seqlens_q.numel() - 1
    S = _va_split(bs)
    if bs < _VA_MINBS or S * bs > _VA_CAP:
        return call()
    why = _why(q, k, v, cu_seqlens_q, mq, seqused_k, causal, window_size, block_table, q_descale, k_descale, v_descale,
               bf16_kv_unit=True)
    if why:
        if ("va", why) not in _st["fb"]:
            _st["fb"].add(("va", why))
            print(f"[verify_attn] 不接管（{why}），走原实现 pid={os.getpid()}", flush=True)
        return call()
    lib = _va_load()
    if _va["po"] is None:
        _va["po"] = torch.empty(_VA_CAP * 2 * 60 * 256, dtype=torch.float32, device=q.device)
        _va["pl"] = torch.empty(_VA_CAP * 2 * 60, dtype=torch.float32, device=q.device)
    qc = q.contiguous()
    bt = block_table.contiguous()
    out = torch.empty_like(qc)
    r = lib.va_attn_run(out.data_ptr(), _va["po"].data_ptr(), _va["pl"].data_ptr(), qc.data_ptr(), k.data_ptr(), v.data_ptr(),
                        bt.data_ptr(), cu_seqlens_q.data_ptr(), seqused_k.data_ptr(), bs, bt.shape[1], mq, 12, 2,
                        float(softmax_scale), S, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if r != 0:
        raise RuntimeError(f"[verify_attn] 自写核返回 {r}")
    _st["va"] += 1
    cap = torch.cuda.is_current_stream_capturing()
    if _st["va"] == 1 or (cap and bs in (16, 32, 48) and ("vacap", bs) not in _st["fb"]):
        _st["fb"].add(("vacap", bs))
        print(f"[verify_attn] ✅ 验证步切分注意力首次真正生效：q{tuple(q.shape)} bs={bs} max_q={mq} 切分 S={S} "
              f"capturing={cap} pid={os.getpid()}", flush=True)
    if _st["vchk"] < _CHECK and not cap:
        _st["vchk"] += 1
        ref = call().float()
        rel = ((out.float() - ref).norm() / (ref.norm() + 1e-12)).item()
        print(f"[verify_attn] 取证 {_st['vchk']}/{_CHECK}：对厂商相对差 {rel:.3e} q{tuple(q.shape)} bs={bs} S={S} "
              f"finite={bool(torch.isfinite(out).all())} pid={os.getpid()}", flush=True)
    return out


def install(mod):
    if getattr(mod, "_pf_attn_installed", False):
        return
    orig = mod.vllm_flash_attn_varlen_func

    def vllm_flash_attn_varlen_func(q, k, v, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k, softmax_scale, causal,
                                    window_size, block_table, fa_version, q_descale, k_descale, v_descale):
        call = lambda: orig(q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q, seqused_k=seqused_k,
                            max_seqlen_k=max_seqlen_k, softmax_scale=softmax_scale, causal=causal, window_size=window_size,
                            block_table=block_table, fa_version=fa_version, q_descale=q_descale, k_descale=k_descale,
                            v_descale=v_descale)
        mq = int(max_seqlen_q)
        if _VA and mq <= 10:
            return _verify(call, q, k, v, cu_seqlens_q, mq, seqused_k, max_seqlen_k, softmax_scale, causal, window_size,
                           block_table, q_descale, k_descale, v_descale)
        if mq < _MINQ:
            return call()
        why = _why(q, k, v, cu_seqlens_q, max_seqlen_q, seqused_k, causal, window_size, block_table, q_descale, k_descale, v_descale)
        if why:
            if why not in _st["fb"]:
                _st["fb"].add(why)
                print(f"[pf_attn] 不接管（{why}），走原实现 pid={os.getpid()}", flush=True)
            return call()
        qc = q.contiguous()
        bt = block_table.contiguous()
        out = torch.empty_like(qc)
        r = _load().pf_attn_run(out.data_ptr(), qc.data_ptr(), k.data_ptr(), v.data_ptr(), bt.data_ptr(),
                                cu_seqlens_q.data_ptr(), seqused_k.data_ptr(), cu_seqlens_q.numel() - 1, bt.shape[1],
                                int(max_seqlen_q), 12, 2, float(softmax_scale),
                                ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        if r != 0:
            raise RuntimeError(f"[pf_attn] 自写核返回 {r}")
        _st["run"] += 1
        if _st["run"] == 1:
            print(f"[pf_attn] ✅ 自写预填充注意力首次真正生效：q{tuple(q.shape)} bs={cu_seqlens_q.numel() - 1} "
                  f"max_q={int(max_seqlen_q)} max_k={int(max_seqlen_k)} capturing={torch.cuda.is_current_stream_capturing()} pid={os.getpid()}", flush=True)
        if _st["chk"] < _CHECK and not torch.cuda.is_current_stream_capturing():
            _st["chk"] += 1
            ref = call().float()
            rel = ((out.float() - ref).norm() / (ref.norm() + 1e-12)).item()
            print(f"[pf_attn] 取证 {_st['chk']}/{_CHECK}：对厂商相对差 {rel:.3e} q{tuple(q.shape)} max_k={int(max_seqlen_k)} "
                  f"finite={bool(torch.isfinite(out).all())} pid={os.getpid()}", flush=True)
        return out

    mod.vllm_flash_attn_varlen_func = vllm_flash_attn_varlen_func
    mod._pf_attn_installed = True
    _load()
    if _VA:
        _va_load()
    print(f"[pf_attn] 已安装到 {mod.__name__}（MINQ={_MINQ}，.so {_SO}；验证步分支 {'开 MINBS=' + str(_VA_MINBS) if _VA else '关'}） pid={os.getpid()}", flush=True)

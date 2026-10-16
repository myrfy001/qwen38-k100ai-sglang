// GDN 预填充因果卷积（核宽 4，逐通道）流式核。
// 为什么（10-10 probe/conv1d_hcu_vs_triton.py，每卡 dim=5120）：生产走厂商 causal_conv1d_fn_hcu，T=4096 297 µs（282 GB/s）、
//   T=16384 1474 µs（228 GB/s）；sglang Triton 版加分档也只有 200–330 GB/s。数据量只有「读 x 一遍 + 写 out 一遍」，
//   带宽下限约 T·dim·4 B / 850 GB/s（T=4096 约 100 µs）。
// 做法：x 与 out 的内存都是 (T, dim) 行连续（gdn_backend 传入的是 mixed_qkv(T,dim).transpose(0,1)）。每线程负责 8 个相邻通道
//   （16 B 向量），沿 token 方向走一个块（TC 个 token），前 3 个 token 的历史留在寄存器里：每个元素只读一次、写一次，
//   一个 wave 一条指令读一行里连续 1 KB。
// 语义与 sglang Triton 版 / 厂商版一致：
//   - 序列 b 的 token 区间 [qsl[b], qsl[b+1])；cache_indices[b] == pad 时整条序列跳过（不写输出、不写状态）；
//   - 历史：块起点在序列中间时取前 3 个 token；序列开头时 has_initial_state[b] 为真取 conv_states[ci][c][0..2]（下标 2 最新），否则 0；
//   - 终态：序列的第一个块的线程先读旧状态，再把序列最后 3 个 token（长度 <3 时左侧用旧状态移位补）写回 conv_states[ci]，
//     读在写前、同一线程，无竞争（与 Triton 版「chunk_offset==0 的程序写状态」相同）。
// 数值变体 NV（位）：1 = 每项乘积先舍入 bf16 再加（Triton 写法）；2 = 用 fmaf；4 = 快速 __expf；8 = silu 写成 acc·(1/(1+e))；
//   16 = 累加顺序反向（最新 token 先加）；32 = 快速除法 __fdividef。
#include <hip/hip_runtime.h>
#include <stdint.h>

using u32x4 = __attribute__((ext_vector_type(4))) unsigned;

__device__ __forceinline__ float bf2f(unsigned short h) { return __uint_as_float(((unsigned)h) << 16); }
__device__ __forceinline__ unsigned short f2bf(float f) {
    unsigned u = __float_as_uint(f);
    u = u + 0x7fffu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}
__device__ __forceinline__ unsigned short ch(const u32x4& v, int i) {
    unsigned w = v[i >> 1];
    return (unsigned short)((i & 1) ? (w >> 16) : (w & 0xffffu));
}

template <int NV>
__device__ __forceinline__ float term(float acc, float x, float w) {
    if (NV & 1) return acc + bf2f(f2bf(x * w));
    if (NV & 2) return fmaf(x, w, acc);
    return acc + x * w;
}
template <int NV>
__device__ __forceinline__ float silu(float a) {
    float e = (NV & 4) ? __expf(-a) : expf(-a);
    if (NV & 32) return (NV & 8) ? a * __fdividef(1.0f, 1.0f + e) : __fdividef(a, 1.0f + e);
    return (NV & 8) ? a * (1.0f / (1.0f + e)) : a / (1.0f + e);
}

template <int NV, int TC>
__global__ __launch_bounds__(64) void conv1d_kernel(
    const unsigned short* __restrict__ x, const unsigned short* __restrict__ w, const unsigned short* __restrict__ bias,
    unsigned short* __restrict__ st, const int* __restrict__ qsl, const int* __restrict__ ci, const bool* __restrict__ his,
    unsigned short* __restrict__ out, int dim, int B, long sxt, long sot, long s_seq, long s_dim, long s_tok,
    int w_sd, int w_sw, int pad, int silu_on, int seg1, int seg2, long off1, long off2, long sot1, long sot2) {
    const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (c0 >= dim) return;
    // 块号 → (序列 b, 块内起点)：线性扫 qsl（B 小）
    int chunk = blockIdx.y, b = 0, beg = 0, len = 0;
    for (; b < B; ++b) {
        beg = qsl[b]; len = qsl[b + 1] - beg;
        int nb = (len + TC - 1) / TC;
        if (chunk < nb) break;
        chunk -= nb;
    }
    if (b >= B) return;
    const int slot = ci ? ci[b] : b;
    if (slot == pad) return;
    const int t0 = chunk * TC, t1 = min(len, t0 + TC);

    float wf[4][8], bf[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int k = 0; k < 4; ++k) wf[k][i] = bf2f(w[(long)(c0 + i) * w_sd + k * w_sw]);
        bf[i] = bias ? bf2f(bias[c0 + i]) : 0.0f;
    }
    // 历史 h[0..2]（h[2] 最新）
    float h[3][8];
    unsigned short* sp = st + (long)slot * s_seq;
    if (t0 > 0) {
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            u32x4 v = *(const u32x4*)(x + (long)(beg + t0 - 3 + k) * sxt + c0);
#pragma unroll
            for (int i = 0; i < 8; ++i) h[k][i] = bf2f(ch(v, i));
        }
    } else {
        const bool init = his ? his[b] : false;
        unsigned short old[3][8];
#pragma unroll
        for (int k = 0; k < 3; ++k)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                old[k][i] = init ? sp[(long)(c0 + i) * s_dim + k * s_tok] : (unsigned short)0;
                h[k][i] = bf2f(old[k][i]);
            }
        // 终态：序列末 3 个 token；长度不足 3 时左侧用旧状态移位
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            const int idx = len - 3 + k;
            u32x4 v = {0u, 0u, 0u, 0u};
            if (idx >= 0) v = *(const u32x4*)(x + (long)(beg + idx) * sxt + c0);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int j = k + len;   // 只在 idx<0（len<3）时用；用 select 链取，动态下标会把 old 放进 scratch
                unsigned short nv = idx >= 0 ? ch(v, i) : (j == 0 ? old[0][i] : j == 1 ? old[1][i] : old[2][i]);
                sp[(long)(c0 + i) * s_dim + k * s_tok] = nv;
            }
        }
    }

    const unsigned short* xp = x + (long)beg * sxt + c0;
    // 分段输出（10-12）：通道 [0,seg1) 写 out（行跨步 sot），[seg1,seg2) 写 out+off1（sot1），[seg2,dim) 写 out+off2（sot2）。
    // 用于把 q/k/v 直接写成三块各自连续的张量，省掉下游 GDN 内核入口的 .contiguous()；seg1=seg2=dim 时与原来相同。
    // 段界是 8 的倍数，一个线程的 8 个通道不会跨段。
    unsigned short* op;
    if (c0 < seg1) op = out + (long)beg * sot + c0;
    else if (c0 < seg2) { op = out + off1 + (long)beg * sot1 + (c0 - seg1); sot = sot1; }
    else { op = out + off2 + (long)beg * sot2 + (c0 - seg2); sot = sot2; }
    int t = t0;
    for (; t + 4 <= t1; t += 4) {
        u32x4 v[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) v[u] = *(const u32x4*)(xp + (long)(t + u) * sxt);
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            u32x4 o;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                unsigned short r[2];
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const int i = 2 * p + q;
                    const float xc = bf2f(ch(v[u], i));
                    float a = bf[i];
                    if (NV & 16) {
                        a = term<NV>(a, xc, wf[3][i]); a = term<NV>(a, h[2][i], wf[2][i]);
                        a = term<NV>(a, h[1][i], wf[1][i]); a = term<NV>(a, h[0][i], wf[0][i]);
                    } else {
                        a = term<NV>(a, h[0][i], wf[0][i]); a = term<NV>(a, h[1][i], wf[1][i]);
                        a = term<NV>(a, h[2][i], wf[2][i]); a = term<NV>(a, xc, wf[3][i]);
                    }
                    if (silu_on) a = silu<NV>(a);
                    r[q] = f2bf(a);
                    h[0][i] = h[1][i]; h[1][i] = h[2][i]; h[2][i] = xc;
                }
                o[p] = (unsigned)r[0] | ((unsigned)r[1] << 16);
            }
            *(u32x4*)(op + (long)(t + u) * sot) = o;
        }
    }
    for (; t < t1; ++t) {
        u32x4 v = *(const u32x4*)(xp + (long)t * sxt), o;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            unsigned short r[2];
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int i = 2 * p + q;
                const float xc = bf2f(ch(v, i));
                float a = bf[i];
                if (NV & 16) {
                    a = term<NV>(a, xc, wf[3][i]); a = term<NV>(a, h[2][i], wf[2][i]);
                    a = term<NV>(a, h[1][i], wf[1][i]); a = term<NV>(a, h[0][i], wf[0][i]);
                } else {
                    a = term<NV>(a, h[0][i], wf[0][i]); a = term<NV>(a, h[1][i], wf[1][i]);
                    a = term<NV>(a, h[2][i], wf[2][i]); a = term<NV>(a, xc, wf[3][i]);
                }
                if (silu_on) a = silu<NV>(a);
                r[q] = f2bf(a);
                h[0][i] = h[1][i]; h[1][i] = h[2][i]; h[2][i] = xc;
            }
            o[p] = (unsigned)r[0] | ((unsigned)r[1] << 16);
        }
        *(u32x4*)(op + (long)t * sot) = o;
    }
}

template <int NV, int TC>
static int go(const void* x, const void* w, const void* bias, void* st, const void* qsl, const void* ci, const void* his, void* out,
              int dim, int B, int nchunks, long sxt, long sot, long s_seq, long s_dim, long s_tok, int w_sd, int w_sw, int pad,
              int silu_on, int bw, int seg1, int seg2, long off1, long off2, long sot1, long sot2, hipStream_t s) {
    // 块宽固定 64：实测 64 最优（10-10 扫 0/64/128/256/640）；__launch_bounds__ 再大编译器就压 VGPR 致溢出（1024 慢 4 倍、640 溢出 112 B、256 溢出 64 B）
    const int nb = 64;
    dim3 g((dim / 8 + nb - 1) / nb, nchunks);
    hipLaunchKernelGGL((conv1d_kernel<NV, TC>), g, dim3(nb), 0, s, (const unsigned short*)x, (const unsigned short*)w,
                       (const unsigned short*)bias, (unsigned short*)st, (const int*)qsl, (const int*)ci, (const bool*)his,
                       (unsigned short*)out, dim, B, sxt, sot, s_seq, s_dim, s_tok, w_sd, w_sw, pad, silu_on,
                       seg1, seg2, off1, off2, sot1, sot2);
    return (int)hipGetLastError();
}

#define TCS(NV_)                                                                                                     \
    case NV_:                                                                                                        \
        switch (tc) {                                                                                                \
            case 4: return go<NV_, 4>(x, w, bias, st, qsl, ci, his, out, dim, B, nchunks, sxt, sot, s_seq, s_dim, s_tok, w_sd, w_sw, pad, silu_on, bw, seg1, seg2, off1, off2, sot1, sot2, s); \
            case 8: return go<NV_, 8>(x, w, bias, st, qsl, ci, his, out, dim, B, nchunks, sxt, sot, s_seq, s_dim, s_tok, w_sd, w_sw, pad, silu_on, bw, seg1, seg2, off1, off2, sot1, sot2, s); \
            case 16: return go<NV_, 16>(x, w, bias, st, qsl, ci, his, out, dim, B, nchunks, sxt, sot, s_seq, s_dim, s_tok, w_sd, w_sw, pad, silu_on, bw, seg1, seg2, off1, off2, sot1, sot2, s); \
            case 32: return go<NV_, 32>(x, w, bias, st, qsl, ci, his, out, dim, B, nchunks, sxt, sot, s_seq, s_dim, s_tok, w_sd, w_sw, pad, silu_on, bw, seg1, seg2, off1, off2, sot1, sot2, s); \
            case 64: return go<NV_, 64>(x, w, bias, st, qsl, ci, his, out, dim, B, nchunks, sxt, sot, s_seq, s_dim, s_tok, w_sd, w_sw, pad, silu_on, bw, seg1, seg2, off1, off2, sot1, sot2, s); \
            case 128: return go<NV_, 128>(x, w, bias, st, qsl, ci, his, out, dim, B, nchunks, sxt, sot, s_seq, s_dim, s_tok, w_sd, w_sw, pad, silu_on, bw, seg1, seg2, off1, off2, sot1, sot2, s); \
        }                                                                                                            \
        return -2;

// 入口：x/out 为 bf16，(T, dim) 行跨步 sxt/sot（元素）；w (dim, 4) 跨步 w_sd/w_sw；st = conv_states（slot, dim, 3）跨步；
// qsl int32 [B+1]、ci int32 [B]（可空）、his bool [B]（可空）；nchunks = Σ ceil(len_b / tc)（主机侧由 seq_lens_cpu 算）。dim % 8 == 0。
static int launch(const void* x, const void* w, const void* bias, void* st, const void* qsl, const void* ci, const void* his,
                  void* out, int dim, int B, int nchunks, long sxt, long sot, long s_seq, long s_dim, long s_tok,
                  int w_sd, int w_sw, int pad, int silu_on, int nv, int tc, int seg1, int seg2, long off1, long off2,
                  long sot1, long sot2, void* stream) {
    hipStream_t s = (hipStream_t)stream;
    const int bw = (tc >> 16) & 0xffff; tc &= 0xffff;   // tc 高 16 位 = 块宽（0 = 整行）
    if (dim % 8 || seg1 % 8 || seg2 % 8 || seg1 > seg2 || seg2 > dim) return -1;
    switch (nv) { TCS(0) TCS(5) TCS(13) TCS(37) }
    return -3;
}

extern "C" int conv1d_launch(const void* x, const void* w, const void* bias, void* st, const void* qsl, const void* ci, const void* his,
                             void* out, int dim, int B, int nchunks, long sxt, long sot, long s_seq, long s_dim, long s_tok,
                             int w_sd, int w_sw, int pad, int silu_on, int nv, int tc, void* stream) {
    return launch(x, w, bias, st, qsl, ci, his, out, dim, B, nchunks, sxt, sot, s_seq, s_dim, s_tok, w_sd, w_sw, pad, silu_on,
                  nv, tc, dim, dim, 0, 0, sot, sot, stream);
}

// 分段输出入口：out 为一块缓冲，三段起点（元素）0 / off1 / off2，各自行跨步 sot / sot1 / sot2。
extern "C" int conv1d_launch_seg(const void* x, const void* w, const void* bias, void* st, const void* qsl, const void* ci,
                                 const void* his, void* out, int dim, int B, int nchunks, long sxt, long sot, long s_seq,
                                 long s_dim, long s_tok, int w_sd, int w_sw, int pad, int silu_on, int nv, int tc, int seg1,
                                 int seg2, long off1, long off2, long sot1, long sot2, void* stream) {
    return launch(x, w, bias, st, qsl, ci, his, out, dim, B, nchunks, sxt, sot, s_seq, s_dim, s_tok, w_sd, w_sw, pad, silu_on,
                  nv, tc, seg1, seg2, off1, off2, sot1, sot2, stream);
}

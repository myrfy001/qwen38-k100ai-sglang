// 验证步分页注意力（由 kernels/pf_attn.hip 派生，10-10）：q ≤ 10 token/请求，沿 KV 切 nsplit 段 + lse 合并（flash-decoding）。
// 原预填充核每 (请求, KV 头) 只有一块、串行走完整段 KV，bs48×4K 约 1 ms；厂商 paged_attention_kernel 只到 281–336 GB/s。
// 以下为原文件说明：自写分页预填充注意力（底层专项 L4）：Qwen3.8-27B 每卡 12 个 q 头 / 2 个 KV 头（GQA6）、D=256、页 64、bf16 KV、因果。
// 服务调用（fa3 HCU 分支 → vllm_flash_attn_varlen_func）：K 缓存 [页][2][64][256]，V 缓存转置 [页][2][256][64]。
// 骨架移植自 Qwen3.6 的 pp_v5（K100AI-QWEN3.6/kernels/pp_v5.cpp：fp8 KV、1 个 KV 头、GQA4），改动：
//   · Q 行打包：一个 KV 头的 6 个 q 头与 token 展平成行 r = tok·6 + h，每 wave 取连续 16 行——GQA6 不整除 16，按 token×头 拼会浪费；
//     展平后行 r 的因果位置 = prefix + r/6，掩码逐行算，零浪费。
//   · K/V 是 bf16：装载直接拷进 LDS（不解包），mmac 换 bf16 版；P 以 bf16 进 PV（厂商 bf16 核同精度）。
//   · grid.z = KV 头；PF_NW 个 wave 一块（每块 16·PF_NW 行共享一份 K/V 半页）。
//     量级：每读 1 KB K/V（一个 key token）做 1024·行数 FLOP；L2 约 945 GB/s（Qwen3.6 PMC）→ 64 行上限约 60 TFLOPS、128 行约 120。
#include <hip/hip_runtime.h>
#include <cmath>
using u16 = unsigned short;
using bf16x4 = __attribute__((ext_vector_type(4))) __bf16;
using f32x4 = __attribute__((ext_vector_type(4))) float;
using u32x4 = __attribute__((ext_vector_type(4))) unsigned;
using u32x2 = __attribute__((ext_vector_type(2))) unsigned;

__device__ __forceinline__ bf16x4 as_bf(u32x2 v) { return __builtin_bit_cast(bf16x4, v); }
__device__ __forceinline__ u16 f2bf(float f) { unsigned u = __float_as_uint(f); if (!(f == f)) return 0x7fc0; return (u16)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
__device__ __forceinline__ float fexp(float x) { return __builtin_amdgcn_exp2f(x * __uint_as_float(0x3fb8aa3bu)); }
__device__ __forceinline__ unsigned shfl_lane(unsigned v, int src) { return __builtin_amdgcn_ds_bpermute(src << 2, v); }

#define HD   256
#define PAGE 64
#define GQA  6
#ifndef PF_NW
#define PF_NW 4
#endif
#ifndef PF_MT
#define PF_MT 1                      // 每 wave 几个 16 行片（2 = 每 wave 32 行，寄存器翻倍、K/V 每读一次多用一倍）
#endif
#define NT   (PF_NW * 64)            // 线程数
#define RB   (PF_NW * 16 * PF_MT)    // 每块 Q 行数（展平行）
#define HP   32                      // 半页 token 数
#define KSTR 130                     // K 阶段 [32 token][128 dword + 2]：dword 跨步 ≡ 2 mod 32，读侧 b64 两拍铺满
#define VSTR 18                      // V 阶段 [256 dim][16 dword + 2]
#define KB   (HP * HD * 2 / NT)      // 每线程每半页 K 字节：NW4 64、NW8 32
#define NKV  (KB / 16)
#define TPT  (HD * 2 / KB)           // 每 token 几个线程
#define VTPD (NT / HD)               // 每 dim 几个线程
#define VB   (HP * 2 / VTPD)         // 每线程 V 字节
#define S16_BYTES ((HD * VSTR * 4) > (PF_NW * 16 * 136 * 2) ? (HD * VSTR * 4) : (PF_NW * 16 * 136 * 2))
static_assert(KB == VB, "K/V 预取共用寄存器");

extern "C" __global__ __launch_bounds__(NT)
void va_attn(u16* __restrict__ out, float* __restrict__ po, float* __restrict__ plse, const u16* __restrict__ q,
             const u16* __restrict__ kc, const u16* __restrict__ vc,
             const int* __restrict__ bt, const int* __restrict__ cu_q, const int* __restrict__ slen,
             int n_per, int nhq, int nhk, float scale, int nsplit, int prow) {
    const int sp = blockIdx.x, qb = 0, b = blockIdx.y, g = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 63, w = tid >> 6;
    const int L = lane & 15, H = lane >> 4;
    const int q0 = cu_q[b], qlen = cu_q[b + 1] - q0;
    const int seqlen = slen[b];
    const int prefix = seqlen - qlen;
    int tok_local[PF_MT], hq[PF_MT], qpos[PF_MT];
    bool qvalid[PF_MT];
#pragma unroll
    for (int mt = 0; mt < PF_MT; ++mt) {
        const int row = qb * RB + (w * PF_MT + mt) * 16 + L;
        tok_local[mt] = row / GQA; hq[mt] = row % GQA;
        qvalid[mt] = tok_local[mt] < qlen;
        qpos[mt] = prefix + (qvalid[mt] ? tok_local[mt] : 0);
    }
    const int qmax = prefix + min((qb * RB + RB - 1) / GQA, qlen - 1);
    const int nhalf = qmax / HP + 1;
    const int qmin = prefix + (qb * RB) / GQA;
    const int hp_full = (qmin >= HP - 1) ? ((qmin - (HP - 1)) / HP + 1) : 0;
    const int per = (nhalf + nsplit - 1) / nsplit;
    const int hs = sp * per, he = min(nhalf, hs + per);
    // 本 wave 有无有效行（verify q=4 → 24 行，NW4 的后两个 wave 全空：只帮装载、不算）
    const bool wact = (qb * RB + w * PF_MT * 16) < qlen * GQA;

    u32x2 qreg[PF_MT][16];
#pragma unroll
    for (int mt = 0; mt < PF_MT; ++mt) {
        const u16* qp = q + ((size_t)(q0 + (qvalid[mt] ? tok_local[mt] : 0)) * nhq + g * GQA + hq[mt]) * HD;
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            const u32x2 t = *(const u32x2*)&qp[kk * 16 + 4 * H];
            qreg[mt][kk] = qvalid[mt] ? t : (u32x2){0u, 0u};
        }
    }

    float m_run[PF_MT], l_run[PF_MT];
    f32x4 acc[PF_MT][16];
#pragma unroll
    for (int mt = 0; mt < PF_MT; ++mt) {
        m_run[mt] = -INFINITY; l_run[mt] = 0.f;
#pragma unroll
        for (int jj = 0; jj < 16; ++jj) acc[mt][jj] = (f32x4){0.f, 0.f, 0.f, 0.f};
    }

    __shared__ __align__(16) unsigned char s16[S16_BYTES];
    __shared__ u16 s_p[PF_NW][PF_MT][16][18];
    unsigned* s16d = (unsigned*)s16;

    const int ktk = tid / TPT, ksg = (tid % TPT) * KB;        // K：token、字节偏移
    const int vd = tid / VTPD, vpt = (tid % VTPD) * VB;       // V：dim、半页内字节偏移
    const size_t kvh = (size_t)nhk;
    u32x4 pre[NKV];
    {
        const int hs0 = hs < he ? hs : 0;
        const unsigned char* kp = (const unsigned char*)(kc + (((size_t)bt[b * n_per + (hs0 >> 1)] * kvh + g) * PAGE
                                                             + (hs0 & 1) * HP + ktk) * HD) + ksg;
#pragma unroll
        for (int c = 0; c < NKV; ++c) pre[c] = *(const u32x4*)(kp + c * 16);
    }

    for (int hp = hs; hp < he; ++hp) {
        const int pg = bt[b * n_per + (hp >> 1)];
        const int hoff = (hp & 1) * HP;
        const int ntok = min(HP, seqlen - hp * HP);
        const unsigned char* vp = (const unsigned char*)(vc + (((size_t)pg * kvh + g) * HD + vd) * PAGE + hoff) + vpt;

        // ① K → s16[token][dim]
        __syncthreads();
        {
            unsigned* dst = s16d + ktk * KSTR + ksg / 4;
#pragma unroll
            for (int c = 0; c < NKV; ++c) {
                *(u32x2*)&dst[c * 4] = (u32x2){pre[c][0], pre[c][1]};
                *(u32x2*)&dst[c * 4 + 2] = (u32x2){pre[c][2], pre[c][3]};
            }
        }
        __syncthreads();
        // ② 本半页 V 的全局读（QK 期间在途；必须在 __syncthreads 之后，否则被 vmcnt(0) 等完）
#pragma unroll
        for (int c = 0; c < NKV; ++c) pre[c] = *(const u32x4*)(vp + c * 16);

        // ③ S = Q·Kᵀ（K 片段一次读、MT 个 Q 片共用）
        f32x4 s[PF_MT][2];
        if (wact) {
#pragma unroll
        for (int mt = 0; mt < PF_MT; ++mt)
#pragma unroll
            for (int tb = 0; tb < 2; ++tb) s[mt][tb] = (f32x4){0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int r4 = 0; r4 < 4; ++r4)
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
#pragma unroll
                for (int tb = 0; tb < 2; ++tb) {
                    const u32x2 bfrag = *(const u32x2*)&s16d[(16 * tb + L) * KSTR + 32 * r4 + 8 * kk + 2 * H];
#pragma unroll
                    for (int mt = 0; mt < PF_MT; ++mt)
                        s[mt][tb] = __builtin_amdgcn_mmac_f32_16x16x16bf16(as_bf(qreg[mt][r4 * 4 + kk]), as_bf(bfrag), s[mt][tb]);
                }

        // ④ 在线 softmax（延迟缩放阈值 8，同 pp_v5）
#pragma unroll
        for (int mt = 0; mt < PF_MT; ++mt) {
            float mb = -INFINITY;
            if (hp < hp_full) {
#pragma unroll
                for (int tb = 0; tb < 2; ++tb)
#pragma unroll
                    for (int j = 0; j < 4; ++j) { s[mt][tb][j] *= scale; mb = fmaxf(mb, s[mt][tb][j]); }
            } else {
#pragma unroll
                for (int tb = 0; tb < 2; ++tb)
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int tok = 16 * tb + H + 4 * j;
                        const int gtok = hp * HP + tok;
                        float v = (tok < ntok && gtok <= qpos[mt]) ? s[mt][tb][j] * scale : -INFINITY;
                        s[mt][tb][j] = v; mb = fmaxf(mb, v);
                    }
            }
#pragma unroll
            for (int d = 16; d <= 32; d <<= 1)
                mb = fmaxf(mb, __uint_as_float(shfl_lane(__float_as_uint(mb), lane ^ d)));
            const bool need = mb > m_run[mt] + 8.0f;
            const float m_new = need ? mb : m_run[mt];
            if (__ballot(need)) {
                const float corr = (m_run[mt] == -INFINITY) ? 0.f : fexp(m_run[mt] - m_new);
                l_run[mt] *= corr;
#pragma unroll
                for (int jj = 0; jj < 16; ++jj)
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[mt][jj][j] *= corr;
            }
            m_run[mt] = m_new;
            float lb = 0.f;
#pragma unroll
            for (int tb = 0; tb < 2; ++tb)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    s[mt][tb][j] = (s[mt][tb][j] == -INFINITY) ? 0.f : fexp(s[mt][tb][j] - m_new);
                    lb += s[mt][tb][j];
                }
#pragma unroll
            for (int d = 16; d <= 32; d <<= 1)
                lb += __uint_as_float(shfl_lane(__float_as_uint(lb), lane ^ d));
            l_run[mt] += lb;
        }
        }

        // ⑤ V → s16[dim][token]
        __syncthreads();
        {
            unsigned* dst = s16d + vd * VSTR + vpt / 4;
#pragma unroll
            for (int c = 0; c < NKV; ++c) {
                *(u32x2*)&dst[c * 4] = (u32x2){pre[c][0], pre[c][1]};
                *(u32x2*)&dst[c * 4 + 2] = (u32x2){pre[c][2], pre[c][3]};
            }
        }
        __syncthreads();
        // ⑥ 下一半页 K 的全局读（PV 期间在途）
        if (hp + 1 < he) {
            const int hn = hp + 1;
            const unsigned char* kp = (const unsigned char*)(kc + (((size_t)bt[b * n_per + (hn >> 1)] * kvh + g) * PAGE
                                                                 + (hn & 1) * HP + ktk) * HD) + ksg;
#pragma unroll
            for (int c = 0; c < NKV; ++c) pre[c] = *(const u32x4*)(kp + c * 16);
        }

        // ⑦ O += P·V（V 片段一次读、MT 个 P 片共用）
        if (wact)
#pragma unroll
        for (int tb = 0; tb < 2; ++tb) {
            u32x2 pa[PF_MT];
#pragma unroll
            for (int mt = 0; mt < PF_MT; ++mt)
#pragma unroll
                for (int j = 0; j < 4; ++j) s_p[w][mt][L][H + 4 * j] = f2bf(s[mt][tb][j]);
            __builtin_amdgcn_s_waitcnt(0xC07F);                 // 只等 LDS（lgkmcnt 0），不等在途的 K 预取
#pragma unroll
            for (int mt = 0; mt < PF_MT; ++mt) {
                pa[mt][0] = (unsigned)s_p[w][mt][L][4 * H] | ((unsigned)s_p[w][mt][L][4 * H + 1] << 16);
                pa[mt][1] = (unsigned)s_p[w][mt][L][4 * H + 2] | ((unsigned)s_p[w][mt][L][4 * H + 3] << 16);
            }
            __builtin_amdgcn_s_waitcnt(0xC07F);
#pragma unroll
            for (int jj = 0; jj < 16; ++jj) {
                const u32x2 bfrag = *(const u32x2*)&s16d[(16 * jj + L) * VSTR + 8 * tb + 2 * H];
#pragma unroll
                for (int mt = 0; mt < PF_MT; ++mt)
                    acc[mt][jj] = __builtin_amdgcn_mmac_f32_16x16x16bf16(as_bf(pa[mt]), as_bf(bfrag), acc[mt][jj]);
            }
        }
    }

    if (nsplit > 1) {
        if (!wact) return;
#pragma unroll
        for (int mt = 0; mt < PF_MT; ++mt) {
            const int row = qb * RB + (w * PF_MT + mt) * 16 + L;
            if (!qvalid[mt]) continue;
            const float inv = (l_run[mt] > 0.f) ? (1.f / l_run[mt]) : 0.f;
            const size_t rbase = (((size_t)sp * gridDim.y + b) * nhk + g) * prow + row;
            float* dst = po + rbase * HD;
#pragma unroll
            for (int jj = 0; jj < 16; ++jj)
#pragma unroll
                for (int j = 0; j < 4; ++j) dst[16 * jj + H + 4 * j] = acc[mt][jj][j] * inv;
            if (H == 0) plse[rbase] = (l_run[mt] > 0.f) ? (m_run[mt] + __logf(l_run[mt])) : -INFINITY;
        }
        return;
    }
    // 归一化 + 经 LDS 转置成行连续后合并写出（每 wave 16 行 × 128 dim，分 2 批）
#pragma unroll
    for (int mt = 0; mt < PF_MT; ++mt) {
    const float inv = (l_run[mt] > 0.f) ? (1.f / l_run[mt]) : 0.f;
    u16* op = out + ((size_t)(q0 + (qvalid[mt] ? tok_local[mt] : 0)) * nhq + g * GQA + hq[mt]) * HD;
    for (int batch = 0; batch < 2; ++batch) {
        __syncthreads();
        u16* so = (u16*)s16 + w * (16 * 136);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int jj = batch * 8 + t;
#pragma unroll
            for (int j = 0; j < 4; ++j) so[L * 136 + H + 4 * j + 16 * t] = f2bf(acc[mt][jj][j] * inv);
        }
        __syncthreads();
        const unsigned op_lo = (unsigned)((size_t)op & 0xffffffffu), op_hi = (unsigned)((size_t)op >> 32);
        const unsigned qv = qvalid[mt] ? 1u : 0u;
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            const unsigned lo = shfl_lane(op_lo, r), hi = shfl_lane(op_hi, r), v = shfl_lane(qv, r);
            if (v) {
                u16* dst = (u16*)(((size_t)hi << 32) | lo) + batch * 128 + lane * 2;
                *(unsigned*)dst = *(const unsigned*)&so[r * 136 + lane * 2];
            }
        }
    }
    }
}


// 合并：每 (行, 请求, KV 头) 一块 256 线程（线程 = dim），按 lse 加权各段。
// （v1 每块串行走 24 行、只有 bs·nhk 块，S≥8 时合并本身成瓶颈）
extern "C" __global__ __launch_bounds__(256)
void va_combine(u16* __restrict__ out, const float* __restrict__ po, const float* __restrict__ plse,
                const int* __restrict__ cu_q, int nhq, int nhk, int nsplit, int prow, int bs) {
    const int row = blockIdx.x, b = blockIdx.y, g = blockIdx.z, d = threadIdx.x;
    const int q0 = cu_q[b], qlen = cu_q[b + 1] - q0;
    if (row >= qlen * GQA) return;
    float mx = -INFINITY;
    for (int s = 0; s < nsplit; ++s) mx = fmaxf(mx, plse[(((size_t)s * bs + b) * nhk + g) * prow + row]);
    float num = 0.f, den = 0.f;
    if (mx > -INFINITY)
        for (int s = 0; s < nsplit; ++s) {
            const size_t rb = (((size_t)s * bs + b) * nhk + g) * prow + row;
            const float ls = plse[rb];
            if (ls == -INFINITY) continue;
            const float wgt = fexp(ls - mx);
            num += wgt * po[rb * HD + d]; den += wgt;
        }
    const int tok = row / GQA, h = row % GQA;
    out[((size_t)(q0 + tok) * nhq + g * GQA + h) * HD + d] = f2bf(den > 0.f ? num / den : 0.f);
}

// 验证步入口：q 每请求 ≤ RB/GQA 个 token；nsplit 段（1 = 不切、直接写 bf16）；po ≥ nsplit·bs·nhk·prow·256 float，plse ≥ nsplit·bs·nhk·prow。
extern "C" int va_attn_run(void* out, void* po, void* plse, const void* q, const void* kc, const void* vc,
                           const void* bt, const void* cu_q, const void* slen,
                           int bs, int n_per, int max_qlen, int nhq, int nhk, float scale, int nsplit, void* stream) {
    if (nhq != nhk * GQA || max_qlen * GQA > RB || nsplit < 1) return -1;
    const int prow = max_qlen * GQA;
    hipLaunchKernelGGL(va_attn, dim3(nsplit, bs, nhk), dim3(NT), 0, (hipStream_t)stream,
                       (u16*)out, (float*)po, (float*)plse, (const u16*)q, (const u16*)kc, (const u16*)vc,
                       (const int*)bt, (const int*)cu_q, (const int*)slen, n_per, nhq, nhk, scale, nsplit, prow);
    if (nsplit > 1)
        hipLaunchKernelGGL(va_combine, dim3(prow, bs, nhk), dim3(256), 0, (hipStream_t)stream,
                           (u16*)out, (const float*)po, (const float*)plse, (const int*)cu_q, nhq, nhk, nsplit, prow, bs);
    return (int)hipGetLastError();
}

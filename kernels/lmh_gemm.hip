// LM-head 瘦长 GEMM（底层专项 L1）：y[M,N] = x[M,K] · W[N,K]ᵀ，bf16 进 / fp32 累加 / bf16 出。
// 形状：N=124160（每卡词表）、K=5120、M=1–192（decode 草稿 M=bs，验证 M=bs×4）。
// 为什么自写（10-09 tools/lmh2.sh）：rocBLAS 逐 M 最快解在 M≥33 时 3.7–4.5 ms，是 W 只读一遍的带宽下界
//   1.27 GB / 896 GB/s = 1.42 ms 的 2.6–3.1 倍；M≤16 已到 1.5–1.8 ms。
// 结构：
//   · 一个 block 4 个 wave，每 wave 负责 NT 个 16 行的 W 片（共 64·NT 行），K 全长；W 从显存直读进寄存器，每行只读一次。
//   · x 每次取 KC=64 列进 LDS（双缓冲），4 个 wave 共享；M 方向 MT 个 16 行片全部在寄存器累加。
//   · 片段 k 序置换：mmac 16x16x16 的 lane(L,H) 持 k=4H..4H+3；A、B 用同一置换不改结果，
//     这里让 lane 在一个 KC=64 块里持 k∈[8H,8H+8)∪[32+8H,32+8H+8)：一条 16 B 读指令里 4 个 lane 读满一行连续 64 B，
//     4 步 mmac 依次用其中每 4 个。
//   · W 当矩阵 A（行=n），x 当矩阵 B（列=m）：累加器 lane(L,H) 的 4 个值是 C[n=L][m=H+4t]，
//     16 个 lane 写同一 m 行的 16 个连续 n，输出合并写。
//   · x 在 LDS 里按片段顺序存放 [H][m][16]，一个 lane 读自己的 32 B，16 个 lane 连续，无 bank 冲突。
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include <stdlib.h>

using bf16x4  = __attribute__((ext_vector_type(4))) __bf16;
using floatx4 = __attribute__((ext_vector_type(4))) float;
using u32x4   = __attribute__((ext_vector_type(4))) unsigned;
using u32x2   = __attribute__((ext_vector_type(2))) unsigned;

#define NW 4
#ifndef STAG
#define STAG 1
#endif

static __device__ __forceinline__ bf16x4 as_bf16x4(u32x2 v) { return __builtin_bit_cast(bf16x4, v); }

// KC：每轮每行读的 k 数。一行在一轮里由 4 个 lane 各读 KC/32 条 16 B 指令；第 i 条指令 lane H 读元素 [32i+8H, 32i+8H+8)，
//     同一条指令 4 个 lane 读满一行连续 64 B，KC/32 条指令连起来是一行连续 2·KC 字节。
template <int MT, int NT, int KC>
__global__ __launch_bounds__(256) void lmh_kernel(const __bf16* __restrict__ X, const __bf16* __restrict__ W,
                                                  __bf16* __restrict__ Y, int M, int N, int K) {
    constexpr int IW = KC / 32;                  // 每行每轮的 16 B 指令数
    __shared__ u32x4 xs[2][IW][4][MT * 16];      // [buf][指令 i][H][m]：与 W 的 lane 持有方式一一对应
    const int tid = threadIdx.x, lane = tid & 63, w = tid >> 6;
    const int L = lane & 15, H = lane >> 4;
    // 常驻网格：块 g 负责连续行段 [g·N/G, (g+1)·N/G)，按 64·NT 行一片顺序处理。G=gridDim.x 直接决定全卡同时在读的行流数
    const int G = gridDim.x;
    const int r0 = (int)((long long)blockIdx.x * N / G), r1 = (int)((long long)(blockIdx.x + 1) * N / G);
    for (int tile0 = r0; tile0 < r1; tile0 += NW * NT * 16) {
    const int n_base = tile0 + w * NT * 16;

    const __bf16* wp[NT];
    bool wv[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        int n = n_base + j * 16 + L;
        wv[j] = n < r1;
        wp[j] = W + (size_t)(wv[j] ? n : r0) * K + 8 * H;
    }
    // x 装载：MT*16 行 × KC 列 = MT*2*KC 个 16 B 向量，256 线程分摊
    constexpr int XV = MT * 16 * (KC / 8);
    constexpr int XR = (XV + 255) / 256;
    auto xload = [&](int kc, u32x4 (&r)[XR]) {
#pragma unroll
        for (int q = 0; q < XR; ++q) {
            int v = tid + q * 256;
            int m = v / (KC / 8), k8 = v % (KC / 8);
            r[q] = (v < XV && m < M) ? *(const u32x4*)(X + (size_t)m * K + kc + k8 * 8) : u32x4{0, 0, 0, 0};
        }
    };
    auto xstore = [&](int buf, u32x4 (&r)[XR]) {
#pragma unroll
        for (int q = 0; q < XR; ++q) {
            int v = tid + q * 256;
            if (v < XV) { int m = v / (KC / 8), k8 = v % (KC / 8); xs[buf][k8 >> 2][k8 & 3][m] = r[q]; }
        }
    };
    auto wload = [&](int kc, u32x4 (&r)[NT][IW]) {
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int i = 0; i < IW; ++i) r[j][i] = *(const u32x4*)(wp[j] + kc + 32 * i);
    };

    floatx4 acc[NT][MT];
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int i = 0; i < MT; ++i) acc[j][i] = floatx4{0, 0, 0, 0};

    auto compute = [&](int buf, u32x4 (&wr)[NT][IW]) {
#pragma unroll
        for (int i = 0; i < IW; ++i)
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                u32x4 b = xs[buf][i][H][mt * 16 + L];
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    acc[j][mt] = __builtin_amdgcn_mmac_f32_16x16x16bf16(as_bf16x4(u32x2{wr[j][i].x, wr[j][i].y}), as_bf16x4(u32x2{b.x, b.y}), acc[j][mt]);
                    acc[j][mt] = __builtin_amdgcn_mmac_f32_16x16x16bf16(as_bf16x4(u32x2{wr[j][i].z, wr[j][i].w}), as_bf16x4(u32x2{b.z, b.w}), acc[j][mt]);
                }
            }
    };

    const int nk = K / KC;   // K 须为 2*KC 的倍数
    // 错开起点（StaggerU）：10-09 实测对带宽无影响，保留为旋钮
    const int k0 = (int)((blockIdx.x * (unsigned)STAG) % (unsigned)nk);
    auto kof = [&](int k) { int c = k + k0; return (c >= nk ? c - nk : c) * KC; };
    u32x4 xr[XR], wa[NT][IW], wb[NT][IW];
    xload(kof(0), xr); wload(kof(0), wa);
    xstore(0, xr);
    for (int k = 0; k < nk; k += 2) {
        __syncthreads();
        if (k + 1 < nk) { xload(kof(k + 1), xr); wload(kof(k + 1), wb); }
        compute(0, wa);
        if (k + 1 < nk) xstore(1, xr);
        __syncthreads();
        if (k + 2 < nk) { xload(kof(k + 2), xr); wload(kof(k + 2), wa); }
        if (k + 1 < nk) compute(1, wb);
        if (k + 2 < nk) xstore(0, xr);
    }

    // 输出：acc[j][i] 的第 t 个值是 C[n = n_base+16j+L][m = 16i+H+4t]
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        int n = n_base + j * 16 + L;
        if (wv[j]) {
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    int m = i * 16 + H + 4 * t;
                    if (m < M) {
                        unsigned u = __float_as_uint(acc[j][i][t]);
                        u = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;   // 就近偶数舍入，与 torch 一致
                        ((unsigned short*)Y)[(size_t)m * N + n] = (unsigned short)u;
                    }
                }
        }
    }
    __syncthreads();   // 下一片要覆盖 LDS
    }
}

static int g_pad = 0;   // 额外动态 LDS 字节：只为压低每 CU 驻留块数（并发行流数）
static int g_grid = 0;  // 常驻网格块数；0 = 每 64·NT 行一块（非常驻）

template <int MT, int NT, int KC>
static int launch(const void* x, const void* w, void* y, int M, int N, int K, hipStream_t s) {
    if (K % (2 * KC) != 0) return -1;
    int rows = NW * NT * 16;
    hipLaunchKernelGGL((lmh_kernel<MT, NT, KC>), dim3(g_grid > 0 ? g_grid : (N + rows - 1) / rows), dim3(256), g_pad, s,
                       (const __bf16*)x, (const __bf16*)w, (__bf16*)y, M, N, K);
    return (int)hipGetLastError();
}

template <int MT, int NT>
static int launch_kc(const void* x, const void* w, void* y, int M, int N, int K, int kc, hipStream_t s) {
    // LDS = 2 缓冲 × KC × MT × 32 B，超 64 KB 的组合不实例化
    switch (kc) {
        case 64: if constexpr (64 * MT * 64 <= 65536) return launch<MT, NT, 64>(x, w, y, M, N, K, s); break;
        case 128: if constexpr (128 * MT * 64 <= 65536) return launch<MT, NT, 128>(x, w, y, M, N, K, s); break;
        case 256: if constexpr (256 * MT * 64 <= 65536) return launch<MT, NT, 256>(x, w, y, M, N, K, s); break;
        case 512: if constexpr (512 * MT * 64 <= 65536) return launch<MT, NT, 512>(x, w, y, M, N, K, s); break;
    }
    return -3;
}

#define CASE_NT(MT_)                                                       \
    case MT_:                                                              \
        switch (nt) {                                                      \
            case 1: return launch_kc<MT_, 1>(x, w, y, M, N, K, kc, s);     \
            case 2: return launch_kc<MT_, 2>(x, w, y, M, N, K, kc, s);     \
            default: return -2;                                            \
        }

// 返回 0 成功；-1 形状不支持，-2 NT 不支持，-3 (MT,KC) 组合超 LDS
extern "C" int lmh_gemm_own(const void* x, const void* w, void* y, int M, int N, int K, int nt, int kc, void* stream) {
    if (M < 1 || M > 192) return -1;
    { static int init = 0; if (!init) { init = 1; const char* e = getenv("LMH_LDS_PAD"); g_pad = e ? atoi(e) : 0;
                                 e = getenv("LMH_GRID"); g_grid = e ? atoi(e) : 0; } }
    hipStream_t s = (hipStream_t)stream;
    int mt = (M + 15) / 16;
    switch (mt) {
        CASE_NT(1) CASE_NT(2) CASE_NT(3) CASE_NT(4) CASE_NT(5) CASE_NT(6)
        CASE_NT(7) CASE_NT(8) CASE_NT(9) CASE_NT(10) CASE_NT(11) CASE_NT(12)
        default: return -1;
    }
}

// ---------------- v2（10-10）：W 经 LDS 中转 ----------------
// 定因（probe/rowread_bw.hip，纯读 1.27 GB）：带宽由「一条读指令每行连续多少字节」决定——每行 64 B（v1 的片段直读）
//   不论在途深度与网格都锁在 300–360 GB/s；每行 256 B 起 800+，每行 1 KB 且每 CU 1 块 879 GB/s（峰值 98%）。
//   片段布局要求 16 个 lane 各持不同行，直读做不到宽行，所以 W 先按宽行合并读进 LDS，再按片段从 LDS 取。
// 每块 64 行（4 wave × 16），每轮 KC 列；W 一行 KC 个元素由 KC/8 个 lane 并排读（KC=256 时每行每指令 512 B）。
// 单缓冲 + 寄存器预取：下一轮的 W、x 全局读在本轮计算期间在途。
// 块内同步只等 LDS、不等在途全局读：__syncthreads() 展开为 s_waitcnt vmcnt(0) lgkmcnt(0); s_barrier，
// gfx928 无回退屏障，编译器对任何 s_barrier（含 __builtin_amdgcn_s_barrier）都补 vmcnt(0)，预取被等光（10-10 汇编实证，
// 双寄存器预取前后 M=1 2.48→2.52 ms 无变化）。写成内联汇编，编译器只按寄存器依赖在 ds_write 前插 vmcnt(k)。
#define LDS_BAR() asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")

template <int MT, int KC>
__global__ __launch_bounds__(256) void lmh2_kernel(const __bf16* __restrict__ X, const __bf16* __restrict__ W,
                                                   __bf16* __restrict__ Y, int M, int N, int K) {
    constexpr int IW = KC / 32;
    constexpr int RB = 64;
    constexpr int S = KC * 2 + 16;               // LDS 行跨步（字节）：dword 跨步 ≡ 4 (mod 32)，8 个 lane 一拍无冲突
    constexpr int WV = RB * KC / 8, WR = WV / 256;   // W 每轮 16 B 向量数 / 每线程
    constexpr int XV = MT * 16 * (KC / 8), XR = (XV + 255) / 256;   // 余数线程截断到最后一个向量重复读写（同值，无害）
    static_assert(WV % 256 == 0, "W 每线程整数个向量");
    extern __shared__ __attribute__((aligned(16))) unsigned char smem[];
    unsigned char* ws = smem;                                   // [RB][S]
    u32x4 (*xs)[4][MT * 16] = (u32x4 (*)[4][MT * 16])(smem + RB * S);   // [IW][H][m]
    const int tid = threadIdx.x, lane = tid & 63, w = tid >> 6;
    const int L = lane & 15, H = lane >> 4;
    const int n0 = blockIdx.x * RB;

    // 两组寄存器轮换：第 k 轮写 LDS 时，第 k+1 轮的读已在途，随即发第 k+2 轮——每 CU 同时两块在途。
    // 单组时每 CU 任一时刻只有一块（32 KB）在途，按 Little 定律只能到约 520 GB/s（10-10 v2 实测 450–630）。
    u32x4 wr[2][WR], xr[2][XR];
    auto gload = [&](int kc, u32x4 (&wq)[WR], u32x4 (&xq)[XR]) {
#pragma unroll
        for (int q = 0; q < WR; ++q) {
            int v = tid + q * 256, r = v / (KC / 8), c = v % (KC / 8);
            int n = n0 + r; if (n >= N) n = N - 1;
            wq[q] = *(const u32x4*)(W + (size_t)n * K + kc + c * 8);
        }
#pragma unroll
        for (int q = 0; q < XR; ++q) {
            // 无分支：m≥M 的行截断到 M−1 照读（这些列的结果不写回）；带条件的读会让编译器跨分支数不清在途数，退化成 vmcnt(0)
            int v = min(tid + q * 256, XV - 1), m = v / (KC / 8), c = v % (KC / 8);
            m = m < M ? m : M - 1;
            xq[q] = *(const u32x4*)(X + (size_t)m * K + kc + c * 8);
        }
    };
    auto lstore = [&](u32x4 (&wq)[WR], u32x4 (&xq)[XR]) {
#pragma unroll
        for (int q = 0; q < WR; ++q) {
            int v = tid + q * 256, r = v / (KC / 8), c = v % (KC / 8);
            *(u32x4*)(ws + r * S + c * 16) = wq[q];
        }
#pragma unroll
        for (int q = 0; q < XR; ++q) {
            int v = min(tid + q * 256, XV - 1);
            int m = v / (KC / 8), c = v % (KC / 8); xs[c >> 2][c & 3][m] = xq[q];
        }
    };
    floatx4 acc[MT];
#pragma unroll
    for (int i = 0; i < MT; ++i) acc[i] = floatx4{0, 0, 0, 0};
    const unsigned char* wrow = ws + (w * 16 + L) * S + 16 * H;
    auto compute = [&]() {
#pragma unroll
        for (int i = 0; i < IW; ++i) {
            u32x4 a = *(const u32x4*)(wrow + 64 * i);
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                u32x4 b = xs[i][H][mt * 16 + L];
                acc[mt] = __builtin_amdgcn_mmac_f32_16x16x16bf16(as_bf16x4(u32x2{a.x, a.y}), as_bf16x4(u32x2{b.x, b.y}), acc[mt]);
                acc[mt] = __builtin_amdgcn_mmac_f32_16x16x16bf16(as_bf16x4(u32x2{a.z, a.w}), as_bf16x4(u32x2{b.z, b.w}), acc[mt]);
            }
        }
    };
    const int nk = K / KC;   // K/KC 须为偶数
    gload(0, wr[0], xr[0]);
    gload(KC, wr[1], xr[1]);
    for (int k = 0; k < nk; k += 2) {
        lstore(wr[0], xr[0]);
        LDS_BAR();
        if (k + 2 < nk) gload((k + 2) * KC, wr[0], xr[0]);
        compute();
        LDS_BAR();
        lstore(wr[1], xr[1]);
        LDS_BAR();
        if (k + 3 < nk) gload((k + 3) * KC, wr[1], xr[1]);
        compute();
        LDS_BAR();
    }
    const int n = n0 + w * 16 + L;
    if (n < N) {
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                int m = i * 16 + H + 4 * t;
                if (m < M) {
                    unsigned u = __float_as_uint(acc[i][t]);
                    u = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
                    ((unsigned short*)Y)[(size_t)m * N + n] = (unsigned short)u;
                }
            }
    }
}

template <int MT, int KC>
static int launch2(const void* x, const void* w, void* y, int M, int N, int K, hipStream_t s) {
    constexpr int lds = 64 * (KC * 2 + 16) + KC * MT * 32;
    if constexpr (lds > 65536) { return -3; } else {
        if (K % (2 * KC) != 0) return -1;
        int dyn = lds + g_pad; if (dyn > 65536) dyn = 65536;
        static bool attr = false;
        if (!attr) { attr = true; hipFuncSetAttribute((const void*)lmh2_kernel<MT, KC>, hipFuncAttributeMaxDynamicSharedMemorySize, 65536); }
        hipLaunchKernelGGL((lmh2_kernel<MT, KC>), dim3((N + 63) / 64), dim3(256), dyn, s,
                           (const __bf16*)x, (const __bf16*)w, (__bf16*)y, M, N, K);
        return (int)hipGetLastError();
    }
}

#define CASE2(MT_) case MT_: switch (kc) { case 64: return launch2<MT_, 64>(x, w, y, M, N, K, s); \
    case 128: return launch2<MT_, 128>(x, w, y, M, N, K, s); case 256: return launch2<MT_, 256>(x, w, y, M, N, K, s); } return -3;

// v2 入口：kc ∈ {64,128,256}；LMH_LDS_PAD 额外动态 LDS（压驻留）
extern "C" int lmh_gemm_own2(const void* x, const void* w, void* y, int M, int N, int K, int kc, void* stream) {
    if (M < 1 || M > 192) return -1;
    hipStream_t s = (hipStream_t)stream;
    switch ((M + 15) / 16) {
        CASE2(1) CASE2(2) CASE2(3) CASE2(4) CASE2(5) CASE2(6) CASE2(7) CASE2(8) CASE2(9) CASE2(10) CASE2(11) CASE2(12)
    }
    return -1;
}

// ---------------- v3（10-10）：每块 16 行、4 个 wave 在 K 方向分段 ----------------
// v2 每块 64 行 → 每 CU 64 条、全卡 7680 条行流同时在读，M=1 只到 435–440 GB/s；探针（rowread_bw）879 GB/s 那一档
// 是每 CU 8 行、每行每指令 512 B–1 KB。v3 每块只取 16 行（mmac 片段的最小行数），每轮 KC 列由全块合并读进 LDS
// （KC=512 时一条指令读一整行 1 KB），4 个 wave 各算本轮的 KC/4 列，结束时 4 份部分和经 LDS 相加。
template <int MT, int KC>
__global__ __launch_bounds__(256) void lmh3_kernel(const __bf16* __restrict__ X, const __bf16* __restrict__ W,
                                                   __bf16* __restrict__ Y, int M, int N, int K) {
    constexpr int RB = 16, IWW = KC / 128;               // 每 wave 每轮 KC/4 列 = IWW 条 32 列
    constexpr int S = KC * 2 + 16;
    constexpr int WV = RB * KC / 8, WR = WV / 256;
    constexpr int XV = MT * 16 * (KC / 8), XR = (XV + 255) / 256;
    static_assert(WV % 256 == 0, "W 每线程整数个向量");
    extern __shared__ __attribute__((aligned(16))) unsigned char smem[];
    unsigned char* ws = smem;
    u32x4 (*xs)[4][MT * 16] = (u32x4 (*)[4][MT * 16])(smem + RB * S);
    const int tid = threadIdx.x, lane = tid & 63, w = tid >> 6;
    const int L = lane & 15, H = lane >> 4;
    const int n0 = blockIdx.x * RB;

    u32x4 wr[2][WR], xr[2][XR];
    auto gload = [&](int kc, u32x4 (&wq)[WR], u32x4 (&xq)[XR]) {
#pragma unroll
        for (int q = 0; q < WR; ++q) {
            int v = tid + q * 256, r = v / (KC / 8), c = v % (KC / 8);
            int n = n0 + r; n = n < N ? n : N - 1;
            wq[q] = *(const u32x4*)(W + (size_t)n * K + kc + c * 8);
        }
#pragma unroll
        for (int q = 0; q < XR; ++q) {
            int v = min(tid + q * 256, XV - 1), m = v / (KC / 8), c = v % (KC / 8);
            m = m < M ? m : M - 1;
            xq[q] = *(const u32x4*)(X + (size_t)m * K + kc + c * 8);
        }
    };
    auto lstore = [&](u32x4 (&wq)[WR], u32x4 (&xq)[XR]) {
#pragma unroll
        for (int q = 0; q < WR; ++q) {
            int v = tid + q * 256, r = v / (KC / 8), c = v % (KC / 8);
            *(u32x4*)(ws + r * S + c * 16) = wq[q];
        }
#pragma unroll
        for (int q = 0; q < XR; ++q) {
            int v = min(tid + q * 256, XV - 1), m = v / (KC / 8), c = v % (KC / 8);
            xs[c >> 2][c & 3][m] = xq[q];
        }
    };
    floatx4 acc[MT];
#pragma unroll
    for (int i = 0; i < MT; ++i) acc[i] = floatx4{0, 0, 0, 0};
    const unsigned char* wrow = ws + L * S + 16 * H;
    auto compute = [&]() {
#pragma unroll
        for (int ii = 0; ii < IWW; ++ii) {
            const int i = w * IWW + ii;
            u32x4 a = *(const u32x4*)(wrow + 64 * i);
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                u32x4 b = xs[i][H][mt * 16 + L];
                acc[mt] = __builtin_amdgcn_mmac_f32_16x16x16bf16(as_bf16x4(u32x2{a.x, a.y}), as_bf16x4(u32x2{b.x, b.y}), acc[mt]);
                acc[mt] = __builtin_amdgcn_mmac_f32_16x16x16bf16(as_bf16x4(u32x2{a.z, a.w}), as_bf16x4(u32x2{b.z, b.w}), acc[mt]);
            }
        }
    };
    const int nk = K / KC;   // 须为偶数
    gload(0, wr[0], xr[0]);
    gload(KC, wr[1], xr[1]);
    for (int k = 0; k < nk; k += 2) {
        lstore(wr[0], xr[0]);
        LDS_BAR();
        if (k + 2 < nk) gload((k + 2) * KC, wr[0], xr[0]);
        compute();
        LDS_BAR();
        lstore(wr[1], xr[1]);
        LDS_BAR();
        if (k + 3 < nk) gload((k + 3) * KC, wr[1], xr[1]);
        compute();
        LDS_BAR();
    }
    // 4 个 wave 的部分和相加：wave 1..3 写 LDS，wave 0 读回累加（固定顺序 0+1+2+3，结果确定）
    floatx4* red = (floatx4*)smem;          // [3][MT][64]
    if (w > 0) {
#pragma unroll
        for (int i = 0; i < MT; ++i) red[((w - 1) * MT + i) * 64 + lane] = acc[i];
    }
    LDS_BAR();
    if (w == 0) {
        const int n = n0 + L;
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            floatx4 v = acc[i];
#pragma unroll
            for (int j = 0; j < 3; ++j) v += red[(j * MT + i) * 64 + lane];
            if (n < N) {
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    int m = i * 16 + H + 4 * t;
                    if (m < M) {
                        unsigned u = __float_as_uint(v[t]);
                        u = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
                        ((unsigned short*)Y)[(size_t)m * N + n] = (unsigned short)u;
                    }
                }
            }
        }
    }
}

template <int MT, int KC>
static int launch3(const void* x, const void* w, void* y, int M, int N, int K, hipStream_t s) {
    constexpr int lds0 = 16 * (KC * 2 + 16) + KC * MT * 32;
    constexpr int lds = lds0 > 3 * MT * 64 * 16 ? lds0 : 3 * MT * 64 * 16;
    if constexpr (lds > 65536) { return -3; } else {
        if (K % (2 * KC) != 0) return -1;
        int dyn = lds + g_pad; if (dyn > 65536) dyn = 65536;
        static bool attr = false;
        if (!attr) { attr = true; hipFuncSetAttribute((const void*)lmh3_kernel<MT, KC>, hipFuncAttributeMaxDynamicSharedMemorySize, 65536); }
        hipLaunchKernelGGL((lmh3_kernel<MT, KC>), dim3((N + 15) / 16), dim3(256), dyn, s,
                           (const __bf16*)x, (const __bf16*)w, (__bf16*)y, M, N, K);
        return (int)hipGetLastError();
    }
}

#define CASE3(MT_) case MT_: switch (kc) { case 128: return launch3<MT_, 128>(x, w, y, M, N, K, s); \
    case 256: return launch3<MT_, 256>(x, w, y, M, N, K, s); case 512: return launch3<MT_, 512>(x, w, y, M, N, K, s); } return -3;

// v3 入口：kc ∈ {128,256,512}
extern "C" int lmh_gemm_own3(const void* x, const void* w, void* y, int M, int N, int K, int kc, void* stream) {
    if (M < 1 || M > 192) return -1;
    hipStream_t s = (hipStream_t)stream;
    switch ((M + 15) / 16) {
        CASE3(1) CASE3(2) CASE3(3) CASE3(4) CASE3(5) CASE3(6) CASE3(7) CASE3(8) CASE3(9) CASE3(10) CASE3(11) CASE3(12)
    }
    return -1;
}

// v2/v3 的 LDS 填充：库加载时取 LMH_LDS_PAD，服务内按 M 选解时由 lmh_set_pad 逐次设置（单线程发射，launch 时读取）
static struct PadInit { PadInit() { const char* e = getenv("LMH_LDS_PAD"); if (e) g_pad = atoi(e); } } g_pad_init;
extern "C" void lmh_set_pad(int p) { g_pad = p; }

// ---------------- v4（10-10）：v3 + 常驻网格、跨片连续流水 ----------------
// v3 M=1 776 GB/s，探针 879。每块只读 160 KB（≈22 µs @ 每 CU 7.3 GB/s），开头要等两轮读延迟（约 4–6 µs）才有数据，
// 每 CU 只驻 1 块时块间不重叠，灌流水开销约 15–25%。v4 让 G 个常驻块各自按 g, g+G, … 取 16 行片，
// 把「片 × K 段」拉平成一条序列，预取始终领先两段、跨片不断；片尾 4 个 wave 的部分和经独立 LDS 区归约后写出。
static int g_grid4 = 0;
extern "C" void lmh_set_grid(int g) { g_grid4 = g; }

template <int MT, int KC>
__global__ __launch_bounds__(256) void lmh4_kernel(const __bf16* __restrict__ X, const __bf16* __restrict__ W,
                                                   __bf16* __restrict__ Y, int M, int N, int K) {
    constexpr int RB = 16, IWW = KC / 128;
    constexpr int S = KC * 2 + 16;
    constexpr int WV = RB * KC / 8, WR = WV / 256;
    constexpr int XV = MT * 16 * (KC / 8), XR = (XV + 255) / 256;
    static_assert(WV % 256 == 0, "W 每线程整数个向量");
    extern __shared__ __attribute__((aligned(16))) unsigned char smem[];
    unsigned char* ws = smem;
    u32x4 (*xs)[4][MT * 16] = (u32x4 (*)[4][MT * 16])(smem + RB * S);
    floatx4* red = (floatx4*)(smem + RB * S + KC * MT * 32);      // [3][MT][64]，独立区：归约不必等下一片写 W
    const int tid = threadIdx.x, lane = tid & 63, w = tid >> 6;
    const int L = lane & 15, H = lane >> 4;
    const int G = gridDim.x, g = blockIdx.x;
    const int ntiles = (N + RB - 1) / RB;
    const int mine = g < ntiles ? (ntiles - g + G - 1) / G : 0;
    const int nk = K / KC;
    const int T = mine * nk;

    u32x4 wr[2][WR], xr[2][XR];
    auto gload = [&](int f, u32x4 (&wq)[WR], u32x4 (&xq)[XR]) {
        const int n0 = (g + (f / nk) * G) * RB, kc = (f % nk) * KC;
#pragma unroll
        for (int q = 0; q < WR; ++q) {
            int v = tid + q * 256, r = v / (KC / 8), c = v % (KC / 8);
            int n = n0 + r; n = n < N ? n : N - 1;
            wq[q] = *(const u32x4*)(W + (size_t)n * K + kc + c * 8);
        }
#pragma unroll
        for (int q = 0; q < XR; ++q) {
            int v = min(tid + q * 256, XV - 1), m = v / (KC / 8), c = v % (KC / 8);
            m = m < M ? m : M - 1;
            xq[q] = *(const u32x4*)(X + (size_t)m * K + kc + c * 8);
        }
    };
    auto lstore = [&](u32x4 (&wq)[WR], u32x4 (&xq)[XR]) {
#pragma unroll
        for (int q = 0; q < WR; ++q) {
            int v = tid + q * 256, r = v / (KC / 8), c = v % (KC / 8);
            *(u32x4*)(ws + r * S + c * 16) = wq[q];
        }
#pragma unroll
        for (int q = 0; q < XR; ++q) {
            int v = min(tid + q * 256, XV - 1), m = v / (KC / 8), c = v % (KC / 8);
            xs[c >> 2][c & 3][m] = xq[q];
        }
    };
    floatx4 acc[MT];
#pragma unroll
    for (int i = 0; i < MT; ++i) acc[i] = floatx4{0, 0, 0, 0};
    const unsigned char* wrow = ws + L * S + 16 * H;
    auto compute = [&]() {
#pragma unroll
        for (int ii = 0; ii < IWW; ++ii) {
            const int i = w * IWW + ii;
            u32x4 a = *(const u32x4*)(wrow + 64 * i);
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                u32x4 b = xs[i][H][mt * 16 + L];
                acc[mt] = __builtin_amdgcn_mmac_f32_16x16x16bf16(as_bf16x4(u32x2{a.x, a.y}), as_bf16x4(u32x2{b.x, b.y}), acc[mt]);
                acc[mt] = __builtin_amdgcn_mmac_f32_16x16x16bf16(as_bf16x4(u32x2{a.z, a.w}), as_bf16x4(u32x2{b.z, b.w}), acc[mt]);
            }
        }
    };
    auto epilogue = [&](int f) {   // 片 f/nk 的 4 份部分和相加并写出（固定顺序 0+1+2+3）
        if (w > 0) {
#pragma unroll
            for (int i = 0; i < MT; ++i) red[((w - 1) * MT + i) * 64 + lane] = acc[i];
        }
        LDS_BAR();
        if (w == 0) {
            const int n = (g + (f / nk) * G) * RB + L;
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                floatx4 v = acc[i];
#pragma unroll
                for (int j = 0; j < 3; ++j) v += red[(j * MT + i) * 64 + lane];
                if (n < N) {
#pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        int m = i * 16 + H + 4 * t;
                        if (m < M) {
                            unsigned u = __float_as_uint(v[t]);
                            u = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
                            ((unsigned short*)Y)[(size_t)m * N + n] = (unsigned short)u;
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int i = 0; i < MT; ++i) acc[i] = floatx4{0, 0, 0, 0};
    };
    if (T == 0) return;
    gload(0, wr[0], xr[0]);
    if (T > 1) gload(1, wr[1], xr[1]);
    for (int f = 0; f < T; f += 2) {
        lstore(wr[0], xr[0]);
        LDS_BAR();
        if (f + 2 < T) gload(f + 2, wr[0], xr[0]);
        compute();
        if ((f + 1) % nk == 0) epilogue(f);
        LDS_BAR();
        if (f + 1 >= T) break;
        lstore(wr[1], xr[1]);
        LDS_BAR();
        if (f + 3 < T) gload(f + 3, wr[1], xr[1]);
        compute();
        if ((f + 2) % nk == 0) epilogue(f + 1);
        LDS_BAR();
    }
}

template <int MT, int KC>
static int launch4(const void* x, const void* w, void* y, int M, int N, int K, hipStream_t s) {
    constexpr int lds = 16 * (KC * 2 + 16) + KC * MT * 32 + 3 * MT * 64 * 16;
    if constexpr (lds > 65536) { return -3; } else {
        if (K % KC != 0) return -1;
        int dyn = lds + g_pad; if (dyn > 65536) dyn = 65536;
        static bool attr = false;
        if (!attr) { attr = true; hipFuncSetAttribute((const void*)lmh4_kernel<MT, KC>, hipFuncAttributeMaxDynamicSharedMemorySize, 65536); }
        int ntiles = (N + 15) / 16, G = g_grid4 > 0 ? g_grid4 : 120;
        hipLaunchKernelGGL((lmh4_kernel<MT, KC>), dim3(G < ntiles ? G : ntiles), dim3(256), dyn, s,
                           (const __bf16*)x, (const __bf16*)w, (__bf16*)y, M, N, K);
        return (int)hipGetLastError();
    }
}

#define CASE4(MT_) case MT_: switch (kc) { case 128: return launch4<MT_, 128>(x, w, y, M, N, K, s); \
    case 256: return launch4<MT_, 256>(x, w, y, M, N, K, s); case 512: return launch4<MT_, 512>(x, w, y, M, N, K, s); } return -3;

// v4 入口：kc ∈ {128,256,512}；常驻块数由 lmh_set_grid 设（默认 120）
extern "C" int lmh_gemm_own4(const void* x, const void* w, void* y, int M, int N, int K, int kc, void* stream) {
    if (M < 1 || M > 64) return -1;
    hipStream_t s = (hipStream_t)stream;
    switch ((M + 15) / 16) { CASE4(1) CASE4(2) CASE4(3) CASE4(4) }
    return -1;
}

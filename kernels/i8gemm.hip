// Qwen3.8 W8A8 预填充 INT8 GEMM（自写，HIP + 海光矩阵核 v_mmac_i32_16x16x32_i8）
//   C[M,N](bf16) = ((float)(A[M,K]·W[N,K]ᵀ) * sa[M]) * sb[N]
//   A：per-token 量化后的 int8 激活，行主序；W：int8 权重，按 [N,K] 行主序存（sglang 里是 [K,N] 的转置视图，stride(0)==1）
//   尾处理的乘法顺序与 lightop matmul_int8 的 Triton 内核一致（int32→f32，先乘 sa 再乘 sb，RNE 转 bf16），可逐位对拍。
// 结构移植自 docs/代码/32_GEMM复刻/mygemm5.cpp（BF16 Tensile 复刻）：
//   双缓冲 LDS、全局读先落寄存器（PGR1）、workgroup 重映射（WGM）、矩阵指令段 s_setprio。
//   BF16 一条 mmac 吃 K=16，INT8 吃 K=32；同样每行 64 字节的 K 块（BK=64），每轮每 wave 的 mmac 数不变。
#include <hip/hip_runtime.h>
#include <stdint.h>

using i32x4 = __attribute__((ext_vector_type(4))) int;
typedef long i8x8;   // mmac_i32_16x16x32i8 的 A/B 操作数：8 个 int8 打包成 64 位
using u32x4 = __attribute__((ext_vector_type(4))) unsigned;

#ifndef WM
#define WM 4            // M 方向 wave 数
#endif
#ifndef FM
#define FM 4            // 每 wave 在 M 方向的片段数（16 行一个）
#endif
#ifndef WN
#define WN 2            // N 方向 wave 数
#endif
#ifndef FN
#define FN 4            // 每 wave 在 N 方向的片段数（16 列一个）
#endif
#ifndef WGM
#define WGM 8
#endif
#ifndef PAD
#define PAD 16          // LDS 行补白（字节）
#endif
#ifndef PF2
#define PF2 0           // DBUF=0 时：两级寄存器预取
#endif
#ifndef EPI_LDS
#define EPI_LDS 1       // 尾处理经 LDS 转置后按行连续写
#endif
#ifndef SETPRIO
#define SETPRIO 1
#endif
#define BM (WM * 16 * FM)
#define BN (WN * 16 * FN)
#ifndef BK
#define BK 64           // 字节 = int8 个数
#endif
#ifndef DBUF
#define DBUF 1          // 1：双缓冲 LDS，每 BK 一个 barrier；0：单缓冲 + 寄存器预取，每 BK 两个 barrier（省一半 LDS，可上 BK=128 / 256×256）
#endif
#define NTHR (WM * WN * 64)
#define LD (BK + PAD)
#define TPR (BK / 16)   // 每行几个线程（每线程 16B）
#define RPI (NTHR / TPR) // 每遍装载覆盖的行数；A、B 的行统一编号（0..BM-1 为 A，BM..BM+BN-1 为 B）
#define NP ((BM + BN) / RPI)
static_assert((BM + BN) % RPI == 0 && BM % RPI == 0 || (RPI % BM == 0 && (BM + BN) % RPI == 0), "tile 与线程数不匹配");

// __syncthreads() 编译成 s_waitcnt vmcnt(0) lgkmcnt(0) + s_barrier：每个 barrier 都把在路上的全局预取等光。
// 块内同步只需要 LDS 读写完成：等 lgkmcnt(0) 再 s_barrier；全局读的等待交给编译器按数据依赖插 vmcnt(N)。
#ifndef LDSBAR
#define LDSBAR 1
#endif
__device__ __forceinline__ void lds_barrier() {
#if LDSBAR
    // 内建 s_barrier 在 gfx9（无 AutoWaitcntBeforeBarrier）前会被插全量 s_waitcnt，必须用内联汇编发
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
#else
    __syncthreads();
#endif
}

__device__ __forceinline__ unsigned short f2bf(float f) {
    unsigned u = __float_as_uint(f);
    u = u + 0x7fffu + ((u >> 16) & 1u);   // RNE（输入不会是 NaN）
    return (unsigned short)(u >> 16);
}

// VGPR 预算：编译器按 256 池算（1024 线程只给 64 个，溢出），硬件实为 512 池。
// amdgpu_num_vgpr 抬不过线程数对应的上限（实测 1024 线程仍是 64、溢出 250）。只能把 launch_bounds 报小（LBOUND），
// 但编译器会据此假定 threadIdx.x < LBOUND，把依赖线程号的分支折叠掉（实测一半输出错）——所以线程号要经内联汇编洗一遍。
#ifndef LBOUND
#define LBOUND NTHR
#endif
extern "C" __global__ __launch_bounds__(LBOUND) void i8gemm_kernel(
    const int8_t* __restrict__ A, const int8_t* __restrict__ W,
    const float* __restrict__ sa, const float* __restrict__ sb,
    unsigned short* __restrict__ C, int M, int N, int K) {
    __shared__ int8_t S[1 + DBUF][(BM + BN) * LD];   // 前 BM 行是 A，后 BN 行是 B

    int tid;
    asm volatile("v_mov_b32 %0, %1" : "=v"(tid) : "v"((int)threadIdx.x));   // 阻断编译器对线程号的范围推断（见 LBOUND）
    const int lane = tid & 63, w = tid >> 6;
    const int L = lane & 15, H = lane >> 4;
    const int wr = w / WN, wc = w % WN;

    int nb = gridDim.x, mb = gridDim.y, m_tile, n_tile;
#if WGM > 1
    { int bid = blockIdx.y * nb + blockIdx.x, per = WGM * nb;
      int grp = bid / per, rem = bid % per, ms = grp * WGM, gsz = min(WGM, mb - ms);
      m_tile = ms + rem % gsz; n_tile = rem / gsz; }
#else
    m_tile = blockIdx.y; n_tile = blockIdx.x;
#endif
    const int m0 = m_tile * BM, n0 = n_tile * BN;

    const int ldRow = tid / TPR, ldCol = (tid % TPR) * 16;
    const int8_t* gp[NP];
    #pragma unroll
    for (int r = 0; r < NP; ++r) {
        int g = ldRow + r * RPI;
        if (g < BM) gp[r] = A + (size_t)min(m0 + g, M - 1) * K + ldCol;   // M 尾块：越界行读最后一行（结果不写回）
        else        gp[r] = W + (size_t)(n0 + g - BM) * K + ldCol;
    }

    i32x4 acc[FN][FM];
    #pragma unroll
    for (int j = 0; j < FN; ++j)
        #pragma unroll
        for (int i = 0; i < FM; ++i) acc[j][i] = (i32x4){0, 0, 0, 0};

    u32x4 rg[NP], rg2[NP];
    auto gload = [&](int k) {
        #pragma unroll
        for (int r = 0; r < NP; ++r) rg[r] = *(const u32x4*)(gp[r] + k);
    };
    auto gload2 = [&](int k) {
        #pragma unroll
        for (int r = 0; r < NP; ++r) rg2[r] = *(const u32x4*)(gp[r] + k);
    };
    auto sstore = [&](int buf) {
        #pragma unroll
        for (int r = 0; r < NP; ++r) *(u32x4*)&S[buf][(ldRow + r * RPI) * LD + ldCol] = rg[r];
    };
    auto sstore2 = [&](int buf) {
        #pragma unroll
        for (int r = 0; r < NP; ++r) *(u32x4*)&S[buf][(ldRow + r * RPI) * LD + ldCol] = rg2[r];
    };
    auto compute = [&](int buf) {
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 32) {
            i8x8 fa[FM], fb[FN];
            #pragma unroll
            for (int i = 0; i < FM; ++i) fa[i] = *(const i8x8*)&S[buf][(wr * 16 * FM + 16 * i + L) * LD + kk + 8 * H];
            #pragma unroll
            for (int j = 0; j < FN; ++j) fb[j] = *(const i8x8*)&S[buf][(BM + wc * 16 * FN + 16 * j + L) * LD + kk + 8 * H];
#if SETPRIO
            asm volatile("s_setprio 1");
#endif
            #pragma unroll
            for (int j = 0; j < FN; ++j)
                #pragma unroll
                for (int i = 0; i < FM; ++i)
                    acc[j][i] = __builtin_amdgcn_mmac_i32_16x16x32i8(fa[i], fb[j], acc[j][i]);
#if SETPRIO
            asm volatile("s_setprio 0");
#endif
        }
    };

    const int nk = K / BK;                  // 要求 K % (2*BK) == 0
#if DBUF
    gload(0); sstore(0);
    for (int k = 0; k < nk; k += 2) {
        lds_barrier();
        if (k + 1 < nk) gload((k + 1) * BK);
        compute(0);
        if (k + 1 < nk) sstore(1);
        lds_barrier();
        if (k + 2 < nk) gload((k + 2) * BK);
        compute(1);
        if (k + 2 < nk) sstore(0);
    }
#elif PF2
    // 单缓冲 + 两级寄存器预取：同时有两个 K 块的全局读在路上，提高 L2 并发请求数（PMC：单级预取时 L2 只到上限 76%）
    gload(0); gload2(BK);
    for (int k = 0; k < nk; k += 2) {
        lds_barrier();
        sstore(0);
        lds_barrier();
        if (k + 2 < nk) gload((k + 2) * BK);
        compute(0);
        lds_barrier();
        sstore2(0);
        lds_barrier();
        if (k + 3 < nk) gload2((k + 3) * BK);
        compute(0);
    }
#else
    gload(0);
    for (int k = 0; k < nk; ++k) {
        lds_barrier();
        sstore(0);
        lds_barrier();
        if (k + 1 < nk) gload((k + 1) * BK);
        compute(0);
    }
#endif

    // 尾处理：片段 C[m][n] → lane = m + 16*(n%4)，元素 = n/4（行 = L，列 = H + 4e）
    float s_b[FN][4];
    #pragma unroll
    for (int j = 0; j < FN; ++j)
        #pragma unroll
        for (int e = 0; e < 4; ++e) s_b[j][e] = sb[n0 + wc * 16 * FN + 16 * j + H + 4 * e];
#if EPI_LDS
    // 直接写时每条写指令只有 2 字节/lane、跨步 4 列，拆成 16 个 L2 请求（PMC：占 L2 请求 40%）。
    // 改为：每 wave 每次把一个 16 行 × (16*FN) 列的片段行转成 bf16 写进 LDS，再按行连续 16 字节/lane 写回。
    constexpr int WC = 16 * FN;            // 每 wave 的列数
    constexpr int SLD = WC + 8;            // bf16 个数，补白 16 字节
    static_assert(NTHR / 64 * 16 * SLD * 2 <= (1 + DBUF) * (BM + BN) * LD, "尾处理暂存放不进 LDS");
    unsigned short* stg = (unsigned short*)&S[0][0] + w * 16 * SLD;
    __syncthreads();                       // 主循环的 LDS 读全部完成后才能复用
    #pragma unroll
    for (int i = 0; i < FM; ++i) {
        int row = m0 + wr * 16 * FM + 16 * i + L;
        float s_a = row < M ? sa[row] : 0.f;
        #pragma unroll
        for (int j = 0; j < FN; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                float v = (float)acc[j][i][e] * s_a;
                v = v * s_b[j][e];
                stg[L * SLD + 16 * j + H + 4 * e] = f2bf(v);
            }
        // 同一 wave 内写后读：LDS 按 wave 顺序执行，等 lgkmcnt 即可，不需要块级 barrier
        __builtin_amdgcn_s_waitcnt(0xc07f);
        constexpr int LPR = WC / 8;        // 每行几个 lane（每 lane 8 个 bf16 = 16B）
        constexpr int RPP = 64 / LPR;      // 每遍几行
        #pragma unroll
        for (int rr = 0; rr < 16; rr += RPP) {
            int r = rr + lane / LPR, c = (lane % LPR) * 8;
            int grow = m0 + wr * 16 * FM + 16 * i + r;
            u32x4 v = *(const u32x4*)&stg[r * SLD + c];
            if (grow < M) *(u32x4*)&C[(size_t)grow * N + n0 + wc * WC + c] = v;
        }
        __builtin_amdgcn_s_waitcnt(0xc07f);
    }
#else
    #pragma unroll
    for (int i = 0; i < FM; ++i) {
        int row = m0 + wr * 16 * FM + 16 * i + L;
        if (row >= M) continue;
        float s_a = sa[row];
        unsigned short* crow = C + (size_t)row * N;
        #pragma unroll
        for (int j = 0; j < FN; ++j) {
            int cb = n0 + wc * 16 * FN + 16 * j + H;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                float v = (float)acc[j][i][e] * s_a;
                v = v * s_b[j][e];
                crow[cb + 4 * e] = f2bf(v);
            }
        }
    }
#endif
}

extern "C" int i8gemm_launch(const void* A, const void* W, const void* sa, const void* sb, void* C,
                             int M, int N, int K, void* stream) {
    if (N % BN || K % (2 * BK) || M <= 0) return -1;
    dim3 grid(N / BN, (M + BM - 1) / BM), blk(NTHR);
    hipLaunchKernelGGL(i8gemm_kernel, grid, blk, 0, (hipStream_t)stream,
                       (const int8_t*)A, (const int8_t*)W, (const float*)sa, (const float*)sb,
                       (unsigned short*)C, M, N, K);
    return (int)hipGetLastError();
}

extern "C" int i8gemm_tile(int* bm, int* bn) { *bm = BM; *bn = BN; return NTHR; }

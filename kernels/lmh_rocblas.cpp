// LM-head bf16 GEMM 直调 rocBLAS 指定解（不经 TunableOp，避免它把其余 bf16 GEMM 一并改走默认解）。
// out[M,N] = x[M,K] · W[N,K]ᵀ；列主序下即 C[N,M] = op(A)=Wᵀ…：transA=T, transB=N, m=N, n=M, k=K, lda=ldb=K, ldc=N
// （与 TunableOp 键 tn_N_M_K_ld_K_K_N 同一调用）。sol=0 走 rocblas 默认启发式。
// 编译：hipcc -O2 -shared -fPIC -o lmh_rocblas.so lmh_rocblas.cpp -lrocblas
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

static rocblas_handle g_h[64];

extern "C" int lmh_init() {
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return -2;
    if (!g_h[dev] && rocblas_create_handle(&g_h[dev]) != rocblas_status_success) return -1;
    return 0;
}

extern "C" int lmh_gemm(const void* x, const void* w, void* out, int M, int N, int K, int sol, void* stream) {
    int dev = 0;
    hipGetDevice(&dev);
    if (!g_h[dev]) { int r = lmh_init(); if (r) return r; }
    rocblas_handle h = g_h[dev];
    rocblas_set_stream(h, (hipStream_t)stream);
    const float a = 1.f, b = 0.f;
    rocblas_status s = rocblas_gemm_ex(
        h, rocblas_operation_transpose, rocblas_operation_none, N, M, K, &a,
        w, rocblas_datatype_bf16_r, K, x, rocblas_datatype_bf16_r, K, &b,
        out, rocblas_datatype_bf16_r, N, out, rocblas_datatype_bf16_r, N,
        rocblas_datatype_f32_r, sol > 0 ? rocblas_gemm_algo_solution_index : rocblas_gemm_algo_standard, sol, 0);
    return (int)s;
}

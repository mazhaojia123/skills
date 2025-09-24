#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#ifndef CEIL_DIV
#define CEIL_DIV(x,y) (((x)+((y)-1))/(y))
#endif

// WMMA tile size (Ampere 常用 16x16x16)
constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;

// 一个 block = 1 个 warp，计算一个 16x16 输出 tile
__global__ void gemm_wmma_tc(
    const half* __restrict__ A,   // [M x K], row-major
    const half* __restrict__ B,   // [K x N], row-major
    half* __restrict__ C,         // [M x N], row-major, 初始为 C_initial
    int M, int N, int K,
    float alpha, float beta)
{
    // tile 起点
    const int tile_m = blockIdx.y * WM;
    const int tile_n = blockIdx.x * WN;

    // 共享内存：A 与 B 的子块（half，零填充）+ 结果暂存（float）
    extern __shared__ unsigned char smem_raw[];
    half*  As = reinterpret_cast<half*>(smem_raw);
    half*  Bs = As + WM * WK;
    float* Cs = reinterpret_cast<float*>(Bs + WK * WN);

    // 结果 fragment（FP32 累加）
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // 单个 warp 的 lane
    const int lane = threadIdx.x & 31;

    // K 维分块
    for (int k0 = 0; k0 < K; k0 += WK) {
        // 1) 把 A[tile_m:tile_m+16, k0:k0+16] 读入 As（带边界零填充）
        for (int i = lane; i < WM * WK; i += 32) {
            int r = i / WK;
            int c = i % WK;
            int gr = tile_m + r;
            int gc = k0 + c;
            half v = __float2half(0.0f);
            if (gr < M && gc < K) v = A[gr * K + gc];
            As[r * WK + c] = v;
        }
        // 2) 把 B[k0:k0+16, tile_n:tile_n+16] 读入 Bs（带边界零填充）
        for (int i = lane; i < WK * WN; i += 32) {
            int r = i / WN;
            int c = i % WN;
            int gr = k0 + r;
            int gc = tile_n + c;
            half v = __float2half(0.0f);
            if (gr < K && gc < N) v = B[gr * N + gc];
            Bs[r * WN + c] = v;
        }
        __syncthreads();

        // 3) 载入 fragment（row-major）
        wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> b_frag;

        wmma::load_matrix_sync(a_frag, As, WK);
        wmma::load_matrix_sync(b_frag, Bs, WN);

        // 4) Tensor Core 计算：c_frag += a_frag * b_frag
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads(); // 复用共享内存
    }

    // 5) 把 c_frag 存到共享内存 Cs（row-major，float）
    wmma::store_matrix_sync(Cs, c_frag, WN, wmma::mem_row_major);
    __syncthreads();

    // 6) 与 C_initial 做 alpha/beta 融合并写回（FP16）
    for (int i = lane; i < WM * WN; i += 32) {
        int r = i / WN;
        int c = i % WN;
        int gr = tile_m + r;
        int gc = tile_n + c;
        if (gr < M && gc < N) {
            float ab   = Cs[i];                         // A*B 的 FP32 结果
            float cold = __half2float(C[gr * N + gc]);  // C_initial
            float out  = alpha * ab + beta * cold;
            C[gr * N + gc] = __float2half_rn(out);
        }
    }
}

// A,B,C 均为 device 指针（half）
extern "C" void solve(const half* A, const half* B, half* C,
                      int M, int N, int K, float alpha, float beta)
{
    dim3 grid(CEIL_DIV(N, WN), CEIL_DIV(M, WM));
    dim3 block(32, 1, 1); // 一个 warp 一个 tile

    size_t shmem_bytes = (WM*WK + WK*WN) * sizeof(half) + (WM*WN) * sizeof(float);
    gemm_wmma_tc<<<grid, block, shmem_bytes>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define WM 16
#define WN 16
#define WK 16

__global__ void gemm_kernel(
    const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta)
{
    const int tileM = blockIdx.x * WM;
    const int tileN = blockIdx.y * WN;

    extern __shared__ unsigned char smemRaw[];
    half *sA = reinterpret_cast<half*>(smemRaw);        // WM * WK
    half *sB = sA + WM * WK;                            // WK * WN
    float *sC = reinterpret_cast<float*>(sB + WK * WN); // WM * WN

    const int laneId = threadIdx.x & 31;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> fragC;
    wmma::fill_fragment(fragC, 0.0f);

    for (int k0 = 0; k0 < K; k0 += WK) {
        // load A to sA
        for (int i = laneId; i < WM * WK; i += 32) {
            int c = i % WK;
            int r = i / WK;
            int gc = k0 + c;
            int gr = tileM + r;
            half rA = 0.0f;
            if (gc < K && gr < M) rA = A[gc + gr * K];
            sA[c + r * WK] = rA;
        }

        // load B to sB
        for (int i = laneId; i < WN * WK; i += 32) {
            int c = i % WN;
            int r = i / WN;
            int gc = c + tileN;
            int gr = r + k0;
            half rB = 0.0f;
            if (gc < N && gr < K) rB = B[gc + gr * N];
            sB[c + r * WN] = rB;
        }

        __syncthreads();

        wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> fragA;
        wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> fragB;

        wmma::load_matrix_sync(fragA, sA, WK);
        wmma::load_matrix_sync(fragB, sB, WN);

        wmma::mma_sync(fragC, fragA, fragB, fragC);

        __syncthreads();
    }

    wmma::store_matrix_sync(sC, fragC, WN, wmma::mem_row_major);
    __syncthreads();

    for (int i = laneId; i < WM * WN; i += 32) {
        int r = i / WN;
        int c = i % WN;
        int gr = tileM + r;
        int gc = tileN + c;

        if (gr < M && gc < N) {
            float ab = sC[r * WN + c];
            float cold = __half2float(C[gr * N + gc]);
            C[gr * N + gc] = __float2half_rn(alpha * ab + beta * cold);
        }
    }
}

extern "C" void solve(
    const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta)
{
    dim3 grid((M + WM - 1) / WM, (N + WN - 1) / WN);
    dim3 block(32, 1, 1);
    size_t shmem_bytes = (WM * WK + WK * WN) * sizeof(half) + (WM * WN) * sizeof(float);
    gemm_kernel<<<grid, block, shmem_bytes>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}

extern "C" float bench(
    const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta)
{
    dim3 grid((M + WM - 1) / WM, (N + WN - 1) / WN);
    dim3 block(32, 1, 1);
    size_t shmem_bytes = (WM * WK + WK * WN) * sizeof(half) + (WM * WN) * sizeof(float);

    cudaEvent_t start;
    cudaEvent_t stop;
    bool start_created = false;
    bool stop_created = false;
    float ms = -1.0f;

    cudaError_t err = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        return -static_cast<float>(err);
    }
    start_created = true;

    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        cudaEventDestroy(start);
        return -static_cast<float>(err);
    }
    stop_created = true;

    err = cudaEventRecord(start);
    if (err != cudaSuccess) {
        ms = -static_cast<float>(err);
        return ms;
    }

    gemm_kernel<<<grid, block, shmem_bytes>>>(A, B, C, M, N, K, alpha, beta);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        ms = -static_cast<float>(err);
        return ms;
    }

    err = cudaEventRecord(stop);
    if (err != cudaSuccess) {
        ms = -static_cast<float>(err);
        return ms;
    }

    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) {
        ms = -static_cast<float>(err);
        return ms;
    }

    err = cudaEventElapsedTime(&ms, start, stop);
    if (err != cudaSuccess) {
        ms = -static_cast<float>(err);
    }

    if (stop_created) {
        cudaEventDestroy(stop);
    }
    if (start_created) {
        cudaEventDestroy(start);
    }
    return ms;
}

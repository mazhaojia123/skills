#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
using namespace nvcuda;

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;

constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;

constexpr int WARP_M = BM / WM;
constexpr int WARP_N = BN / WN;
constexpr int WARP_K = BK / WK;

#define CEIL_DIV(x,y) (((x)+(y)-1) / (y))

struct SmemGemm 
{
    half *sa[2];
    half *sb[2];
    float *sc;
};

__device__ __forceinline__ SmemGemm setSmem(unsigned char *smemRaw) 
{
    SmemGemm s;
    s.sa[0] = reinterpret_cast<half*>(smemRaw); // BM * BK
    s.sa[1] = s.sa[0] + BM * BK;                // BM * BK
    s.sb[0] = s.sa[1] + BM * BK;                // BN * BK
    s.sb[1] = s.sb[0] + BN * BK;                // BN * BK
    s.sc = reinterpret_cast<float*>(s.sb[1] + BN * BK);
    return s;
}

__device__ __forceinline__ void cp_async_16B(void *smemDst, const void *gmemSrc) 
{
    unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smemDst));
    unsigned long long src = reinterpret_cast<unsigned long long>(gmemSrc);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16; \n"
        :: "r"(dst), "l"(src)
    );
}

__device__ __forceinline__ void cp_async_wait_all() 
{
    asm volatile("cp.async.wait_group 0; \n" ::);
}

__device__ __forceinline__ void cp_async_commit() 
{
    asm volatile("cp.async.commit_group;\n" ::);
}

__global__ void gemm_tc_db(const half* A, const half* B, half* C,
                      int M, int N, int K, float alpha, float beta) 
{
    extern __shared__ unsigned char smemRaw[];
    SmemGemm smem = setSmem(smemRaw);

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int wc = wid % WARP_N;
    const int wr = wid / WARP_N;
    const int TPB = blockDim.x;

    const int tileM = blockIdx.x * BM;
    const int tileN = blockIdx.y * BN;

    {
        // load 8 个 A 中的 half == 16B
        const int sa_n16B = BM * BK * 2 / 16;
        for (int i = tid; i < sa_n16B; i += TPB) {
            int elem8 = i * 8;
            int c = elem8 % BK;
            int r = elem8 / BK;
            int gc = c + 0;   // k0 = 0
            int gr = r + tileM;

            bool valid_raw = gr < M;
            bool valid_tile = valid_raw && (gc + 7 < K);
            if (valid_tile) {
                cp_async_16B(smem.sa[0] + r * BK + c, A + gr * K + gc);
            } else if (valid_raw) {
                for (int t0 = 0; t0 < 8; t0++) {
                    int gc1 = gc + t0;
                    int c1 = c + t0;
                    if (c1 >= BK) break;
                    smem.sa[0][r * BK + c1] = gc1 < K ? A[gr * K + gc1] : static_cast<half>(0.0f);
                }
            } else {
                for (int t0 = 0; t0 < 8; t0++) {
                    int c1 = c + t0;
                    if (c1 >= BK) break;
                    smem.sa[0][r * BK + c1] = static_cast<half>(0.0f);
                }
            }
        }

        const int sb_n16B = BN * BK * 2 / 16;
        for (int i = tid; i < sb_n16B; i += TPB) {
            int elem8 = i * 8;
            int c = elem8 % BN;
            int r = elem8 / BN;
            int gc = c + tileN;
            int gr = r + 0;       // k0 = 0;

            bool valid_raw = gr < K;
            bool valid_tile = valid_raw && (gc + 7 < N);

            if (valid_tile) {
                cp_async_16B(smem.sb[0] + r * BN + c, B + gr * N + gc);
            } else if (valid_raw) {
                for (int t0 = 0; t0 < 8; t0++) {
                    int gc1 = gc + t0;
                    int c1 = c + t0;
                    if (c1 >= BN) break;
                    smem.sb[0][r * BN + c1] = gc1 < N ? B[gr * N + gc1] : static_cast<half>(0.0f);
                }
            } else {
                for (int t0 = 0; t0 < 8; t0++) {
                    int c1 = c + t0;
                    if (c1 >= BN) break;
                    smem.sb[0][r * BN + c1] = static_cast<half>(0.0f);
                }
            }
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads(); 
    }
    
    int readBuf = 0, writeBuf = 1;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> fragC;
    wmma::fill_fragment(fragC, 0.0f);

    // mainloop
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K)
        {
            int tileK = k0 + BK;
            // load 8 个 A 中的 half == 16B
            const int sa_n16B = BM * BK * 2 / 16;
            for (int i = tid; i < sa_n16B; i += TPB) {
                int elem8 = i * 8;
                int c = elem8 % BK;
                int r = elem8 / BK;
                int gc = c + tileK;   // k0 = 0
                int gr = r + tileM;

                bool valid_raw = gr < M;
                bool valid_tile = valid_raw && (gc + 7 < K);
                if (valid_tile) {
                    cp_async_16B(smem.sa[writeBuf] + r * BK + c, A + gr * K + gc);
                } else if (valid_raw) {
                    for (int t0 = 0; t0 < 8; t0++) {
                        int gc1 = gc + t0;
                        int c1 = c + t0;
                        if (c1 >= BK) break;
                        smem.sa[writeBuf][r * BK + c1] = gc1 < K ? A[gr * K + gc1] : static_cast<half>(0.0f);
                    }
                } else {
                    for (int t0 = 0; t0 < 8; t0++) {
                        int c1 = c + t0;
                        if (c1 >= BK) break;
                        smem.sa[writeBuf][r * BK + c1] = static_cast<half>(0.0f);
                    }
                }
            }

            const int sb_n16B = BN * BK * 2 / 16;
            for (int i = tid; i < sb_n16B; i += TPB) {
                int elem8 = i * 8;
                int c = elem8 % BN;
                int r = elem8 / BN;
                int gc = c + tileN;
                int gr = r + tileK;       // k0 = 0;

                bool valid_raw = gr < K;
                bool valid_tile = valid_raw && (gc + 7 < N);

                if (valid_tile) {
                    cp_async_16B(smem.sb[writeBuf] + r * BN + c, B + gr * N + gc);
                } else if (valid_raw) {
                    for (int t0 = 0; t0 < 8; t0++) {
                        int gc1 = gc + t0;
                        int c1 = c + t0;
                        if (c1 >= BN) break;
                        smem.sb[writeBuf][r * BN + c1] = gc1 < N ? B[gr * N + gc1] : static_cast<half>(0.0f);
                    }
                } else {
                    for (int t0 = 0; t0 < 8; t0++) {
                        int c1 = c + t0;
                        if (c1 >= BN) break;
                        smem.sb[writeBuf][r * BN + c1] = static_cast<half>(0.0f);
                    }
                }
            }
            cp_async_commit();
        }
    
        wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> fragA;
        wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> fragB;
        
        const half *sa = smem.sa[readBuf] + wr * WM * BK;
        const half *sb = smem.sb[readBuf] + wc * WN;
        wmma::load_matrix_sync(fragA, sa, BK);
        wmma::load_matrix_sync(fragB, sb, BN);

        wmma::mma_sync(fragC, fragA, fragB, fragC);

        if (k0 + BK < K) {
            cp_async_wait_all();
            __syncthreads();
            readBuf ^= 1;
            writeBuf ^= 1;
        }
    }

    wmma::store_matrix_sync(smem.sc + wr * WM * BN + wc * WN, fragC, BN, wmma::mem_row_major);
    __syncthreads();
    
    for (int i = tid; i < BN * BM; i += TPB) {
        int c = i % BN; 
        int r = i / BN;
        int gc = c + tileN;
        int gr = r + tileM;
        if (gc < N && gr < M) {
            float rc = smem.sc[r * BN + c];
            float oldC = __half2float(C[gc + gr * N]);
            C[gc + gr * N] = __float2half_rn(alpha * rc + beta * oldC);
        }
    }
}

extern "C" void solve(const half* A, const half* B, half* C,
                      int M, int N, int K, float alpha, float beta)
{
    dim3 grid(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    dim3 block(32 * 16, 1, 1);
    size_t shmSize = (BM * BK + BN * BK) * sizeof(half) * 2 + BM * BN * sizeof(float);

    gemm_tc_db<<<grid, block, shmSize>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}

extern "C" float bench(const half* A, const half* B, half* C,
                      int M, int N, int K, float alpha, float beta)
{
    dim3 grid(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    dim3 block(32 * 16, 1, 1);
    size_t shmSize = (BM * BK + BN * BK) * sizeof(half) * 2 + BM * BN * sizeof(float);

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

    gemm_tc_db<<<grid, block, shmSize>>>(A, B, C, M, N, K, alpha, beta);
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

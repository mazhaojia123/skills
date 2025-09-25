#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
using namespace nvcuda;

#ifndef CEIL_DIV
#define CEIL_DIV(x,y) (((x)+((y)-1))/(y))
#endif

// WMMA tile 固定 16×16×16
constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;

// Block 级 tile：64×64 输出，K 方向步长 16
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;

// 一个 block 工作的 warp 拓扑：4×4 warp 正好覆盖 64×64（每个 warp 16×16）
constexpr int WARPS_M = BM / WM; // 4
constexpr int WARPS_N = BN / WN; // 4
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N; // 16

// 线程组织：512 线程 = 16 个 warp
[[maybe_unused]] constexpr int THREADS_PER_BLOCK = 32 * WARPS_PER_BLOCK; // 512

// A/B 双缓冲所需共享内存（half） + C 临时缓存（float）
struct SmemLayout {
    // A: 2 * (BM * BK), B: 2 * (BK * BN), C: (BM * BN) float
    // 为了简单按紧凑平铺，不做额外 padding；WMMA 加载按 row-major。
    half* A_buf[2];
    half* B_buf[2];
    float* C_buf;
};

__device__ __forceinline__ SmemLayout smem_partition(void* smem_base) {
    auto* hptr = reinterpret_cast<half*>(smem_base);
    SmemLayout s;
    s.A_buf[0] = hptr;                                  // [BM * BK]
    s.A_buf[1] = s.A_buf[0] + BM * BK;                  // [BM * BK]
    s.B_buf[0] = s.A_buf[1] + BM * BK;                  // [BK * BN]
    s.B_buf[1] = s.B_buf[0] + BK * BN;                  // [BK * BN]
    auto* fptr = reinterpret_cast<float*>(s.B_buf[1] + BK * BN);
    s.C_buf = fptr;                                     // [BM * BN]
    return s;
}

// 将 (bytes=16) 的一段从 gmem 复制到 smem（sm_80 用 cp.async，其他架构普通拷贝）
__device__ __forceinline__
void cp_async_16B(void* smem_dst, const void* gmem_src) {
#if __CUDA_ARCH__ >= 800
    unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    unsigned long long src = reinterpret_cast<unsigned long long>(gmem_src);
    asm volatile (
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(dst), "l"(src)
    );
#else
    *reinterpret_cast<uint4*>(smem_dst) = *reinterpret_cast<const uint4*>(gmem_src);
#endif
}

#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}
#endif

// 多 warp + 双缓冲 WMMA GEMM 内核
__global__ void gemm_wmma_tc_db(
    const half* __restrict__ A,   // [M x K] row-major
    const half* __restrict__ B,   // [K x N] row-major
    half* __restrict__ C,         // [M x N] row-major (含 C_initial)
    int M, int N, int K,
    float alpha, float beta)
{
    extern __shared__ unsigned char smem_uchar[];
    SmemLayout smem = smem_partition(smem_uchar);

    // 本 block 的输出 tile 起点（全局坐标）
    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;

    // warp id / lane id（支持任意 block 维度排列）
    const int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    const int linear_tid = threadIdx.x;
    const half zero = __float2half(0.0f);
    const int warpId = (linear_tid >> 5); // 0..15 （我们用 blockDim = (32*16,1,1) 或 (128,4,1) 等）
    const int warp_m = warpId / WARPS_N;   // 0..3
    const int warp_n = warpId % WARPS_N;   // 0..3

    // 累加 fragment
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // K 方向分块数
    const int stages = CEIL_DIV(K, BK);

    // ===== 预取 stage 0 到 buffer 0 =====
    // 以 16B 为单位协作拷贝：A_sub: [BM x BK]，B_sub: [BK x BN]
    // A_sub 行主序（ld=BK），B_sub 行主序（ld=BN）
    {
        // A_sub
        int total_16B_A = (BM * BK * sizeof(half)) / 16; // 每 16B 8 个 half
        for (int idx = linear_tid; idx < total_16B_A; idx += threads_per_block) {
            int elem8 = idx * 8;                    // half 元素索引
            int r = elem8 / BK;                     // 0..BM-1
            int c = elem8 % BK;                     // 0..BK-1
            int gr = block_m + r;
            int gc = c;                              // k0 = 0
            half* row_dst = &smem.A_buf[0][r * BK];
            half* dst = row_dst + c;

            bool row_valid = (gr < M);
            bool full_tile = row_valid && (gc + 7 < K);

            if (full_tile) {
                const half* src = A + gr * K + gc;
                cp_async_16B(dst, src);
            } else if (row_valid) {
                for (int t = 0; t < 8; ++t) {
                    int sc = c + t;
                    if (sc >= BK) break;
                    int gcol = gc + t;
                    row_dst[sc] = (gcol < K) ? A[gr * K + gcol] : zero;
                }
            } else {
                for (int t = 0; t < 8; ++t) {
                    int sc = c + t;
                    if (sc >= BK) break;
                    row_dst[sc] = zero;
                }
            }
        }
        // B_sub
        int total_16B_B = (BK * BN * sizeof(half)) / 16;
        for (int idx = linear_tid; idx < total_16B_B; idx += threads_per_block) {
            int elem8 = idx * 8;
            int r = elem8 / BN;                     // 0..BK-1
            int c = elem8 % BN;                     // 0..BN-1
            int gr = 0 + r;                         // k0 = 0
            int gc = block_n + c;
            half* row_dst = &smem.B_buf[0][r * BN];
            half* dst = row_dst + c;

            bool row_valid = (gr < K);
            bool full_tile = row_valid && (gc + 7 < N);

            if (full_tile) {
                const half* src = B + gr * N + gc;
                cp_async_16B(dst, src);
            } else if (row_valid && gc < N) {
                for (int t = 0; t < 8; ++t) {
                    int sc = c + t;
                    if (sc >= BN) break;
                    int gcol = gc + t;
                    row_dst[sc] = (gcol < N) ? B[gr * N + gcol] : zero;
                }
            } else {
                for (int t = 0; t < 8; ++t) {
                    int sc = c + t;
                    if (sc >= BN) break;
                    row_dst[sc] = zero;
                }
            }
        }
#if __CUDA_ARCH__ >= 800
        cp_async_commit();
        cp_async_wait_all();
#endif
        __syncthreads();
    }

    // ==== 主循环 ====
    int read_buf = 0, write_buf = 1;
    for (int s = 0; s < stages; ++s) {

        // 预取下一 stage 到 write_buf（如果有）
        if (s + 1 < stages) {
            int k0 = (s + 1) * BK;

            // A_sub(next)
            int total_16B_A = (BM * BK * sizeof(half)) / 16;
            for (int idx = linear_tid; idx < total_16B_A; idx += threads_per_block) {
                int elem8 = idx * 8;
                int r = elem8 / BK;
                int c = elem8 % BK;
                int gr = block_m + r;
                int gc = k0 + c;
                half* row_dst = &smem.A_buf[write_buf][r * BK];
                half* dst = row_dst + c;

                bool row_valid = (gr < M);
                bool full_tile = row_valid && (gc + 7 < K);

                if (full_tile) {
                    const half* src = A + gr * K + gc;
                    cp_async_16B(dst, src);
                } else if (row_valid) {
                    for (int t = 0; t < 8; ++t) {
                        int sc = c + t;
                        if (sc >= BK) break;
                        int gcol = gc + t;
                        row_dst[sc] = (gcol < K) ? A[gr * K + gcol] : zero;
                    }
                } else {
                    for (int t = 0; t < 8; ++t) {
                        int sc = c + t;
                        if (sc >= BK) break;
                        row_dst[sc] = zero;
                    }
                }
            }
            // B_sub(next)
            int total_16B_B = (BK * BN * sizeof(half)) / 16;
            for (int idx = linear_tid; idx < total_16B_B; idx += threads_per_block) {
                int elem8 = idx * 8;
                int r = elem8 / BN;
                int c = elem8 % BN;
                int gr = k0 + r;
                int gc = block_n + c;
                half* row_dst = &smem.B_buf[write_buf][r * BN];
                half* dst = row_dst + c;

                bool row_valid = (gr < K);
                bool full_tile = row_valid && (gc + 7 < N);

                if (full_tile) {
                    const half* src = B + gr * N + gc;
                    cp_async_16B(dst, src);
                } else if (row_valid && gc < N) {
                    for (int t = 0; t < 8; ++t) {
                        int sc = c + t;
                        if (sc >= BN) break;
                        int gcol = gc + t;
                        row_dst[sc] = (gcol < N) ? B[gr * N + gcol] : zero;
                    }
                } else {
                    for (int t = 0; t < 8; ++t) {
                        int sc = c + t;
                        if (sc >= BN) break;
                        row_dst[sc] = zero;
                    }
                }
            }
#if __CUDA_ARCH__ >= 800
            cp_async_commit();
#endif
        }

        // ===== 计算当前 read_buf 上的 64×64×16 =====
        // 每个 warp 取自己 16×16 的子块
        wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> b_frag;

        const half* As = smem.A_buf[read_buf] + (warp_m * WM) * BK; // ld = BK
        const half* Bs = smem.B_buf[read_buf] + (warp_n * WN);      // ld = BN

        wmma::load_matrix_sync(a_frag, As, BK);
        wmma::load_matrix_sync(b_frag, Bs, BN);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        // 等待预取完成 & 切换缓冲
        if (s + 1 < stages) {
#if __CUDA_ARCH__ >= 800
            cp_async_wait_all();
#endif
            __syncthreads();
            read_buf ^= 1;
            write_buf ^= 1;
        }
    }

    // ===== 把每个 warp 的 c_frag 写到共享内存 C_buf（float） =====
    // 这样可以统一做 alpha/beta，最后半精度写回
    wmma::store_matrix_sync(
        smem.C_buf + (warp_m * WM) * BN + (warp_n * WN), // 该 warp 在 C_buf 的起点
        c_frag, BN, wmma::mem_row_major
    );
    __syncthreads();

    // ===== 融合 alpha/beta 并写回全局内存（FP16） =====
    // 让全体线程协作把 64×64 的 float 缓冲写回 C（边界判定）
    for (int idx = linear_tid; idx < BM * BN; idx += threads_per_block) {
        int r = idx / BN;   // 0..63
        int c = idx % BN;   // 0..63
        int gr = block_m + r;
        int gc = block_n + c;
        if (gr < M && gc < N) {
            float ab   = smem.C_buf[r * BN + c];
            float cold = __half2float(C[gr * N + gc]);
            float out  = alpha * ab + beta * cold;
            C[gr * N + gc] = __float2half_rn(out);
        }
    }
}

// A,B,C 为 device 指针（half）
extern "C" void solve(const half* A, const half* B, half* C,
                      int M, int N, int K, float alpha, float beta)
{
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    // 512 线程，组织成 (128,4,1) 便于观察 warp 排布；(32*16,1,1) 也可
    dim3 block(128*4, 1, 1);

    size_t smem_bytes =
        // 双缓冲的 A/B（half）
        (2 * BM * BK + 2 * BK * BN) * sizeof(half)
        // C 临时（float）
        + (BM * BN) * sizeof(float);

    // 对于较大的共享内存（> 48KB）时，可提升动态共享内存上限（A100 默认充足，此处一般 < 32KB）
    // cudaFuncSetAttribute(gemm_wmma_tc_db, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    gemm_wmma_tc_db<<<grid, block, smem_bytes>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}

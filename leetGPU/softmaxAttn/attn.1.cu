#include <cuda_runtime.h>
#include <math.h>

// block 内做归约求点积；要求 blockDim.x 为 2 的幂（下方已按此选择）
static __device__ float blockReduceSum(float* shm, float v) {
    int tid = threadIdx.x;
    shm[tid] = v;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) shm[tid] += shm[tid + stride];
        __syncthreads();
    }
    return shm[0];
}

// 每个 block 处理 Q 的一行：i
static __global__ void softmax_attention_kernel(
    const float* __restrict__ Q,   // [M, d]
    const float* __restrict__ K,   // [N, d]
    const float* __restrict__ V,   // [N, d]
    float* __restrict__ O,         // [M, d]
    int M, int N, int d)
{
    int i = blockIdx.x;
    if (i >= M) return;

    extern __shared__ float smem[];
    // 布局: q_sh[d] | sum_sh[d] | red[blockDim.x] | scalars[4]
    float* q_sh   = smem;                 // 缓存 Q[i]
    float* red    = q_sh + d;           // 归约缓冲
    float* scal   = red + blockDim.x;     // 标量共享: [max, e, denom, tmp]
    const unsigned int rep = 10;
    float sum_sh[rep];             // 累加 ∑ e * V
    for (int i = 0; i < rep; ++i) sum_sh[i] = 0.0f;
    

    const int tid = threadIdx.x;
    const float inv_sqrt_d = rsqrtf((float)d);

    // 载入 Q[i], 并清零 sum_sh
    for (int k = tid; k < d; k += blockDim.x) {
        q_sh[k]   = Q[i * d + k];
    }
    __syncthreads();

    // ---------- Pass 1: 求该行 logits 的最大值（数值稳定） ----------
    float row_max = -1.0e20f;
    for (int j = 0; j < N; ++j) {
        // 计算 dot = q_i · k_j
        float part = 0.0f;
        const float* Kj = K + j * d;
        for (int k = tid; k < d; k += blockDim.x) {
            part += q_sh[k] * Kj[k];
        }
        float dot = blockReduceSum(red, part);
        if (tid == 0) {
            float z = dot * inv_sqrt_d;
            if (z > row_max) row_max = z;
        }
        __syncthreads();
    }
    if (tid == 0) scal[0] = row_max; // 广播 max
    __syncthreads();
    row_max = scal[0];

    // ---------- Pass 2: 累加分子向量与分母 ----------
    float denom = 0.0f;  // ∑_j exp((qk)/√d - max)
    for (int j = 0; j < N; ++j) {
        // 仍需点积
        float part = 0.0f;
        const float* Kj = K + j * d;
        for (int k = tid; k < d; k += blockDim.x) {
            part += q_sh[k] * Kj[k];
        }
        float dot = blockReduceSum(red, part);

        if (tid == 0) {
            float z = dot * inv_sqrt_d;
            float e = expf(z - row_max);
            scal[1] = e;      // 广播 e
            denom  += e;
        }
        __syncthreads();
        float e = scal[1];

        // sum_sh += e * V[j]
        const float* Vj = V + j * d;
        for (int k = tid; k < d; k += blockDim.x) {
            sum_sh[k/blockDim.x] += e * Vj[k];
        }
        __syncthreads();
    }
    if (tid == 0) scal[2] = denom;
    __syncthreads();
    denom = scal[2];

    // ---------- 写回输出：O[i] = (∑ e * V) / denom ----------
    for (int k = tid; k < d; k += blockDim.x) {
        O[i * d + k] = sum_sh[k/blockDim.x] / denom;
    }
}

// Q, K, V, output 均为 device 指针
extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int M, int N, int d) {
    // 选择 2 的幂线程数（≤256），便于归约
    int threads = 1;
    while (threads < d && threads < 256) threads <<= 1; // 1,2,4,...,256

    dim3 grid(M);
    dim3 block(threads);

    // 共享内存需求: 2*d + blockDim.x + 4 个 float
    size_t shmem_bytes = sizeof(float) * (2 * (size_t)d + (size_t)threads + 4);

    softmax_attention_kernel<<<grid, block, shmem_bytes>>>(Q, K, V, output, M, N, d);
    cudaDeviceSynchronize();
}

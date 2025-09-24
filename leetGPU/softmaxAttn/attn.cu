#include <cuda_runtime.h>
#include <float.h>

static __device__ float blockReduceSum(float *red, float v) {
    // red 是 reduction 用的共享内存
    // v 是当前线程所掌握的数据量
    unsigned int tId = threadIdx.x;
    red[tId] = v;
    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tId < i) red[tId] += red[tId + i];
        __syncthreads();
    }
    
    return red[0];
}

__global__ void softmaxAttn(
    const float *Q,
    const float *K,
    const float *V, 
    float *O, 
    const int m, const int n, const int d)
{
    unsigned int bId = blockIdx.x;
    unsigned int tId = threadIdx.x;
    unsigned int TPB = blockDim.x;
    unsigned int BPG = gridDim.x;
    if (bId >= m) return;

    extern __shared__ float smem[];
    float *q_sh = smem;             // [d], 缓存Q[i,:]
    float *red = smem + d;          // [blockDim.x]
    float *scal = red + blockDim.x; // [1]

    float ro[16];
    for (int i = 0; i < 16; i++) ro[i] = 0.0f;

    float inv_sqrt_d = rsqrtf((float)d);

    const float *q0 = Q + bId * d;
    for (int k0 = 0; k0 < d; k0++) {
        q_sh[k0] = q0[k0];
    }

    __syncthreads();

    float row_max = -FLT_MAX;
    for (int col = 0; col < n; col++) {
        const float *k0 = K + col * d;
        float acc = 0.0f;
        for (unsigned int idx = tId; idx < d; idx += TPB) {
            acc += q_sh[idx] * k0[idx];
        }
        acc = blockReduceSum(red, acc);
        if (tId == 0) {
            float logit = acc * inv_sqrt_d;
            row_max = fmaxf(row_max, logit);
        }
    }

    if (tId == 0) scal[0] = row_max;
    __syncthreads();
    row_max = scal[0];

    float denom = 0.0f;
    for (int col = 0; col < n; col++) {
        const float *k0 = K + col * d;
        float acc = 0.0f;
        for (unsigned int idx = tId; idx < d; idx += TPB) {
            acc += q_sh[idx] * k0[idx];
        }
        acc = blockReduceSum(red, acc);
        if (tId == 0) {
            float logit = acc * inv_sqrt_d;
            float e = expf(logit - row_max);
            denom += e;
            scal[1] = e;
        }
        __syncthreads();
        float p = scal[1];
        const float *v0 = V + col * d;
        for (int i = tId; i < d; i += TPB) {
            ro[i / TPB] += p * v0[i];
        }
    }

    if (tId == 0) scal[2] = denom;
    __syncthreads();
    denom = scal[2];

    float *o0 = O + bId * d;
    for (int i = tId; i < d; i += TPB) {
        o0[i] = ro[i / TPB] / denom;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    // Q (Mxd), K (Nxd), V (Nxd)

    int threads = 1;
    while (threads < d && threads < 256) threads <<= 1; // 1,2,4,...,256
    dim3 grid(M);
    dim3 block(threads);

    size_t smemSize = (threads + d + 3) * sizeof(float);
    softmaxAttn<<<grid, block, smemSize>>>(Q, K, V, output, M, N, d);
    cudaDeviceSynchronize();
}

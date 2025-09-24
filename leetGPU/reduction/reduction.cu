#include <cuda_runtime.h>

#define TPB 256

__global__ void reduce_sum(const float *input, float *res_sum, int N) {
    int tId = threadIdx.x;
    int bId = blockIdx.x;
    int gTId = bId * blockDim.x + tId;
    int stride = blockDim.x * gridDim.x;
    int bDim = blockDim.x;
    __shared__ float sAcc[TPB];

    float acc = 0.0f;
    for (int i = gTId; i < N; i += stride) {
        acc += input[i];
    }
    sAcc[tId] = acc;
    __syncthreads();

    for (int offset = bDim / 2; offset > 0; offset >>= 1) {
        if (tId < offset) {
            sAcc[tId] += sAcc[tId + offset];
        }
        __syncthreads();
    }

    if (tId == 0) {
        atomicAdd(res_sum, sAcc[0]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {  
    cudaMemset(output, 0, sizeof(float));
    cudaDeviceSynchronize();

    int numBlock = (N + TPB - 1) / TPB;
    numBlock = numBlock > 65535 ? 65535 : numBlock;
    reduce_sum<<<numBlock, TPB>>>(input, output, N);
    cudaDeviceSynchronize();
}
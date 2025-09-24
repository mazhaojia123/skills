#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// Threads per block
#define TPB 256

// Per-block reduction to compute local maxima
__global__ void block_reduce_max(const float *input, float *blockMax, int n) {
    int tId = threadIdx.x;
    int bId = blockIdx.x;
    int totalThreads = blockDim.x * gridDim.x;
    __shared__ float sMax[TPB];

    int cg = (n + totalThreads - 1) / totalThreads;
    sMax[tId] = -FLT_MAX;
    float curMax = -FLT_MAX;
    for (int cg0 = 0; cg0 < cg; cg0++) {
        int idx = cg0 * totalThreads + bId * blockDim.x + tId;
        if (idx < n) {
            float cur = input[idx];
            curMax = curMax > cur ? curMax : cur;
        }
    }

    sMax[tId] = curMax;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tId < s) {
            float rCurMax = sMax[tId];
            float rCur = sMax[tId + s];
            sMax[tId] = rCurMax > rCur ? rCurMax : rCur;
        }
        __syncthreads();
    }

    if (tId == 0) {
        blockMax[bId] = sMax[0];
    }
}

// Final reduction to get global max from per-block maxima
__global__ void finalize_reduce_max(const float *blockMax, float *maxOut, int cnb) {
    int tId = threadIdx.x;
    __shared__ float sMax[TPB];

    float localMax = -FLT_MAX;
    for (int i = tId; i < cnb; i += blockDim.x) {
        float v = blockMax[i];
        localMax = localMax > v ? localMax : v;
    }
    sMax[tId] = localMax;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tId < s) {
            float a = sMax[tId];
            float b = sMax[tId + s];
            sMax[tId] = a > b ? a : b;
        }
        __syncthreads();
    }

    if (tId == 0) maxOut[0] = sMax[0];
}

// Compute exp(x - max) into output and per-block partial sums
__global__ void exp_and_partial_sum(const float *input, float *output, const float *theMax, float *partial, int n) {
    int tId = threadIdx.x;
    int bId = blockIdx.x;
    int globalTid = bId * blockDim.x + tId;

    __shared__ float sAcc[TPB];
    float acc = 0.0f;

    int totalThreads = blockDim.x * gridDim.x;
    for (int idx = globalTid; idx < n; idx += totalThreads) {
        float e = expf(input[idx] - *theMax);
        output[idx] = e;
        acc += e;
    }

    sAcc[tId] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tId < s) sAcc[tId] += sAcc[tId + s];
        __syncthreads();
    }

    if (tId == 0) partial[bId] = sAcc[0];
}

// Reduce per-block partial sums to a single total sum
__global__ void finalize_reduce_sum(const float *partial, float *sumOut, int cnb) {
    int tId = threadIdx.x;
    __shared__ float sSum[TPB];

    float local = 0.0f;
    for (int i = tId; i < cnb; i += blockDim.x) local += partial[i];
    sSum[tId] = local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tId < s) sSum[tId] += sSum[tId + s];
        __syncthreads();
    }

    if (tId == 0) sumOut[0] = sSum[0];
}

// Normalize output by total sum
__global__ void normalize(float *output, const float *sum, int n) {
    int totalThreads = blockDim.x * gridDim.x;
    int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = globalTid; idx < n; idx += totalThreads) {
        output[idx] = output[idx] / (*sum);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    const int threadsPerBlock = TPB;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *dBlockMax = nullptr;
    float *dMax = nullptr;
    float *dPartialSums = nullptr;
    float *dTotalSum = nullptr;

    cudaMalloc(&dBlockMax, blocksPerGrid * sizeof(float));
    cudaMalloc(&dMax, sizeof(float));
    cudaMalloc(&dPartialSums, blocksPerGrid * sizeof(float));
    cudaMalloc(&dTotalSum, sizeof(float));

    // 1) per-block max -> final max
    block_reduce_max<<<blocksPerGrid, threadsPerBlock>>>(input, dBlockMax, N);
    finalize_reduce_max<<<1, threadsPerBlock>>>(dBlockMax, dMax, blocksPerGrid);

    // 2) compute exp and per-block sums
    exp_and_partial_sum<<<blocksPerGrid, threadsPerBlock>>>(input, output, dMax, dPartialSums, N);

    // 3) reduce sums to total
    finalize_reduce_sum<<<1, threadsPerBlock>>>(dPartialSums, dTotalSum, blocksPerGrid);

    // 4) normalize
    normalize<<<blocksPerGrid, threadsPerBlock>>>(output, dTotalSum, N);

    cudaDeviceSynchronize();

    cudaFree(dBlockMax);
    cudaFree(dMax);
    cudaFree(dPartialSums);
    cudaFree(dTotalSum);
}

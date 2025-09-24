#include <cuda_runtime.h>

// (kR, kC) : (kC, 1) : (kr0, kc0)
__constant__ float cKernel[32 * 32];

__global__ void conv2d(const float *in, float *out,
                       int inR, int inC, int kR, int kC, int outR, int outC)
{

    // 0) load to smem
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bdimx = blockDim.x;
    int bdimy = blockDim.y;
    int smemRows = bdimy + kR - 1;
    int smemCols = bdimx + kC - 1;
    int smemColsPad = ((32 - (smemCols & 31)) & 31) + smemCols;

    // smem: (smemRows, smemColsPad): (smemColsPad, 1): (ty, tx)
    extern __shared__ float smem[];

    // in (gmem): (inR, inC): (inC, 1): (by*bdimy+ty, bx*bdimx+tx)
    for (int ty = threadIdx.y; ty < smemRows; ty += bdimy)
    {
        int sidx0 = ty * smemColsPad;
        int gx0 = bx * bdimx;
        int gy = by * bdimy + ty;
        for (int tx = threadIdx.x; tx < smemCols; tx += bdimx)
        {
            int gx = gx0 + tx;
            int sidx = sidx0 + tx;
            int gidx = gy * inC + gx;
            if (gy < inR && gx < inC)
                smem[sidx] = in[gidx];
            else 
                smem[sidx] = 0.0f;
        }
    }

    __syncthreads();

    // 1) accumulate
    // out (gmem)
    // (outR, outC): (outC, 1): (gIdY, gIdX)
    int tIdX = threadIdx.x;
    int tIdY = threadIdx.y;
    int gIdX = bx * bdimx + threadIdx.x;
    int gIdY = by * bdimy + threadIdx.y;

    if (gIdX >= outC || gIdY >= outR)
        return;

    float acc = 0.0f;
    #pragma unroll
    for (int kr0 = 0; kr0 < kR; kr0++)
    {
        #pragma unroll
        for (int kc0 = 0; kc0 < kC; kc0++)
        {
            acc += smem[(kr0 + tIdY) * smemColsPad + kc0 + tIdX] * cKernel[kr0 * kC + kc0];
        }
    }
    out[gIdY * outC + gIdX] = acc;
}

// input, kernel, output are device pointers
extern "C" void solve(const float *input, const float *kernel, float *output,
                      int input_rows, int input_cols, int kernel_rows, int kernel_cols)
{

    const int out_rows = input_rows - kernel_rows + 1;
    const int out_cols = input_cols - kernel_cols + 1;

    const size_t kBytes = (size_t)kernel_rows * kernel_cols * sizeof(float);
    cudaMemcpyToSymbol(cKernel, kernel, kBytes, 0, cudaMemcpyDeviceToDevice);

    // 1 个 block 负责算 16x16 个输出
    dim3 blockDim(16, 16);
    dim3 numBlock(
        (out_cols + blockDim.x - 1) / blockDim.x,
        (out_rows + blockDim.y - 1) / blockDim.y);

    int smemRows = blockDim.y + kernel_rows - 1;
    int smemCols = blockDim.x + kernel_cols - 1;
    int smemColsPad = ((32 - (smemCols & 31)) & 31) + smemCols;
    int smemBytes = smemColsPad * smemRows * sizeof(float);

    conv2d<<<numBlock, blockDim, smemBytes>>>(
        input, output, input_rows, input_cols, kernel_rows, kernel_cols, out_rows, out_cols);

    cudaDeviceSynchronize();
}
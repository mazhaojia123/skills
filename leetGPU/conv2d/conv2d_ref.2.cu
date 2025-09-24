#include <cuda_runtime.h>

// 题面约束 kernel <= 31x31，这里给 32x32 的上限
__constant__ float cKernel[32 * 32];

static __global__ void conv2d_tiled_kernel(const float* __restrict__ in,
                                           float* __restrict__ out,
                                           int inR, int inC,
                                           int kR,  int kC,
                                           int outR, int outC)
{
    // ---- 计算共享内存 tile 尺寸（含 halo）与行对齐填充
    const int tileW = blockDim.x + kC - 1;
    const int tileH = blockDim.y + kR - 1;
    const int pad   = (32 - (tileW & 31)) & 31;   // 行宽对齐到 32，避开 bank 冲突
    const int pitch = tileW + pad;

    extern __shared__ float smem[]; // 大小 = pitch * tileH * sizeof(float)

    const int ox0 = blockIdx.x * blockDim.x; // 该 block 输出起点（左上角）
    const int oy0 = blockIdx.y * blockDim.y;

    // ---- 协作把输入 tile 搬到共享内存（越界补 0）
    for (int ty = threadIdx.y; ty < tileH; ty += blockDim.y) {
        const int gy = oy0 + ty;
        const int sbase = ty * pitch;
        const int gbase = gy * inC + ox0;
        for (int tx = threadIdx.x; tx < tileW; tx += blockDim.x) {
            const int gx = ox0 + tx;
            smem[sbase + tx] = (gy < inR && gx < inC) ? in[gbase + tx] : 0.0f;
        }
    }
    __syncthreads();

    // ---- 计算每个线程对应的输出像素
    const int ox = ox0 + threadIdx.x;
    const int oy = oy0 + threadIdx.y;
    if (ox >= outC || oy >= outR) return;

    float acc = 0.f;
    #pragma unroll
    for (int ky = 0; ky < kR; ++ky) {
        const int srow = (threadIdx.y + ky) * pitch + threadIdx.x;
        const int krow = ky * kC;
        #pragma unroll
        for (int kx = 0; kx < kC; ++kx) {
            acc += smem[srow + kx] * cKernel[krow + kx]; // valid：不翻转 kernel
        }
    }
    out[oy * outC + ox] = acc;
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_rows, int input_cols, int kernel_rows, int kernel_cols)
{
    const int out_rows = input_rows - kernel_rows + 1;
    const int out_cols = input_cols - kernel_cols + 1;
    if (out_rows <= 0 || out_cols <= 0) return;

    // 1) 把 kernel 复制到常量内存（D2D，因为 kernel 已在 device）
    const size_t kBytes = (size_t)kernel_rows * kernel_cols * sizeof(float);
    cudaMemcpyToSymbol(cKernel, kernel, kBytes, 0, cudaMemcpyDeviceToDevice);

    // 2) 选择 block / grid
    dim3 block(16, 16); // x 方向保证合并访问；可调 (32,8)、(8,32) 做试验
    dim3 grid((out_cols + block.x - 1) / block.x,
              (out_rows + block.y - 1) / block.y);

    // 3) 动态共享内存大小（含对齐 padding）
    const int tileW = block.x + kernel_cols - 1;
    const int tileH = block.y + kernel_rows - 1;
    const int pad   = (32 - (tileW & 31)) & 31;
    const size_t smemBytes = (size_t)(tileW + pad) * tileH * sizeof(float);

    // 4) launch
    conv2d_tiled_kernel<<<grid, block, smemBytes>>>(
        input, output,
        input_rows, input_cols,
        kernel_rows, kernel_cols,
        out_rows, out_cols
    );
    cudaDeviceSynchronize();
}

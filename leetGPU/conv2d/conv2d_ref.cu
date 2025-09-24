#include <cuda_runtime.h>

static __global__ void conv2d_valid_kernel(const float* __restrict__ input,
                                           const float* __restrict__ kernel,
                                           float* __restrict__ output,
                                           int in_rows, int in_cols,
                                           int k_rows, int k_cols,
                                           int out_rows, int out_cols) {
    int oy = blockIdx.y * blockDim.y + threadIdx.y; // 输出行
    int ox = blockIdx.x * blockDim.x + threadIdx.x; // 输出列
    if (oy >= out_rows || ox >= out_cols) return;

    float acc = 0.0f;
    // 有效(valid)卷积：不翻转kernel，只在完全覆盖时计算
    for (int ky = 0; ky < k_rows; ++ky) {
        int in_r = oy + ky;
        int in_base = in_r * in_cols + ox;     // 这一行窗口的起点
        int k_base  = ky * k_cols;
        #pragma unroll
        for (int kx = 0; kx < k_cols; ++kx) {
            acc += input[in_base + kx] * kernel[k_base + kx];
        }
    }
    output[oy * out_cols + ox] = acc;
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {

    const int out_rows = input_rows - kernel_rows + 1;
    const int out_cols = input_cols - kernel_cols + 1;
    if (out_rows <= 0 || out_cols <= 0) return; // 题面已保证不会发生，这里稳一手

    dim3 block(16, 16);
    dim3 grid((out_cols + block.x - 1) / block.x,
              (out_rows + block.y - 1) / block.y);

    conv2d_valid_kernel<<<grid, block>>>(input, kernel, output,
                                         input_rows, input_cols,
                                         kernel_rows, kernel_cols,
                                         out_rows, out_cols);
    cudaDeviceSynchronize();
}

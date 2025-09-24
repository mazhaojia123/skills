#pragma once
#include <cuda_runtime.h>

// Basic GEMV kernel - one thread per output element
// y = alpha * A * x + beta * y
// A: m x n matrix (row-major)
// x: n x 1 vector
// y: m x 1 vector

__global__ void gemv_kernel_v0(
    int m, int n,
    float alpha,
    const float* A, int lda,
    const float* x, int incx,
    float beta,
    float* y, int incy
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m) {
        float sum = 0.0f;
        for (int col = 0; col < n; col++) {
            sum += A[row * lda + col] * x[col * incx];
        }
        y[row * incy] = alpha * sum + beta * y[row * incy];
    }
}

// CUDA wrapper function for kernel v0
void launch_gemv_v0(
    int m, int n,
    float alpha,
    const float* d_A, int lda,
    const float* d_x, int incx,
    float beta,
    float* d_y, int incy
) {
    const int block_size = 256;
    const int grid_size = (m + block_size - 1) / block_size;
    
    gemv_kernel_v0<<<grid_size, block_size>>>(
        m, n, alpha, d_A, lda, d_x, incx, beta, d_y, incy
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error in v0: %s\n", cudaGetErrorString(err));
    }
}
#pragma once
#include <cuda_runtime.h>

// Optimized GEMV kernel with shared memory
// y = alpha * A * x + beta * y
// A: m x n matrix (row-major)
// x: n x 1 vector
// y: m x 1 vector

#define TILE_SIZE 256

__global__ void gemv_kernel_v1(
    int m, int n,
    float alpha,
    const float* A, int lda,
    const float* x, int incx,
    float beta,
    float* y, int incy
) {
    __shared__ float shared_x[TILE_SIZE];
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    float sum = 0.0f;
    
    // Process x vector in tiles
    for (int tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
        // Load a tile of x into shared memory
        int x_idx = tile_start + tid;
        if (x_idx < n) {
            shared_x[tid] = x[x_idx * incx];
        } else {
            shared_x[tid] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product for this tile
        if (row < m) {
            int tile_end = min(tile_start + TILE_SIZE, n);
            for (int col = tile_start; col < tile_end; col++) {
                int shared_idx = col - tile_start;
                sum += A[row * lda + col] * shared_x[shared_idx];
            }
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < m) {
        y[row * incy] = alpha * sum + beta * y[row * incy];
    }
}

// CUDA wrapper function for kernel v1
void launch_gemv_v1(
    int m, int n,
    float alpha,
    const float* d_A, int lda,
    const float* d_x, int incx,
    float beta,
    float* d_y, int incy
) {
    const int block_size = TILE_SIZE;
    const int grid_size = (m + block_size - 1) / block_size;
    
    gemv_kernel_v1<<<grid_size, block_size>>>(
        m, n, alpha, d_A, lda, d_x, incx, beta, d_y, incy
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error in v1: %s\n", cudaGetErrorString(err));
    }
}

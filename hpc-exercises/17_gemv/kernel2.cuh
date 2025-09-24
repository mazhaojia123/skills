#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// Vectorized GEMV kernel using float4 loads
// y = alpha * A * x + beta * y
// A: m x n matrix (row-major)
// x: n x 1 vector
// y: m x 1 vector

__global__ void gemv_kernel_v2(
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
        
        // Process 4 elements at a time when possible
        int col = 0;
        const float* A_row = &A[row * lda];
        
        // Vectorized loads for aligned data
        if (n >= 4 && ((uintptr_t)A_row % 16 == 0) && ((uintptr_t)x % 16 == 0)) {
            for (; col <= n - 4; col += 4) {
                float4 a_vec = *reinterpret_cast<const float4*>(&A_row[col]);
                float4 x_vec = *reinterpret_cast<const float4*>(&x[col * incx]);
                
                sum += a_vec.x * x_vec.x;
                sum += a_vec.y * x_vec.y;
                sum += a_vec.z * x_vec.z;
                sum += a_vec.w * x_vec.w;
            }
        }
        
        // Handle remaining elements
        for (; col < n; col++) {
            sum += A_row[col] * x[col * incx];
        }
        
        y[row * incy] = alpha * sum + beta * y[row * incy];
    }
}

// Alternative implementation using warp-level reduction
__global__ void gemv_kernel_v2_warp_reduce(
    int m, int n,
    float alpha,
    const float* A, int lda,
    const float* x, int incx,
    float beta,
    float* y, int incy
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_size = 32;
    
    if (row < m) {
        float sum = 0.0f;
        
        // Each thread in the warp processes different columns
        for (int col = tid; col < n; col += blockDim.x) {
            sum += A[row * lda + col] * x[col * incx];
        }
        
        // Warp-level reduction
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        // Only the first thread writes the result
        if (tid == 0) {
            y[row * incy] = alpha * sum + beta * y[row * incy];
        }
    }
}

// CUDA wrapper function for kernel v2
void launch_gemv_v2(
    int m, int n,
    float alpha,
    const float* d_A, int lda,
    const float* d_x, int incx,
    float beta,
    float* d_y, int incy
) {
    // Choose kernel based on problem size
    if (n >= 1024) {
        // Use warp reduction for large n
        const int block_size = 32;  // One warp per block
        const int grid_size = m;    // One block per row
        
        gemv_kernel_v2_warp_reduce<<<grid_size, block_size>>>(
            m, n, alpha, d_A, lda, d_x, incx, beta, d_y, incy
        );
    } else {
        // Use vectorized loads for smaller n
        const int block_size = 256;
        const int grid_size = (m + block_size - 1) / block_size;
        
        gemv_kernel_v2<<<grid_size, block_size>>>(
            m, n, alpha, d_A, lda, d_x, incx, beta, d_y, incy
        );
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error in v2: %s\n", cudaGetErrorString(err));
    }
}

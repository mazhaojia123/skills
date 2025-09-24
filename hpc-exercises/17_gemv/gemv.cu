#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sys/time.h>
#include "kernel0.cuh"
#include "kernel1.cuh"
#include "kernel2.cuh"

// Default problem size
#define DEFAULT_M 4096
#define DEFAULT_N 4096

// CUDA error checking macro
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// CUBLAS error checking macro
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "CUBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Timer utility
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// CPU reference implementation of GEMV
// y = alpha * A * x + beta * y
void cpu_gemv_reference(
    int m, int n,
    float alpha,
    const float* A, int lda,
    const float* x, int incx,
    float beta,
    float* y, int incy
) {
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += A[i * lda + j] * x[j * incx];
        }
        y[i * incy] = alpha * sum + beta * y[i * incy];
    }
}

// Initialize matrix with random values
void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;  // [-1, 1]
    }
}

// Initialize vector with random values
void init_vector(float* vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;  // [-1, 1]
    }
}

// Verify results by comparing with reference
bool verify_result(const float* result, const float* reference, int size, float tolerance = 1e-5f) {
    float max_error = 0.0f;
    float max_ref = 0.0f;
    
    for (int i = 0; i < size; i++) {
        float error = fabs(result[i] - reference[i]);
        float ref_val = fabs(reference[i]);
        
        if (error > max_error) max_error = error;
        if (ref_val > max_ref) max_ref = ref_val;
    }
    
    float relative_error = max_error / (max_ref + 1e-10f);
    printf("Max absolute error: %e, Max relative error: %e\n", max_error, relative_error);
    
    return relative_error < tolerance;
}

// Performance benchmark
double benchmark_kernel(
    void (*kernel_launcher)(int, int, float, const float*, int, const float*, int, float, float*, int),
    int m, int n,
    float alpha,
    const float* d_A, int lda,
    const float* d_x, int incx,
    float beta,
    float* d_y, int incy,
    int warmup_runs = 5,
    int benchmark_runs = 100
) {
    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        kernel_launcher(m, n, alpha, d_A, lda, d_x, incx, beta, d_y, incy);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark
    double start_time = get_time();
    for (int i = 0; i < benchmark_runs; i++) {
        kernel_launcher(m, n, alpha, d_A, lda, d_x, incx, beta, d_y, incy);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    double end_time = get_time();
    
    return (end_time - start_time) / benchmark_runs;
}

int main(int argc, char** argv) {
    // Parse command line arguments
    int m = DEFAULT_M;
    int n = DEFAULT_N;
    if (argc >= 3) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
    }
    
    printf("=== GEMV Performance Test ===\n");
    printf("Matrix size: %d x %d\n", m, n);
    printf("Vector size: %d\n", n);
    printf("Output size: %d\n", m);
    
    // Parameters
    float alpha = 1.0f;
    float beta = 0.0f;
    int lda = n;  // Row-major storage
    int incx = 1;
    int incy = 1;
    
    // Allocate host memory
    float* h_A = (float*)malloc(m * n * sizeof(float));
    float* h_x = (float*)malloc(n * sizeof(float));
    float* h_y = (float*)malloc(m * sizeof(float));
    float* h_y_ref = (float*)malloc(m * sizeof(float));
    float* h_y_gpu = (float*)malloc(m * sizeof(float));
    
    if (!h_A || !h_x || !h_y || !h_y_ref || !h_y_gpu) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    
    // Initialize data
    srand(42);  // For reproducible results
    init_matrix(h_A, m, n);
    init_vector(h_x, n);
    init_vector(h_y, m);
    memcpy(h_y_ref, h_y, m * sizeof(float));
    
    // Allocate device memory
    float *d_A, *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_A, m * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, m * sizeof(float)));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, m * sizeof(float), cudaMemcpyHostToDevice));
    
    printf("\n=== Computing Reference (CPU) ===\n");
    double cpu_start = get_time();
    cpu_gemv_reference(m, n, alpha, h_A, lda, h_x, incx, beta, h_y_ref, incy);
    double cpu_time = get_time() - cpu_start;
    printf("CPU time: %.6f seconds\n", cpu_time);
    
    // Calculate theoretical FLOPS and memory bandwidth
    long long flops = 2LL * m * n;  // Each element requires n multiply-adds
    long long bytes = (long long)(m * n + n + m) * sizeof(float);  // A + x + y
    double cpu_gflops = flops / (cpu_time * 1e9);
    double cpu_bandwidth = bytes / (cpu_time * 1e9);  // GB/s
    printf("CPU performance: %.2f GFLOPS, %.2f GB/s\n", cpu_gflops, cpu_bandwidth);
    
    printf("\n=== Testing CUDA Kernels ===\n");
    
    // Test cuBLAS reference
    printf("\n--- cuBLAS Reference ---\n");
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    // Reset device memory
    CHECK_CUDA(cudaMemcpy(d_y, h_y, m * sizeof(float), cudaMemcpyHostToDevice));
    
    double cublas_time = 0.0;
    {
        // Warmup
        for (int i = 0; i < 5; i++) {
            // For row-major A (m x n), we compute y = alpha * A^T * x + beta * y
            // Since A is row-major but cuBLAS expects column-major, we use CUBLAS_OP_T
            CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_T, n, m, &alpha, d_A, n, d_x, incx, &beta, d_y, incy));
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Benchmark
        double start_time = get_time();
        for (int i = 0; i < 10; i++) {
            // For row-major A (m x n), we compute y = alpha * A^T * x + beta * y
            // Since A is row-major but cuBLAS expects column-major, we use CUBLAS_OP_T
            CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_T, n, m, &alpha, d_A, n, d_x, incx, &beta, d_y, incy));
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        cublas_time = (get_time() - start_time) / 10.0;
    }
    
    CHECK_CUDA(cudaMemcpy(h_y_gpu, d_y, m * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("cuBLAS time: %.6f seconds\n", cublas_time);
    double cublas_gflops = flops / (cublas_time * 1e9);
    double cublas_bandwidth = bytes / (cublas_time * 1e9);
    printf("cuBLAS performance: %.2f GFLOPS, %.2f GB/s\n", cublas_gflops, cublas_bandwidth);
    printf("cuBLAS speedup vs CPU: %.2fx\n", cpu_time / cublas_time);
    
    if (verify_result(h_y_gpu, h_y_ref, m)) {
        printf("✓ cuBLAS result verified\n");
    } else {
        printf("✗ cuBLAS result verification failed\n");
    }
    
    // Test custom kernels
    printf("\n--- Custom Kernel v0 (Basic) ---\n");
    // Reset device memory
    CHECK_CUDA(cudaMemcpy(d_y, h_y, m * sizeof(float), cudaMemcpyHostToDevice));
    
    double kernel_v0_time = benchmark_kernel(launch_gemv_v0, m, n, alpha, d_A, lda, d_x, incx, beta, d_y, incy);
    
    CHECK_CUDA(cudaMemcpy(h_y_gpu, d_y, m * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("Kernel v0 time: %.6f seconds\n", kernel_v0_time);
    double kernel_v0_gflops = flops / (kernel_v0_time * 1e9);
    double kernel_v0_bandwidth = bytes / (kernel_v0_time * 1e9);
    printf("Kernel v0 performance: %.2f GFLOPS, %.2f GB/s\n", kernel_v0_gflops, kernel_v0_bandwidth);
    printf("Kernel v0 speedup vs CPU: %.2fx\n", cpu_time / kernel_v0_time);
    printf("Kernel v0 efficiency vs cuBLAS: %.2f%%\n", (kernel_v0_bandwidth / cublas_bandwidth) * 100.0);
    
    if (verify_result(h_y_gpu, h_y_ref, m)) {
        printf("✓ Kernel v0 result verified\n");
    } else {
        printf("✗ Kernel v0 result verification failed\n");
    }
    
    // Test kernel v1
    printf("\n--- Custom Kernel v1 (Shared Memory) ---\n");
    // Reset device memory
    CHECK_CUDA(cudaMemcpy(d_y, h_y, m * sizeof(float), cudaMemcpyHostToDevice));
    
    double kernel_v1_time = benchmark_kernel(launch_gemv_v1, m, n, alpha, d_A, lda, d_x, incx, beta, d_y, incy);
    
    CHECK_CUDA(cudaMemcpy(h_y_gpu, d_y, m * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("Kernel v1 time: %.6f seconds\n", kernel_v1_time);
    double kernel_v1_gflops = flops / (kernel_v1_time * 1e9);
    double kernel_v1_bandwidth = bytes / (kernel_v1_time * 1e9);
    printf("Kernel v1 performance: %.2f GFLOPS, %.2f GB/s\n", kernel_v1_gflops, kernel_v1_bandwidth);
    printf("Kernel v1 speedup vs CPU: %.2fx\n", cpu_time / kernel_v1_time);
    printf("Kernel v1 efficiency vs cuBLAS: %.2f%%\n", (kernel_v1_bandwidth / cublas_bandwidth) * 100.0);
    
    if (verify_result(h_y_gpu, h_y_ref, m)) {
        printf("✓ Kernel v1 result verified\n");
    } else {
        printf("✗ Kernel v1 result verification failed\n");
    }
    
    // Test kernel v2
    printf("\n--- Custom Kernel v2 (Vectorized) ---\n");
    // Reset device memory
    CHECK_CUDA(cudaMemcpy(d_y, h_y, m * sizeof(float), cudaMemcpyHostToDevice));
    
    double kernel_v2_time = benchmark_kernel(launch_gemv_v2, m, n, alpha, d_A, lda, d_x, incx, beta, d_y, incy);
    
    CHECK_CUDA(cudaMemcpy(h_y_gpu, d_y, m * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("Kernel v2 time: %.6f seconds\n", kernel_v2_time);
    double kernel_v2_gflops = flops / (kernel_v2_time * 1e9);
    double kernel_v2_bandwidth = bytes / (kernel_v2_time * 1e9);
    printf("Kernel v2 performance: %.2f GFLOPS, %.2f GB/s\n", kernel_v2_gflops, kernel_v2_bandwidth);
    printf("Kernel v2 speedup vs CPU: %.2fx\n", cpu_time / kernel_v2_time);
    printf("Kernel v2 efficiency vs cuBLAS: %.2f%%\n", (kernel_v2_bandwidth / cublas_bandwidth) * 100.0);
    
    if (verify_result(h_y_gpu, h_y_ref, m)) {
        printf("✓ Kernel v2 result verified\n");
    } else {
        printf("✗ Kernel v2 result verification failed\n");
    }
    
    // Summary
    printf("\n=== Performance Summary ===\n");
    printf("%-20s %12s %12s %12s %12s\n", "Implementation", "Time (s)", "GFLOPS", "GB/s", "Speedup");
    printf("%-20s %12.6f %12.2f %12.2f %12.2fx\n", "CPU Reference", cpu_time, cpu_gflops, cpu_bandwidth, 1.0);
    printf("%-20s %12.6f %12.2f %12.2f %12.2fx\n", "cuBLAS", cublas_time, cublas_gflops, cublas_bandwidth, cpu_time / cublas_time);
    printf("%-20s %12.6f %12.2f %12.2f %12.2fx\n", "Custom Kernel v0", kernel_v0_time, kernel_v0_gflops, kernel_v0_bandwidth, cpu_time / kernel_v0_time);
    printf("%-20s %12.6f %12.2f %12.2f %12.2fx\n", "Custom Kernel v1", kernel_v1_time, kernel_v1_gflops, kernel_v1_bandwidth, cpu_time / kernel_v1_time);
    printf("%-20s %12.6f %12.2f %12.2f %12.2fx\n", "Custom Kernel v2", kernel_v2_time, kernel_v2_gflops, kernel_v2_bandwidth, cpu_time / kernel_v2_time);
    
    // Cleanup
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    
    free(h_A);
    free(h_x);
    free(h_y);
    free(h_y_ref);
    free(h_y_gpu);
    
    printf("\n=== Test completed successfully ===\n");
    return EXIT_SUCCESS;
}

// All kernel implementations are now in separate header files:
// - kernel0.cuh: Basic implementation
// - kernel1.cuh: Shared memory optimization
// - kernel2.cuh: Vectorized loads and warp reduction

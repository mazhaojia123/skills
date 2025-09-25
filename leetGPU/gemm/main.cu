#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(expr)                                                          \
    do {                                                                          \
        cudaError_t err__ = (expr);                                               \
        if (err__ != cudaSuccess) {                                               \
            std::cerr << "CUDA error " << cudaGetErrorString(err__)              \
                      << " (" << static_cast<int>(err__) << ") at "               \
                      << __FILE__ << ":" << __LINE__ << std::endl;                \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

#define solve solve_v1
#define bench bench_v1
#include "gemm_v1.cuh"
#undef solve
#undef bench
#undef WM
#undef WN
#undef WK

#define solve solve_v2
#define bench bench_v2
#include "gemm_v2.cuh"
#undef solve
#undef bench
#undef CEIL_DIV

using BenchFn = float (*)(const half*, const half*, half*, int, int, int, float, float);

float benchmark_kernel(const char* name,
                       BenchFn bench,
                       const half* dA,
                       const half* dB,
                       half* dC,
                       int M,
                       int N,
                       int K,
                       float alpha,
                       float beta,
                       int warmup = 5,
                       int runs = 20) {
    const size_t bytesC = static_cast<size_t>(M) * N * sizeof(half);

    for (int i = 0; i < warmup; ++i) {
        CUDA_CHECK(cudaMemset(dC, 0, bytesC));
        float ms = bench(dA, dB, dC, M, N, K, alpha, beta);
        if (ms < 0.0f) {
            int code = static_cast<int>(-ms);
            if (code <= 0) {
                std::cerr << name << " warmup failed with unknown CUDA error" << std::endl;
            } else {
                std::cerr << name << " warmup failed with CUDA error "
                          << cudaGetErrorString(static_cast<cudaError_t>(code))
                          << std::endl;
            }
            std::exit(EXIT_FAILURE);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float total_ms = 0.0f;
    for (int i = 0; i < runs; ++i) {
        CUDA_CHECK(cudaMemset(dC, 0, bytesC));
        float ms = bench(dA, dB, dC, M, N, K, alpha, beta);
        if (ms < 0.0f) {
            int code = static_cast<int>(-ms);
            if (code <= 0) {
                std::cerr << name << " run failed with unknown CUDA error" << std::endl;
            } else {
                std::cerr << name << " run failed with CUDA error "
                          << cudaGetErrorString(static_cast<cudaError_t>(code))
                          << std::endl;
            }
            std::exit(EXIT_FAILURE);
        }
        total_ms += ms;
    }

    float avg_ms = total_ms / runs;
    double flops = 2.0 * static_cast<double>(M) * N * K;
    double gflops = flops / (avg_ms * 1e-3) / 1e9;

    std::cout << std::fixed << std::setprecision(3)
              << name << " kernel avg " << avg_ms << " ms, "
              << gflops << " GFLOP/s" << std::defaultfloat << std::endl;
    return static_cast<float>(gflops);
}

void fill_random(std::vector<half>& data, std::mt19937& gen, float low, float high) {
    std::uniform_real_distribution<float> dist(low, high);
    for (auto& v : data) {
        v = __float2half(dist(gen));
    }
}

void report_diff(const std::vector<half>& ref, const std::vector<half>& other) {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        float rf = __half2float(ref[i]);
        float of = __half2float(other[i]);
        float abs_diff = std::fabs(rf - of);
        max_abs = std::max(max_abs, abs_diff);
        float denom = std::max(std::fabs(rf), 1e-6f);
        max_rel = std::max(max_rel, abs_diff / denom);
    }
    std::cout << "Output diff (v2 vs v1) max_abs=" << max_abs
              << ", max_rel=" << max_rel << std::endl;
}

int main(int argc, char** argv) {
    int M = 1024;
    int N = 1024;
    int K = 1024;
    if (argc == 4) {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
        K = std::stoi(argv[3]);
    } else if (argc != 1) {
        std::cerr << "Usage: " << argv[0] << " [M N K]" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "GEMM problem: M=" << M << " N=" << N << " K=" << K << std::endl;

    const size_t elemsA = static_cast<size_t>(M) * K;
    const size_t elemsB = static_cast<size_t>(K) * N;
    const size_t elemsC = static_cast<size_t>(M) * N;
    const size_t bytesA = elemsA * sizeof(half);
    const size_t bytesB = elemsB * sizeof(half);
    const size_t bytesC = elemsC * sizeof(half);

    std::vector<half> hA(elemsA);
    std::vector<half> hB(elemsB);
    std::mt19937 gen(42);
    fill_random(hA, gen, -1.0f, 1.0f);
    fill_random(hB, gen, -1.0f, 1.0f);

    half* dA = nullptr;
    half* dB = nullptr;
    half* dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    float gflops_v1 = benchmark_kernel("gemm_v1", bench_v1, dA, dB, dC, M, N, K, alpha, beta);
    std::vector<half> hC_v1(elemsC);
    CUDA_CHECK(cudaMemcpy(hC_v1.data(), dC, bytesC, cudaMemcpyDeviceToHost));

    float gflops_v2 = benchmark_kernel("gemm_v2", bench_v2, dA, dB, dC, M, N, K, alpha, beta);
    std::vector<half> hC_v2(elemsC);
    CUDA_CHECK(cudaMemcpy(hC_v2.data(), dC, bytesC, cudaMemcpyDeviceToHost));

    report_diff(hC_v1, hC_v2);

    std::cout << std::fixed << std::setprecision(2)
              << "gemm_v1: " << gflops_v1 << " GFLOP/s, "
              << "gemm_v2: " << gflops_v2 << " GFLOP/s\n" << std::defaultfloat;

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}

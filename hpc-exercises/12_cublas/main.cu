#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>

#define N 1024

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

static void simple_sgemm(int n, float alpha, float *A, float *B, float beta, float *C) {
	int i, j, k;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			float prod = 0.0f;
			for (k = 0; k < n; ++k) {
				float t_A = A[k*n+i];
				float t_B = B[j*n+k];
				prod = prod + t_A * t_B;
			}
			C[j*n+i] = C[j*n+i] + prod;
		}
	}
}

int main() {
	float *h_A;
	float *h_B;
	float *h_C;
	float *h_C_ref;
	float *d_A = NULL;
	float *d_B = NULL;
	float *d_C = NULL;
	float alpha = 1.0f;
	float beta = 0.0f;
	int n2 = N*N;
	int i;
	float error_norm;
	float ref_norm;
	float diff;
	cublasStatus_t status;
	cublasHandle_t handle;

	// 1. 初始化 cublas
	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "CUBLAS initialization error\n");
		return EXIT_FAILURE;
	}
	printf("1. initialized\n");
	
	// 2. 分配内存
	h_A = reinterpret_cast<float *>(malloc(n2*sizeof(h_A[0])));
	h_B = reinterpret_cast<float *>(malloc(n2*sizeof(h_B[0])));
	h_C = reinterpret_cast<float *>(malloc(n2*sizeof(h_C[0])));
	h_C_ref = reinterpret_cast<float *>(malloc(n2*sizeof(h_C_ref[0])));
	if (h_A==0 || h_B==0 || h_C == 0 || h_C_ref == 0) {
		fprintf(stderr, "Host memory allocation error\n");
		return EXIT_FAILURE;
	}
	for (i = 0; i < n2; i++) {
		h_A[i] = rand() / static_cast<float>(RAND_MAX);
		h_B[i] = rand() / static_cast<float>(RAND_MAX);
		h_C[i] = rand() / static_cast<float>(RAND_MAX);
		h_C_ref[i] = h_C[i];
	}

	gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&d_A), n2*sizeof(d_A[0])));
	gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&d_B), n2*sizeof(d_B[0])));
	gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&d_C), n2*sizeof(d_C[0])));

	status = cublasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "Device copy error\n");
		return EXIT_FAILURE;
	}
	status = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "Device copy error\n");
		return EXIT_FAILURE;
	}
	status = cublasSetVector(n2, sizeof(h_C[0]), h_C, 1, d_C, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "Device copy error\n");
		return EXIT_FAILURE;
	}
	printf("2. allocated memory\n");
	
	// 3. call gemm
	simple_sgemm(N, alpha, h_A, h_B, beta, h_C_ref);
	printf("3. invoked simple.\n");

	status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, 
						  d_A, CUDA_R_32F, N, 
						  d_B, CUDA_R_32F, N, &beta,
						  d_C, CUDA_R_32F, N, 
						  CUDA_R_32F, 
						  CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "Kernel execution error. \n");
		return EXIT_FAILURE;
	}

	status = cublasGetVector(n2, sizeof(h_C[0]), d_C, 1, h_C, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "Device copy error(d_C->h_C) \n");
		return EXIT_FAILURE;
	}
	printf("3. invoked cublas.\n");

	// 4. 验证程序的正确性
	error_norm = 0;
	ref_norm = 0;
	for (i = 0; i < n2; ++i) {
		diff = h_C_ref[i] - h_C[i];
		error_norm += diff * diff;
		ref_norm += h_C_ref[i] * h_C_ref[i];
	}
	error_norm = static_cast<float>(sqrt(static_cast<double>(error_norm)));
	ref_norm = static_cast<float>(sqrt(static_cast<double>(ref_norm)));
	if (fabs(ref_norm) < 1e-7) {
		fprintf(stderr, "!!!! reference norm is 0\n");
		return EXIT_FAILURE;
	}	
	if (error_norm / ref_norm < 1e-2f) {
		printf("simpleCUBLAS test passed.\n");
		exit(EXIT_SUCCESS);
	} else {
		printf("simpleCUBLAS test failed.\n");
		exit(EXIT_FAILURE);
	}
	printf("4. verified\n");
	
	// 5. clean up
	free(h_A);
	free(h_B);
	free(h_C);
	free(h_C_ref);
	gpuErrchk(cudaFree(d_A))
	gpuErrchk(cudaFree(d_B))
	gpuErrchk(cudaFree(d_C))
	status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! shutdown error (A)\n");
		return EXIT_FAILURE;
	}
}
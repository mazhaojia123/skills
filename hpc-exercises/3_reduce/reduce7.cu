// 因为重新 代码逻辑带来了改变，所以我们从新写个文件

// 1. 看 shared memory, 从 32 个 Bytes   一个 Bank 来看
// 2. 看 threads 	  , 从 32 个 threads 一个 warp 来看


#include <stdio.h>
#include <stdlib.h>
// #define DEBUG 1

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void hReduce(float *hA, float *refOut, int size);
void randomize(float *m, int size);
bool check(float *ref, float *m, int size);

template <int blockSize>
__device__ void warpReduce(volatile float *sdata, int tid) {
	if (blockSize >= 64) 
		sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) 
		sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) 
		sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) 
		sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) 
		sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) 
		sdata[tid] += sdata[tid + 1];
}

template <int blockSize, int elem_per_thread>
__global__ void reduce7(float *g_idata, float *g_odata) {
	extern __shared__ float sdata[];
	int tid = threadIdx.x;
	int i = blockIdx.x * (blockDim.x * elem_per_thread) + threadIdx.x;
	sdata[tid] = 0; 

	#pragma unroll
	for (int iter = 0; iter < elem_per_thread; iter++) {
		sdata[tid] += g_idata[i + iter * blockSize];
	}

	__syncthreads();

	if (blockSize >= 512) {
		if (tid < 256) 
			sdata[tid] += sdata[tid + 256];
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128) 
			sdata[tid] += sdata[tid + 128];
		__syncthreads();
	}
	if (blockSize >= 128) {
		if (tid < 64) 
			sdata[tid] += sdata[tid + 64];
		__syncthreads();
	}

	if (tid < 32) warpReduce<blockSize>(sdata, tid);

	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

const int repeat = 32;
const int N = 16*1024*1024;
const int BLOCK_NUM = 1024;
const int THREAD_NUM_PER_BLOCK = 256;
// const int THREAD_NUM_PER_BLOCK = ELEM_NUM_PER_BLOCK / ELEM_NUM_PER_THREAD;
const int ELEM_NUM_PER_BLOCK = N / BLOCK_NUM;
const int ELEM_NUM_PER_THREAD = ELEM_NUM_PER_BLOCK / THREAD_NUM_PER_BLOCK ;

int main() {


	float *hA, *hOut, *refOut;
	float *dA, *dOut; 
	hA = (float*)malloc(N*sizeof(float));
	hOut = (float*)malloc(BLOCK_NUM*sizeof(float));
	refOut = (float*)malloc(BLOCK_NUM*sizeof(float));
	gpuErrchk(cudaMalloc((void**)&dA, N*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dOut, BLOCK_NUM*sizeof(float)));
	randomize(hA, N);
	gpuErrchk(cudaMemcpy(dA, hA, N*sizeof(float), cudaMemcpyHostToDevice));

	hReduce(hA, refOut, N);

	dim3 gridDim(BLOCK_NUM);
	dim3 blockDim(THREAD_NUM_PER_BLOCK);
	reduce7<THREAD_NUM_PER_BLOCK, ELEM_NUM_PER_THREAD><<<gridDim, blockDim, THREAD_NUM_PER_BLOCK*sizeof(float)>>>(dA, dOut);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaMemcpy(hOut, dOut, BLOCK_NUM*sizeof(float), cudaMemcpyDeviceToHost));

	if (check(refOut, hOut, BLOCK_NUM))
		printf("Pass.\n");
	else
		printf("WA.\n");


	// 进行时间的测试
	cudaEvent_t start, stop;
	float elpasedTime = 0.f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	for (int i = 0; i < repeat; i++) {
		reduce7<THREAD_NUM_PER_BLOCK, ELEM_NUM_PER_THREAD><<<gridDim, blockDim, THREAD_NUM_PER_BLOCK*sizeof(float)>>>(dA, dOut);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);	// NOTE: host 会等待时间执行结束
	cudaEventElapsedTime(&elpasedTime, start, stop);
	elpasedTime *= 1e-3; // ms --> s
	float bandwidth = repeat * N * sizeof(float) / elpasedTime * 1e-9;
	printf("elapsed time per kernel: %f s, bandwidth : %.2f GB/s. \n", elpasedTime/repeat, bandwidth);

	return 0;
}

void randomize(float *m, int size) {
	float tmp;
	int lower = 0, upper = 1; 
	static bool hasSrand = false;
	if (!hasSrand) srand(time(NULL));

	for (int i = 0; i < size; i++) {
		tmp = (float)rand() / RAND_MAX * (upper - lower) + lower;
		// tmp = 1;
		// tmp = rand() % 20;
		m[i] = tmp; 
	}

#ifdef DEBUG
	printf("== DEBUG == randomize(float*, int) \n");
	for (int i = 0; i<size && i<20; i++)
		printf("%.2f ", m[i]);
	printf("\n");
#endif
}

bool check(float *ref, float *m, int size) {
	bool res = true;
	for (int i = 0; i < size; i++) {
#ifdef DEBUG
		printf("== DEUBG == ref[%d]=%.2f\tm[%d]=%.2f\n", i, ref[i], i, m[i]);
#endif
		if (ref[i] - m[i] > 1e-1) {
			res = false;
#ifdef DEBUG
			printf("above error\n");
#else
			break;
#endif
		}
	}
	return res;
}

void hReduce(float *hA, float *refOut, int size) {
	for (int i = 0; i < BLOCK_NUM; i++) {
		refOut[i] = 0;
		for (int j = 0; j < ELEM_NUM_PER_BLOCK; j++) {
			refOut[i] += hA[i * ELEM_NUM_PER_BLOCK + j];
		}
	}
}
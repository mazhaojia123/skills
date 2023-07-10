// 因为重新 代码逻辑带来了改变，所以我们从新写个文件

// 1. 看 shared memory, 从 32 个 Bytes   一个 Bank 来看
// 2. 看 threads 	  , 从 32 个 threads 一个 warp 来看


#include <stdio.h>
#include <stdlib.h>
const int NUM_OF_THREADS_PER_BLOCK = 256;
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

__global__ void reduce1(float *g_idata, float *g_odata) {
	extern __shared__ float sdata[];
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	sdata[tid] = g_idata[i];
	__syncthreads();

	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2*s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// #2
__global__ void reduce2(float *g_idata, float *g_odata) {
	extern __shared__ float sdata[];
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	sdata[tid] = g_idata[i];
	__syncthreads();

	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;
		
		// NOTE: 解决 warp divergent
		// 		我们让尽可能一个 warp 中的代码去走相同的分支
		// 		让线程去连续的处理问题
		if (index < blockDim.x) {
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}

	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// #3
__global__ void reduce3(float *g_idata, float *g_odata) {
	extern __shared__ float sdata[];
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	sdata[tid] = g_idata[i];
	__syncthreads();

	// NOTE: 让数据的读取连续，解决在 shared memory 上的 bank conflict
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid+s];
		}
		__syncthreads();
	}

	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// #4
__global__ void reduce4(float *g_idata, float *g_odata) {
	extern __shared__ float sdata[];
	int tid = threadIdx.x;
	// NOTE: 减少 block 的数量
	int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;
	sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid+s];
		}
		__syncthreads();
	}

	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main() {
	const int repeat = 32;
	const int N = 16*1024*1024;

	float *hA, *hOut, *refOut;
	float *dA, *dOut; 
	hA = (float*)malloc(N*sizeof(float));
	hOut = (float*)malloc((N/NUM_OF_THREADS_PER_BLOCK/2)*sizeof(float));
	refOut = (float*)malloc((N/NUM_OF_THREADS_PER_BLOCK/2)*sizeof(float));
	gpuErrchk(cudaMalloc((void**)&dA, N*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dOut, (N/NUM_OF_THREADS_PER_BLOCK)*sizeof(float)/2));
	randomize(hA, N);
	gpuErrchk(cudaMemcpy(dA, hA, N*sizeof(float), cudaMemcpyHostToDevice));

	hReduce(hA, refOut, N);

	dim3 gridDim(N/NUM_OF_THREADS_PER_BLOCK/2);
	dim3 blockDim(NUM_OF_THREADS_PER_BLOCK);
	reduce4<<<gridDim, blockDim, NUM_OF_THREADS_PER_BLOCK*sizeof(float)>>>(dA, dOut);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaMemcpy(hOut, dOut, (N/NUM_OF_THREADS_PER_BLOCK/2)*sizeof(float), cudaMemcpyDeviceToHost));

	if (check(refOut, hOut, (N/NUM_OF_THREADS_PER_BLOCK/2)))
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
		reduce4<<<gridDim, blockDim, NUM_OF_THREADS_PER_BLOCK*sizeof(float)>>>(dA, dOut);
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
		if (ref[i] - m[i] > 1e-2) {
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
	for (int i = 0; i < size/NUM_OF_THREADS_PER_BLOCK/2; i++) {
		refOut[i] = 0;
		for (int j = 0; j < NUM_OF_THREADS_PER_BLOCK * 2; j++) {
			refOut[i] += hA[i * NUM_OF_THREADS_PER_BLOCK * 2 + j];
		}
	}
}
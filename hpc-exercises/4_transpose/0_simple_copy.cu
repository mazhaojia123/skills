#include <stdio.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

#define NUM_REPS 100


void computeTransposeGold(float *transposeGold, float *h_idata, int size_x, int size_y);

int main(int argc, char **argv) {
	const int size_x = 2048, size_y = 2048;

	void (*kernel)(float *, float *, int, int, int);
	char *kernelName;

	dim3 grid(size_x / TILE_DIM, size_y / TILE_DIM), 
		 threads(TILE_DIM, BLOCK_ROWS);

	cudaEvent_t start, stop;
	const int mem_size = sizeof(float) * size_x * size_y;

	float *h_idata = (float*) malloc(mem_size);
	float *h_odata = (float*) malloc(mem_size);
	float *transposeGold = (float*) malloc(mem_size);
	float *gold;	// TODO: 这个做什么的？？？

	float *d_idata, *d_odata;
	cudaMalloc((void**) &d_idata, mem_size);
	cudaMalloc((void**) &d_odata, mem_size);

	for (int i = 0; i < (size_x * size_y); i++)
		h_idata[i] = i; 

	cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice);

	// host kernel
	computeTransposeGold(transposeGold, h_idata, size_x, size_y);

	printf("\nMatrix size: %dx%d, tile: %dx%d, block: %dx%d\n\n", 
			size_x, size_y, TILE_DIM, TILE_DIM, TILE_DIM, BLOCK_ROWS);
	printf("Kernel\t\t\tLoop over kernel\tLoop within kernel\n"); 
	printf("------\t\t\t----------------\t------------------\n");

	// device kernels
	for (int k = 0; k<8; k++) {
		switch (k) {
			case 0: 
				kernel = &copy; 
				kernelName = "simple copy "; 
				break; 
			case 1: 
				kernel = &copySharedMem; 
				kernelName = "shared memory copy "; 
				break; 
			case 2: 
				kernel = &transposeNaive; 
				kernelName = "naive transpose "; 
				break; 
			case 3: 
				kernel = &transposeCoalesced; 
				kernelName = "coalesced transpose "; 
				break; 
			case 4: 
				kernel = &transposeNoBankConflicts; 
				kernelName = "no bank conflict trans"; 
				break; 
			case 5: 
				kernel = &transposeCoarseGrained; 
				kernelName = "coarse-grained "; 
				break; 
			case 6: 
				kernel = &transposeFineGrained; 
				kernelName = "fine-grained "; 
				break; 
			case 7: 
				kernel = &transposeDiagonal; 
				kernelName = "diagonal transpose "; 
				break;
		}
	}

	// cudaEventDestroy(start);
	// cudaEventDestroy(stop);

	cudaFree(d_idata);
	cudaFree(d_odata);
	free(h_idata);
	free(h_odata);
	free(transposeGold);

	return 0;
}
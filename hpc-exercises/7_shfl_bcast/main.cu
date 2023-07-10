#include <stdio.h>

__global__ void bcast(int arg) {
	int laneId = threadIdx.x &0x1f; 
	int value;
	if (laneId == 0) value = arg;
	value = __shfl_sync(0xffffffff, value, 0);
	if (value != arg)
		printf("Thread %d failed. \n", threadIdx.x);
}

int main() {
	bcast<<<1, 32>>>(1234);
	cudaDeviceSynchronize();

	return 0;
}
#include <stdio.h>

__global__ void warpReduce() {
	int lane_id = threadIdx.x & 0x1f;
	int value = 1;
	
	for (int i = 16; i >= 1; i /= 2)
		value += __shfl_xor_sync(0xffffffff, value, i, 32);

	printf("lane_id=%d\tvalue=%d\n", lane_id, value);
}

int main() {
	warpReduce<<<1, 32>>>();
	cudaDeviceSynchronize();
	return 0;
}
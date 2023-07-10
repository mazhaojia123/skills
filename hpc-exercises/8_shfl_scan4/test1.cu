#include <stdio.h>

__global__ void kernel() {
	int laneId = threadIdx.x & 0x1f;
	int value = laneId;

	for (int i = 1; i <= 4; i ++) {
		int n = -1;	
		n = __shfl_up_sync(0xffffffff, value, i, 8);
		printf("i=%d\tlaneId=%d\tn=%d\n", i, laneId, n);
	}
}

int main() {
	kernel<<<1, 32>>>();
	cudaDeviceSynchronize();
	return 0;
}
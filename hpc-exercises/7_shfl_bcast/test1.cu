#include <stdio.h>

__global__ void bcast(int arg) {
	int lane_id = threadIdx.x & 0x1f;
	int n = -1;
	n = __shfl_sync(0xffffffff, lane_id, 0, 8);

	printf("threadIdx=%d\tlane_id=%d\tn=%d\n", threadIdx.x, lane_id, n);
}

int main() {
	bcast<<<1, 48>>>(1234);
	cudaDeviceSynchronize();

	return 0;
}
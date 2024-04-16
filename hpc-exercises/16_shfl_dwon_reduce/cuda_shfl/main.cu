#include <stdio.h>

__global__ void warpReduce(int *out) {
	int lane_id = threadIdx.x & 0x1f;
	int value = 1;
	
	for (int i = 16; i >= 1; i /= 2)
		value += __shfl_down_sync(0xffffffff, value, i, 32);

	out[lane_id]=value;
}

int main() {
	int tmp_h[32];
	int *tmp_d;
	cudaMalloc((void**) &tmp_d, 32*sizeof(int));
	warpReduce<<<1, 32>>>(tmp_d);
	cudaMemcpy(tmp_h, tmp_d, 32*sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	printf("result: %d\n", tmp_h[0]);
	return 0;
}

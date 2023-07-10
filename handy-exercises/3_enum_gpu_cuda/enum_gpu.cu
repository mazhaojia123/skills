#include <stdio.h>

int main() {
	cudaDeviceProp prop;   // NOTE: 我们能够从这个对象里面看到比较多的信息

	int count;
	cudaGetDeviceCount(&count);

	for (int i = 0; i < count; i++) { 
		// 下面分别从整体信息、memory， multi-processor 的三个角度来查看这个设备
		cudaGetDeviceProperties(&prop, i);
		printf("   --- General Information for device %d ---\n", i);
		printf("Name:  %s\n", prop.name);
		printf("Compute capability:  %d.%d\n", prop.major, prop.minor);
		printf("Clock rate:  %d\n", prop.clockRate);
		printf("Device copy overlap:  ");
		if (prop.deviceOverlap)
			printf("Enable\n");
		else 
			printf("Disabled\n");
		printf("Kernel execution timeout:  ");
		if (prop.kernelExecTimeoutEnabled)
			printf("Enable\n");
		else 
			printf("Disabled\n");

		printf("   --- Memory Information for device %d ---\n", i);
		printf("Total global mem:  %ld\n", prop.totalGlobalMem);
		printf("Total const mem:  %ld\n", prop.totalConstMem);

		printf("   --- MP Information for device %d ---\n", i);
		printf("Multiporcessor count %d\n", prop.multiProcessorCount);
		printf("Shared mem per mp:  %ld\n", prop.sharedMemPerBlock);
		printf("Registers per mp: %d\n", prop.regsPerBlock);
		printf("Threads in warp:  %d\n", prop.warpSize);
		printf("Max threads per block:  %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions: (%d, %d, %d)\n", 
				prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);

		printf("Max grid dimensions: (%d, %d, %d)\n", 
				prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

		printf("\n");
	}

	return 0;
}
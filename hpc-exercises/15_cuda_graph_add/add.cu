#include <iostream>
#include <chrono>
#include <vector>
#include "helper_cuda.h"

#define Clock std::chrono::high_resolution_clock

__global__ void addKernel(int *c, const int *a, const int *b, int n)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    c[gtid] = a[gtid] + b[gtid];
}

int main() {
    const int arraySize = 5;
	int size = arraySize;

    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    int threads = 256;
    int blocks = (arraySize + threads - 1) / threads;

    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // For Graph
    cudaStream_t streamForGraph;
    cudaGraph_t graph;
    std::vector<cudaGraphNode_t> nodeDependencies;
    cudaGraphNode_t memcpyNode, kernelNode;
    cudaKernelNodeParams kernelNodeParams = { 0 };
    cudaMemcpy3DParms memcpyParams = { 0 };


	// NOTE: 分配 device memory
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	cudaStatus = cudaMalloc((void**)&dev_c, arraySize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&dev_b, arraySize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	// NOTE: 创建图
    checkCudaErrors(cudaGraphCreate(&graph, 0));
    checkCudaErrors(cudaStreamCreateWithFlags(&streamForGraph, cudaStreamNonBlocking));

	// NOTE: 添加 memcpy 节点
    // Add memcpy nodes for copying input vectors from host memory to GPU buffers
    memset(&memcpyParams, 0, sizeof(memcpyParams));
    memcpyParams.srcArray = NULL;
    memcpyParams.srcPos = make_cudaPos(0, 0, 0);
    memcpyParams.srcPtr = make_cudaPitchedPtr((void*)a, size * sizeof(int), size, 1);
    memcpyParams.dstArray = NULL;
    memcpyParams.dstPos = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr = make_cudaPitchedPtr(dev_a, size * sizeof(float), size, 1);
    memcpyParams.extent = make_cudaExtent(size * sizeof(float), 1, 1);
    memcpyParams.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &memcpyParams));
    nodeDependencies.push_back(memcpyNode);

    memset(&memcpyParams, 0, sizeof(memcpyParams));
    memcpyParams.srcArray = NULL;
    memcpyParams.srcPos = make_cudaPos(0, 0, 0);
    memcpyParams.srcPtr = make_cudaPitchedPtr((void*)b, size * sizeof(int), size, 1);
    memcpyParams.dstArray = NULL;
    memcpyParams.dstPos = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr = make_cudaPitchedPtr(dev_b, size * sizeof(float), size, 1);
    memcpyParams.extent = make_cudaExtent(size * sizeof(float), 1, 1);
    memcpyParams.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &memcpyParams));
    nodeDependencies.push_back(memcpyNode);

	// NOTE: 添加 kenrel 节点
    // Add a kernel node for launching a kernel on the GPU
    memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
    kernelNodeParams.func = (void*)addKernel;
    kernelNodeParams.gridDim = dim3(blocks, 1, 1);
    kernelNodeParams.blockDim = dim3(threads, 1, 1);
    kernelNodeParams.sharedMemBytes = 0;
    void* kernelArgs[4] = { (void*)&dev_c, (void*)&dev_a, (void*)&dev_b, &size };
    kernelNodeParams.kernelParams = kernelArgs;
    kernelNodeParams.extra = NULL;
    checkCudaErrors(cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(), nodeDependencies.size(), &kernelNodeParams));
    nodeDependencies.clear();
    nodeDependencies.push_back(kernelNode);

	// NOTE: 添加 memcpy 节点
    // Add memcpy node for copying output vector from GPU buffers to host memory
    memset(&memcpyParams, 0, sizeof(memcpyParams));
    memcpyParams.srcArray = NULL;
    memcpyParams.srcPos = make_cudaPos(0, 0, 0);
    memcpyParams.srcPtr = make_cudaPitchedPtr(dev_c, size * sizeof(int), size, 1);
    memcpyParams.dstArray = NULL;
    memcpyParams.dstPos = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr = make_cudaPitchedPtr(c, size * sizeof(int), size, 1);
    memcpyParams.extent = make_cudaExtent(size * sizeof(int), 1, 1);
    memcpyParams.kind = cudaMemcpyDeviceToHost;
    checkCudaErrors(cudaGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(), nodeDependencies.size(), &memcpyParams)); 

	// NOTE: 输出有用的信息
	cudaGraphNode_t* nodes = NULL;
	size_t numNodes = 0;
	checkCudaErrors(cudaGraphGetNodes(graph, nodes, &numNodes));
	printf("Num of nodes in the graph created manually = %zu\n", numNodes);

	// NOTE: 初始化图
    // Create an executable graph from a graph
    cudaGraphExec_t graphExec;
    checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    auto t1 = Clock::now();
    for (int i = 0; i < 100; ++i) {
		// NOTE: 在某个流上执行图
        checkCudaErrors(cudaGraphLaunch(graphExec, streamForGraph));
		// NOTE: 同步流
        checkCudaErrors(cudaStreamSynchronize(streamForGraph));
    }
    auto t2 = Clock::now();
    auto us_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    printf("Looped %d time(s) in %lld microseconds\n", 100, us_elapsed.count());

	printf("C:\t");
	for (int i = 0; i < arraySize; i++){
		printf("%d\t", c[i]);
	}
	printf("\n");

    // Clean up
    checkCudaErrors(cudaGraphExecDestroy(graphExec));
    checkCudaErrors(cudaGraphDestroy(graph));
    checkCudaErrors(cudaStreamDestroy(streamForGraph));

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
}
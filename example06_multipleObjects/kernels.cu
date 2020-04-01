
#include "kernels.cuh"

int currDevice = -1;
cudaDeviceProp prop;
int numThreads = 256;

__global__ void fillZeros(float* buf, int size) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = threadID; i < size; i += numThreads)
	{
		buf[i] = 0.0f;
	}
}
void fillWithZeroesKernel(float* buf, int size, cudaStream_t s) {
	

	// TODO: Implement this when there are several GPUs/
	// Assuming one GPU for now
	// This code is a scalability measure that 
	// ensures that the kernel can fit on the GPU
	// since the histogram can get very large
	/*if (currDevice == -1) {
		checkCudaErrors(cudaGetDevice(&currDevice));
		cudaGetDeviceProperties(&prop, currDevice);
		int maxGridSize[3] = { prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] };

	}*/
	/*
	int device;
	checkCudaErrors(cudaGetDevice(&device));
	if (currDevice == -1 || currDevice != device) {
		currDevice = device;
		cudaGetDeviceProperties(&prop, device);
	}*/

	
	int numBlocks = (size + numThreads - 1) / numThreads;
	if (s) {
		fillZeros << < numThreads, numBlocks, 0, s >> > (buf, size);
	}
	else {
		fillZeros << < numThreads, numBlocks >> > (buf, size);
	}
	getLastCudaError("Kernel Launch Failure");

}
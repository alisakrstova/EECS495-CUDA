#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

#define T 12

__global__ void opt_2dhistoKernel(uint32_t *input, size_t height, size_t width, uint32_t* bins);
__global__ void opt_32to8Kernel(uint32_t *input, uint8_t* output, size_t length);

void opt_2dhisto(uint32_t* input, size_t height, size_t width, uint8_t* bins)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */
	uint32_t* g_bins;
	cudaMalloc(&g_bins, HISTO_HEIGHT * HISTO_WIDTH * sizeof(uint32_t));

    cudaMemset(bins, 0, HISTO_HEIGHT * HISTO_WIDTH * sizeof(bins[0]));
    cudaMemset(g_bins, 0, HISTO_HEIGHT * HISTO_WIDTH * sizeof(g_bins[0]));
    
    
    // Kernel to calculate the bins
    // We use 1024 * T threads so that more streaming multiprocessors can be used
    opt_2dhistoKernel<<<2 * T, 512>>>(input, height, width, g_bins);
  
    // Convert 332 bit to 8 bit
    opt_32to8Kernel<<<HISTO_HEIGHT * HISTO_WIDTH / 512, 512>>>(g_bins, bins, 1024);

    cudaThreadSynchronize();
    cudaFree(g_bins);
}

/* Include below the implementation of any other functions you need */

__global__ void opt_2dhistoKernel(uint32_t *input, size_t height, size_t width, uint32_t* bins){
	
	// Shared memory to hold the sub-histogram
	__shared__ int sub_hist[1024];
	
	// No bank conflict as threads access the shared memory sequentially
	sub_hist[threadIdx.x] = 0;
	sub_hist[threadIdx.x + 512] = 0;
	
	// Global thread id
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
		
	// Step 1: Divide each row into different groups so that each thread can concurrently work on a portion of the row
	// 	   Acheived by dividing the row into 'T' groups each having (width/T) elements.
	// Step 2: Calculate the row id and column id according to this new configuration
	// 	   Row id = (idx / T) * height of the input data
	// 	   Column id = (idx % T) * (width / T) , as each thread accesses data strided in the columns by (width / T)
	// Step 3: Each thread performs work only in its assigned group
	// 	   The loop runs from 0 to (width / T)
	
	for (int i = 0; i < (width / T); ++i)
	{
		atomicAdd(sub_hist + input[((idx / T) * height) + ((idx % T) * (width / T)) +  i], 1);
	}
	__syncthreads();
	
	// Update the global memory - storing is again sequential so no bank conflict
	atomicAdd(bins + threadIdx.x, sub_hist[threadIdx.x]);
	atomicAdd(bins + threadIdx.x + 512, sub_hist[threadIdx.x + 512]);
}

__global__ void opt_32to8Kernel(uint32_t *input, uint8_t* output, size_t length){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	output[idx] = (uint8_t)((input[idx] < UINT8_MAX) * input[idx]) + (input[idx] >= UINT8_MAX) * UINT8_MAX;

	__syncthreads();
}

void* AllocateDevice(size_t size){
	void* ret;
	cudaMalloc(&ret, size);
	return ret;
}

void CopyToDevice(void* D_device, void* D_host, size_t size){
	cudaMemcpy(D_device, D_host, size, 
					cudaMemcpyHostToDevice);
}

void CopyFromDevice(void* D_host, void* D_device, size_t size){
	cudaMemcpy(D_host, D_device, size, 
					cudaMemcpyDeviceToHost);
}

void FreeDevice(void* D_device){
	cudaFree(D_device);
}
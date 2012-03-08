#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

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

    opt_2dhistoKernel<<<2, 512>>>(input, height, width, g_bins);

    opt_32to8Kernel<<<HISTO_HEIGHT * HISTO_WIDTH / 512, 512>>>(g_bins, bins, 1024);

    cudaThreadSynchronize();
    cudaFree(g_bins);
}

/* Include below the implementation of any other functions you need */

__global__ void opt_2dhistoKernel(uint32_t *input, size_t height, size_t width, uint32_t* bins){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    //__shared__ uint s_bins[1024];
    //s_bins[idx] = 0;
    //__syncthreads();
	for (int i = 0; i < width; ++i)
	{
		atomicAdd(bins + input[idx * 1024 + i], 1);
	}
	
	__syncthreads();

		//++bins[input[j * height + idx]];
	//if (s_bins[input[idx + 512]] < UINT8_MAX)
	//atomicAdd(s_bins + input[idx + 512], 1);
		//++bins[input[j * height + idx + width / 2]];

	//if(s_bins[idx] > UINT8_MAX) s_bins[idx] = UINT8_MAX;
	//if(s_bins[idx + 512] > UINT8_MAX) s_bins[idx + 512] = UINT8_MAX;

    //__syncthreads();
    //bins[idx] = (uint8_t)(s_bins[idx] & 0xFF);
    //__syncthreads();
    //bins[idx + 512] = (uint8_t)s_bins[idx + 512];
    //bins[idx + 512] = (uint8_t)s_bins[idx + 512];
    //__syncthreads();
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
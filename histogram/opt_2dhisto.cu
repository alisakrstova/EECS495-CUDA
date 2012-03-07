#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

__global__ void opt_2dhistoKernel(uint32_t *input, size_t height, size_t width, uint8_t* bins);

void opt_2dhisto(uint32_t* input, size_t height, size_t width, uint8_t* bins)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */

    //cudaMemset	(bins, 0, HISTO_HEIGHT * HISTO_WIDTH * sizeof(bins[0]));

    opt_2dhistoKernel<<<1, 512>>>(input, height, width, bins);

    cudaThreadSynchronize();
}

/* Include below the implementation of any other functions you need */

__global__ void opt_2dhistoKernel(uint32_t *input, size_t height, size_t width, uint8_t* bins){

    int idx = threadIdx.x;
    __shared__ uint s_bins[HISTO_HEIGHT*HISTO_WIDTH];
    s_bins[idx] = 0;
    s_bins[idx + width / 2] = 0;
    __syncthreads();

	if (s_bins[input[idx]] < UINT8_MAX)
		atomicAdd(s_bins + input[idx], 1);
	__syncthreads();
		//++bins[input[j * height + idx]];
	if (s_bins[input[idx + width / 2]] < UINT8_MAX)
		atomicAdd(s_bins + input[idx + width / 2], 1);
		//++bins[input[j * height + idx + width / 2]];

    __syncthreads();
    bins[idx] = (uint8_t)s_bins[idx];
    __syncthreads();
    bins[idx + width / 2] = (uint8_t)s_bins[idx + width / 2];
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
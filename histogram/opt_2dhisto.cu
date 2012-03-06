#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

__global__ void opt_2dhistoKernel(uint32_t *input[], size_t height, size_t width, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH]);

void opt_2dhisto(uint32_t *input[], size_t height, size_t width, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH])
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */

    cudaMemset	(bins, 0, HISTO_HEIGHT * HISTO_WIDTH * sizeof(bins[0]));

    opt_2dhistoKernel<<<1, 1024>>>(input, height, width, bins);
}

/* Include below the implementation of any other functions you need */

__global__ void opt_2dhistoKernel(uint32_t *input[], size_t height, size_t width, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH]){

    for (size_t j = 0; j < height; ++j)
    {
        for (size_t i = 0; i < width; ++i)
        {
            const uint32_t value = input[j][i];

            uint8_t *p = (uint8_t*)bins;

            // Increment the appropriate bin, but do not roll-over the max value
            if (p[value] < UINT8_MAX)
                ++p[value];
        }
    }
}

void* AllocateDevice(size_t size){
	void* ret;
	cudaMalloc(ret, size);
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
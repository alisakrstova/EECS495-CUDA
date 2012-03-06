#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto(uint32_t *input[], size_t height, size_t width, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH]);

/* Include below the function headers of any other functions that you implement */

__global__ void opt_2dhistoKernel(uint32_t *input[], size_t height, size_t width, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH]);

#include <cutil.h>

#endif

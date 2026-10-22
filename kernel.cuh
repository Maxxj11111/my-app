/**
 * kernel.cuh — Forward declaration of main kernel
 */

#ifndef PSC_KERNEL_CUH
#define PSC_KERNEL_CUH

#include "common.h"

__global__ void __launch_bounds__(THREADS_PER_BLOCK, BLOCKS_PER_SM)
kangaroo_kernel_v57(Point target, uint64_t seed, uint32_t dp_bits, uint32_t dp_table_size, int use_mod3);

#endif // PSC_KERNEL_CUH

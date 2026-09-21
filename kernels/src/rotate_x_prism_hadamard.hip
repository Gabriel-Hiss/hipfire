// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Blockwise normalized Sylvester-Walsh-Hadamard transform used by PrismML
// Hadamard-folded GGUF weights. One workgroup handles one block from one row.
// `inverse == 0` is the fold-before-matmul order (signs, then H) that every
// folded weight expects. `inverse == 1` is the after-lookup order (H, then
// signs) that returns a latent embedding row to the primal basis.
extern "C" __launch_bounds__(256, 1)
__global__ void rotate_x_prism_hadamard(
    const float* __restrict__ x,
    const float* __restrict__ signs,
    float* __restrict__ out,
    int k,
    int block_size,
    int batch_size,
    int inverse
) {
    extern __shared__ float tile[];
    const int tid = (int) threadIdx.x;
    const int blocks_per_row = k / block_size;
    const int block = (int) blockIdx.x;
    const int row = block / blocks_per_row;
    const int group = block - row * blocks_per_row;
    if (row >= batch_size) return;

    const int base = row * k + group * block_size;
    for (int i = tid; i < block_size; i += (int)blockDim.x) {
        const float sign = signs[group * block_size + i];
        tile[i] = inverse ? x[base + i] : x[base + i] * sign;
    }
    __syncthreads();

    for (int stride = 1; stride < block_size; stride <<= 1) {
        for (int pair = tid; pair < block_size / 2; pair += (int)blockDim.x) {
            const int lane = pair % stride;
            const int group_base = (pair / stride) * (stride << 1);
            const int a_idx = group_base + lane;
            const int b_idx = a_idx + stride;
            const float a = tile[a_idx];
            const float b = tile[b_idx];
            tile[a_idx] = a + b;
            tile[b_idx] = a - b;
        }
        __syncthreads();
    }

    const float scale = rsqrtf((float)block_size);
    for (int i = tid; i < block_size; i += (int)blockDim.x) {
        const float sign = signs[group * block_size + i];
        out[base + i] = inverse ? tile[i] * scale * sign : tile[i] * scale;
    }
}

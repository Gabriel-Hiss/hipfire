// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// One embedding row of a TQ2-G128 table (34 B per 128-weight group:
// [FP16 d][32 B of 2-bit codes], value = (code - 1) * d) decoded to F32.
// Rows are contiguous, so the row base is token * (dim / 128) * 34.
extern "C" __launch_bounds__(256, 1)
__global__ void embedding_lookup_tq2g128(
    const unsigned char* __restrict__ table,
    float* __restrict__ out,
    int token,
    int dim
) {
    const int groups = dim / 128;
    const long long row = (long long) token * groups * 34;
    for (int g = blockIdx.x; g < groups; g += gridDim.x) {
        const unsigned char* block = table + row + (long long) g * 34;
        const float d = (float)(*(const _Float16*) block);
        const unsigned char* qs = block + 2;
        for (int i = threadIdx.x; i < 128; i += blockDim.x) {
            const int code = (qs[i >> 2] >> ((i & 3) * 2)) & 0x3;
            out[g * 128 + i] = (float) (code - 1) * d;
        }
    }
}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

static __device__ __forceinline__ int ptq1_trit(const unsigned char* block, int e) {
    const unsigned char* qs = block;
    const unsigned char* qh = block + 24;
    unsigned int v;
    int n;
    if (e < 80) { v = qs[e & 15]; n = e >> 4; }
    else if (e < 120) { const int t = e - 80; v = qs[16 + (t & 7)]; n = t >> 3; }
    else { const int t = e - 120; v = qh[t & 1]; n = t >> 1; }
    #pragma unroll
    for (int i = 0; i < 4; ++i) if (i < n) v = (v * 3) & 0xFF;
    return (int)((v * 3) >> 8) - 1;
}

extern "C" __global__ void embedding_lookup_ptq1g128(
    const unsigned char* __restrict__ table,
    float* __restrict__ out,
    int token, int dim
) {
    const int groups = dim / 128;
    const long long row = (long long)token * groups * 28;
    for (int g = blockIdx.x; g < groups; g += gridDim.x) {
        const unsigned char* q = table + row + (long long)g * 28;
        const float d = (float)(*(const _Float16*)(q + 26));
        for (int e = threadIdx.x; e < 128; e += blockDim.x) {
            out[g * 128 + e] = d * ptq1_trit(q, e);
        }
    }
}

// Prefill form: token blockIdx.y of `tokens` into row blockIdx.y of `out`.
// Grid: [dim / 128, n_tokens]. Same values as the single-token kernel.
extern "C" __global__ void embedding_lookup_ptq1g128_batched(
    const unsigned char* __restrict__ table,
    float* __restrict__ out,
    const int* __restrict__ tokens,
    int dim
) {
    const int groups = dim / 128;
    const int g = blockIdx.x;
    const unsigned char* q = table + ((long long)tokens[blockIdx.y] * groups + g) * 28;
    const float d = (float)(*(const _Float16*)(q + 26));
    float* o = out + (long long)blockIdx.y * dim + g * 128;
    for (int e = threadIdx.x; e < 128; e += blockDim.x) {
        o[e] = d * ptq1_trit(q, e);
    }
}

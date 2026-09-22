// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Probe: achievable read bandwidth. The decode ceiling rests on this number, so
// it has to be measured on this card rather than read off a spec sheet.
//
// Grid-stride float4 read with a reduction that cannot be eliminated. The store
// is unreachable in practice (the sum of a real buffer is never exactly this
// constant) but the compiler cannot prove that, so the loads survive.
__launch_bounds__(256, 8)
extern "C" __global__ void probe_dram_bw(
    const float4* __restrict__ src,
    float* __restrict__ out,
    long long n4
) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        const float4 v = src[i];
        acc.x += v.x;
        acc.y += v.y;
        acc.z += v.z;
        acc.w += v.w;
    }
    const float s = acc.x + acc.y + acc.z + acc.w;
    if (s == 12345.678f) out[0] = s;
}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Probe: achievable read bandwidth. The decode ceiling rests on this number, so
// it has to be measured on this card rather than read off a spec sheet.
//
// Grid-stride float4 read, PROBE_UNROLL independent loads in flight per thread
// per iteration: one load per iteration leaves the memory pipe short of
// outstanding requests and under-reports the device. The reduction cannot be
// eliminated: the store is unreachable in practice (the sum of a real buffer
// is never exactly this constant) but the compiler cannot prove that.
#define PROBE_UNROLL 8
typedef float probe_f4 __attribute__((ext_vector_type(4)));
__launch_bounds__(256, 8)
extern "C" __global__ void probe_dram_bw(
    const float4* __restrict__ src,
    float* __restrict__ out,
    long long n4
) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + (PROBE_UNROLL - 1) * stride < n4; i += PROBE_UNROLL * stride) {
        const probe_f4* s4 = (const probe_f4*)src;
        probe_f4 v[PROBE_UNROLL];
        #pragma unroll
        for (int u = 0; u < PROBE_UNROLL; ++u) v[u] = __builtin_nontemporal_load(s4 + i + u * stride);
        #pragma unroll
        for (int u = 0; u < PROBE_UNROLL; ++u) {
            acc.x += v[u].x;
            acc.y += v[u].y;
            acc.z += v[u].z;
            acc.w += v[u].w;
        }
    }
    for (; i < n4; i += stride) {
        const float4 v = src[i];
        acc.x += v.x;
        acc.y += v.y;
        acc.z += v.z;
        acc.w += v.w;
    }
    const float s = acc.x + acc.y + acc.z + acc.w;
    if (s == 12345.678f) out[0] = s;
}

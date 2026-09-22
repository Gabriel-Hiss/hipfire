// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// What is the real iu4 : iu8 throughput ratio on this part?
//
// The INT4 route for ternary weights only pays if `v_wmma_i32_16x16x16_iu4` is
// more than 2x `iu8`, because an 8-bit activation cannot be fed to iu4
// directly: x = 16*a + b needs TWO iu4 MMAs to reproduce one iu8 MMA's dot
// exactly (the reference quantizes activations to int8 with 32-element
// sub-blocks, and a 4-bit activation costs 18.1x the RMS error). The spec
// sheet says 183.9 vs 91.96 TOPS, exactly 2x, which would cancel the split
// exactly. This measures it rather than trusting the sheet.
//
// Both kernels issue 8 INDEPENDENT accumulator chains so the loop measures
// throughput, not latency, and the same instruction count for each.
typedef int __attribute__((ext_vector_type(4))) i8x16; // 16 int8  (iu8 operand)
typedef int __attribute__((ext_vector_type(2))) i8x8;  // 16 int4  (iu4 operand)
typedef int __attribute__((ext_vector_type(8))) i32x8;

#define UNROLL 8
#define ITERS 2048

extern "C" __launch_bounds__(32, 1)
__global__ void probe_wmma_iu8_rate(i32x8* __restrict__ out, int seed) {
    i8x16 a, b;
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        a[w] = seed + w;
        b[w] = seed - w;
    }
    i32x8 acc[UNROLL];
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) acc[u] = (i32x8){0, 0, 0, 0, 0, 0, 0, 0};
    for (int it = 0; it < ITERS; ++it) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            acc[u] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, acc[u], false);
        }
    }
    i32x8 s = (i32x8){0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) s[j] += acc[u][j];
    }
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}

extern "C" __launch_bounds__(32, 1)
__global__ void probe_wmma_iu4_rate(i32x8* __restrict__ out, int seed) {
    i8x8 a, b;
    #pragma unroll
    for (int w = 0; w < 2; ++w) {
        a[w] = seed + w;
        b[w] = seed - w;
    }
    i32x8 acc[UNROLL];
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) acc[u] = (i32x8){0, 0, 0, 0, 0, 0, 0, 0};
    for (int it = 0; it < ITERS; ++it) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            acc[u] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, acc[u], false);
        }
    }
    i32x8 s = (i32x8){0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) s[j] += acc[u][j];
    }
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Throwaway probe: what IS the wave32 `v_wmma_i32_16x16x16_iu8` fragment layout?
//
// The f16 sibling's layout is documented in kernels/src/gemm_q8_0_wmma.hip
// (A/B = 16 halves per lane, C[j] = C[2j + (tid>>4)][tid & 15]), but nothing in
// the tree uses iu8, so its layout is unverified. Assuming it would mean
// building a whole prefill GEMM on a guess.
//
// This kernel takes A and B as PLAIN row-major [16 x 16] int8 and does the
// fragment gather internally under the f16-sibling assumption (each lane owns
// one row of A / one column of B, 16 K-values each). If the assumption is right
// C matches A @ B. If not, C is a permutation of it and the pattern identifies
// the real layout.
#include <hip/hip_runtime.h>

using i8x16 = int __attribute__((ext_vector_type(4)));
using i32x8 = int __attribute__((ext_vector_type(8)));

extern "C" __launch_bounds__(32, 1)
__global__ void probe_wmma_iu8(
    const signed char* __restrict__ A,  // [16 x 16] row-major, row = M, col = K
    const signed char* __restrict__ B,  // [16 x 16] row-major, row = K, col = N
    int* __restrict__ C                 // [16 x 16] row-major, row = M, col = N
) {
    const int tid = threadIdx.x;
    // Assumption under test: lane's low 4 bits select the A row and the B
    // column; each lane holds 16 consecutive K values (4 int32 = 16 int8).
    const int row = tid & 15;
    const int col = tid & 15;

    i8x16 a = *reinterpret_cast<const i8x16*>(A + row * 16);
    // B is [K][N]; lane owns column `col`, so its 16 K-values are strided by N.
    signed char b_vals[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) b_vals[k] = B[k * 16 + col];
    i8x16 b = *reinterpret_cast<const i8x16*>(b_vals);

    i32x8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
    acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, acc, false);

    // Output mapping under test, copied from the f16 sibling.
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int out_row = 2 * j + (tid >> 4);
        const int out_col = tid & 15;
        C[out_row * 16 + out_col] = acc[j];
    }
}

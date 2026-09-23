// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Batched PTQ1_0 prefill GEMM on the INT8 matrix unit.
//
// Y[N, M] = X[N, K] @ A_ptq1[M, K]^T
//   A: PTQ1_0 packed ternary, 28 B per 128-weight group: qs[24] + qh[2] +
//      fp16 scale at offset 26.
//   X: Q8_1 (block_q8_1_mmq) -- int8 qs[128] + 4 fp16 scales per 128-group,
//      layout [K/128 group][N]. This is the buffer `ensure_q8_1_mmq_x` already
//      produces for the scalar PTQ1 prefill kernel, consumed as-is.
//   Y: FP32 [N, M] row-major.
//
// WHY THE MATRIX UNIT. The scalar `gemm_ptq1g128_prefill` runs at 1.38e12 MAC/s
// (measured), the same rate as the decode GEMV: both are bound by the vector
// integer multiply, not by bytes. gfx1100 ships INT8/INT4/FP16/BF16 matrix ops
// (91.96 / 183.9 / 91.96 / 91.96 TOPS) and this tree used none of them for
// ternary. 91.96 TOPS / 2 = 45.98e12 MAC/s against 27e9 MACs/token bounds
// prefill at 1702 tok/s.
//
// WHY iu8 AND NOT FP16. `v_wmma_i32_16x16x16_iu8` takes BOTH operands at 8 bits
// and accumulates in int32, which is exactly the integer dot llama.cpp's
// `vec_dot_ptq1_0_q8_1` computes: an exact trit in {-1,0,1} against the Q8_1
// int8 activation. The FP16 WMMA family would force the activation through
// fp16, which is a different (and non-matching) quantization of X. `iu4` would
// double the rate but requires 4-bit activations, i.e. 16 levels against 256,
// and every exact recovery scheme (nibble split, sign-plane split) costs 2x the
// MACs for exactly the 2x rate it gains.
//
// Fragment layout is MEASURED, not assumed: crates/rdna-compute/examples/
// test_wmma_iu8_layout.rs compares this gather against the CPU int8 product and
// passes with 0/256 mismatches. A/B are 4x int32 (16 int8) per lane with lane's
// low 4 bits selecting the A row and the B column; C[j] lands at
// C[2*j + (tid>>4)][tid & 15].
//
// Grid:  [ceil(M/16), ceil(N/(16*PTQ1_BATCH_TILES))]
// Block: [32] (wave32, one wave per workgroup).
typedef int __attribute__((ext_vector_type(4))) i8x16;
// gfx1100 benefits from sharing one ternary decode across two token tiles.
// Keep the original single-tile path on unmeasured architectures.
#ifndef PTQ1_BATCH_TILES
#if defined(__gfx1100__)
#define PTQ1_BATCH_TILES 2
#else
#define PTQ1_BATCH_TILES 1
#endif
#endif
typedef int __attribute__((ext_vector_type(8))) i32x8;

#define QK8_1 32
struct block_q8_1_mmq {
    _Float16 ds4[8];
    signed char qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad Q8_1 layout");

// Trit decode. The group-at-a-time branchless form below replaces the
// per-element `ptq1_trit` the scalar kernel uses: extracting digit t of a byte
// needs t conditional multiplies, while walking a byte's digits in order costs
// one unconditional multiply each. The channel test pins both to the same
// values (test_gemv_ptq1g128, 7.6e-6 on 257x512x17).

// Pack 16 consecutive values into the 4x int32 iu8 fragment. Byte b of word w is
// the value at w*4+b, matching the layout test_wmma_iu8_layout.rs pins.
#define PTQ1_PACK16(dst, EXPR)                                   \
    do {                                                         \
        _Pragma("unroll")                                        \
        for (int w = 0; w < 4; ++w) {                            \
            unsigned int word = 0;                               \
            _Pragma("unroll")                                    \
            for (int b = 0; b < 4; ++b) {                        \
                word |= ((unsigned int)(unsigned char)(EXPR)) << (8 * b); \
            }                                                    \
            (dst)[w] = (int)word;                                \
        }                                                        \
    } while (0)

#ifndef PTQ1_WMMA_NAME
#define PTQ1_WMMA_NAME gemm_ptq1g128_wmma
#endif
__launch_bounds__(32, 2)
extern "C" __global__ void PTQ1_WMMA_NAME(
    const unsigned char* __restrict__ A,   // [M, K/128 * 28] packed PTQ1_0
    const block_q8_1_mmq* __restrict__ Xq, // [K/128 group][N]
    float* __restrict__ Y,                 // [N, M] row-major
    int M, int K, int N
) {
    const int row_start = blockIdx.x * 16;
    const int col_start = blockIdx.y * (16 * PTQ1_BATCH_TILES);
    const int tid = threadIdx.x;
    if (row_start >= M || col_start >= N) return;

    const int my_row = row_start + (tid & 15);
    const int safe_row = (my_row < M) ? my_row : (M - 1);

    const int groups = K / 128;
    const unsigned char* row_base = A + (long long)safe_row * groups * 28;

    float acc[PTQ1_BATCH_TILES][8];
    #pragma unroll
    for (int bt = 0; bt < PTQ1_BATCH_TILES; ++bt) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc[bt][j] = 0.0f;
    }

    for (int g = 0; g < groups; ++g) {
        const unsigned char* q = row_base + g * 28;
        // The A fragment row a lane holds is NOT the row its `acc[j]` belongs to:
        // lane L holds A row (L&15) but writes output row 2j+(L>>4), column L&15.
        // So the weight scale must be fetched for the OUTPUT row, per j. Using
        // the lane's own row applies dw[column] to every row, which is invisible
        // while dw is uniform and wrong the moment it varies (measured: W[0]
        // constant at dw[0]*partial across all 16 output rows).
        float dwrow[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int orow = row_start + 2 * j + (tid >> 4);
            const int srow = (orow < M) ? orow : (M - 1);
            const unsigned char* qq = A + (long long)srow * groups * 28 + g * 28;
            _Float16 h;
            unsigned short bits;
            __builtin_memcpy(&bits, qq + 26, 2);
            __builtin_memcpy(&h, &bits, 2);
            dwrow[j] = (float)h;
        }
        // Walk each packed byte's base-3 digits once for the eight WMMA fragments.
        signed char trits[128];
        #pragma unroll
        for (int m = 0; m < 16; ++m) {
            unsigned int v = q[m];
            #pragma unroll
            for (int d = 0; d < 5; ++d) {
                const unsigned int w = v * 3;
                v = w & 0xFF;
                trits[d * 16 + m] = (signed char)((int)(w >> 8) - 1);
            }
        }
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            unsigned int v = q[16 + m];
            #pragma unroll
            for (int d = 0; d < 5; ++d) {
                const unsigned int w = v * 3;
                v = w & 0xFF;
                trits[80 + d * 8 + m] = (signed char)((int)(w >> 8) - 1);
            }
        }
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            unsigned int v = q[24 + h];
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                const unsigned int w = v * 3;
                v = w & 0xFF;
                trits[120 + d * 2 + h] = (signed char)((int)(w >> 8) - 1);
            }
        }
        // Reuse the decoded ternary row across token tiles on gfx1100.
        const block_q8_1_mmq* xq[PTQ1_BATCH_TILES];
        #pragma unroll
        for (int bt = 0; bt < PTQ1_BATCH_TILES; ++bt) {
            const int col = col_start + bt * 16 + (tid & 15);
            xq[bt] = Xq + (long long)g * N + (col < N ? col : 0);
        }

        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            float dx[PTQ1_BATCH_TILES];
            #pragma unroll
            for (int bt = 0; bt < PTQ1_BATCH_TILES; ++bt) dx[bt] = (float)xq[bt]->ds4[2 * s];
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int t = s * 2 + half;
                i8x16 av;
                PTQ1_PACK16(av, trits[t * 16 + w * 4 + b]);
                #pragma unroll
                for (int bt = 0; bt < PTQ1_BATCH_TILES; ++bt) {
                    const i8x16 bv = ((const i8x16*)xq[bt]->qs)[t];
                    i32x8 part = {0, 0, 0, 0, 0, 0, 0, 0};
                    part = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                        true, av, true, bv, part, false);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                        acc[bt][j] += (dwrow[j] * dx[bt]) * (float)part[j];
                }
            }
        }
    }

    #pragma unroll
    for (int bt = 0; bt < PTQ1_BATCH_TILES; ++bt) {
        const int my_col = col_start + bt * 16 + (tid & 15);
        if (my_col >= N) continue;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y[(long long)my_col * M + out_row] = acc[bt][j];
            }
        }
    }
}

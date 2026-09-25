// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched BF16-weight x F32-activation GEMM: Y[N x M] = X[N x K] @ W[M x K]^T.
//
// Why this kernel exists. Prism's ternary checkpoints keep the DeltaNet gate
// projections (w_alpha/w_beta) UNQUANTIZED as BF16, and no gfx11 arch has a
// BF16 *weight* GEMM: the only BF16 GEMM in the tree is the gfx942 MFMA
// calibration-teacher path. `is_batchable_la` therefore rejects BF16, and
// `qwen35_layer_batch_admissible` rejecting ONE layer drops the WHOLE model to
// the per-token prefill loop. Measured on Ternary-Bonsai-2-27B before this:
// prefill flat at 34 tok/s from 64 to 1024 prompt tokens, i.e. one full weight
// pass per prompt token instead of a batched GEMM.
//
// Widening those weights to F32 at load would also unblock it, but it moves the
// per-token GEMV onto a different accumulation order and shifts the llama.cpp
// parity metric (RMSE 0.012999 -> 0.013725 on the pinned prompt). This kernel
// leaves the decode path bit-identical.
//
// BF16 widens to F32 losslessly (bf16 IS the top 16 bits of an f32), so the
// inner loop is a plain FMA with no decode and no scale handling.
#define BF16_BATCH_TILE 8
// Rows per wave. The activation matrix is read once per row tile, so with one
// row per wave a [512 x 5120] prefill re-read 10.5 MB from cache for each of
// the 48 gate rows (~500 MB per call, 209 us measured). Four rows cut that
// traffic 4x. Each output keeps its lane's k sequence and the shuffle tree;
// the compiler's FMA contraction differs from the one-row form, which the
// CPU-oracle channel test measures as equal or smaller error.
#define BF16_ROW_TILE 4

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_bf16_xf32_batched(
    const unsigned short* __restrict__ W,  // [M x K] bf16
    const float* __restrict__ X,           // [N x K] f32
    float* __restrict__ Y,                 // [N x M] f32
    int M, int K, int N
) {
    const int row0 = blockIdx.x * BF16_ROW_TILE;
    const int n0 = blockIdx.y * BF16_BATCH_TILE;
    if (row0 >= M || n0 >= N) return;
    const int lane = threadIdx.x;

    // Rows past M read row M-1; their stores are suppressed.
    const unsigned short* wrow[BF16_ROW_TILE];
    #pragma unroll
    for (int r = 0; r < BF16_ROW_TILE; ++r) wrow[r] = W + (long long)min(row0 + r, M - 1) * K;
    // Columns past N read column N-1; their stores are suppressed.
    const float* xcol[BF16_BATCH_TILE];
    #pragma unroll
    for (int nb = 0; nb < BF16_BATCH_TILE; ++nb) xcol[nb] = X + (long long)min(n0 + nb, N - 1) * K;

    float acc[BF16_ROW_TILE][BF16_BATCH_TILE];
    #pragma unroll
    for (int r = 0; r < BF16_ROW_TILE; ++r)
        #pragma unroll
        for (int nb = 0; nb < BF16_BATCH_TILE; ++nb) acc[r][nb] = 0.0f;

    // One weight element feeds all BF16_BATCH_TILE columns and one activation
    // element all BF16_ROW_TILE rows.
    for (int k = lane; k < K; k += 32) {
        float w[BF16_ROW_TILE];
        #pragma unroll
        for (int r = 0; r < BF16_ROW_TILE; ++r) w[r] = __uint_as_float((unsigned int)wrow[r][k] << 16);
        #pragma unroll
        for (int nb = 0; nb < BF16_BATCH_TILE; ++nb) {
            const float x = xcol[nb][k];
            #pragma unroll
            for (int r = 0; r < BF16_ROW_TILE; ++r) acc[r][nb] += w[r] * x;
        }
    }

    #pragma unroll
    for (int r = 0; r < BF16_ROW_TILE; ++r) {
        #pragma unroll
        for (int nb = 0; nb < BF16_BATCH_TILE; ++nb) {
            float s = acc[r][nb];
            for (int off = 16; off > 0; off >>= 1) s += __shfl_down(s, off);
            if (lane == 0 && row0 + r < M && n0 + nb < N) Y[(long long)(n0 + nb) * M + row0 + r] = s;
        }
    }
}

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

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_bf16_xf32_batched(
    const unsigned short* __restrict__ W,  // [M x K] bf16
    const float* __restrict__ X,           // [N x K] f32
    float* __restrict__ Y,                 // [N x M] f32
    int M, int K, int N
) {
    const int row = blockIdx.x;
    const int n0 = blockIdx.y * BF16_BATCH_TILE;
    if (row >= M || n0 >= N) return;
    const int lane = threadIdx.x;
    const unsigned short* wrow = W + (long long)row * K;

    float acc[BF16_BATCH_TILE];
    #pragma unroll
    for (int nb = 0; nb < BF16_BATCH_TILE; ++nb) acc[nb] = 0.0f;

    // One weight element feeds all BF16_BATCH_TILE columns, so the weight load
    // is amortised over the tile — the same reuse `gemm_ptq1g128_prefill` gets
    // from decoding a group once for eight columns.
    for (int k = lane; k < K; k += 32) {
        const float w = __uint_as_float((unsigned int)wrow[k] << 16);
        #pragma unroll
        for (int nb = 0; nb < BF16_BATCH_TILE; ++nb) {
            const int n = n0 + nb;
            if (n >= N) break;
            acc[nb] += w * X[(long long)n * K + k];
        }
    }

    #pragma unroll
    for (int nb = 0; nb < BF16_BATCH_TILE; ++nb) {
        const int n = n0 + nb;
        if (n >= N) break;
        float s = acc[nb];
        for (int off = 16; off > 0; off >>= 1) s += __shfl_down(s, off);
        if (lane == 0) Y[(long long)n * M + row] = s;
    }
}

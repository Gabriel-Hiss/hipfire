// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched PTQ1_0 prefill GEMM on the INT8 matrix unit, 64-row x 64-token
// workgroup tile. Same contract as `gemm_ptq1g128_wmma`:
//   Y[N, M] = X[N, K] @ A_ptq1[M, K]^T
//   A: PTQ1_0 packed ternary, 28 B per 128-weight group (qs[24], qh[2], fp16 d).
//   X: Q8_1 `block_q8_1_mmq`, layout [K/128 group][N].
//   Y: FP32 [N, M] row-major.
//
// Why a workgroup tile. `gemm_ptq1g128_wmma` gives each wave 16 rows x 32
// tokens and has every lane decode its own 128-trit row byte by byte for each
// group, then scales every WMMA result separately. On gfx1100 the WMMA and the
// surrounding VALU/LDS work do not overlap within a wave (measured: removing
// the WMMAs halves the time), so the work around each WMMA is what bounds it.
// Here 4 waves share one decoded 64-row slab and one 64-token activation slab
// in LDS:
//   - the trit decode runs once per 64 x 64 tile, on packed dwords: two
//     v_perm_b32 gather the five base-3 digits of four bytes at a time (the
//     recurrence of PrismML's vec_dot_ptq1_0_q8_1), and every lane of a wave
//     decodes the same dword position, so the wave never diverges;
//   - the two K16 WMMAs of a 32-element Q8_1 sub-block accumulate in int32 and
//     are scaled once: sumi_s * dx_s summed over the four sub-blocks, then
//     times the weight scale, which is the order PrismML's llama.cpp vec_dot
//     uses.
// Measured against `gemm_ptq1g128_wmma` at N=512: 1.46-1.91x (24-27e12 MAC/s,
// ~57% of the 47e12 MAC/s iu8 peak measured on this part). Below 64 tokens
// the tile is mostly padding and the 16x32 kernel wins, so the host dispatches
// this one for N > 32.
//
// Grid: [ceil(M/64), ceil(N/64)]. Block: [128] (4 x wave32).

typedef int __attribute__((ext_vector_type(4))) t64_i8x16;
typedef int __attribute__((ext_vector_type(8))) t64_i32x8;

struct t64_block_q8_1 {
    _Float16 ds4[8];
    signed char qs[128];
};
static_assert(sizeof(t64_block_q8_1) == 144, "bad Q8_1 layout");

// Digit t of bytes 0..3 of `w`, one byte each, values {0,1,2}.
static __device__ __forceinline__ void t64_digits5(unsigned w, unsigned D[5]) {
    unsigned lo = __builtin_amdgcn_perm(0u, w, 0x0c010c00u); // [b0, 0, b1, 0]
    unsigned hi = __builtin_amdgcn_perm(0u, w, 0x0c030c02u); // [b2, 0, b3, 0]
    #pragma unroll
    for (int t = 0; t < 5; ++t) {
        const unsigned lo3 = lo * 3u;
        const unsigned hi3 = hi * 3u;
        lo = lo3 & 0x00FF00FFu;
        hi = hi3 & 0x00FF00FFu;
        D[t] = __builtin_amdgcn_perm(hi3, lo3, 0x07050301u);
    }
}

// Bytes {0,1,2} -> signed trits {-1,0,1} with no borrow between bytes.
static __device__ __forceinline__ unsigned t64_signed(unsigned d) {
    return ((d | 0x80808080u) - 0x01010101u) ^ 0x80808080u;
}

#define T64_ROWS 64
#define T64_TOKS 64
#define T64_STRIDE 36 // 32 words of int8 per row/token + 4 words of bank padding

extern "C" __launch_bounds__(128, 2)
__global__ void gemm_ptq1g128_wmma_t64(
    const unsigned char* __restrict__ A,
    const t64_block_q8_1* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N
) {
    __shared__ unsigned a_lds[T64_ROWS * T64_STRIDE];
    __shared__ unsigned x_lds[T64_TOKS * T64_STRIDE];
    __shared__ float dw_lds[T64_ROWS];
    __shared__ float dx_lds[T64_TOKS * 4];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int row0 = blockIdx.x * T64_ROWS;
    const int tok0 = blockIdx.y * T64_TOKS;
    const int groups = K >> 7;

    // Activation staging: two threads per token, four uint4 each.
    const int s_tok = tid >> 1;
    const int s_part = tid & 1;
    const int x_tok = min(tok0 + s_tok, N - 1);

    // Each wave computes 32 rows x 32 tokens = 2 x 2 WMMA tiles.
    const int w_row = (wave & 1) * 32;
    const int w_tok = (wave >> 1) * 32;

    float acc[2][2][8];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            #pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.0f;

    for (int g = 0; g < groups; ++g) {
        // Weights: 64 rows x 7 dwords decoded as units u = tid + 128 i with
        // dword k = u / 64 and row u % 64. Trit words land in the element
        // order the vec_dot traversal defines: qs[0..15] digit t -> elements
        // 16t + m, qs[16..23] -> 80 + 8t + m, qh -> 120 + 2t + h.
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int u = tid + 128 * i;
            if (u < T64_ROWS * 7) {
                const int k = u >> 6;
                const int r = u & 63;
                const int arow = min(row0 + r, M - 1);
                const unsigned w =
                    ((const unsigned*)(A + (long long)arow * groups * 28))[g * 7 + k];
                unsigned* dst = a_lds + r * T64_STRIDE;
                if (k < 6) {
                    unsigned D[5];
                    t64_digits5(w, D);
                    #pragma unroll
                    for (int t = 0; t < 5; ++t)
                        dst[k < 4 ? 4 * t + k : 20 + 2 * t + (k - 4)] = t64_signed(D[t]);
                } else {
                    unsigned v = __builtin_amdgcn_perm(0u, w, 0x0c010c00u); // [qh0, 0, qh1, 0]
                    #pragma unroll
                    for (int t = 0; t < 4; t += 2) {
                        const unsigned w0 = v * 3u;
                        v = w0 & 0x00FF00FFu;
                        const unsigned w1 = v * 3u;
                        v = w1 & 0x00FF00FFu;
                        dst[30 + t / 2] = t64_signed(__builtin_amdgcn_perm(w1, w0, 0x07050301u));
                    }
                    const unsigned short bits = (unsigned short)(w >> 16);
                    dw_lds[r] = (float)(*(const _Float16*)&bits);
                }
            }
        }
        // Activations: 64 tokens of 128 int8 plus the four sub-block scales.
        {
            const t64_block_q8_1* xb = Xq + (long long)g * N + x_tok;
            const uint4* src = (const uint4*)xb->qs;
            uint4* dst = (uint4*)(x_lds + s_tok * T64_STRIDE);
            #pragma unroll
            for (int j = 0; j < 4; ++j) dst[s_part * 4 + j] = src[s_part * 4 + j];
            if (s_part == 0) {
                #pragma unroll
                for (int s = 0; s < 4; ++s) dx_lds[s_tok * 4 + s] = (float)xb->ds4[2 * s];
            }
        }
        __syncthreads();

        // A fragment: lane's row (lane & 15), 16 int8 of K. B fragment: lane's
        // token (lane & 15). C[e] lands at row 2e + (lane >> 4), token lane & 15.
        float gacc[2][2][8];
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                #pragma unroll
                for (int e = 0; e < 8; ++e) gacc[i][j][e] = 0.0f;

        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            t64_i8x16 af[2][2], bf[2][2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const unsigned* ar = a_lds + (w_row + i * 16 + (lane & 15)) * T64_STRIDE;
                const unsigned* xr = x_lds + (w_tok + i * 16 + (lane & 15)) * T64_STRIDE;
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    af[i][h] = *(const t64_i8x16*)(ar + (2 * s + h) * 4);
                    bf[i][h] = *(const t64_i8x16*)(xr + (2 * s + h) * 4);
                }
            }
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const float dx = dx_lds[(w_tok + j * 16 + (lane & 15)) * 4 + s];
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    t64_i32x8 part = {0, 0, 0, 0, 0, 0, 0, 0};
                    part = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af[i][0], true, bf[j][0], part, false);
                    part = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af[i][1], true, bf[j][1], part, false);
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) gacc[i][j][e] += dx * (float)part[e];
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float dw = dw_lds[w_row + i * 16 + 2 * e + (lane >> 4)];
                #pragma unroll
                for (int j = 0; j < 2; ++j) acc[i][j][e] += dw * gacc[i][j][e];
            }
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int tok = tok0 + w_tok + j * 16 + (lane & 15);
        if (tok >= N) continue;
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row = row0 + w_row + i * 16 + 2 * e + (lane >> 4);
                if (row < M) Y[(long long)tok * M + row] = acc[i][j][e];
            }
    }
}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Prism PTQ1_0: 128 ternary values in 28 bytes:
// qs[24] (five base-3 trits/byte), qh[2] (four trits/byte), fp16 scale d.
//
// Decode GEMV layout: PTQ1_LPR lanes split one row's K groups, so a wave
// covers 32/PTQ1_LPR rows at once, and each lane walks PTQ1_RL of them while
// the activation block it loaded stays in registers. Grid is
// ceil(M / PTQ1_WG_ROWS); the host mirrors PTQ1_WG_ROWS.
#define PTQ1_LPR 8
#define PTQ1_RL 1
#define PTQ1_WG_ROWS ((32 / PTQ1_LPR) * PTQ1_RL)

// Per-element trit decode used by the prefill kernel, which keeps the
// four-elements-per-lane Q8_1 mapping (one lane's four elements never share a
// source byte, so the group-at-a-time form below does not apply).
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

#define QK8_1 32
struct block_q8_1_mmq {
    _Float16 ds4[8];
    signed char qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad Q8_1 layout");

// Four packed PTQ1 bytes -> the five digit dwords of their base-3 recurrence.
// This is the traversal of PrismML-Eng/llama.cpp's CUDA vec_dot_ptq1_0_q8_1
// (MIT): widen the bytes into 16-bit lanes so the multiply by three cannot
// carry between them, and gather the digit byte ((3v) >> 8) of every lane with
// one byte permute. D[t] holds digit t of bytes 0..3 in byte order, which is
// exactly four consecutive activation elements, so one dot4 consumes it.
static __device__ __forceinline__ void ptq1_digits5(unsigned w, unsigned D[5]) {
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

// Unsigned digits {0,1,2} against signed int8 activations (v_dot4_i32_iu8 on
// gfx11/gfx12; the byte-wise form elsewhere computes the same integer).
static __device__ __forceinline__ int ptq1_dot(unsigned digits, int x, int acc) {
#if defined(__GFX11__) || defined(__GFX12__)
    return __builtin_amdgcn_sudot4(false, (int)digits, true, x, acc, false);
#else
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        acc += (int)((digits >> (8 * b)) & 0xFFu) * (int)(signed char)(x >> (8 * b));
    }
    return acc;
#endif
}

// One 128-weight block against its Q8_1 block. Digits are stored offset by
// one, so sum((d-1)*x) = sum(d*x) - sum(x) per 32-element sub-block; the
// caller supplies sum(x) because every row of the lane shares it. The integer
// sub-block sums equal the reference's exactly, and the float epilogue keeps
// its order: acc = sum_k ds_k * sumi_k, result = d * acc.
static __device__ __forceinline__ float ptq1_block_dot(
    const unsigned w[7],
    const int xv[32],
    const int sx[4],
    const float dx[4]
) {
    int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    unsigned D[5];
    // qs[0..15]: element e = t*16 + m. Dword k, digit t -> x dword 4t + k.
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        ptq1_digits5(w[k], D);
        s0 = ptq1_dot(D[0], xv[k], s0);
        s0 = ptq1_dot(D[1], xv[4 + k], s0);
        s1 = ptq1_dot(D[2], xv[8 + k], s1);
        s1 = ptq1_dot(D[3], xv[12 + k], s1);
        s2 = ptq1_dot(D[4], xv[16 + k], s2);
    }
    // qs[16..23]: element e = 80 + t*8 + m. Dword 4+k, digit t -> 20 + 2t + k.
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        ptq1_digits5(w[4 + k], D);
        s2 = ptq1_dot(D[0], xv[20 + k], s2);
        s2 = ptq1_dot(D[1], xv[22 + k], s2);
        s3 = ptq1_dot(D[2], xv[24 + k], s3);
        s3 = ptq1_dot(D[3], xv[26 + k], s3);
        s3 = ptq1_dot(D[4], xv[28 + k], s3);
    }
    // qh[0..1]: element e = 120 + t*2 + h. Two digits of both bytes per dword.
    unsigned v = __builtin_amdgcn_perm(0u, w[6], 0x0c010c00u); // [qh0, 0, qh1, 0]
    #pragma unroll
    for (int t = 0; t < 4; t += 2) {
        const unsigned w0 = v * 3u;
        v = w0 & 0x00FF00FFu;
        const unsigned w1 = v * 3u;
        v = w1 & 0x00FF00FFu;
        s3 = ptq1_dot(__builtin_amdgcn_perm(w1, w0, 0x07050301u), xv[30 + t / 2], s3);
    }

    float acc = 0.0f;
    acc += dx[0] * (float)(s0 - sx[0]);
    acc += dx[1] * (float)(s1 - sx[1]);
    acc += dx[2] * (float)(s2 - sx[2]);
    acc += dx[3] * (float)(s3 - sx[3]);
    const unsigned short dbits = (unsigned short)(w[6] >> 16);
    return (float)(*(const _Float16*)&dbits) * acc;
}

// Activation block g: four sub-block scales, 32 packed int8 dwords, and the
// per-sub-block sum(x) that folds the digit offset.
struct ptq1_xblock {
    float dx[4];
    int xv[32];
    int sx[4];
};

static __device__ __forceinline__ void ptq1_load_x(const block_q8_1_mmq* __restrict__ Xq, int g, ptq1_xblock& x) {
    const uint4* xb = (const uint4*)(Xq + g);
    const uint4 h = xb[0];
    const unsigned hw[4] = {h.x, h.y, h.z, h.w};
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const unsigned short b = (unsigned short)(hw[k] & 0xFFFFu);
        x.dx[k] = (float)(*(const _Float16*)&b);
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint4 q = xb[1 + j];
        x.xv[4 * j + 0] = (int)q.x;
        x.xv[4 * j + 1] = (int)q.y;
        x.xv[4 * j + 2] = (int)q.z;
        x.xv[4 * j + 3] = (int)q.w;
    }
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int a = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) a = ptq1_dot(0x01010101u, x.xv[8 * k + j], a);
        x.sx[k] = a;
    }
}

// Rows [block * PTQ1_WG_ROWS, +PTQ1_WG_ROWS) of y = A * x. With `residual`
// the row result is added to the existing y (x + W*x_rot, in place), in the
// same operand order as the add_inplace_f32 it replaces.
static __device__ __forceinline__ void ptq1_gemv_rows(
    const unsigned char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ y,
    int M, int K, int block, int residual
) {
    const int lane = threadIdx.x;
    const int s = lane % PTQ1_LPR;
    const int groups = K >> 7;
    const long long row_words = (long long)groups * 7; // 28 B per group
    const int row_base = block * PTQ1_WG_ROWS + lane / PTQ1_LPR;

    // Rows past M read row M-1: redundant work instead of predicated loads.
    const unsigned* rp[PTQ1_RL];
    #pragma unroll
    for (int i = 0; i < PTQ1_RL; ++i) {
        const int row = min(row_base + i * (32 / PTQ1_LPR), M - 1);
        rp[i] = (const unsigned*)A + (long long)row * row_words;
    }

    float acc[PTQ1_RL];
    #pragma unroll
    for (int i = 0; i < PTQ1_RL; ++i) acc[i] = 0.0f;

    for (int g = s; g < groups; g += PTQ1_LPR) {
        unsigned w[PTQ1_RL][7];
        #pragma unroll
        for (int i = 0; i < PTQ1_RL; ++i) {
            const unsigned* wp = rp[i] + (long long)g * 7;
            // Plain loads (b128 + b96): nontemporal loads measured 350 GB/s
            // against 450 GB/s for these on gfx1100 at the model's shapes.
            #pragma unroll
            for (int j = 0; j < 7; ++j) w[i][j] = wp[j];
        }
        ptq1_xblock x;
        ptq1_load_x(Xq, g, x);
        #pragma unroll
        for (int i = 0; i < PTQ1_RL; ++i) acc[i] += ptq1_block_dot(w[i], x.xv, x.sx, x.dx);
    }

    #pragma unroll
    for (int i = 0; i < PTQ1_RL; ++i) {
        float a = acc[i];
        #pragma unroll
        for (int off = PTQ1_LPR / 2; off > 0; off >>= 1) a += __shfl_xor(a, off);
        const int row = row_base + i * (32 / PTQ1_LPR);
        if (s == 0 && row < M) y[row] = residual ? y[row] + a : a;
    }
}

// PTQ1 x Q8_1 decode GEMV. See PTQ1_LPR / PTQ1_RL for the lane layout.
__launch_bounds__(32, 16)
extern "C" __global__ void gemv_ptq1g128(
    const unsigned char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ y,
    int M, int K
) {
    ptq1_gemv_rows(A, Xq, y, M, K, blockIdx.x, 0);
}

// One row of a BF16-weight x F32-input GEMV; the arithmetic of
// gemv_bf16_xf32 (kernels/src/gemv_bf16_xf32.hip) verbatim, so the result is
// bit-identical to that kernel's.
static __device__ __forceinline__ void ptq1_bf16_row(
    const unsigned short* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ y,
    int row, int K
) {
    const int tid = threadIdx.x;
    const unsigned short* w_row = W + (size_t)row * K;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    if ((K & 7) == 0) {
        for (int i = tid * 8; i + 8 <= K; i += 32 * 8) {
            const uint4 wv = *(const uint4*)(w_row + i);
            const float4 xa = *(const float4*)(x + i);
            const float4 xb = *(const float4*)(x + i + 4);
            a0 += __uint_as_float(wv.x << 16) * xa.x;
            a1 += __uint_as_float(wv.x & 0xFFFF0000u) * xa.y;
            a2 += __uint_as_float(wv.y << 16) * xa.z;
            a3 += __uint_as_float(wv.y & 0xFFFF0000u) * xa.w;
            a0 += __uint_as_float(wv.z << 16) * xb.x;
            a1 += __uint_as_float(wv.z & 0xFFFF0000u) * xb.y;
            a2 += __uint_as_float(wv.w << 16) * xb.z;
            a3 += __uint_as_float(wv.w & 0xFFFF0000u) * xb.w;
        }
    }
    for (int i = ((K & 7) == 0 ? (K & ~7) : 0) + tid; i < K; i += 32) {
        a0 += __uint_as_float(((unsigned int)w_row[i]) << 16) * x[i];
    }
    float sum = (a0 + a1) + (a2 + a3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[row] = sum;
}

// Up to three PTQ1 matrices that read the same activation (qkv+z, q+k+v,
// gate+up) in one launch, so the short matrices share one ramp and tail and
// the launch count drops. Segment i owns ceil(M_i / PTQ1_WG_ROWS) blocks, in
// order; an unused segment has M_i = 0. The trailing 2*BM blocks are one row
// each of two optional BF16 matrices over the un-rotated input `xf` (the
// DeltaNet beta/alpha projections, which are not Hadamard-folded).
__launch_bounds__(32, 16)
extern "C" __global__ void gemv_ptq1g128_multi(
    const unsigned char* __restrict__ A0, float* __restrict__ y0, int M0,
    const unsigned char* __restrict__ A1, float* __restrict__ y1, int M1,
    const unsigned char* __restrict__ A2, float* __restrict__ y2, int M2,
    const block_q8_1_mmq* __restrict__ Xq,
    int K, int residual,
    const unsigned short* __restrict__ B0, float* __restrict__ yb0,
    const unsigned short* __restrict__ B1, float* __restrict__ yb1,
    int BM, const float* __restrict__ xf
) {
    int b = blockIdx.x;
    const int b0 = (M0 + PTQ1_WG_ROWS - 1) / PTQ1_WG_ROWS;
    if (b < b0) { ptq1_gemv_rows(A0, Xq, y0, M0, K, b, residual); return; }
    b -= b0;
    const int b1 = (M1 + PTQ1_WG_ROWS - 1) / PTQ1_WG_ROWS;
    if (b < b1) { ptq1_gemv_rows(A1, Xq, y1, M1, K, b, residual); return; }
    b -= b1;
    const int b2 = (M2 + PTQ1_WG_ROWS - 1) / PTQ1_WG_ROWS;
    if (b < b2) { ptq1_gemv_rows(A2, Xq, y2, M2, K, b, residual); return; }
    b -= b2;
    if (b < BM) ptq1_bf16_row(B0, xf, yb0, b, K);
    else ptq1_bf16_row(B1, xf, yb1, b - BM, K);
}

#define PTQ1_BATCH_TILE 8
__launch_bounds__(32, 8)
extern "C" __global__ void gemm_ptq1g128_prefill(
    const unsigned char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq, // [K/128 block, N]
    float* __restrict__ y,
    int M, int K, int N
) {
    const int row = blockIdx.x;
    const int n0 = blockIdx.y * PTQ1_BATCH_TILE;
    if (row >= M || n0 >= N) return;
    const int lane = threadIdx.x;
    const int groups = K / 128;
    const unsigned char* row_ptr = A + (long long)row * groups * 28;
    float acc[PTQ1_BATCH_TILE];
    #pragma unroll
    for (int nb=0; nb<PTQ1_BATCH_TILE; ++nb) acc[nb]=0.0f;
    for (int g = 0; g < groups; ++g) {
        const unsigned char* q = row_ptr + g * 28;
        const float dw = (float)(*(const _Float16*)(q + 26));
        const int e = lane * 4;
        const int sub = lane >> 3;
        const int w0=ptq1_trit(q,e), w1=ptq1_trit(q,e+1), w2=ptq1_trit(q,e+2), w3=ptq1_trit(q,e+3);
        #pragma unroll
        for (int nb=0; nb<PTQ1_BATCH_TILE; ++nb) {
            const int n=n0+nb;
            if (n>=N) break;
            const block_q8_1_mmq* xb = Xq + (long long)g*N + n;
            const float dx = (float)xb->ds4[2*sub];
            const signed char* x4=xb->qs+e;
            const int idot=w0*(int)x4[0]+w1*(int)x4[1]+w2*(int)x4[2]+w3*(int)x4[3];
            acc[nb] += dw*dx*(float)idot;
        }
    }
    #pragma unroll
    for (int nb=0; nb<PTQ1_BATCH_TILE; ++nb) {
        const int n=n0+nb;
        if (n>=N) break;
        float sum=acc[nb];
        for (int off=16;off>0;off>>=1) sum+=__shfl_down(sum,off);
        if (lane==0) y[(long long)n*M+row]=sum;
    }
}

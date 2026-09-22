// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Prism PTQ1_0: 128 ternary values in 28 bytes:
// qs[24] (five base-3 trits/byte), qh[2] (four trits/byte), fp16 scale d.
//
// One weight block is 28 B while its matching Q8_1 activation block is 144 B —
// the activation is 5.1x the weight. That ratio is the exact inverse of
// HFQ4/MQ4 (136 B weight vs 144 B activation), which is why row tiling
// amortises the wrong operand there and the right one here: every extra row
// per workgroup reuses the same 144 B Xq block.
#define PTQ1_ROW_TILE 2

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

// Decode PTQ1_ROW_TILE rows and dot each against one shared Q8_1 block. The
// base-3 recurrence is the traversal published by PrismML-Eng/llama.cpp (MIT):
// one running byte per row per region, advanced by v = (v*3) & 0xFF with the
// trit read from (v*3) >> 8. Rows are interleaved so each activation byte is
// loaded once and consumed by every row, which removes R-1 activation loads
// per group AND gives the scheduler R independent decode chains to cover VMEM
// latency. Each row's accumulation order is unchanged from the single-row
// kernel, so results are bit-identical to it.
static __device__ __forceinline__ void ptq1_block_dot_rows(
    const unsigned char* const* q,
    const block_q8_1_mmq* __restrict__ xq,
    float* out
) {
    int sumi[PTQ1_ROW_TILE][4];
    unsigned int v[PTQ1_ROW_TILE];
    #pragma unroll
    for (int r = 0; r < PTQ1_ROW_TILE; ++r) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) sumi[r][k] = 0;
    }

    // qs[0..15]: element e = t*16 + m, five trits per byte.
    #pragma unroll
    for (int m = 0; m < 16; ++m) {
        #pragma unroll
        for (int r = 0; r < PTQ1_ROW_TILE; ++r) v[r] = q[r][m];
        #pragma unroll
        for (int t = 0; t < 5; ++t) {
            const int e = t * 16 + m;
            const int x = (int)xq->qs[e];
            #pragma unroll
            for (int r = 0; r < PTQ1_ROW_TILE; ++r) {
                const unsigned int w = v[r] * 3;
                v[r] = w & 0xFF;
                sumi[r][e >> 5] += ((int)(w >> 8) - 1) * x;
            }
        }
    }

    // qs[16..23]: element e = 80 + t*8 + m.
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        #pragma unroll
        for (int r = 0; r < PTQ1_ROW_TILE; ++r) v[r] = q[r][16 + m];
        #pragma unroll
        for (int t = 0; t < 5; ++t) {
            const int e = 80 + t * 8 + m;
            const int x = (int)xq->qs[e];
            #pragma unroll
            for (int r = 0; r < PTQ1_ROW_TILE; ++r) {
                const unsigned int w = v[r] * 3;
                v[r] = w & 0xFF;
                sumi[r][e >> 5] += ((int)(w >> 8) - 1) * x;
            }
        }
    }

    // qh[0..1]: element e = 120 + t*2 + h, four trits per byte.
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        #pragma unroll
        for (int r = 0; r < PTQ1_ROW_TILE; ++r) v[r] = q[r][24 + h];
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int e = 120 + t * 2 + h;
            const int x = (int)xq->qs[e];
            #pragma unroll
            for (int r = 0; r < PTQ1_ROW_TILE; ++r) {
                const unsigned int w = v[r] * 3;
                v[r] = w & 0xFF;
                sumi[r][3] += ((int)(w >> 8) - 1) * x;
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < PTQ1_ROW_TILE; ++r) {
        float acc = 0.0f;
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            acc += (float)xq->ds4[2 * sub] * (float)sumi[r][sub];
        }
        out[r] = (float)(*(const _Float16*)(q[r] + 26)) * acc;
    }
}

// PTQ1 x Q8_1 decode GEMV. One wave owns PTQ1_ROW_TILE output rows; each lane
// owns whole 128-weight groups and strides across K.
//
// minBlocksPerMultiprocessor is deliberately permissive: this kernel is
// latency-bound (measured ~41% of the card's DRAM peak and ~25% of its integer
// issue capacity at large M), so resident waves — not registers or shared
// memory — are the binding constraint on hiding the trit decode's dependent
// integer chain and the weight loads behind it.
__launch_bounds__(32, 24)
extern "C" __global__ void gemv_ptq1g128(
    const unsigned char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ y,
    int M, int K
) {
    const int row0 = blockIdx.x * PTQ1_ROW_TILE;
    if (row0 >= M) return;
    const int lane = threadIdx.x;
    const int groups = K / 128;
    const long long row_stride = (long long)groups * 28;
    const unsigned char* base = A + (long long)row0 * row_stride;

    // Rows past M fold onto row0: the redundant arithmetic costs less than
    // predicating every load in the hot decode. Their stores are suppressed.
    const unsigned char* q[PTQ1_ROW_TILE];
    #pragma unroll
    for (int r = 0; r < PTQ1_ROW_TILE; ++r) {
        q[r] = (row0 + r < M) ? base + r * row_stride : base;
    }

    float acc[PTQ1_ROW_TILE];
    #pragma unroll
    for (int r = 0; r < PTQ1_ROW_TILE; ++r) acc[r] = 0.0f;

    for (int g = lane; g < groups; g += 32) {
        const unsigned char* qg[PTQ1_ROW_TILE];
        float dot[PTQ1_ROW_TILE];
        #pragma unroll
        for (int r = 0; r < PTQ1_ROW_TILE; ++r) qg[r] = q[r] + g * 28;
        ptq1_block_dot_rows(qg, Xq + g, dot);
        #pragma unroll
        for (int r = 0; r < PTQ1_ROW_TILE; ++r) acc[r] += dot[r];
    }

    #pragma unroll
    for (int r = 0; r < PTQ1_ROW_TILE; ++r) {
        float a = acc[r];
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down(a, off);
        if (lane == 0 && row0 + r < M) y[row0 + r] = a;
    }
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

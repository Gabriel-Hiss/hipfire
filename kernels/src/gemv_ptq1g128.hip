// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Prism PTQ1_0: 128 ternary values in 28 bytes:
// qs[24] (five base-3 trits/byte), qh[2] (four trits/byte), fp16 scale d.
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

// PTQ1 x Q8_1, matching PrismML llama.cpp's GPU vecdot. One lane owns four
// consecutive K elements; every eight lanes share one 32-value Q8_1 scale.
__launch_bounds__(32, 16)
extern "C" __global__ void gemv_ptq1g128(
    const unsigned char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const int groups = K / 128;
    const unsigned char* row_ptr = A + (long long)row * groups * 28;
    float acc = 0.0f;
    for (int g = 0; g < groups; ++g) {
        const unsigned char* q = row_ptr + g * 28;
        const float dw = (float)(*(const _Float16*)(q + 26));
        const int e = lane * 4;
        const int sub = lane >> 3;
        const float dx = (float)Xq[g].ds4[2*sub];
        const signed char* x4 = Xq[g].qs + e;
        const int idot =
            ptq1_trit(q, e+0)*(int)x4[0] + ptq1_trit(q, e+1)*(int)x4[1] +
            ptq1_trit(q, e+2)*(int)x4[2] + ptq1_trit(q, e+3)*(int)x4[3];
        acc += dw * dx * (float)idot;
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    if (lane == 0) y[row] = acc;
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

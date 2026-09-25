// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Non-causal attention for the DFlash draft's shape: at most 16 queries
// against a long context (L in the thousands), F32 Q/K/V, head_dim 128, GQA.
// `attention_dflash_wmma_f32` launches one wave per (head, 16-query tile),
// i.e. n_heads waves (32 on the Qwen3.5 draft), each walking every key tile
// in sequence through LDS round trips, so the device sits mostly idle.
//
// Here a workgroup is one KV head and one contiguous range of 16-key tiles
// (the key axis is split `gridDim.y` ways). Its `rep` waves are the query
// heads sharing that KV head; K and V tiles are staged once into LDS (f16,
// V transposed) and read by all of them. Each wave computes S^T = K Q^T with
// the f16 WMMA, so a query's 16 scores sit in one lane pair and the online
// softmax needs one cross-half shuffle; P is rebuilt in the A layout with one
// more, and O stays in registers. Each workgroup writes the unnormalised
// partial (m, l, O[128]) per (query, head, split) and
// `attention_dflash_split_merge` folds the splits.
//
// Sliding window: key kj is visible to query q iff kj >= key_lo0 + q (the
// DFlash2 non-causal window `|ctx_span + q - kj| < W` gives
// key_lo0 = ctx_span - W + 1; its upper side never binds for B < W). Full
// attention passes a key_lo0 far below zero.
//
// Partials: F32 [B][n_heads][n_splits][130] = m, l, O.
// Grid:  [n_kv_heads, n_splits]; block [32 * (n_heads / n_kv_heads)], rep <= 8.
//
// WMMA (gfx11 wave32, f16 in, f32 acc): D = A B^T. Lane t holds row t & 15 of
// A and of B (lanes 16..31 replicate); acc[j] at lane t is D[2j + (t >> 4)][t & 15].

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define DGS_HD 128
#define DGS_KT 16

extern "C" __launch_bounds__(256, 1)
__global__ void attention_dflash_gqa_split_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ partials,
    int B, int L, int n_heads, int n_kv_heads,
    float scale, int tiles_per_split, int key_lo0
) {
    __shared__ _Float16 K_lds[DGS_KT][DGS_HD];
    __shared__ _Float16 Vt_lds[DGS_HD][DGS_KT];

    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int n_splits = gridDim.y;
    const int rep = n_heads / n_kv_heads;
    const int wave = threadIdx.x >> 5;
    const int t = threadIdx.x & 31;
    const int row = t & 15;          // query (B operand / softmax owner) or key (A operand)
    const int hid = t >> 4;
    const int head = kvh * rep + wave;
    const int q_stride = n_heads * DGS_HD;
    const int kv_stride = n_kv_heads * DGS_HD;

    // Q fragments: lane holds query `row`'s 128 dims as 8 half16 chunks.
    half16_t qf[DGS_HD / 16];
    {
        const float* qr = q + (long long)(row < B ? row : 0) * q_stride + head * DGS_HD;
        #pragma unroll
        for (int c = 0; c < DGS_HD / 16; ++c) {
            #pragma unroll
            for (int j = 0; j < 16; ++j) qf[c][j] = row < B ? (_Float16)qr[c * 16 + j] : (_Float16)0.0f;
        }
    }

    float8_t o[DGS_HD / 16];
    #pragma unroll
    for (int c = 0; c < DGS_HD / 16; ++c) o[c] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
    float m_run = -INFINITY;   // for query `row`
    float l_run = 0.0f;

    // No query sees a key below key_lo0, so ranges wholly below it are skipped.
    const int k_begin = split * tiles_per_split * DGS_KT;
    int k_end = k_begin + tiles_per_split * DGS_KT;
    if (k_end > L) k_end = L;
    int kt0 = k_begin;
    if (key_lo0 > kt0) kt0 = k_begin + ((key_lo0 - k_begin) / DGS_KT) * DGS_KT;

    for (int kt = kt0; kt < k_end; kt += DGS_KT) {
        // Stage K [16 x 128] and V^T [128 x 16] as f16; keys past k_end are zero.
        for (int i = threadIdx.x; i < DGS_KT * DGS_HD / 4; i += blockDim.x) {
            const int key = i / (DGS_HD / 4);
            const int d = (i % (DGS_HD / 4)) * 4;
            const bool ok = kt + key < k_end;
            const long long off = (long long)(kt + key) * kv_stride + kvh * DGS_HD + d;
            const float4 kk = ok ? *(const float4*)(k + off) : float4{0, 0, 0, 0};
            const float4 vv = ok ? *(const float4*)(v + off) : float4{0, 0, 0, 0};
            K_lds[key][d + 0] = (_Float16)kk.x;
            K_lds[key][d + 1] = (_Float16)kk.y;
            K_lds[key][d + 2] = (_Float16)kk.z;
            K_lds[key][d + 3] = (_Float16)kk.w;
            Vt_lds[d + 0][key] = (_Float16)vv.x;
            Vt_lds[d + 1][key] = (_Float16)vv.y;
            Vt_lds[d + 2][key] = (_Float16)vv.z;
            Vt_lds[d + 3][key] = (_Float16)vv.w;
        }
        __syncthreads();

        if (wave < rep) {
            // S^T[key][query]: A = K rows (lane = key), B = Q rows (lane = query).
            float8_t s = {0, 0, 0, 0, 0, 0, 0, 0};
            #pragma unroll
            for (int c = 0; c < DGS_HD / 16; ++c) {
                const half16_t a = *(const half16_t*)&K_lds[row][c * 16];
                s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, qf[c], s);
            }
            // s[j] at lane t: key kt + 2j + hid, query row.
            float tmax = -INFINITY;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int key = kt + 2 * j + hid;
                s[j] = (key < k_end && key >= key_lo0 + row) ? s[j] * scale : -INFINITY;
                tmax = fmaxf(tmax, s[j]);
            }
            tmax = fmaxf(tmax, __shfl_xor(tmax, 16));
            const float m_new = fmaxf(m_run, tmax);
            // A query whose keys are all masked so far keeps m = -inf: no rescale, p = 0.
            const bool none = m_new == -INFINITY;
            const float alpha = none ? 1.0f : expf(m_run - m_new);
            float p[8];
            float psum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                p[j] = none ? 0.0f : expf(s[j] - m_new);
                psum += p[j];
            }
            psum += __shfl_xor(psum, 16);
            l_run = alpha * l_run + psum;
            m_run = m_new;

            // A = P rows (lane = query): key 2j + hid is local, the other parity
            // lives in the partner half.
            half16_t pa;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float other = __shfl_xor(p[j], 16);
                pa[2 * j + hid] = (_Float16)p[j];
                pa[2 * j + (hid ^ 1)] = (_Float16)other;
            }
            // O rows are queries 2j + hid: their rescale factors live on lane 2j + hid.
            float arow[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) arow[j] = __shfl(alpha, 2 * j + hid);
            #pragma unroll
            for (int c = 0; c < DGS_HD / 16; ++c) {
                float8_t acc = o[c];
                #pragma unroll
                for (int j = 0; j < 8; ++j) acc[j] *= arow[j];
                const half16_t b = *(const half16_t*)&Vt_lds[c * 16 + row][0];
                o[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, b, acc);
            }
        }
        __syncthreads();
    }

    if (wave >= rep) return;
    const int stride = 2 + DGS_HD;
    if (t < 16 && row < B) {
        float* pr = partials + ((long long)(row * n_heads + head) * n_splits + split) * stride;
        pr[0] = m_run;
        pr[1] = l_run;
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int qi = 2 * j + hid;
        if (qi >= B) continue;
        float* pr = partials + ((long long)(qi * n_heads + head) * n_splits + split) * stride + 2;
        #pragma unroll
        for (int c = 0; c < DGS_HD / 16; ++c) pr[c * 16 + row] = o[c][j];
    }
}

// out[q, head, :] = sum_s O_s e^{m_s - M} / sum_s l_s e^{m_s - M}, M = max_s m_s.
// Grid: [n_heads, B]; block: [head_dim].
extern "C" __global__ void attention_dflash_split_merge(
    const float* __restrict__ partials,
    float* __restrict__ out,
    int n_heads, int head_dim, int n_splits
) {
    const int head = blockIdx.x;
    const int qi = blockIdx.y;
    const int d = threadIdx.x;
    const int stride = 2 + head_dim;
    const float* p = partials + (long long)(qi * n_heads + head) * n_splits * stride;
    float m = -INFINITY;
    for (int s = 0; s < n_splits; ++s) m = fmaxf(m, p[s * stride]);
    float l = 0.0f, o = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        const float ms = p[s * stride];
        if (ms == -INFINITY) continue;
        const float w = expf(ms - m);
        l += w * p[s * stride + 1];
        o += w * p[s * stride + 2 + d];
    }
    if (d < head_dim) out[(long long)(qi * n_heads + head) * head_dim + d] = l > 0.0f ? o / l : 0.0f;
}

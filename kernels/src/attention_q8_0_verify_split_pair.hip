// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Q8_0-KV causal attention for a speculative verify block (at most 16 queries)
// on gfx11 at head_dim 256, split over the key axis.
//
// attention_q8_0_flash_prefill_wmma_pair serves 64 queries of one head per
// workgroup; a 16-query verify leaves three of its four wave pairs idle and
// launches n_heads workgroups in total, each walking every key tile, so at a
// 1.3k-token context the Bonsai target spent ~0.33 ms per attention layer.
// Here a workgroup is one KV head and one range of 16-key tiles: its
// n_heads / n_kv_heads wave pairs are the query heads sharing that KV head and
// read each K/V tile dequantized once into LDS. Per pair, the arithmetic is
// the pair kernel's: each wave owns half of head_dim for Q and O, the two
// halves of S = Q K^T are summed through LDS (low + high), the online softmax
// runs in registers and P moves to the A layout through LDS. Each workgroup
// writes the unnormalised partial (m, l, O[256]) per (query, head, range) and
// attention_dflash_split_merge folds the ranges.
//
// The key ranges come from positions[] on the device (the causal bound is the
// last query's position), so the grid depends only on the partials capacity
// and the launch stays valid under graph replay.
//
// A range is at least `min_tiles` tiles: fewer, longer ranges trade workgroup
// count for less partial traffic (16 x n_heads x 258 floats per range).
//
// Partials: F32 [batch][n_heads][n_splits][258]. Grid: [n_kv_heads, n_splits].
// Block: [64 * (n_heads / n_kv_heads)], at most 8 pairs.

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(16))) vs_half16;
typedef float __attribute__((ext_vector_type(8))) vs_float8;

#define VS_HD 256
#define VS_HALF 128
#define VS_DC 8
#define VS_MAXP 8
#define VS_KS (VS_HD + 8)
#define VS_VS 24
#define VS_PS 24

extern "C" __global__ __launch_bounds__(512, 1) void attention_q8_0_verify_split_pair(
    const float* __restrict__ q,               // [batch x n_heads x 256]
    const unsigned char* __restrict__ k_cache, // [pos][n_kv_heads x 8][34]
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ positions,
    int n_heads, int n_kv_heads, int batch_size, float scale_attn, int min_tiles
) {
    __shared__ _Float16 K_lds[16 * VS_KS];
    __shared__ _Float16 V_lds_T[VS_HD * VS_VS];
    __shared__ float S_x[VS_MAXP][2][8][32];
    __shared__ _Float16 P_lds[2 * VS_MAXP][16 * VS_PS];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int half = lane & 15;
    const int half_id = lane >> 4;
    const int pair = wave >> 1;
    const int hsel = wave & 1;
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int n_splits = gridDim.y;
    const int rep = n_heads / n_kv_heads;
    const int h = kvh * rep + pair;
    const int row_stride = n_kv_heads * 8 * 34;
    const int kv_off = kvh * 8 * 34;
    const int q_stride = n_heads * VS_HD;
    const int dbase = hsel * VS_HALF;

    vs_half16 qf[VS_DC];
    {
        const bool ok = half < batch_size;
        const float* src = q + (long long)min(half, batch_size - 1) * q_stride + h * VS_HD + dbase;
#pragma unroll
        for (int dc = 0; dc < VS_DC; ++dc)
#pragma unroll
            for (int j = 0; j < 16; ++j) qf[dc][j] = (_Float16)(ok ? src[dc * 16 + j] : 0.0f);
    }
    int rpos[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int r = 2 * j + half_id;
        rpos[j] = r < batch_size ? positions[r] : -1;
    }
    int seq_len = 0;
    for (int r = 0; r < batch_size; ++r) seq_len = max(seq_len, positions[r] + 1);
    const int tiles = (seq_len + 15) / 16;
    const int tps = max(min_tiles, (tiles + n_splits - 1) / n_splits);
    const int t_begin = split * tps;
    const int t_end = min(tiles, t_begin + tps);

    vs_float8 O[VS_DC];
#pragma unroll
    for (int dc = 0; dc < VS_DC; ++dc) O[dc] = (vs_float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float m_run[8], l_run[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        m_run[j] = -INFINITY;
        l_run[j] = 0.0f;
    }

    for (int ti = t_begin; ti < t_end; ++ti) {
        const int kt = ti * 16;
        const int t_len = min(16, seq_len - kt);
        // Stage: thread -> (K or V, key, 32-dim block), f16(scale * int8).
        if (tid < 256) {
            const bool is_v = tid >= 128;
            const int jj = tid & 127;
            const int kr = jj >> 3;
            const int blk = jj & 7;
            _Float16 vals[32];
            if (kr < t_len) {
                const unsigned short* src = (const unsigned short*)((is_v ? v_cache : k_cache)
                    + (long long)(kt + kr) * row_stride + kv_off + blk * 34);
                unsigned short raw[17];
#pragma unroll
                for (int e = 0; e < 17; ++e) raw[e] = src[e];
                const float s = (float)*(const _Float16*)&raw[0];
#pragma unroll
                for (int e = 0; e < 16; ++e) {
                    vals[2 * e] = (_Float16)(s * (float)(signed char)(raw[e + 1] & 0xFF));
                    vals[2 * e + 1] = (_Float16)(s * (float)(signed char)(raw[e + 1] >> 8));
                }
            } else {
#pragma unroll
                for (int e = 0; e < 32; ++e) vals[e] = (_Float16)0.0f;
            }
            if (is_v) {
#pragma unroll
                for (int e = 0; e < 32; ++e) V_lds_T[(blk * 32 + e) * VS_VS + kr] = vals[e];
            } else {
                vs_half16* dst = (vs_half16*)(K_lds + kr * VS_KS + blk * 32);
                vs_half16 a, b;
#pragma unroll
                for (int e = 0; e < 16; ++e) {
                    a[e] = vals[e];
                    b[e] = vals[16 + e];
                }
                dst[0] = a;
                dst[1] = b;
            }
        }
        __syncthreads();

        vs_float8 s_part = (vs_float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int dc = 0; dc < VS_DC; ++dc) {
            const vs_half16 kb = *(const vs_half16*)(K_lds + half * VS_KS + dbase + dc * 16);
            s_part = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qf[dc], kb, s_part);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) S_x[pair][hsel][j][lane] = s_part[j];
        __syncthreads();

        float p[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float s = (S_x[pair][0][j][lane] + S_x[pair][1][j][lane]) * scale_attn;
            const bool valid = (half < t_len) && (kt + half <= rpos[j]);
            const float sv = valid ? s : -INFINITY;
            float tm = sv;
#pragma unroll
            for (int o = 1; o < 16; o <<= 1) tm = fmaxf(tm, __shfl_xor(tm, o, 32));
            const float m_new = fmaxf(m_run[j], tm);
            const float alpha = (m_run[j] == -INFINITY) ? 0.0f : __expf(m_run[j] - m_new);
            const float e = (m_new == -INFINITY) ? 0.0f : __expf(sv - m_new);
            float sum = e;
#pragma unroll
            for (int o = 1; o < 16; o <<= 1) sum += __shfl_xor(sum, o, 32);
            m_run[j] = m_new;
            l_run[j] = l_run[j] * alpha + sum;
            p[j] = e;
#pragma unroll
            for (int dc = 0; dc < VS_DC; ++dc) O[dc][j] *= alpha;
        }
        _Float16* pw = P_lds[wave];
#pragma unroll
        for (int j = 0; j < 8; ++j) pw[(2 * j + half_id) * VS_PS + half] = (_Float16)p[j];
        __syncthreads();
        const vs_half16 pa = *(const vs_half16*)(pw + half * VS_PS);
#pragma unroll
        for (int dc = 0; dc < VS_DC; ++dc) {
            const vs_half16 vb = *(const vs_half16*)(V_lds_T + (dbase + dc * 16 + half) * VS_VS);
            O[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, vb, O[dc]);
        }
        __syncthreads();
    }

    // Partial per row r = 2j + half_id: m, l once per row (wave 0 of the pair,
    // lane half 0), O by every lane for its dims. An empty range writes only
    // m = -inf, which the merge skips without reading O.
    const int stride = 2 + VS_HD;
    const bool empty = t_begin >= t_end;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int r = 2 * j + half_id;
        if (r >= batch_size) continue;
        float* pr = partials + ((long long)(r * n_heads + h) * n_splits + split) * stride;
        if (hsel == 0 && half == 0) {
            pr[0] = m_run[j];
            pr[1] = l_run[j];
        }
        if (empty) continue;
#pragma unroll
        for (int dc = 0; dc < VS_DC; ++dc) pr[2 + dbase + dc * 16 + half] = O[dc][j];
    }
}

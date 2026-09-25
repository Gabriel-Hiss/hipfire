// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Q8_0 flash prefill attention for gfx11 at head_dim 256, the default over
// attention_q8_0_flash_prefill_wmma for that shape.
//
// Why a different layout. The one-wave kernel keeps a 16-query x 256-dim f16
// Q tile, the V^T tile and S in 18 KB of LDS per wave, so a CU holds three
// waves (1.5 per SIMD) and every phase of the online softmax waits on the
// previous one; it measured ~7% of the f16 WMMA rate at 512-token chunks.
// Here a PAIR of waves serves 16 queries, each wave owning half of head_dim:
// its Q half (8 A fragments) and O half (8 accumulators) stay in registers.
// Four pairs (64 queries of one head) share each 16-key K/V tile, dequantized
// once per workgroup by all 256 threads with 2-byte loads. The two halves of
// S = Q K^T are summed through LDS in a fixed order (low half + high half),
// so both waves of a pair hold the same S and run the same online softmax in
// registers: row max and row sum reduce across the 16 lanes of a half-wave,
// and the rescale factor lands on O's rows without leaving the lane. Only P
// goes through LDS, to move it from the C to the A fragment layout.
//
// Precision: K and V are dequantized to f16 exactly as the one-wave kernel
// does; S and the softmax stay f32 (the one-wave kernel rounds S to f16
// before the max and the exponent), P is rounded to f16 for the P x V WMMA.
// Measured against an f64 CPU reference on random Q/K/V: relative L2 5.5e-4
// (one-wave kernel 1.1e-3) at 512 queries over 512 and 2048 keys.
//
// Grid: [ceil(batch_size / 64), n_heads]. Block: [256]. Static LDS ~34 KB.
// Single-slot addressing: positions[] carries each row's causal bound.

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(16))) ap_half16;
typedef float __attribute__((ext_vector_type(8))) ap_float8;

#define AP_HD 256
#define AP_HALF 128            // dims per wave
#define AP_DC 8                // d-chunks per wave
#define AP_PAIRS 4
#define AP_Q (16 * AP_PAIRS)   // queries per workgroup
#define AP_KS (AP_HD + 8)      // K row stride (f16)
#define AP_VS 24               // V^T row stride (f16)
#define AP_PS 24               // P row stride (f16)

extern "C" __global__ __launch_bounds__(256, 1) void attention_q8_0_flash_prefill_wmma_pair(
    const float* __restrict__ q,               // [batch x n_heads x 256]
    const unsigned char* __restrict__ k_cache, // [pos][n_kv_heads x 8][34]
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ positions,
    int n_heads, int n_kv_heads, int batch_size, float scale_attn
) {
    __shared__ _Float16 K_lds[16 * AP_KS];
    __shared__ _Float16 V_lds_T[AP_HD * AP_VS];
    __shared__ float S_x[AP_PAIRS][2][8][32];
    __shared__ _Float16 P_lds[8][16 * AP_PS];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int half = lane & 15;
    const int half_id = lane >> 4;
    const int pair = wave >> 1;
    const int hsel = wave & 1;
    const int h = blockIdx.y;
    const int wg_q0 = blockIdx.x * AP_Q;
    const int q_base = wg_q0 + pair * 16;
    const int row_stride = n_kv_heads * 8 * 34;
    const int kv_off = (h / (n_heads / n_kv_heads)) * 8 * 34;
    const int q_stride = n_heads * AP_HD;
    const int dbase = hsel * AP_HALF;

    // Q fragments: this lane's query row (half), 16 dims per d-chunk.
    ap_half16 qf[AP_DC];
    {
        const int qr = q_base + half;
        const bool ok = qr < batch_size;
        const float* src = q + (long long)min(qr, batch_size - 1) * q_stride + h * AP_HD + dbase;
#pragma unroll
        for (int dc = 0; dc < AP_DC; ++dc)
#pragma unroll
            for (int j = 0; j < 16; ++j) qf[dc][j] = (_Float16)(ok ? src[dc * 16 + j] : 0.0f);
    }
    // Positions of this lane's 8 C-layout rows (2j + half_id); -1 = no row.
    int rpos[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int r = q_base + 2 * j + half_id;
        rpos[j] = r < batch_size ? positions[r] : -1;
    }
    int seq_len = 0;
    for (int r = 0; r < AP_Q && wg_q0 + r < batch_size; ++r) seq_len = max(seq_len, positions[wg_q0 + r] + 1);

    ap_float8 O[AP_DC];
#pragma unroll
    for (int dc = 0; dc < AP_DC; ++dc) O[dc] = (ap_float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float m_run[8], l_run[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        m_run[j] = -INFINITY;
        l_run[j] = 0.0f;
    }

    for (int kt = 0; kt < seq_len; kt += 16) {
        const int t_len = min(16, seq_len - kt);
        // Stage: thread -> (K or V, key, 32-dim block), f16(scale * int8).
        {
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
                for (int e = 0; e < 32; ++e) V_lds_T[(blk * 32 + e) * AP_VS + kr] = vals[e];
            } else {
                ap_half16* dst = (ap_half16*)(K_lds + kr * AP_KS + blk * 32);
                ap_half16 a, b;
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

        // Partial S over this wave's 128 dims.
        ap_float8 s_part = (ap_float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int dc = 0; dc < AP_DC; ++dc) {
            const ap_half16 kb = *(const ap_half16*)(K_lds + half * AP_KS + dbase + dc * 16);
            s_part = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qf[dc], kb, s_part);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) S_x[pair][hsel][j][lane] = s_part[j];
        __syncthreads();

        // Full S (low half + high half, same order in both waves), mask,
        // online softmax per row across the 16 lanes of this half-wave.
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
            for (int dc = 0; dc < AP_DC; ++dc) O[dc][j] *= alpha;
        }
        // P from C layout (lane = key, j = row) to A layout (lane = row).
        _Float16* pw = P_lds[wave];
#pragma unroll
        for (int j = 0; j < 8; ++j) pw[(2 * j + half_id) * AP_PS + half] = (_Float16)p[j];
        __syncthreads();
        const ap_half16 pa = *(const ap_half16*)(pw + half * AP_PS);
#pragma unroll
        for (int dc = 0; dc < AP_DC; ++dc) {
            const ap_half16 vb = *(const ap_half16*)(V_lds_T + (dbase + dc * 16 + half) * AP_VS);
            O[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, vb, O[dc]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int dc = 0; dc < AP_DC; ++dc) {
        const int d = dbase + dc * 16 + half;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int r = q_base + 2 * j + half_id;
            if (r >= batch_size) continue;
            const float inv = (l_run[j] > 0.0f) ? (1.0f / l_run[j]) : 0.0f;
            out[(long long)r * q_stride + h * AP_HD + d] = O[dc][j] * inv;
        }
    }
}

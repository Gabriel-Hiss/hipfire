// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// conv1d + silu + Q/K/V split fused into one kernel.
//
// Replaces the sequence
//   conv1d_silu_f32(dn_conv_out, dn_qkv, weight, state, qkv_dim);
//   memcpy_dtod_at(q_out, 0, dn_conv_out, 0,           k_dim * 4);
//   memcpy_dtod_at(k_out, 0, dn_conv_out, k_dim*4,     k_dim * 4);
//   memcpy_dtod_at(v_out, 0, dn_conv_out, k_dim*2*4,   v_dim * 4);
//
// with a single launch. Each thread processes one channel: does the 1D
// conv + SiLU in registers (bit-identical to conv1d_silu.hip), then
// writes to q_out / k_out / v_out depending on channel index.
//
// Channel layout: [Q (k_dim) | K (k_dim) | V (v_dim)] in the `input`
// buffer, matching the wqkv projection output. Total n_channels =
// 2*k_dim + v_dim.
//
// blockIdx.y selects an independent sequence lane.  The normal prefill path
// launches grid.y=1 and is byte-identical to the original implementation.
// Continuous-batch decode launches grid.y=batch with n_tokens=1; input/output
// are `[lane, token, channel]` and state is `[lane, channel, 3]`.

extern "C" __global__ void conv1d_silu_split_f32(
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ state,
    int k_dim,
    int v_dim,
    int n_tokens
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = blockIdx.y;
    int n_channels = 2 * k_dim + v_dim;
    if (c >= n_channels) return;

    // Loop over tokens, performing exactly the same math and memory ops as
    // the single-token version did before the loop was added. Keeping the
    // weight[] indexing in-place (not hoisted) is LOAD-BEARING for
    // byte-exact output against the single-token baseline — hoisting the
    // loads changes the compiler's FMA fusion decisions and nudges the
    // numerics by ~1 ULP, which cascades over 2k+ greedy tokens.
    for (int t = 0; t < n_tokens; t++) {
        // Same math as conv1d_silu.hip — 4-tap causal depthwise conv + SiLU.
        int row = lane * n_tokens + t;
        int state_base = lane * n_channels * 3;
        float x = input[row * n_channels + c];
        float s0 = state[state_base + c * 3];
        float s1 = state[state_base + c * 3 + 1];
        float s2 = state[state_base + c * 3 + 2];

        float y = weight[c * 4 + 3] * x + weight[c * 4 + 2] * s0
                + weight[c * 4 + 1] * s1 + weight[c * 4] * s2;

        // SiLU
        float result = y / (1.0f + expf(-y));

        // Update ring buffer state
        state[state_base + c * 3 + 2] = s1;
        state[state_base + c * 3 + 1] = s0;
        state[state_base + c * 3]     = x;

        // Write to Q/K/V based on channel index.
        if (c < k_dim) {
            q_out[row * k_dim + c] = result;
        } else if (c < 2 * k_dim) {
            k_out[row * k_dim + (c - k_dim)] = result;
        } else {
            v_out[row * v_dim + (c - 2 * k_dim)] = result;
        }
    }
}

// Masked sibling: same conv1d + SiLU + Q/K/V split but lane-masked.
// Derives physical lane from blockIdx.y (existing grid y) and returns before
// any inactive-lane read/write. Full-mask callers use the unmasked
// `conv1d_silu_split_f32` path for exact parity.
extern "C" __global__ void conv1d_silu_split_f32_masked(
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ state,
    int k_dim,
    int v_dim,
    int n_tokens,
    unsigned long long active_mask
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = blockIdx.y;
    if (((active_mask >> lane) & 1ULL) == 0) return;
    int n_channels = 2 * k_dim + v_dim;
    if (c >= n_channels) return;

    for (int t = 0; t < n_tokens; t++) {
        int row = lane * n_tokens + t;
        int state_base = lane * n_channels * 3;
        float x = input[row * n_channels + c];
        float s0 = state[state_base + c * 3];
        float s1 = state[state_base + c * 3 + 1];
        float s2 = state[state_base + c * 3 + 2];

        float y = weight[c * 4 + 3] * x + weight[c * 4 + 2] * s0
                + weight[c * 4 + 1] * s1 + weight[c * 4] * s2;

        float result = y / (1.0f + expf(-y));

        state[state_base + c * 3 + 2] = s1;
        state[state_base + c * 3 + 1] = s0;
        state[state_base + c * 3]     = x;

        if (c < k_dim) {
            q_out[row * k_dim + c] = result;
        } else if (c < 2 * k_dim) {
            k_out[row * k_dim + (c - k_dim)] = result;
        } else {
            v_out[row * v_dim + (c - 2 * k_dim)] = result;
        }
    }
}

// Token-parallel prefill form of conv1d_silu_split_f32 (single sequence):
// thread (c, t) computes token t of channel c from input rows t-3..t, falling
// back to the ring state for rows before the chunk. The convolution has no
// recurrence, only the ring state carries across launches, so the serial
// token loop above (40 workgroups for Bonsai's 10240 channels, the state read
// and rewritten in global memory every token) is unnecessary here. Same
// expression and FMA order, bit-identical outputs and state; measured
// 142 -> 51 us for 512 tokens x 10240 channels on gfx1100.
// Grid: [ceil(n_channels / 256), n_tokens]. conv1d_state_advance must run
// after it: it is the only writer of the ring.
extern "C" __global__ void conv1d_silu_split_f32_tp(
    float* __restrict__ q_out, float* __restrict__ k_out, float* __restrict__ v_out,
    const float* __restrict__ input, const float* __restrict__ weight, const float* __restrict__ state,
    int k_dim, int v_dim, int n_tokens
) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    const int n_channels = 2 * k_dim + v_dim;
    if (c >= n_channels || t >= n_tokens) return;
    const float x = input[(long long)t * n_channels + c];
    const float s0 = t >= 1 ? input[(long long)(t - 1) * n_channels + c] : state[c * 3];
    const float s1 = t >= 2 ? input[(long long)(t - 2) * n_channels + c] : state[c * 3 + (1 - t)];
    const float s2 = t >= 3 ? input[(long long)(t - 3) * n_channels + c] : state[c * 3 + (2 - t)];
    float y = weight[c * 4 + 3] * x + weight[c * 4 + 2] * s0
            + weight[c * 4 + 1] * s1 + weight[c * 4] * s2;
    float result = y / (1.0f + expf(-y));
    if (c < k_dim) q_out[(long long)t * k_dim + c] = result;
    else if (c < 2 * k_dim) k_out[(long long)t * k_dim + (c - k_dim)] = result;
    else v_out[(long long)t * v_dim + (c - 2 * k_dim)] = result;
}

// Ring state after n_tokens inputs: [x_{n-1}, x_{n-2}, x_{n-3}], shifting in
// the old state when n_tokens < 3.
extern "C" __global__ void conv1d_state_advance(
    const float* __restrict__ input, float* __restrict__ state, int n_channels, int n_tokens
) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_channels) return;
    const float o0 = state[c * 3], o1 = state[c * 3 + 1], o2 = state[c * 3 + 2];
    float r[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int t = n_tokens - 1 - j;  // row feeding slot j
        if (t >= 0) r[j] = input[(long long)t * n_channels + c];
        else {
            const int back = -t - 1;      // how far into the old ring
            r[j] = back == 0 ? o0 : back == 1 ? o1 : o2;
        }
    }
    state[c * 3] = r[0];
    state[c * 3 + 1] = r[1];
    state[c * 3 + 2] = r[2];
}

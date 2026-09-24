// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Decode producers for PTQ1 (Prism Hadamard) GEMV inputs. Each kernel fuses
// one elementwise producer with the Prism rotation and the Q8_1 activation
// quantization that the PTQ1 GEMV reads, replacing three launches (producer,
// rotate_x_prism_hadamard, quantize_q8_1_mmq_ds4) with one.
//
// Every stage repeats the arithmetic of the kernel it replaces in the same
// order, so the Q8_1 output is bit-identical to the unfused chain:
//   producer   rmsnorm_f32 / silu_mul_f32 / gated_norm_f32 / sigmoid_mul_f32
//   rotation   rotate_x_prism_hadamard (signs, then H, then 1/sqrt(block))
//   quantize   quantize_q8_1_mmq_ds4 (8 lanes x 4 values per 32-value scale)
//
// Grid: one 256-thread workgroup per 1024-wide Hadamard block; thread t owns
// block elements [4t, 4t+4). A block never reads another block's values after
// the producer, so blocks are independent except for rmsnorm's whole-vector
// sum, which each workgroup recomputes.
//
// The butterflies run in the reference order (stride 1, 2, ..., 512) with the
// reference pairing, so they are bit-identical to the LDS loop they replace:
// strides 1-2 stay inside a thread, 4-64 cross lanes of one wave by shuffle,
// and 128-512 cross waves through a double-buffered LDS tile (one barrier
// per stage instead of one per stride).

#define PRQ_BLOCK 1024
#define PRQ_THREADS 256

struct prq_block_q8_1_mmq {
    _Float16 ds4[8];
    signed char qs[128];
};
static_assert(sizeof(prq_block_q8_1_mmq) == 144, "bad Q8_1 layout");

// v holds sign-applied elements [4t, 4t+4) of block `blk`: normalized H, then
// the Q8_1 quantization into Y. `tile` is 2 * PRQ_BLOCK floats of LDS.
static __device__ __forceinline__ void prq_rotate_quantize(
    float v[4],
    float* tile,
    prq_block_q8_1_mmq* __restrict__ Y,
    int blk
) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    // stride 1: (0,1), (2,3); stride 2: (0,2), (1,3)
    {
        const float a0 = v[0], a1 = v[1], a2 = v[2], a3 = v[3];
        v[0] = a0 + a1; v[1] = a0 - a1; v[2] = a2 + a3; v[3] = a2 - a3;
    }
    {
        const float a0 = v[0], a1 = v[1], a2 = v[2], a3 = v[3];
        v[0] = a0 + a2; v[2] = a0 - a2; v[1] = a1 + a3; v[3] = a1 - a3;
    }
    // strides 4..64: partner lane = lane ^ (stride / 4), same element slot.
    #pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const bool upper = (lane & m) != 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float p = __shfl_xor(v[j], m, 32);
            v[j] = upper ? p - v[j] : v[j] + p;
        }
    }
    // strides 128..512: across waves, double-buffered so each stage needs
    // one barrier.
    #pragma unroll
    for (int st = 0; st < 3; ++st) {
        float* buf = tile + (st & 1) * PRQ_BLOCK;
        const int stride = 128 << st;
        #pragma unroll
        for (int j = 0; j < 4; ++j) buf[4 * tid + j] = v[j];
        __syncthreads();
        const bool upper = ((4 * tid) & stride) != 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float p = buf[(4 * tid + j) ^ stride];
            v[j] = upper ? p - v[j] : v[j] + p;
        }
    }

    const float scale = rsqrtf((float)PRQ_BLOCK);
    const float x0 = v[0] * scale;
    const float x1 = v[1] * scale;
    const float x2 = v[2] * scale;
    const float x3 = v[3] * scale;

    float amax = fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3)));
    for (int offset = 4; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    }
    float sum = x0 + x1 + x2 + x3;
    for (int offset = 4; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset, 32);
    }
    const float d = amax > 1.0e-20f ? amax / 127.0f : 0.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    char4 q;
    q.x = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, x0 * id)));
    q.y = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, x1 * id)));
    q.z = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, x2 * id)));
    q.w = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, x3 * id)));

    const int e = blk * PRQ_BLOCK + 4 * tid;
    prq_block_q8_1_mmq* out = Y + e / 128;
    const int iqs = e & 127;
    ((char4*)out->qs)[iqs / 4] = q;
    if ((iqs & 31) == 0) {
        out->ds4[2 * (iqs / 32)] = (_Float16)d;
        out->ds4[2 * (iqs / 32) + 1] = (_Float16)sum;
    }
}

// rmsnorm_f32 -> rotate -> Q8_1. The sum of squares keeps rmsnorm_f32's
// order (256-thread strided partials, then the halving tree, whose last five
// levels are the wave-local shfl_down below). `plain`, when non-null,
// receives the un-rotated normalized vector for projections that are not
// Hadamard-folded (Bonsai's BF16 alpha/beta).
extern "C" __launch_bounds__(PRQ_THREADS, 1)
__global__ void ptq1_rmsnorm_rotate_q8(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ signs,
    float* __restrict__ plain,
    prq_block_q8_1_mmq* __restrict__ Y,
    int n, float eps
) {
    __shared__ float tile[2 * PRQ_BLOCK];
    __shared__ float sdata[PRQ_THREADS];
    const int tid = threadIdx.x;
    const int blk = blockIdx.x;
    const int base = blk * PRQ_BLOCK + 4 * tid;

    const float4 xv = *(const float4*)(x + base);
    const float4 wv = *(const float4*)(weight + base);
    const float4 sv = *(const float4*)(signs + base);

    // Loads issued four at a time, summed in the reference order.
    float sum_sq = 0.0f;
    int i = tid;
    for (; i + 3 * PRQ_THREADS < n; i += 4 * PRQ_THREADS) {
        const float t0 = x[i];
        const float t1 = x[i + PRQ_THREADS];
        const float t2 = x[i + 2 * PRQ_THREADS];
        const float t3 = x[i + 3 * PRQ_THREADS];
        sum_sq += t0 * t0;
        sum_sq += t1 * t1;
        sum_sq += t2 * t2;
        sum_sq += t3 * t3;
    }
    for (; i < n; i += PRQ_THREADS) {
        const float t = x[i];
        sum_sq += t * t;
    }
    sdata[tid] = sum_sq;
    __syncthreads();
    #pragma unroll
    for (int s = PRQ_THREADS / 2; s >= 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float r = tid < 32 ? sdata[tid] : 0.0f;
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) r += __shfl_down(r, s, 32);
    if (tid == 0) sdata[0] = r;
    __syncthreads();
    const float rms = rsqrtf(sdata[0] / (float)n + eps);

    float v[4] = {xv.x * wv.x * rms, xv.y * wv.y * rms, xv.z * wv.z * rms, xv.w * wv.w * rms};
    if (plain) *(float4*)(plain + base) = make_float4(v[0], v[1], v[2], v[3]);
    v[0] *= sv.x; v[1] *= sv.y; v[2] *= sv.z; v[3] *= sv.w;
    prq_rotate_quantize(v, tile, Y, blk);
}

// silu_mul_f32 -> rotate -> Q8_1 (the FFN down-projection input).
extern "C" __launch_bounds__(PRQ_THREADS, 1)
__global__ void ptq1_silu_mul_rotate_q8(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ signs,
    prq_block_q8_1_mmq* __restrict__ Y
) {
    __shared__ float tile[2 * PRQ_BLOCK];
    const int base = blockIdx.x * PRQ_BLOCK + 4 * threadIdx.x;
    const float4 g = *(const float4*)(gate + base);
    const float4 u = *(const float4*)(up + base);
    const float4 sv = *(const float4*)(signs + base);
    float v[4] = {
        ((g.x / (1.0f + expf(-g.x))) * u.x) * sv.x,
        ((g.y / (1.0f + expf(-g.y))) * u.y) * sv.y,
        ((g.z / (1.0f + expf(-g.z))) * u.z) * sv.z,
        ((g.w / (1.0f + expf(-g.w))) * u.w) * sv.w,
    };
    prq_rotate_quantize(v, tile, Y, blockIdx.x);
}

// sigmoid_mul_f32 (attention output gate) -> rotate -> Q8_1 (o_proj input).
extern "C" __launch_bounds__(PRQ_THREADS, 1)
__global__ void ptq1_sigmoid_mul_rotate_q8(
    const float* __restrict__ x,
    const float* __restrict__ gate,
    const float* __restrict__ signs,
    prq_block_q8_1_mmq* __restrict__ Y
) {
    __shared__ float tile[2 * PRQ_BLOCK];
    const int base = blockIdx.x * PRQ_BLOCK + 4 * threadIdx.x;
    const float4 xv = *(const float4*)(x + base);
    const float4 g = *(const float4*)(gate + base);
    const float4 sv = *(const float4*)(signs + base);
    float v[4] = {
        (xv.x * (1.0f / (1.0f + expf(-g.x)))) * sv.x,
        (xv.y * (1.0f / (1.0f + expf(-g.y)))) * sv.y,
        (xv.z * (1.0f / (1.0f + expf(-g.z)))) * sv.z,
        (xv.w * (1.0f / (1.0f + expf(-g.w)))) * sv.w,
    };
    prq_rotate_quantize(v, tile, Y, blockIdx.x);
}

// gated_norm_f32 (per-head RMS norm times silu(z)) -> rotate -> Q8_1 (the
// DeltaNet out-projection input). head_dim must be 128 so that wave w owns
// head w of the block; the sum keeps gated_norm_f32's order (lane-strided
// partials, xor-shuffle reduction).
extern "C" __launch_bounds__(PRQ_THREADS, 1)
__global__ void ptq1_gated_norm_rotate_q8(
    const float* __restrict__ x,
    const float* __restrict__ z,
    const float* __restrict__ weight,
    const float* __restrict__ signs,
    prq_block_q8_1_mmq* __restrict__ Y,
    int head_dim, float eps
) {
    __shared__ float tile[2 * PRQ_BLOCK];
    const int lane = threadIdx.x & 31;
    const int base = blockIdx.x * PRQ_BLOCK + 4 * threadIdx.x;
    const int hb = blockIdx.x * PRQ_BLOCK + (threadIdx.x >> 5) * 128;
    float sq_sum = 0.0f;
    #pragma unroll
    for (int i = lane; i < 128; i += 32) sq_sum += x[hb + i] * x[hb + i];
    for (int o = 16; o > 0; o >>= 1) sq_sum += __shfl_xor(sq_sum, o, 32);
    const float inv_rms = rsqrtf(sq_sum / (float)head_dim + eps);

    const float4 xv = *(const float4*)(x + base);
    const float4 zv = *(const float4*)(z + base);
    const float4 wv = *(const float4*)(weight + 4 * lane);
    const float4 sv = *(const float4*)(signs + base);
    const float xs[4] = {xv.x, xv.y, xv.z, xv.w};
    const float zs[4] = {zv.x, zv.y, zv.z, zv.w};
    const float ws[4] = {wv.x, wv.y, wv.z, wv.w};
    const float ss[4] = {sv.x, sv.y, sv.z, sv.w};
    float v[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float normed = xs[j] * inv_rms * ws[j];
        const float silu_z = zs[j] / (1.0f + expf(-zs[j]));
        v[j] = (normed * silu_z) * ss[j];
    }
    prq_rotate_quantize(v, tile, Y, blockIdx.x);
}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Ceiling probe for the PTQ1_0 prefill GEMM on RDNA3 (gfx11 wave32).
// Compile: hipcc -O3 --offload-arch=gfx1100 -o bench_wmma_ptq1_ceiling benchmarks/bench_wmma_ptq1_ceiling.hip
// Run:     bench_wmma_ptq1_ceiling
//
// Three register-only kernels at the production launch shape (128-thread
// workgroups, each wave a 2x2 grid of 16x16 WMMA tiles):
//   wmma       eight independent v_wmma_i32_16x16x16_iu8 chains per wave.
//   format32   the WMMAs plus what one activation scale per 32 values costs:
//              per 32-wide sub-block an int->float conversion and an FMA per
//              element, per 128-weight group an FMA by the weight scale (the
//              fork's Q8_1).
//   format128  the same with one activation scale per 128: the group's eight
//              WMMAs chain in int32, one conversion and FMA per group.
// No trit decode, no LDS, no memory traffic. On RDNA3 the WMMA issues on the
// VALU, so the float work serializes with it. Each probe launches
// back-to-back for 1.5 s before the last ~1 s is timed: the rate a 2 s
// prefill sustains under the board's power limit, not the first-launch boost.
// Each kernel stays far below the ~2 s Windows TDR limit.

#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
typedef int v8i __attribute__((ext_vector_type(8)));
typedef int v4i __attribute__((ext_vector_type(4)));

__global__ __launch_bounds__(128, 2) void p_wmma(int iters, int* out) {
    const v4i a = (int)threadIdx.x, b = (int)threadIdx.x + 1;
    v8i c[8];
    for (int i = 0; i < 8; ++i) c[i] = (v8i)i;
    for (int it = 0; it < iters; ++it)
#pragma unroll
        for (int i = 0; i < 8; ++i) c[i] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c[i], false);
    int s = 0;
    for (int i = 0; i < 8; ++i) s += c[i][0];
    if (s == 0x7fffffff) out[0] = s;
}

template <int S128>
__global__ __launch_bounds__(128, 2) void p_fmt(int groups, int* out) {
    const int lane = threadIdx.x & 31;
    const float dw = 0.5f + lane * 1e-3f;
    float acc[2][2][8] = {};
    for (int g = 0; g < groups; ++g) {
        float gacc[2][2][8] = {};
        v8i iacc[2][2] = {};
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            v4i af[2][2], bf[2][2];
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    af[i][h] = (v4i)(g + 4 * s + 2 * i + h + lane);
                    bf[i][h] = (v4i)(g * 3 + s + i + h);
                }
            const float dx = 0.25f + s * 1e-3f + lane * 1e-4f;
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    if (S128) {
                        iacc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af[i][0], true, bf[j][0], iacc[i][j], false);
                        iacc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af[i][1], true, bf[j][1], iacc[i][j], false);
                    } else {
                        v8i part = {0, 0, 0, 0, 0, 0, 0, 0};
                        part = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af[i][0], true, bf[j][0], part, false);
                        part = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af[i][1], true, bf[j][1], part, false);
#pragma unroll
                        for (int e = 0; e < 8; ++e) gacc[i][j][e] += dx * (float)part[e];
                    }
                }
        }
        const float dx = 0.25f + g * 1e-6f + lane * 1e-4f;
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int e = 0; e < 8; ++e)
                    acc[i][j][e] += S128 ? (dw * dx) * (float)iacc[i][j][e] : dw * gacc[i][j][e];
    }
    float s = 0;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int e = 0; e < 8; ++e) s += acc[i][j][e];
    if (s == 1234.5f) out[0] = 1;
}

template <class F>
double sustained(F launch, double macs_per_launch) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    while (std::chrono::duration<double>(clk::now() - t0).count() < 1.5) { launch(); hipDeviceSynchronize(); }
    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    hipEventRecord(e0);
    int n = 0;
    const auto t1 = clk::now();
    while (std::chrono::duration<double>(clk::now() - t1).count() < 1.0) {
        for (int i = 0; i < 4; ++i, ++n) launch();
        hipDeviceSynchronize();
    }
    hipEventRecord(e1);
    hipEventSynchronize(e1);
    float ms = 0;
    hipEventElapsedTime(&ms, e0, e1);
    return macs_per_launch * n / (ms * 1e-3) / 1e12;
}

int main() {
    int* out;
    hipMalloc(&out, 16);
    const int wgs = 1280, trips = 2048;
    const double macs = (double)wgs * 4 * trips * 32 * 4096.0;
    std::printf("wmma      %6.2f e12 MAC/s\n",
                sustained([&] { hipLaunchKernelGGL(p_wmma, dim3(wgs), dim3(128), 0, 0, trips * 4, out); }, macs));
    std::printf("format32  %6.2f e12 MAC/s\n",
                sustained([&] { hipLaunchKernelGGL(p_fmt<0>, dim3(wgs), dim3(128), 0, 0, trips, out); }, macs));
    std::printf("format128 %6.2f e12 MAC/s\n",
                sustained([&] { hipLaunchKernelGGL(p_fmt<1>, dim3(wgs), dim3(128), 0, 0, trips, out); }, macs));
}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Ceiling probe for the PTQ1_0 x Q8_1 prefill GEMM on RDNA3 (gfx11 wave32).
// Compile: hipcc -O3 --offload-arch=gfx1100 -o bench_wmma_ptq1_ceiling benchmarks/bench_wmma_ptq1_ceiling.hip
// Run:     bench_wmma_ptq1_ceiling [runs=7]
//
// Two register-only kernels, same launch shape as the production tile
// (128-thread workgroups, each wave a 2x2 grid of 16x16 WMMA tiles):
//   wmma      - eight independent v_wmma_i32_16x16x16_iu8 chains: the
//               instruction's sustained rate.
//   format    - per 128-weight group, what the Q8_1 format obliges: two K16
//               WMMAs per 32-wide activation sub-block into a fresh int32
//               fragment, int->float and one FMA by that sub-block's scale per
//               element, one FMA by the weight scale per group. No trit decode,
//               no LDS, no memory traffic. Any kernel that keeps the per-32
//               activation scales does at least this work per WMMA; on RDNA3
//               the WMMA issues on the VALU, so this float work serializes with
//               it instead of hiding behind it.
// Prints the median over `runs` launches of each, in MAC/s.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

typedef int v8i __attribute__((ext_vector_type(8)));
typedef int v4i __attribute__((ext_vector_type(4)));

__global__ __launch_bounds__(128, 2) void probe_wmma(int iters, int* out) {
    const v4i a = (int)threadIdx.x, b = (int)threadIdx.x + 1;
    v8i c[8];
    for (int i = 0; i < 8; ++i) c[i] = (v8i)i;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int i = 0; i < 8; ++i)
            c[i] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c[i], false);
    }
    int s = 0;
    for (int i = 0; i < 8; ++i) s += c[i][0];
    if (s == 0x7fffffff) out[0] = s;
}

__global__ __launch_bounds__(128, 2) void probe_format(int groups, int* out) {
    const int lane = threadIdx.x & 31;
    const float dw = 0.5f + lane * 1e-3f;
    float acc[2][2][8] = {};
    for (int g = 0; g < groups; ++g) {
        float gacc[2][2][8] = {};
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            // Operands vary with g and s so no WMMA is loop-invariant.
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
                    v8i part = {0, 0, 0, 0, 0, 0, 0, 0};
                    part = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af[i][0], true, bf[j][0], part, false);
                    part = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af[i][1], true, bf[j][1], part, false);
#pragma unroll
                    for (int e = 0; e < 8; ++e) gacc[i][j][e] += dx * (float)part[e];
                }
        }
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[i][j][e] += dw * gacc[i][j][e];
    }
    float s = 0;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int e = 0; e < 8; ++e) s += acc[i][j][e];
    if (s == 1234.5f) out[0] = 1;
}

int main(int argc, char** argv) {
    const int runs = argc > 1 ? std::atoi(argv[1]) : 7;
    int* out = nullptr;
    if (hipMalloc(&out, 16) != hipSuccess) return 1;
    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    // Both kernels issue 32 WMMAs per loop trip per wave.
    const int trips = 2048;
    for (int wgs : {640, 1280, 2560}) {
        auto measure = [&](auto kernel, int arg) {
            hipLaunchKernelGGL(kernel, dim3(wgs), dim3(128), 0, 0, arg, out);
            hipDeviceSynchronize();
            std::vector<double> rates;
            for (int r = 0; r < runs; ++r) {
                hipEventRecord(e0);
                hipLaunchKernelGGL(kernel, dim3(wgs), dim3(128), 0, 0, arg, out);
                hipEventRecord(e1);
                hipEventSynchronize(e1);
                float ms = 0;
                hipEventElapsedTime(&ms, e0, e1);
                const double macs = (double)wgs * 4 * trips * 32 * 4096.0;
                rates.push_back(macs / (ms * 1e-3) / 1e12);
            }
            std::sort(rates.begin(), rates.end());
            return rates[rates.size() / 2];
        };
        // probe_wmma runs 8 WMMAs per iteration: trips*4 iterations = 32*trips.
        const double w = measure(probe_wmma, trips * 4);
        const double f = measure(probe_format, trips);
        std::printf("workgroups %5d  wmma %6.2f e12 MAC/s  format %6.2f e12 MAC/s  (%.0f%%)\n",
                    wgs, w, f, 100.0 * f / w);
    }
    hipFree(out);
    return 0;
}

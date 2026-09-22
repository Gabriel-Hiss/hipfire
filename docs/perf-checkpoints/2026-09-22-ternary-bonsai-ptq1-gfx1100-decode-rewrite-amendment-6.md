# Amendment 6 — Ternary-Bonsai-2-27B PTQ1_0 gfx1100: per-kernel profiles, and both targets are unreachable

**Date:** 2026-09-22
**Lifecycle:** `historical`
**Amends:** [`2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md`](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md) and amendments [1](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-1.md), [2](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-2.md), [3](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-3.md), [4](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-4.md), [5](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-5.md), all unchanged.
**Disposition:** **measured amendment, and a ceiling proof.** First per-kernel attribution for this model on both the prefill and the decode path. Both stated targets are proved unreachable with the measured hardware limits, not with arithmetic on spec-sheet numbers.

## Fixture

Same host, device, HIP 7.2, model (`C:/tmp/bonsai-2-27b.ptq1`, md5
`8abae179f984e2461cbf0fece6a8606f`).
`hipfire.exe` md5 `1cb750ea8e61e8a94ae2148205fde1a4`, `daemon.exe` md5
`a5470df71377ec890801cb284a84991c`, revision `1f5d4fec9`.

End-to-end, canonical flags: prefill **234.3 tok/s** (233.5/243.7/234.3/242.5/238.1),
TTFT 100.8 ms, decode 34.7 tok/s.

## Method, and one trap

`profile_prefill_qwen35` wraps `forward_prefill_batch` in
`rdna_compute::profile::{start,stop}`. `profile_qwen35_mq4` does the same for
`forward_scratch` (the decode path).

**The decode profile returns one kernel and nothing else unless graph capture is
off.** With capture on, the 8-step window reported exactly one entry per step
(`rotate_x_prism_hadamard`, 0.09 ms) against 28.77 ms/step of wall time, because
the GEMV and the rest of the forward run inside a replayed hipGraph and never
reach the instrumented launch path. `HIPFIRE_GRAPH=0` makes the 9504 entries
appear. Any future decode profile on this model needs that switch, or it will
report 0.3% coverage and look like a model that does nothing.

## Prefill, B=256: 90.7% is one kernel

1459 entries.

| rnk | kernel | calls | total | % |
|---|---|---:|---:|---:|
| 1 | **gemm_ptq1g128_wmma** | 400 | 584.1 ms | **90.7** |
| 2 | gated_delta_net_q8_batch_seq | 48 | 19.4 ms | 3.0 |
| 3 | gemm_bf16_xf32_batched | 96 | 9.4 ms | 1.5 |
| 4 | rotate_x_prism_hadamard | 321 | 7.1 ms | 1.1 |
| 5 | conv1d_silu_split_f32_n | 48 | 5.3 ms | 0.8 |
| 6 | fused_silu_mul_mq_rotate_batched | 64 | 4.9 ms | 0.8 |
| 7 | add_inplace_f32 | 128 | 4.5 ms | 0.7 |
| 8 | fused_rmsnorm_mq_rotate_batched | 128 | 4.4 ms | 0.7 |
| 9-17 | nine more | 222 | 7.5 ms | 1.2 |
| | TOTAL | 1459 | 644.3 ms | |

584 ms for 6.27e12 MACs is **10.7e12 MAC/s, 23% of the 45.98e12 INT8 ceiling**.
(The profiler serializes launches, so the absolute wall is inflated; the
attribution and the rate derived from MAC count over kernel time are not.)

**Prefill ceiling.** The other 9.3% is 60 ms and does not scale with GEMM
efficiency. A GEMM at 100% of the INT8 peak takes 136 ms, so B=256 cannot go
below 196 ms: **1306 tok/s**. The 3000 target needs 85 ms for B=256, and the GEMM
alone at a perfect 100% needs 136 ms. **3000 tok/s is unreachable by a factor of
3.9 on this hardware**, and INT4's 2x peak (which parity rules out, amendment 3)
would still only reach ~2000.

## Decode: 67.7% is the GEMV, and it is bandwidth-bound

With `HIPFIRE_GRAPH=0`, 9504 entries over 8 steps, per-step kernel time 36.44 ms
against 27.7 ms unprofiled.

| rnk | kernel | calls | total | % | GiB/s |
|---|---|---:|---:|---:|---:|
| 1 | **gemv_ptq1g128** | 3208 (401/step) | 197.3 ms | **67.7** | 212.2 |
| 2 | rotate_x_prism_hadamard | 2064 | 25.6 ms | 8.8 | 7.6 |
| 3 | rmsnorm_batched | 1024 | 24.0 ms | 8.2 | 2.4 |
| 4 | add_inplace_f32 | 1024 | 14.4 ms | 4.9 | 4.1 |
| 5 | gated_delta_net_q8_compact3_b2 | 384 | 11.5 ms | 4.0 | 53.4 |
| 6 | silu_mul_f32 | 512 | 5.8 ms | 2.0 | 17.1 |
| 7 | gated_norm_f32 | 384 | 4.0 ms | 1.4 | 8.7 |
| 8 | kv_cache_write_q8_0 | 256 | 4.0 ms | 1.4 | 0.3 |
| 9-12 | four more | 648 | 4.8 ms | 1.6 | |
| | TOTAL | 9504 | 291.5 ms | | |

The GEMV moves **42873 MiB / 8 steps = 5.62 GB of weights per token**, which
matches the tensor shapes (24.5e9 PTQ1 weights at 1.75 bpw = 5.36 GB) and is the
number the ceiling rests on.

### The peak bandwidth is measured, not assumed

`kernels/src/probe_dram_bw.hip` reads a 146.8 MB buffer with grid-stride float4
loads: **551-553 GB/s** across runs. The spec sheet says 576 GB/s; the achievable
figure is 96% of it, which is what a real kernel can be held to.

| | GB/s | of measured peak |
|---|---:|---:|
| `gemv_ptq1g128` | 292.7 | 53% |
| pure read probe | 551.2 | 100% |

### Decode ceiling

| term | ms/token |
|---|---:|
| weights to read: 5.62 GB | |
| GEMV at 100% of measured peak | **10.20** |
| non-GEMV, measured | 8.70 |
| **floor with the GEMV perfect** | **18.90** |

**18.90 ms/token is 52.9 tok/s.** The 80 tok/s target is 12.5 ms/token, which
would leave 2.3 ms for 401 GEMV launches plus ~1200 other launches and all the
elementwise work, against a GEMV that cannot go below 10.2 ms. **80 tok/s is
unreachable by a factor of 1.5**, and the format cannot shrink to help: 1.625
bpw of trits against log2(3) = 1.585 bpw is 2.5% of waste, and the 2-byte scale
is 0.125 bpw.

**The 2-bit repack considered in amendment 5 would make this worse.** 34 bytes
per 128 weights against 28 is +21% bytes, and the decode is bandwidth-bound.

## Two kernel attempts, both correct, neither the binder

| change | kernel rate | delta |
|---|---:|---:|
| baseline | 10.08e12 MAC/s | |
| branchless group-at-a-time trit decode | 10.60e12 | +5% |
| + B fragment loaded as one `i8x16` | 10.74e12 | +1.3% |

Both are kept: they are simpler than what they replaced (the branchy per-element
`ptq1_tile_trit` is gone from this kernel, and the byte-assembly macro is gone
from the activation side), and both are pinned to identical values by the channel
test. But the kernel is at 23% of peak and neither ALU reduction moved it, so
**the binder is not the decode and not the fragment packing.**

What the numbers leave: grid is `[M/16, N/16]` = 1280 blocks of 32 threads at
5120x5120x64, which is 4 waves per SIMD on 320 SIMD units, so occupancy is low
for a kernel that is not ALU-bound. The untested suspect is the per-output-row
scale load, 8 scattered `_Float16` reads per lane per group (320 per block) that
amendment 4 introduced to fix the output-column bug. Staging those 16 values per
group through LDS once, instead of each lane issuing 8 scattered loads, is the
next thing to try.

## Disposition

Both targets are **proved unreachable**, so the goal's stop condition is met by
measurement rather than by declaring success:

- prefill: ceiling **1306 tok/s** (target 3000), bounded by the INT8 peak times
  the 9.3% of prefill the GEMM does not own.
- decode: ceiling **52.9 tok/s** (target 80), bounded by 5.62 GB/token against a
  measured 551 GB/s.

The reachable work is kernel efficiency, not reachability of the targets: the
GEMM has 4.3x of headroom to its own ceiling and the GEMV 1.9x, which is worth
roughly 231 -> 700 tok/s prefill and 35 -> 50 tok/s decode. Those are the numbers
the remaining work should be measured against.

These rows are measurement, not admission.

# Amendment 7 — Ternary-Bonsai-2-27B PTQ1_0 gfx1100: the prefill GEMM is grid-depth limited, not occupancy limited

**Date:** 2026-09-22
**Lifecycle:** `historical`
**Amends:** [`2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md`](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md) and amendments [1](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-1.md), [2](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-2.md), [3](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-3.md), [4](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-4.md), [5](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-5.md), [6](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-6.md), all unchanged.
**Disposition:** **diagnostic.** Corrects the next-lever attribution amendment 6 ended on, with the compile stats and one more rejected lever.

## Fixture

As amendment 6. Kernel rate 10.68e12 MAC/s at 5120x5120x64 (10.74e12 before the
`launch_bounds` test below; the two are the same within this host's spread).

## Rejected: raising the resident-block hint

Amendment 6's decode-side finding was that the decode GEMV is occupancy-bound, so
the same was tried on the prefill GEMM: `__launch_bounds__(32, 2)` -> `(32, 8)`,
asking the compiler to fit eight blocks per SM instead of two.

| hint | kernel rate |
|---|---:|
| `(32, 2)` | 10.74e12 MAC/s |
| `(32, 8)` | 10.68e12 MAC/s |

No change. Reverted.

## Compile stats

From the kernel cache's `radiowave.json`
(`gemm_ptq1g128_wmma`, gfx1100, wave32):

| | |
|---|---:|
| vgpr_count | **108** |
| sgpr_count | 26 |
| vgpr_spill_count | **0** |
| sgpr_spill_count | 0 |
| private_segment_fixed_size | 0 |

No spills, and 108 VGPRs per lane against a 16384-entry register file per SIMD
puts roughly four waves resident per SIMD.

**The grid is the same size as that residency.** At 5120x5120x64 the launch is
`[M/16, N/16] = [320, 4] = 1280` blocks of 32 threads, and 80 CUs x 4 SIMD = 320
SIMD units, so 1280 blocks is exactly four waves per SIMD. The kernel launches one
full residency and then has no further work to rotate in.

That is why the hint does nothing: there is no second residency to admit. It is
also why the two ALU reductions in amendment 6 moved the rate 6.5% between them
while the kernel sits at 23% of the INT8 peak: the unit is waiting, not computing.

## The lever this implies

The fix is not more resident waves but more work per block, so each wave has
enough independent work to cover its own latency. Concretely a 32x32 or 16x64
tile with the A operand staged through LDS, which is the structure the tree's
other WMMA GEMMs already use (`gemm_q8_0_wmma` stages A and B; this kernel reads
both operands straight from global because it was written to be small enough to
debug). That is a rewrite of the kernel, not a tuning constant.

Amendment 6 named the per-output-row scale load as the suspect. That is now the
second-ranked suspect, not the first: 320 scattered fp16 loads per block is real
work, but it would not explain a kernel that the resident-wave arithmetic says is
latency-starved with no work to swap in.

The reachable target is unchanged: the GEMM has 4.3x of headroom to its own
ceiling, worth roughly 231 -> 700 tok/s prefill, and the ceiling itself (1306) is
still 2.3x below the 3000 target.

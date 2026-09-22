# Amendment 16 — Ternary-Bonsai-2-27B PTQ1_0 gfx1100: the DFlash verify's extra cost is not the draft

**Date:** 2026-09-22
**Lifecycle:** `historical`
**Amends:** [`2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md`](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md) and amendments [1](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-1.md) through [15](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-15.md), all unchanged.
**Disposition:** **correction to amendment 14, and the slope decomposition that replaces it.** Amendment 14 attributed 91.3 ms of the DFlash cycle to the draft's forward. The profiler and a slope measurement disagree with that, and the slope is the more trustworthy of the two.

## The disagreement

Amendment 14 measured T_verify two ways and read the difference as the draft:

| | T_verify (17 tokens) |
|---|---:|
| n-gram, no neural draft | 46.3 ms |
| DFlash, neural draft | 137.6 ms |

and concluded "the neural draft's forward costs 91.3 ms of the cycle".

The profiler says the draft's kernels are a small share: 19.29 ms
(`gemm_hfq4g256_residual_wmma_ksplit_det`) + 10.33 ms (`_k2`) + 17.43 ms
(`mq_rotate_x_batched`) = 47 ms over 5 cycles, **9.4 ms per cycle** against a
~95 ms cycle.

**The profiler is not hiding anything.** The kernel count is identical with and
without graph capture (6510 over 3 cycles either way), so the captured verify
does not swallow launches the profiler would otherwise see. And the draft's
compute is small on its own terms: 5 layers x (4*5120^2 + 3*5120*17408) =
1.86e9 MACs per slot, 0.19 ms/slot at the measured 10e12 MAC/s, against 0.92 GB
of weights = 1.7 ms to read. Neither supports 91 ms.

## The slope decomposition

Two K sweeps, each in one window, T_verify = E / tok/s:

| K | plain verify (n-gram, tau~0.1) | DFlash |
|---:|---:|---:|
| 2 | 40.7 ms | 73.8 ms |
| 4 | 43.0 | 81.0 |
| 8 | 44.3 | 94.8 |
| 16 | — | 116.7 |
| 32 | 48.1 | 199.2 |

| | fixed | marginal |
|---|---:|---:|
| plain verify | ~40 ms | **0.25 ms/slot** |
| DFlash | ~61 ms | **3.06 ms/slot** (K<=16), 5.16 past it |

The DFlash path carries ~21 ms of fixed overhead and **~2.8 ms/slot that is not
the draft** (the draft's own profiled share is ~0.42 ms/slot at K=8). The
superlinear regime past K=16 is the draft losing accuracy against its trained
block, which is also where E stops paying.

So amendment 14's number is wrong in its attribution and right in its
conclusion: something in the DFlash verify costs ~2.8 ms per block slot, and it
is not a profiled GPU kernel. The candidates left are host-side per-cycle work
and unprofiled device copies — the DeltaNet snapshot/rewind that partial
acceptance requires is the obvious one, and it is exactly the cost the SSSD brief
predicted ("cada branch precisa de estado proprio; ramificar estado e o custo
escondido").

## What stands

- **DFlash 54.0 tok/s at K=8 vs AR 29.20 = 1.85x.** Measured in one window, both
  legs, reproducible.
- The K optimum is 8 and it is now a config knob (`speculation.dflash_block`).
- The parity gate and the lossless check are unaffected by any of this.
- The q8_1 cache is inert (amendment 15).

## What is left

One well-scoped question with a 2x payoff: **what costs 2.8 ms per block slot in
the DFlash verify?** It is not the draft's kernels, so the lever is not "make the
draft faster". A per-cycle host/GPU split (the existing `HIPFIRE_HOST_TIMING`
only reports wall) or a copy-counter would settle it, and that is the next
measurement rather than a guess.

These rows are measurement, not admission.

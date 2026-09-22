# Amendment 17 — Ternary-Bonsai-2-27B PTQ1_0 gfx1100: the DFlash cycle is GPU-bound, and the draft is 4% of it

**Date:** 2026-09-22
**Lifecycle:** `historical`
**Amends:** [`2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md`](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md) and amendments [1](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-1.md) through [16](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-16.md), all unchanged.
**Disposition:** **closes the open question amendment 16 left, and rules the profiler back in.**

## The host breakdown

`HIPFIRE_HOST_TIMING=1`, mean over 41 cycles, K=8:

```
wall=107908 us
  launch=2548 (n=2379)  h2d=3652 (n=3)  d2h=27370 (n=15, 14600KB)
  d2d=417 (n=443)  memset=42 (n=48, 10MB)  glaunch=0 (n=0)
  ssync=67025 (n=1)  esync=0  dsync=0  -> other=6854
```

**62% of the cycle is one stream sync** — that is the host waiting for the GPU
forward, not overhead. **25% is a 14.6 MB device-to-host transfer** across 15
copies: the full-logits download (17 x 248320 x 4 B = 16.9 MB). The launch path
is 2.5 ms for 2379 launches, so launch overhead is not a factor.

## What this settles

The GPU work is ~67 ms of the 108 ms cycle. The profiler's serialized total is
240 ms per cycle for the same kernels, so its 3.6x inflation scales down
proportionally, and the draft's kernels (47 ms of 1198 ms profiled) are **2.6 ms
of the 67 ms, ~4%.** The profiler was right and amendment 14's "the draft costs
91.3 ms" was wrong in a second way: the two-point subtraction it used also
carried the demo's 27 ms full-logits D2H on the DFlash leg only, which is a
measurement artifact of `dflash_spec_demo` (its seed-oracle needs logits) and not
a property of the verify.

Amendment 16's slope decomposition stands as arithmetic (plain verify 0.25
ms/slot, DFlash 3.06 ms/slot) but its attribution does not: the ~2.8 ms/slot is
in the target's side of the verify, not in the draft, and it is inside the 67 ms
of GPU work rather than outside the profiler.

## What is actually left, and what is not

Not the draft. Not launch overhead (2.5 ms of 108). Not the q8_1 cache
(inert, amendment 15).

The candidate is the target's own batched verify at (b+1) rows, whose per-slot
cost is 12x the plain verify's 0.25 ms/slot. The batched verify carries the
hidden-state ring capture and the DeltaNet snapshot/rewind that partial
acceptance needs — the cost the SSSD brief predicted — and both scale with the
row count while the profiler's kernel list does not name them (they are `d2d`
copies and ring writes, 417 us and 10 MB of memset per cycle respectively, which
is 10x too small on their own but points at the same place).

Settling it needs a per-row accounting inside `verify_dflash_block_inner`, which
is the next measurement.

## What stands

**DFlash 54.0 tok/s at K=8 against AR 29.20 = 1.85x**, lossless, parity intact,
with K now a config knob and its optimum measured rather than assumed.

These rows are measurement, not admission.

# Amendment 10 — Ternary-Bonsai-2-27B PTQ1_0 gfx1100: the draft lm_head change is a tie, and the dflash cycle is K-independent

**Date:** 2026-09-22
**Lifecycle:** `historical`
**Amends:** [`2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md`](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md) and amendments [1](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-1.md) through [9](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-9.md), all unchanged.
**Disposition:** **measurement correction, plus one structural finding.** The apparent lm_head win in this amendment's first revision was machine drift. A controlled interleaved A/B shows a tie. The K-independence of the dflash cycle is the real result, and it bounds every target in the SSD/SSSD line.

## Fixture

| item | value |
|---|---|
| model | `C:/tmp/bonsai-2-27b.ptq1`, md5 `8abae179f984e2461cbf0fece6a8606f`, 5.94 GB |
| draft | `qwen35-27b-dflash-mq4.hfq`, trained block size 16 |
| gpu | gfx1100, HIP 7.2, 17.2 GB VRAM |
| harness | `hipfire bench --spec dflash --backend noslots --workload stateless --max-tokens 128`, fresh process |
| prompt | `def fibonacci(n):` (inline, **not** a committed fixture file; no prompt md5 to compare against) |
| binaries | `old` `b208bc1cbc457f63ad3293d5b7730fc2`, `new` `b345e2fc4b56651d6b6d01b736a539ec` (both archived under `C:/tmp/ab/`) |

`HIPFIRE_MTP_VERIFY_DECOUPLE` and `HIPFIRE_PREFILL_BATCHED` MUST be unset.
Leaving `HIPFIRE_PREFILL_BATCHED=0` set in a shell makes the verify fall back
to the per-token loop and moves the whole measurement: prefill 150 -> 26 tok/s,
K=8 dflash 54 -> 11.5 tok/s, AR decode unchanged at 25.9 tok/s. A first pass at
this A/B was run with both variables still exported from an earlier per-token
test and produced "old == new" for the wrong reason.

## Correction: the lm_head change is a tie

The draft-side lm_head in `spec_step_dflash` carried its own inline dtype list,
separate from the verify-side one, and never got the PTQ1G128H arm. It now uses
the shared `dflash_batched_lm_head_supported` helper with PTQ1G128H added, and
one `gemm_ptq1g128_wmma` replaces `(B-1)` serial gemvs.

Measured as an interleaved A/B, two binaries, alternating, 2 runs / 1 warmup
each, 128 tokens:

| K | old | new |
|---:|---|---|
| 8 | 54.3, 52.5 tok/s | 54.1, 55.6 tok/s |
| 16 | 65.0, 65.2 tok/s | 67.7, 64.5 tok/s |

**Tie at both K.** An earlier sequential before/after pair in this same session
reported K=8 44.7 -> 53.1 and K=16 41.3 -> 66.1. Those "before" rows were taken
in a slower machine state: the same unmodified binary now measures 54.3 at K=8
where it measured 44.7 then. The delta was drift, not the change.

The change is kept anyway, as a maintainability fix: two dtype lists that must
agree are one list now, and that divergence is what produced the per-row
fallback in the first place. It is not a perf claim.

## Structural finding: the dflash cycle does not scale with K

Same fixture, single binary, 3 runs / 2 warmups:

| K | tau | tok/s | windows | ms/cycle | tok/cycle |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.92 | 21.5 | 66 | 90.2 | 1.94 |
| 4 | 2.26 | 34.6 | 39 | 94.9 | 3.28 |
| 8 | 4.08 | 56.3 | 25 | 90.9 | 5.12 |
| 16 | 5.05 | 66.2 | 21 | 92.1 | 6.10 |

**The cycle is ~90 ms at every K from 2 to 16.** The marginal cost of the
speculation rows is below the measurement floor; the cycle is a fixed cost.

Two consequences:

1. **The batched GEMMs amortize.** If they re-read the weights per row, the
   cycle would grow with K. It does not, so the verify's `forward_prefill_batch`
   and both lm_head GEMMs share one weight read across rows. Any theory that
   blames per-row weight re-reads is refuted by this table.
2. **Raising K is free until tau saturates.** K=16 is the optimum because the
   draft's trained block is 16; past it tau falls (K=20 -> 3.38, K=31 -> 2.53)
   while the cycle stays ~90 ms.

## What this bounds

AR decode on this model is 25.9-31 tok/s (32-39 ms per token). The dflash cycle
is 90 ms. So ~50-55 ms per cycle is dflash-specific fixed overhead beyond the
target's own forward, and it does not shrink with K.

Every throughput target in the SSD/SSSD line is `E(K, alpha) / T_verify`:

| target | needs | measured |
|---|---|---|
| SSD K=7-8 -> 120-180 tok/s | T_verify ~43 ms at tau 5.1 | 91 ms at K=8, tau 4.08 |
| n-gram + SSD K=16 -> 200-350 | T_verify ~30 ms at tau 6.1 | 92 ms at K=16, tau 5.05 |

**The speculation side is already at its ceiling for this draft: tau 5.05 at
K=16 is the draft's limit, and K=16 is free.** Reaching any of those targets
requires cutting the ~90 ms cycle, not adding speculation machinery. A megaspec,
an outcome cache, or a branch trie all operate on a term that is already ~0.

## Not established

- **Where the ~50-55 ms of dflash-specific overhead goes.** The cycle is 90 ms;
  the target's own forward is 32-39 ms; the draft forward is ~5.25 ms (measured
  by the `ssd_fan_out` probe in amendment 11's predecessor). The rest is
  unidentified. `HIPFIRE_PROFILE=1` and `HIPFIRE_HOST_TIMING=1` produce no
  output on this path: the PTQ1G128H kernels are not wired to
  `crate::profile::begin_timer`. Instrumenting that is the prerequisite for any
  further cycle work, and it was not done here.
- **Whether AR and DFlash are bit-identical at greedy on this model.** They are
  not: on this prompt they agree for 11 tokens, then diverge and do not
  re-synchronize within 48. That is consistent with one near-tie argmax flip
  cascading through greedy decode, which
  [`docs/investigations/2026-08-03-ds4-cdna3-arch-gate-gaps.md`](../investigations/2026-08-03-ds4-cdna3-arch-gate-gaps.md)
  documents as expected. Acceptance of 5 of 16 drafts per cycle is not
  consistent with a broken verify. No logits comparison was run to settle it.
- **Amendment 9's batched-prefill divergence** is untouched and still open. It
  gates any logits claim on the batched path, which includes the verify.

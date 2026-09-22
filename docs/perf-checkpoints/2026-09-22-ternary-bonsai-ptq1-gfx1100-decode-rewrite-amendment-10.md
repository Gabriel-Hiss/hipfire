# Amendment 10 — Ternary-Bonsai-2-27B PTQ1_0 gfx1100: the draft lm_head was 39% of the DFlash cycle

**Date:** 2026-09-22
**Lifecycle:** `historical`
**Amends:** [`2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md`](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md) and amendments [1](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-1.md), [2](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-2.md), [3](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-3.md), [4](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-4.md), [5](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-5.md), [6](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-6.md), [7](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-7.md), [8](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-8.md), [9](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-9.md), all unchanged.
**Disposition:** **accepted perf change.** Speculative-decode throughput only; the AR decode path, the verify, and the logits parity rows in amendments 1–9 are untouched. This amendment does not address amendment 9's batched-prefill divergence, which remains open and still gates any claim about the batched path's logits.

## Fixture

| item | value |
|---|---|
| model | `C:/tmp/bonsai-2-27b.ptq1`, md5 `8abae179f984e2461cbf0fece6a8606f`, 5.94 GB |
| draft | `qwen35-27b-dflash-mq4.hfq` |
| gpu | gfx1100, HIP 7.2 |
| harness | `hipfire bench --spec dflash --backend noslots --workload stateless --max-tokens 128`, 3 runs / 2 warmups, fresh process |
| binaries | `hipfire.exe` `f106081ce811691042be2b3a2d2b55fc`, `daemon.exe` `c69e3f83bd5434c32c4aae960eddb155` |
| prompt | `def fibonacci(n):` (inline, not a committed fixture file) |

## What was wrong

`spec_step_dflash`'s draft-side lm_head carried its own inline dtype list,
`use_batched_gemm`, separate from `dflash_enqueue_verify_lm_head`'s. The verify
head had gained a PTQ1G128H arm; the draft head had not. So for a PTQ1G128H
trunk the draft head fell through to the per-row loop: `(B-1)` serial gemvs
against the 248320x5120 output head, every cycle.

Decomposing the K=8 cycle (115 ms) from measured quantities:

| term | ms |
|---|---:|
| first verify row (the target's weight read; 1000/31.0 from AR decode) | 32 |
| 8 marginal verify rows (from the K=8 -> K=16 slope, +33 ms / 8 rows) | 33 |
| draft forward (measured by the `ssd_fan_out` probe) | 5 |
| **unexplained** | **45** |

The 45 ms residual is the draft lm_head's per-row loop: 9 serial gemvs plus 9
downloads. The draft forward was 4% of the cycle; the head was 39%.

## Change

Replace the inline `matches!` with the shared
`dflash_batched_lm_head_supported` helper (which already held the same set),
add `PTQ1G128H` to it, and add the matching arm: one Prism-Hadamard rotation
over the `(B-1)` hidden rows via `rotate_x_mq_batched_for` (which routes
PTQ1/TQ2 to `rotate_x_prism_hadamard`), then one `gemm_ptq1g128_wmma`.

## Measured

| K | before | after |
|---:|---|---|
| 8 | 44.7 tok/s, tau 4.08 | **53.1 tok/s, tau 4.08** |
| 12 | — | 52.9 tok/s, tau 4.08 |
| 16 | 41.3 tok/s, tau 5.05 | **66.1 tok/s, tau 5.05** |
| 20 | — | 32.9 tok/s, tau 3.38 |
| 24 | 21.2 tok/s, tau 3.54 | 28.1 tok/s, tau 3.54 |
| 31 | — | 21.6 tok/s, tau 2.53 |

Tau is identical at every K measured on both sides (8, 16, 24), so the batched
head emits exactly the tokens the per-row loop emitted. K=16 is the new
optimum; before this change K=16 lost to K=8 because the head's cost scaled
with the position count while the extra acceptances did not pay for it.

## Caveats

- **Single prompt.** `def fibonacci(n):` is inline, not a committed fixture, so
  this row is not comparable to a benchmark that pins a prompt md5. Re-measure
  against `benchmarks/prompts/` before quoting an absolute number.
- **Not bit-identical to AR at greedy.** AR and DFlash agree for the first 11
  tokens on this prompt, then diverge for the remaining 9 and do not
  re-synchronize within 48 tokens. That is one near-tie argmax flip cascading
  through greedy decode, which
  [`docs/investigations/2026-08-03-ds4-cdna3-arch-gate-gaps.md`](../investigations/2026-08-03-ds4-cdna3-arch-gate-gaps.md)
  documents as expected ("DSpark is not bit-identical to AR at greedy... a
  flipped argmax at a near-tie is an exact-token-id rejection in the accept
  path"). Acceptance of 5 of 16 drafts per cycle (`tau 5.05`) is not consistent
  with a broken verify; a wrong forward would give `alpha ~ 0`. This change
  touches the draft's proposal head only, never the verify, so it cannot have
  introduced the divergence.
- **Tau, not logits, is the correctness signal here.** The draft head affects
  only which tokens are proposed. Acceptance, and therefore the emitted
  sequence, is decided by the verify, which this change does not touch.
- Amendment 9's batched-prefill divergence is untouched and still open. It
  bounds what any throughput row in this series can claim.

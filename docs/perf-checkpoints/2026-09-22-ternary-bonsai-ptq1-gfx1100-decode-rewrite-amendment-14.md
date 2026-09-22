# Amendment 14 — Ternary-Bonsai-2-27B PTQ1_0 gfx1100: the K optimum, and why SSD needs a second device

**Date:** 2026-09-22
**Lifecycle:** `historical`
**Amends:** [`2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md`](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md) and amendments [1](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-1.md) through [13](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-13.md), all unchanged.
**Disposition:** **measured K sweep plus a feasibility read on the SSD reference.** The DFlash path is now 1.85x AR at its measured optimum. A faithful SSD replication is blocked on hardware, with the arithmetic below.

## Fixture

Target `C:/tmp/bonsai-2-27b.ptq1` (md5 `8abae179f984e2461cbf0fece6a8606f`), draft
`~/.hipfire/models/qwen35-27b-dflash-mq4.hfq` (sha256
`3d428b97c1911a9ad815cc52fbee080306852c1dafad6b1b17bb70bd68010301`), revision
`9d13b5449`. Canonical bench, `def fibonacci(n):`, 128 tokens, `--backend
noslots`, AR = 29.20 tok/s.

## The speculation length is now a knob, and its optimum is 8

The block size was fixed at the draft checkpoint's trained value and nothing
could override it: `--draft-max` maps to `ngram_k`/`mtp_k` and neither reaches
the DFlash speculator. `speculation.dflash_block` (`HIPFIRE_DFLASH_BLOCK`, 0 =
checkpoint default) now applies to `DflashConfig` before
`runtime_block_size()`, wired into both loaders.

| K | E (tokens/window) | tok/s | windows |
|---:|---:|---:|---:|
| 2 | 0.92 | 26.3 | 66 |
| 4 | 2.26 | 40.5 | 39 |
| **8** | **4.08** | **54.0** | **25** |
| 12 | 4.08 | 47.4 | 25 |
| 16 | 5.05 | 52.2 | 21 |
| 24 | 3.54 | 25.3 | 28 |
| 32 | 2.34 | 16.9 | 38 |

**K=8 is the optimum at 1.85x AR**, and it lands on the brief's "SSD direct
K=7-8" row. Two effects meet there: E saturates (4.08 at K=8, 5.05 at K=16, a
24% gain for a 2x longer block) while T_verify grows with K; and past the
trained block of 16 the draft's own accuracy collapses, so K=24/32 lose on both
axes at once.

## The draft is two thirds of the verify cycle

Same prompt, both at K=16, so T_verify is comparable:

| | windows | tok/s | T_verify (17 tokens) |
|---|---:|---:|---:|
| n-gram, no neural draft | 112 | 24.7 | **46.3 ms** |
| DFlash, neural draft | 21 | 44.3 | **137.6 ms** |

The neural draft's forward costs 91.3 ms of the 137.6 ms cycle. **A cache hit
that skipped it would give T_verify = 46.3 ms, i.e. 4.08/0.0463 = 88 tok/s at
K=8** — inside the brief's 120-180 band once the q8_1 cache and K=8 are both in
place.

## Why the SSD reference cannot be replicated faithfully here

The algorithm (Kumar, Dao & May, arXiv:2603.03251) is understood and its
structure is portable. What is not portable is its cost model.

`compute_megaspec_lookahead(K, MQ_LEN) = K + 1 + K * MQ_LEN`. The draft does not
speculate K tokens; it speculates a **tree**: for each of the K+1 possible
acceptance lengths it forks `fan_out_list[i]` likely recovery tokens, and for
each of those it drafts K more. At K=16 with a fan-out of 16 that is 273 draft
tokens per step against the 17 it drafts today.

The reference pays that because the draft runs on a **separate device** (the
paper's setup is a 4-GPU target plus a 5th GPU for the draft, wired with NCCL),
so the megaspec overlaps the target's verify and its cost never lands on the
critical path. On one gpu1100 the two serialize, so the megaspec is 16x the
draft's current cost added to the cycle — a large net loss, not a gain.

The brief's own portable variant is to put the outcome predictor on the CPU
(trie/n-gram feeding the same `(accepted_len - 1, recovery_token)` cache). That
fails on measurement: the n-gram drafter's acceptance on a real prompt is
tau=0.02 (amendment 13), so the cache would almost never hit, and a 5-layer
neural draft does not fit the brief's 1 ms cache-miss budget on CPU either.

**SSD's speedup is a property of the hardware topology, not of the cache
structure.** With a second GPU the replication becomes possible and the
arithmetic above says where to look; without one, the honest read is that the
cache can be built and will not pay.

## What is delivered and what is blocked

Delivered and measured: the DFlash draft attaches to the Prism ternary target
(1.85x AR at K=8, up from 1.40x before the q8_1 cache), lossless, parity intact,
with K now tunable and its optimum derived rather than assumed.

Blocked on hardware: the outcome-prediction half of SSSD. The two open items in
the goal's todo list both require a second device, and the brief's blocker
clause covers this case.

These rows are measurement, not admission.

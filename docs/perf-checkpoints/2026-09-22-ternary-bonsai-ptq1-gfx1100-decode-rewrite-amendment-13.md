# Amendment 13 — Ternary-Bonsai-2-27B PTQ1_0 gfx1100: the n-gram bench number is a cache artifact

**Date:** 2026-09-22
**Lifecycle:** `historical`
**Amends:** [`2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md`](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md) and amendments [1](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-1.md) through [12](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-12.md), all unchanged.
**Disposition:** **measurement defect.** Records that `hipfire bench --spec ngram` reports an inflated median, with the per-request evidence. The DFlash numbers in amendment 12 are checked and clean.

## What was seen

`HIPFIRE_NGRAM_DRAFT_K=16 HIPFIRE_NGRAM_MIN_COUNT=1` on the bench's default
prompt gave tau=11.70 and 88-107 tok/s across three runs, against AR 29.20. That
is 3.0-3.7x, and tau=11.70 with K=16 implies 74% per-token acceptance — on a
prose prompt, for a model-free bigram drafter. Suspicious on its face.

## What it is

Per-request tau from one bench invocation (`--runs 3 --warmups 2`):

```
req 1 (warmup):  tau=0.00   tok/s=25.3   (10 tok, 9 windows)
req 2 (run 1):   tau=0.02   tok/s=29.7   (128 tok, 125 windows)
req 3 (run 2):   tau=11.70  tok/s=106.8  (128 tok, 10 windows)
req 4 (run 3):   tau=11.70  tok/s=106.5  (128 tok, 10 windows)
```

**The true number is request 2: tau=0.02, 29.7 tok/s — identical to AR.** The
bench submits the same prompt on every run, and the n-gram drafter's bigram
cache survives across requests within the daemon, so from the third request on
the drafter proposes the model's own previous output and it is accepted
verbatim. The reported median is the median of the two cache-hit runs.

`hipfire run` on the same prompt with the same env gives tau=0.02 and text
byte-identical to AR, which is the same measurement made once.

This is the failure `AGENTS.md` names directly: *"Tight stddev on a spec-decode
bench is SUSPICIOUS, not reassuring ... Always eyeball the decoded output when
tau comes back unusually high — single-token attractor failures pass every
statistical gate as fake wins."* Here it is not an attractor but a drafter cache,
and the tell is the same: a number too good for the drafter that produced it.

## The DFlash numbers are clean

Same per-request check on the DFlash path:

```
req 1: tau=4.00  tok/s=28.3  (10 tok, 2 windows)
req 2: tau=5.05  tok/s=51.6  (128 tok, 21 windows)
req 3: tau=5.05  tok/s=50.9  (128 tok, 21 windows)
```

tau is stable at 5.05 across requests, so amendment 12's **27.90 AR vs 51.60
DFlash (1.77x)** stands. A neural draft conditions on the actual hidden state
per request, so it has no cross-request replay to exploit.

## What this means for the SSD work

Any drafter whose cache is keyed on tokens rather than on target state (n-gram,
trie, and the SSD outcome cache the reference implementation adds) can replay
across requests. **A bench that reuses one prompt cannot measure those drafters.**
The `--spec ngram` row must be read from request 2 of a fresh daemon, or from a
per-request tau log, not from the reported median.

For the SSD replication this is load-bearing: the whole point of the outcome
cache is that a hit returns a speculation without drafting, so a bench that
averages over hits and misses measures the cache's warm-up curve, not the
algorithm. The measurement must report `p_hit` and the tok/s conditioned on it,
which is what the SSSD brief's metric list asks for.

These rows are measurement, not admission.

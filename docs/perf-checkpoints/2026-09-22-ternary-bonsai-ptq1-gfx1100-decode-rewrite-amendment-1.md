# Amendment — Ternary-Bonsai-2-27B PTQ1_0 gfx1100: the device's practical streaming ceiling

**Date:** 2026-09-22
**Lifecycle:** `historical`
**Amends:** [`2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md`](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md) (unchanged).
**Disposition:** **measured amendment.** Adds the device's achievable streaming bandwidth, which the original record did not measure and which changes the strength of its conclusion. The original is not modified; its numbers stand.

## Why this exists

The original record compared the PTQ1 GEMV against the card's **nameplate**
576 GB/s and concluded the 100 tok/s target was unreachable. Nameplate is not a
rate any real kernel achieves, so that bound was weaker than it looked: if the
device actually streamed at, say, 520 GB/s, the GEMV's 343 GiB/s decode-free
rate would have been the pattern's fault and worth attacking.

This amendment measures what the device actually delivers on kernels with no
decode work and perfect coalescing.

## Method

Trivial elementwise kernels over three 1 GiB F32 tensors (3 GiB of traffic:
2 read + 1 write), 20 reps after one warm-up call, HIP events around the loop.
`add_inplace_f32` and `silu_mul_f32` are the two with no reduction and no
decode, so their rate is the device's streaming ceiling for this access shape.
`rmsnorm_f32` is included and is **not** a bandwidth measurement — see the note
below.

Fixture is otherwise the original record's (same host, device, HIP 7.2,
revision `3ec62c3c4`, `hipfire.exe` md5 `2f541a477375570f2f8a51954af17594`).

## Result

| kernel | ms/call | GiB/s | note |
|---|---:|---:|---|
| `add_inplace_f32` | 7.198 | **416.8** | 3 GiB traffic, no decode |
| `silu_mul_f32` | 7.129 | **420.8** | 3 GiB traffic, no decode |
| `rmsnorm_f32` | 648.580 | 4.6 | **not a bandwidth row** — see below |

416.8 and 420.8 GiB/s are 73–78% of the 576 GB/s nameplate. That is the
device's practical streaming rate.

`rmsnorm_f32` at 4.6 GiB/s is two orders of magnitude below the others and is
retained as measured rather than discarded. It is a single-workgroup reduction
over 256 M elements, so it is measuring the reduction's serialisation, not
memory. It is reported here only so the row is not silently dropped, and it is
**not** usable as a bandwidth bound. The 4.6 GiB/s figure that appears for
`rmsnorm_batched` in the original record's profile table is the same artifact:
`rmsnorm_batched` was 7.5% of decode time at that rate, and the decode-time
call is the many-small-rows case, not this one.

## What it changes

The PTQ1 GEMV's decode-free rate (343 GiB/s) is **82% of the device's measured
streaming ceiling** (420 GiB/s), not 60% of a nameplate number. The weight
access pattern is already near the device limit; the remaining 18% is the R=4
multi-stream read, which the R=2 tile (the accepted config) may already improve.

The absolute bound moves from the nameplate roofline to the measured one:

| bound | rate | decode tok/s for a 5.54 GiB model |
|---|---:|---:|
| nameplate 576 GB/s | 536 GiB/s | 96.8 |
| **measured streaming ceiling** | **420 GiB/s** | **75.8** |
| measured decode-free GEMV pattern | 343 GiB/s | 61.9 |
| measured GEMV with decode | 219 GiB/s | 39.5 |

75.8 tok/s is the absolute ceiling and assumes the trit decode, attention, all
norms, and the embedding and head passes are **free** — which they are not. The
100 tok/s target would need 554 GB/s, i.e. 96% of nameplate and 132% of what
the device measurably delivers on a trivial elementwise kernel.

The original conclusion stands and is now bounded twice: by nameplate and by
measurement. Per-stream decode on this model and device cannot reach 100 tok/s.

## What remains reachable

Two levers do not depend on raising the streaming rate, and neither was
attempted in the original record:

1. **Speculation** multiplies the bandwidth bound by the accepted-tokens-per-
   pass rate, because one verification pass reads all weights once and yields
   several tokens. At the measured 219 GiB/s GEMV rate, τ = 2.6 reaches
   100 tok/s. This requires a draft for the artifact; the original record notes
   none exists, and the repo's own n-gram rows are genre-conditional
   (+26–31% on code, −26% on prose).
2. **Continuous batching** amortises one weight read across concurrent streams.
   The daemon currently refuses it for this format, with the exact reason in
   the original record (`embd=PTQ1G128H lm_head=PTQ1G128H`), so aggregate
   throughput cannot exceed single-stream until batched embedding and batched
   `lm_head` exist for PTQ1G128H. The matmuls already take N.

These rows are measurement, not admission.

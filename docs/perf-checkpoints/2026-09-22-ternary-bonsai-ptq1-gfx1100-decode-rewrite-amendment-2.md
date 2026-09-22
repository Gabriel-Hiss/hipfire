# Amendment 2 — Ternary-Bonsai-2-27B PTQ1_0 gfx1100: the decode is ALU-bound, not memory-bound

**Date:** 2026-09-22
**Lifecycle:** `historical`
**Amends:** [`2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md`](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md) and [its amendment 1](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-1.md), both unchanged.
**Disposition:** **measured amendment.** Corrects the causal attribution of the original record. The original said the GEMV was ALU-bound at 57% of the card's bandwidth; amendment 1 raised the streaming ceiling and called the access pattern "near the device limit". This measures what is actually spent, and the answer invalidates the "bandwidth-bound" framing entirely.

## Why this exists

The original record and amendment 1 both reasoned from rate (GiB/s) to cause. That is the wrong inference. This record decomposes the GEMV into memory and arithmetic by abliteration, then tests the conclusion against the kernel's own ISA metadata.

## Fixture

Same host, device, HIP 7.2, and model as the parent records. Isolation used an
offline harness over the measured per-token shape mix, so each configuration
costs one rebuild plus a three-second run instead of a full decode.

The shape mix is measured, not assumed. It was obtained by instrumenting
`gemv_ptq1g128`, running one decode, and normalising by the `lm_head` call
count: **401 calls/token**, summing to 5.22 GiB of the model's 5.54 GiB (the
remainder is the PTQ1 embedding table and the norms).

| shape M×K | calls/token | MB/token |
|---|---:|---:|
| 17408×5120 | 128 | 2500 |
| 5120×17408 | 64 | 1250 |
| 10240×5120 | 48 | 551 |
| 5120×6144 | 64 | 440 |
| 6144×5120 | 48 | 330 |
| 248320×5120 | 1 | 278 |
| 12288×5120 | 16 | 220 |
| 1024×5120 | 32 | 37 |

The 17408×5120 projection alone is 47% of all weight bytes read per token.

## The harness is calibrated against the model

At the accepted configuration the harness reports 21.81 ms/token of GEMV. The
full model's decode profile attributed roughly 22 ms/token to the GEMV. The
harness reproduces the real number, so configurations can be compared offline.

## Abliteration: where the GEMV actually goes

Each row removes arithmetic while keeping every load, so the memory traffic is
identical across rows. All at `M=1048576, K=5120`-equivalent real shapes,
library-wide totals.

| variant | ms/token | GiB/s | vs memory floor |
|---|---:|---:|---:|
| full decode (accepted) | 21.81 | 239 | 2.18× |
| recurrence kept, `(w>>8)-1)*x` removed | **10.02** | **520** | 1.00× |
| only the `-1` removed | 19.43 | 268 | 1.94× |

Two conclusions follow, and they are not the ones the parent records drew.

**The memory side is already finished.** 10.02 ms/token is 520 GiB/s of pure
weight streaming, against a 536 GiB/s nameplate peak on this card. There is no
headroom left in the access pattern. Amendment 1's framing ("82% of the device's
measured streaming ceiling") understated it, because the elementwise kernels it
used as the ceiling were themselves not at the true limit.

**The decode arithmetic is 11.79 ms/token, more than half the GEMV.** The
remaining gap is arithmetic, not bytes.

## The arithmetic is integer-multiply-bound

Removing three operations (`>>8`, `- 1`, `* x`) from a sequence that also keeps
the `v*3` recurrence moves the total by 2.18×. Weighting a `V_MUL_LO_U32` at
quarter rate and every other integer op at full rate predicts 11/5 = 2.2×. The
prediction matches the measurement to 1%, so the model is:

| per element | units |
|---|---:|
| `w = v*3` (multiply) | 4 |
| `v = w & 0xFF` | 1 |
| `w >> 8` | 1 |
| `- 1` | 1 |
| `* x` (multiply) | 4 |
| accumulate | 1 |
| **total** | **12** |

**Two of every twelve units are multiplies consuming eight, and gfx1100 has no
integer dot-product to fold one of them.** `__builtin_amdgcn_sdot4` is rejected
at compile time on this target (`needs target feature dot1-insts`), which the
repo already documents for `FUSED_GATE_UP_HFQ4G256_WAVE64_DP4A_SRC`. The kernel
is at 33 VGPRs with **zero spills**, so it is not register-starved either: the
bound is multiply issue throughput.

## What was tried and rejected in this pass

| lever | measured with | result |
|---|---|---|
| row tile R=1 / R=2 / R=4 | harness | 21.81 / **21.42** / 24.69 ms — R=4 worse despite giving exactly balanced work for both dominant K values |
| `__launch_bounds__` min blocks 16 / 24 / 32 | harness | 24.25 / **21.42** / 23.51 ms — occupancy past 24 is worse, and the ISA shows why: the kernel is not register-limited |
| `v*3` → `v + (v<<1)` | harness | 22.60 ms — no gain, the compiler already strength-reduces it |
| 256-entry decode table in `__constant__` | model | 10.9 → 9.3 tok/s (regression, parent record) |
| `sdot4` packed decode | compile | rejected by hipcc on gfx1100 |

## Accepted — fold the `-1` into a shared sub-block sum

`Σ(digit − 1)·x = Σ digit·x − Σ x`. The second term is per 32-element
sub-block and independent of the weights, so one lane accumulates it once per
group and every tiled row reuses it. This removes the `- 1` from the per-element
chain at the cost of 128 adds per group.

Harness: 21.42 ms/token (from 21.81). Model, `hipfire bench --spec off
--backend noslots --workload stateless --max-tokens 128`, 5 samples after 3
warmups:

| | before | after |
|---|---:|---:|
| decode tok/s | 31.3 (31.4/31.3/31.2/31.2/31.0) | **33.7** (34.1/33.7/33.7/33.6/33.5) |
| wall tok/s | 26.3 | 28.4 |
| TTFT ms | 765 | 706 |

`hipfire.exe` md5 `da94c77c416417c82b1b5d1f0320fd2f`, `daemon.exe` md5
`1ec251119f184077762008567f571b3b`, model md5 `8abae179f984e2461cbf0fece6a8606f`,
revision `4f9833a5c`.

The harness predicted 1.8% on the GEMV and the model moved 7.7%; the harness
runs without graph capture and appears to under-report the arithmetic savings.
The model number is the one to quote.

### Parity is preserved bit-exactly

This was a correctness-neutral rewrite (integer arithmetic, no precision loss),
and it was verified as such rather than assumed. Final-position logits against
`llama-debug` from the reference fork, prompt `[48,25,220,16,10,16,28]`:

| | value |
|---|---|
| correlation | 0.99996763 |
| RMSE | 0.012999 |
| max abs | 0.071036 |
| argmax | 248046 (matches reference) |
| top-10 ordering | identical to reference |
| **vs the previous kernel** | **RMSE 0.0, bit-identical** |

The channel test `test_gemv_ptq1g128` still passes at `max |gpu-cpu| = 1.335e-5`
and `prefill vs scalar = 3.815e-6`.

## The ceiling, and the blocker

Per-token budget at 33.7 tok/s (29.7 ms/token): GEMV 21.4 ms, everything else
about 8.3 ms.

| scenario | ms/token | tok/s |
|---|---:|---:|
| now | 29.7 | 33.7 |
| all decode arithmetic free (memory floor only) | 10.0 + 8.3 = 18.3 | 54.6 |
| all decode arithmetic free and non-GEMV halved | ~14.2 | 70 |
| whole model at 536 GiB/s nameplate, nothing else | 9.7 | 103 |

**The next lever is blocked.** Recovering the 11.4 ms/token of decode arithmetic
requires the trits to be stored in a form that does not need the base-3
recurrence and does not need a per-element multiply against `x`. Every in-format
route was measured or rejected:

- a decode table in `__constant__` regresses (divergent constant access serialises);
- a table in LDS needs a 1280-byte per-block fill that costs about as much as the
  decode it replaces at these block sizes, and the block size cannot grow without
  the R=4 regression above;
- the multiply cannot be folded into an integer dot-product because gfx1100 has none;
- the remaining op-shuffling schemes (select-based trit application) all land at
  6 to 7 issue units against the current 6, so they are sideways or worse.

What would move it is a **load-time repack of the PTQ1 weights into a
directly-addressable form** — two-bit trit codes, or pre-decoded signed bytes —
which removes the base-3 recurrence and one of the two multiplies per element.
That is a loader and format change, not a kernel change: it needs a new on-disk
or in-VRAM layout, a conveyor from `hipfire-quantize`, and the loader path that
materialises it. It also grows the weights by 14% at two bits per trit, which
lowers the memory-bound ceiling from 103 to about 90 tok/s while removing the
arithmetic that currently costs 38% of the frame.

These rows are measurement, not admission.

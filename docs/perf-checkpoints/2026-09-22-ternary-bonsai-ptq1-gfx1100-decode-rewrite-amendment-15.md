# Amendment 15 — Ternary-Bonsai-2-27B PTQ1_0 gfx1100: the q8_1 cache never hits, and the +32% was not its doing

**Date:** 2026-09-22
**Lifecycle:** `historical`
**Amends:** [`2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md`](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md) and amendments [1](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-1.md) through [14](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-14.md), all unchanged.
**Disposition:** **correction.** Amendment 12 credited a 32% DFlash gain to the q8_1 activation cache. The cache never hits. The gain was machine state.

## What was claimed

Amendment 12 recorded: "DFlash 39.20 -> 51.90 tok/s (+32% from the cache alone,
1.78x over AR)". The commit that landed the cache repeated it.

## What the counters say

An env-gated hit/miss counter in `ensure_q8_1_mmq_x`
(`HIPFIRE_Q81_CACHE_TRACE=1`):

```
[q8_1-cache] hits=0 miss=6500 (ptr=0x35b22e0000 b=1 k=5120)
[q8_1-cache] hits=0 miss=9000 (ptr=0x5a0170000 b=1 k=6144)
```

**hits=0.** The same source pointer arrives thousands of times and every one is
a miss. The cache cannot have produced a 32% gain because it never takes the
hit path.

## Why it never hits

`scratch_must_convert` is:

```rust
is_recording || capture_mode || cached_ptr != src_ptr
```

with the documented contract that "if a recorder is active the kernel always
runs" — the recorder has to observe the kernel, so the skip and the record stay
coupled. The Redline retained-replay default is enabled on this model, so
`is_recording` is true on every call and the pointer comparison never decides
anything.

It is not graph capture: `HIPFIRE_GRAPH=0 HIPFIRE_VERIFY_GRAPH=0` gives the same
`hits=0`.

So the cache is correct but inert in this configuration. It stays in the tree
because it is the same predicate the fp16/fp8 caches use and it will hit
wherever no recorder is active — but it is not a lever here.

## The real number

The 39.20 in amendment 12 was a single measurement taken while the host was in
its slow band; the 51.90 that followed was in its fast band. The A/B was never
controlled for state, and amendment 11 had already measured that band at 29.6 to
34.1 tok/s on one binary.

The K sweep in amendment 14 was taken in one window and is internally
consistent, so its shape (optimum at K=8, collapse past K=16) stands, and so does
its level: **AR 29.20 vs DFlash 54.0 tok/s at K=8, 1.85x.** Amendment 12's
"1.77x" and amendment 14's "1.85x" are the same effect measured twice.

The profiler row is also consistent with an inert cache: 2075 quantization calls
over 5 cycles (415/cycle) against 400 PTQ1 GEMMs per cycle, down from 429/cycle
— a 3% change, not the ~50% the pairing of wqkv/wz and gate/up would give if the
cache were live.

## What this changes and what it does not

Does not change: the parity gate, the lossless check, the K optimum, or the
amplitude of the DFlash win (~1.8x AR). All of those were measured independently
of the cache.

Changes: the attribution. There is no measured lever in the q8_1 path on this
configuration, and amendment 14's isolation of the draft's cost (T_verify 46.3 ms
without the neural draft vs 137.6 ms with, at K=16) remains the only large
measured lever left.

**Lesson, same class as amendment 13:** a within-session delta needs its
mechanism confirmed, not just its direction. Amendment 13 caught a drafter cache
inflating tau; this caught a cache that never fires inflating an attribution. Both
were visible only from a counter, not from the tok/s.

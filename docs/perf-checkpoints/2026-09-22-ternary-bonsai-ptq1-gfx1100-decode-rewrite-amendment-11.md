# Amendment 11 — Ternary-Bonsai-2-27B PTQ1_0 gfx1100: the q8_1 activation cache is dead under graph capture

**Date:** 2026-09-22
**Lifecycle:** `historical`
**Amends:** [`2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md`](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md) and amendments [1](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-1.md) through [10](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-10.md), all unchanged.
**Disposition:** **root cause found for a previously unexplained zero hit rate.** No code behavior changes; the finding is recorded in `ScratchState`'s field doc. This closes the open question of why the q8_1 cache never hits, and redirects the launch-reduction work from caching to fusion.

## Fixture

As amendment 10. Instrument: `HIPFIRE_Q81_CACHE_TRACE=1`, which
`ensure_q8_1_mmq_x` already carries.

## What was measured

32-token dflash run, Ternary-Bonsai-2-27B PTQ1G128H + MQ4 draft:

```
[q8_1-cache] hits=0 miss=2000 (ptr=0x4b2380000 b=1   k=5120)
[q8_1-cache] hits=0 miss=2500 (ptr=0x5b5c00000 b=256 k=5120)
[q8_1-cache] hits=0 miss=3000 (ptr=0x5bd4d0000 b=251 k=5120)
[q8_1-cache] hits=0 miss=3500 (ptr=0x4f99d0000 b=16  k=5120)
[q8_1-cache] hits=0 miss=4000 (ptr=0x4fa3d0000 b=16  k=17408)
[q8_1-cache] hits=0 miss=4500 (ptr=0x4f99d0000 b=16  k=5120)
[q8_1-cache] hits=0 miss=5500 (ptr=0x4b2380000 b=1   k=5120)
```

**0 hits, 5500+ misses.** The pointer sequence does repeat (`0x4f99d0000 b=16
k=5120` twice), so the thrashing-single-entry theory is testable and false: a
multi-entry cache would also produce 0 hits.

## Root cause

`scratch_must_convert` (`crates/rdna-compute/src/scratch.rs`):

```text
is_recording || capture_mode || cached_ptr != src_ptr
```

The predicate is `true` whenever either recorder is armed, **by design**: the
doc comment states the convert kernel must always run under a recorder so the
skip and the record stay coupled. If the kernel does not run it is not
recorded, and a recorder that misses a node produces a tape or graph that
replays wrong.

The dflash verify path is graph-captured end to end (`HIPFIRE_VERIFY_GRAPH`
defaults on). So for the entire verify, `capture_mode` is set and the cache is
bypassed: every `ensure_q8_1_mmq_x` call re-quantizes.

The redundancy the cache was written for is real. `ScratchState`'s own field doc
says so: *"a run of projections sharing one activation (wqkv and wz both read
x_rot_batch) paid for the conversion twice."* It is still being paid twice; the
cache only avoids it outside capture, where the verify does not run.

## Why caching cannot fix it

Keeping a lookup would require recording a skip as a node, which is exactly what
the predicate refuses to do. The fix has to remove the second conversion from
the graph, not let it be skipped:

- **Fuse the conversion into the writer.** `rotate_x_prism_hadamard` (and the
  rmsnorm that feeds it) already writes the activation; having it also emit the
  `block_q8_1_mmq` layout gives the graph one node per activation instead of
  two, and makes the sharing structural rather than cache-dependent.
- **Not attempted here.** It is a kernel change on the hottest PTQ1 path.

## Size of the prize, measured

From the amendment-10 profile (8920 calls / 4 cycles):

| kernel | calls/cycle | calls/layer | us/call (profiled) |
|---|---:|---:|---:|
| `gemm_ptq1g128_wmma` | 400 | 6.25 | 222 |
| `quantize_q8_1_mmq_ds4` | 430 | 6.7 | 96 |
| `rotate_x_prism_hadamard` | 317 | 4.95 | 100 |

`quantize_q8_1_mmq_ds4` runs at roughly 1:1 with the PTQ1 GEMMs that consume it.
Its data cost is small: `k=5120, batch=16` moves 327 KB in and 92 KB out, which
at the ~180 GB/s this model sustains is ~2.3 us against 96 us measured. So the
call is dispatch-bound, and the profiled 96 us is inflated by the profiler's own
event recording; the real per-launch cost is lower. Halving 430 calls per cycle
is therefore a single-digit-percent cycle win, not a path to the 200+ tok/s
target. That target needs the ~90 ms cycle roughly halved, and ~2230 launches
per cycle spread over ~35 per layer is where it lives.

## Correction to amendment 3

Amendment 3 recorded a "+32%" q8_1-cache win. This finding explains the
discrepancy the later measurement caught: the cache cannot contribute while
capture is armed, so any measured delta attributed to it was machine state.

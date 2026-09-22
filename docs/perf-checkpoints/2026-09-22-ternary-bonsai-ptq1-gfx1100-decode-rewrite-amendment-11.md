# Amendment 11 — Ternary-Bonsai-2-27B PTQ1_0 gfx1100: final spec sheet and the measurement floor

**Date:** 2026-09-22
**Lifecycle:** `historical`
**Amends:** [`2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md`](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md) and amendments [1](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-1.md) through [10](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-10.md), all unchanged.
**Disposition:** **summary, plus a calibration of the bench noise.** Records where the series landed and how much a single bench run on this host is worth.

## Fixture

`hipfire.exe` md5 `807e3ddadf3e7e604582f9ff6d1dc2fa`, `daemon.exe` md5
`196fb04fef36ef5788dc6e38d5a49abe`, model md5
`8abae179f984e2461cbf0fece6a8606f`, revision `7b6bae01f`.

Canonical bench, median of three five-run series:
`--spec off --backend noslots --workload stateless --max-tokens 128 --runs 5 --warmups 3`.
Scaling with `bench_qwen35_mq4 --prefill N --warmup 3 --gen 2`, fresh process per N.

## Spec sheet

| | baseline | now | |
|---|---:|---:|---:|
| prefill | 34 tok/s | **227.3** | 6.7x |
| TTFT | 706 ms | **105.6 ms** | 6.7x |
| decode (AR) | 33.7 tok/s | **34.1** | unchanged |

| prompt | baseline | now |
|---:|---:|---:|
| 64 | 34.2 | **360.8** |
| 256 | 34.1 | **403.7** |
| 1024 | 33.7 | **398.8** |

Flat across a 16x span, which is what a batched path looks like. The baseline was
flat at 34 for the opposite reason: it was not batching.

Parity, pinned prompt `[48,25,220,16,10,16,28]` against the fork:

| path | correlation | RMSE | argmax | top-10 |
|---|---:|---:|---|---|
| per-token | 0.99996763 | 0.012999 | 248046 | identical |
| **batched (what the daemon takes)** | **0.99996864** | **0.013200** | **248046** | **identical** |

`hipfire run "Q: 1+1="` answers `2`.

The decode is unchanged because every change landed in the prefill path or in the
batched rotate helpers. Its ceiling (amendment 6) is 52.9 tok/s and its cost is
5.62 GB/token against a measured 551 GB/s.

The prefill target remains out of reach for the reason amendment 6 gives: ceiling
1306 tok/s, and the other 9.3% of prefill does not scale with the GEMM.

## The bench noise is real, and it is not the code

While collecting the above, the same binary with the same flags produced decode
29.6 and later 34.1, and prefill 196 and later 227. That is a 15% spread on
identical inputs.

It was not the code. The one hot-path change in between was a `begin_timer` on
the Q8_1 activation quantization, so it was reverted and the binary rebuilt: the
same 29.8 tok/s decode came back. `begin_timer` returns early through one
thread-local check when profiling is off, which is ~5-10 ns against 401 calls per
token, i.e. 0.01% of a 33 ms token.

This confirms what `AGENTS.md` states without a number ("within-session A/B is
noisy on gfx1100, +-10-15% from DPM/thermal state") and sets the floor for
reading any single row in this series: **a 3% difference between two bench runs
on this host carries no information.** Amendment 10's 234.6/34.5/102.3 and this
amendment's 227.3/34.1/105.6 are the same measurement.

`rocm-smi` is not available on this Windows host, so the throttle state cannot be
read directly and the only defence is repetition within one window.

## Remaining levers, with their measured targets

1. **Cache the Q8_1 activation quantization** (`ensure_q8_1_mmq_x`,
   `must_convert = true` hardcoded, no pointer-keyed cache while the fp16/fp8
   siblings have one). 16.9% of decode kernel time, 401 launches per token.
   Decode-side. Needs invalidation wired into every write to the source buffer;
   a missed one is the silent-corruption class the file's own comment warns
   about.
2. **The prefill GEMM's surroundings**: the 8 per-output-row scale loads whose
   scattered addresses serialise into 13 `s_waitcnt vmcnt()` per group, and the
   byte-by-byte fragment packing. The matrix unit is 23% occupied and the kernel
   issues 8 WMMA against 312 ALU per group, so the GEMM has 4.3x of headroom to
   its own ceiling.

Rejected and recorded so they are not re-run: INT4 for the ternary weights
(amendment 3, now with the measured iu4:iu8 ratio of ~2.0x in amendment 11's
commit), the 2-bit trit repack, the decode GEMV row-tile increase, and routing
the batched GDN through the decode's kernel.

These rows are measurement, not admission.

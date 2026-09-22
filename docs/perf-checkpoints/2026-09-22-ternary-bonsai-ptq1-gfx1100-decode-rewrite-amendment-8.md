# Amendment 8 — Ternary-Bonsai-2-27B PTQ1_0 gfx1100: prefill scaling after the matrix-unit GEMM

**Date:** 2026-09-22
**Lifecycle:** `historical`
**Amends:** [`2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md`](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md) and amendments [1](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-1.md), [2](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-2.md), [3](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-3.md), [4](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-4.md), [5](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-5.md), [6](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-6.md), [7](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-7.md), all unchanged.
**Disposition:** **measured amendment.** Fills the scaling row amendment 3 opened but that no amendment after the matrix-unit GEMM had re-measured.

## Fixture

As amendment 6 (`hipfire.exe` md5 `1cb750ea8e61e8a94ae2148205fde1a4`, revision
`1f5d4fec9` plus the two commits after it). Model unchanged.

Method: `bench_qwen35_mq4 --prefill N --warmup 3 --gen 2`, fresh process per N, so
the JIT cost is warm and excluded.

## The series

| prompt | per-token loop (am. 3) | BF16 gate (am. 3) | **matrix-unit GEMM** |
|---:|---:|---:|---:|
| 64 | 34.2 | 20.0 (JIT) | **371.8** |
| 256 | 34.1 | 51.5 | **372.0** |
| 1024 | 33.7 | 51.0 | **373.5** |

Wall: 172.15 / 688.17 / 2741.39 ms.

**Flat at ~372 tok/s across a 16x span of prompt length.** The objective asked for
this row specifically to distinguish "improved at one point" from "scales with
length": the rate is constant, which is what a batched path looks like, and it is
the same flatness the per-token loop had at 34 except 11x higher.

## This number is not the daemon's

The canonical `hipfire bench` reports prefill 234.3 tok/s on its own prompt with
TTFT 100.8 ms (amendment 6). Both are real and they measure different scopes:
this example times `forward_prefill_batch` alone on a synthetic prompt of exactly
N tokens, while the daemon's figure includes prompt templating, tokenization, the
first-token logits, and its own chunking. The scaling conclusion holds on either
path; do not cross-compare the two as if they were the same measurement.

## Disposition

The scaling deliverable is satisfied: prefill is flat and length-independent at
~372 tok/s on the batch path, against a ceiling of 1306 (amendment 6) and a
target of 3000 that the ceiling rules out.

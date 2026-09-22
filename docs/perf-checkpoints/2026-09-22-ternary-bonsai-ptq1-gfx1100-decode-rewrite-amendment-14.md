# Amendment 14 — Ternary-Bonsai-2-27B PTQ1_0 gfx1100: the requant/rotation fusion is feasible but not worth it

**Date:** 2026-09-22
**Lifecycle:** `historical`
**Amends:** [`2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md`](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md) and amendments [1](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-1.md) through [13](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-13.md), all unchanged.
**Disposition:** **design finding, not implemented.** Amendment 11 named fusing the q8_1 conversion into the writer as the fix for the always-missing conversion. This amendment records that it is mechanically tractable and that the arithmetic does not justify it. No code changed.

## It is tractable

`block_q8_1_mmq` (`kernels/src/gemm_hfq4g256_residual_mmq.hip:36`) is 144 bytes:
`half2 ds4[4]` plus `int8_t qs[128]`, indexed `Y[kblock * N + token]`.
`quantize_q8_1_mmq_ds4` covers one 1024-element k-block with 256 threads at four
floats each; per-32-lane groups reduce amax and sum, so one 128-element Q8_1
sub-block lands per 32 lanes.

`rotate_x_prism_hadamard` (`kernels/src/rotate_x_prism_hadamard.hip`) uses the
same geometry: 256 threads, one workgroup per `(row, k-block)`, and the finished
values sit in shared `tile[]` before the store to `out[base + i]`. It could
reduce amax/sum per 32 elements there and write the Q8_1 sub-block alongside the
f32 output, which would put one node per activation in the captured graph
instead of two.

## Why it is not worth taking

The saving is one dispatch per shared activation. Amendment 11's profile puts
`quantize_q8_1_mmq_ds4` at 430 calls per cycle and `rotate_x_prism_hadamard` at
317, against a PTQ1 GEMM count of 400: roughly one conversion per GEMM, of which
the wqkv/wz pair is the redundant share, about 215 calls.

At the real per-dispatch cost (the profiled 96 us is inflated, and the kernel's
own data movement for `k=5120, batch=16` is ~2.3 us at the ~180 GB/s this model
sustains), that is roughly **3 ms of a ~100 ms cycle, about 3%**. Against it, the
rotation kernel would gain a per-32-element amax/sum reduction and 128 extra bytes
written per 128 elements, which consumes most of the saving.

## Why the risk is the deciding argument

The rotation is on the hot path of every PTQ1 GEMM in the verify, and a defect in
the Q8_1 layout is a silent numerical corruption rather than a crash. This model
has already paid that bill twice in this series: the shared-rotation basis bug
(correlation 0.043 against the fork) and the four duplicated dtype lists that
silently dropped arms. A 3% ceiling does not justify another entry in that
column.

If the cycle is to be attacked, the measured shape of the problem (2230 launches
per cycle spread over ~35 per layer, no single large redundancy) points at
whole-graph restructuring rather than one fused kernel.

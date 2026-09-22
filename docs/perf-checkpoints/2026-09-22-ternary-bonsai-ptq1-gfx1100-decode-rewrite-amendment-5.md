# Amendment 5 — Ternary-Bonsai-2-27B PTQ1_0 gfx1100: the decode GEMV row-tile increase is a regression

**Date:** 2026-09-22
**Lifecycle:** `historical`
**Amends:** [`2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md`](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md) and amendments [1](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-1.md), [2](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-2.md), [3](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-3.md), [4](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-4.md), all unchanged.
**Disposition:** **negative result.** One rejected lever, recorded so it is not re-run. No code change; the tree is at `00d8d8d6e`.

## Fixture

Identical to amendment 4 (same model md5 `8abae179f984e2461cbf0fece6a8606f`, same
flags, gfx1100, HIP 7.2).

## The hypothesis

`kernels/src/gemv_ptq1g128.hip`'s own header argues for multi-row on this format
and against it on HFQ4/MQ4:

> One weight block is 28 B while its matching Q8_1 activation block is 144 B — the
> activation is 5.1x the weight. That ratio is the exact inverse of HFQ4/MQ4 (136 B
> weight vs 144 B activation), which is why row tiling amortises the wrong operand
> there and the right one here: every extra row per workgroup reuses the same 144 B
> Xq block.

So `PTQ1_ROW_TILE` 2 -> 4, the one constant in both `kernels/src/gemv_ptq1g128.hip`
and `Gpu::GEMV_PTQ1G128_ROW_TILE` (they must match: the grid is `ceil(M / this)`).

## Result: rejected

| | tile 2 (kept) | tile 4 |
|---|---:|---:|
| decode | **36.1 tok/s** | 32.0 tok/s (-11.4%) |
| prefill | 231.7 tok/s | 239.4 tok/s (+3.3%) |

Decode samples at tile 4: 31.9 / 32.0 / 32.1 / 32.0 / 32.0.

The activation-traffic argument is correct and is not what binds. Doubling the row
tile doubles the live decode state per lane (`v[R]`, `sumi[R][4]`, `q[R]`), and
this kernel is latency-bound with occupancy as the binding constraint — its
`__launch_bounds__(32, 24)` exists to keep resident waves high enough to hide the
dependent integer chain. Amortising a load it was already hiding costs more
occupancy than it saves traffic.

The prefill move (+3.3%) is not a reason to take it: prefill runs the separate
`gemm_ptq1g128_wmma` kernel now, so the two numbers are not coupled, and +3.3% is
inside this host's prefill spread (amendment 4's five runs span 227.1-233.6).

## What this means for the decode target

The decode is GEMV-bound and cannot be closed from the non-GEMV side. From the
measured rates: 27.7 ms/token total against ~19.4 ms of PTQ1 GEMV, so driving
every other kernel to zero yields **~51 tok/s**. The 80 tok/s target needs the GEMV
itself near 10 ms/token, a 2.1x improvement in the kernel, and the only lever that
size is removing the base-3 `v = (v*3) & 0xFF` decode chain from the hot loop —
a weight repack (2 bits per trit) touching the converter, the loader, and every
PTQ1 kernel. That is a separate project, not a tuning pass.

Recorded so the next attempt starts from the occupancy finding rather than
re-deriving the activation-traffic argument and re-running this experiment.

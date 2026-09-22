# Amendment 4 — Ternary-Bonsai-2-27B PTQ1_0 gfx1100: the matrix unit delivers 4.4x prefill

**Date:** 2026-09-22
**Lifecycle:** `historical`
**Amends:** [`2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md`](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md) and amendments [1](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-1.md), [2](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-2.md), [3](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-3.md), all unchanged.
**Disposition:** **measured amendment.** Executes the lever amendment 3 identified and reports the result, the one kernel defect found on the way, and where the remaining headroom is.

## Fixture

Same host, device, HIP 7.2, and model as the parent records
(`C:/tmp/bonsai-2-27b.ptq1`, md5 `8abae179f984e2461cbf0fece6a8606f`).
`hipfire.exe` md5 `31af7812f04897e23d7dff1bb3f82e8e`, `daemon.exe` md5
`92195da1b030ada5a1ef56898f45362c`, revision `00d8d8d6e`.

Canonical bench, byte-identical flags throughout:
`hipfire bench --spec off --backend noslots --workload stateless --max-tokens 128 --runs 5 --warmups 3`.

## Result

| | before (amendment 3) | after | |
|---|---:|---:|---:|
| prefill | 49.8 tok/s | **231.7 tok/s** | 4.65x |
| TTFT | 482 ms | **104.6 ms** | 4.61x |
| decode | 33.9 tok/s | 36.1 tok/s | unchanged (noise) |

Five-run samples, prefill: 233.3 / 231.7 / 227.1 / 229.4 / 233.6.
TTFT: 102.9 / 103.6 / 105.7 / 104.6 / 102.8 ms.
Decode: 35.5 / 35.7 / 36.4 / 36.8 / 36.1.

## iu8, not fp16

Amendment 3 recommended an FP16 WMMA GEMM on the argument that a trit times an
int8 activation is fp16-representable, so the integer dot survives. That argument
holds, but it is the harder of the two available routes.

`v_wmma_i32_16x16x16_iu8` takes **both** operands at 8 bits and accumulates in
int32. That is not an approximation of the integer dot the scalar kernel computes
— it *is* that dot, in the same type, with no intermediate fp16 rounding of the
activation and no dependence on the K=5120 accumulator staying under 2^24. It is
also what the 8-bit activation already in `ensure_q8_1_mmq_x` is for: the WMMA
kernel consumes the same Q8_1 buffer the scalar kernel consumes, unchanged.

The rate is the same as the fp16 route (91.96 TOPS for both on this part), so the
fp16 argument bought nothing that iu8 does not have for free.

## The defect: the weight scale was indexed by output column

The first working kernel produced degenerate output. The kernel was not wrong
about the trits, the activations, the sub-block boundaries, or the fragment
layout — it was wrong about **which row the weight scale belongs to**.

A wave32 WMMA lane holds A row `(L & 15)` but writes output row `2j + (L >> 4)`,
column `L & 15`. The first version loaded one `dw` per lane from the lane's own A
row and applied it to all eight of its `acc[j]`. That is `dw[column]` applied to
every output row.

It is invisible whenever `dw` is uniform. Every channel test written up to that
point used a single scale for all rows — the identity and pair patterns used 1.0,
the mixed-pattern test used a flat 0.5 — so all of them passed. The measurement
that exposed it, on a 16x128x16 case with trits held constant and `dw` varied per
row, printed the ratio of the kernel's output to the scalar's per output row:

```
out_row : W[0]      S[0]      ratio   expected dw
   0    :  -0.2418   -0.2418   1.000   0.100
   1    :  -0.2418   -0.3628   0.666   0.150
   2    :  -0.2418   -0.4836   0.500   0.200
   ...
  15    :  -0.2418   -2.0560   0.118   0.850
```

`W[0]` is constant across all sixteen output rows and equals `dw[0] * partial`.
The fix fetches the scale per output row. Same case after:

| case | before | after |
|---|---:|---:|
| `dw` equal | 0.000e0 | 0.000e0 |
| **`dw` per-row** | **4.252e0** | **7.153e-7** |

**The general lesson, which cost real time twice in this series:** a channel test
whose two operands are indexed by the same lane id, or whose parameters are
uniform across the dimension under test, proves less than it appears to. The
fragment-layout probe passed 0/256 while being unable to distinguish the A-row
index from the B-column index, because both were `tid & 15`.

## Verification

| shape | WMMA vs scalar, max abs |
|---|---:|
| 16x128x16 | 1.431e-6 |
| 257x512x17 | 7.629e-6 |
| 5120x5120x64 | 4.196e-5 |

All with random trits **and** random per-row, per-group scales. The 5120x5120x64
figure is fp32 accumulation noise over 40 groups (values in that test run to
~15), not a systematic difference. The fragment layout itself is pinned
independently by `test_wmma_iu8_layout.rs` (0/256 mismatches).

## Kernel-level rate and the remaining headroom

Measured at 5120x5120x64, 20 iterations, warmup discarded:

| kernel | ms | MAC/s |
|---|---:|---:|
| `gemm_ptq1g128_prefill` (scalar) | 1.181 | 1.42e12 |
| `gemm_ptq1g128_wmma` | 0.164 | **10.23e12** |

**7.20x at the kernel level**, and 22% of the 45.98e12 MAC/s INT8 ceiling. At the
end-to-end level the same change is 4.65x, which is the expected gap: the scalar
GEMM was ~96% of prefill wall time, and the remainder (activation quantization,
attention, norms) did not get faster.

The ceiling from amendment 3 is unchanged: **1702 tok/s at 100% matrix
efficiency**, and 231.7 is 13.6% of it. So ~7.3x of headroom remains, and it is
in the kernel (occupancy, K-tiling, amortising the per-output-row scale loads
against the LDS staging the tree already uses for its other WMMA GEMMs), not in
the dispatch. Amendment 3's finding that INT4 cannot be used at parity also
stands unchanged.

## A note on this model's free generation

`hipfire run` on this checkpoint produces degenerate text (`ledge,,,,,,`). This is
the **model**, not the engine, and it is reproduced by the reference: the fork's
own saved logits for the pinned prompt `[48,25,220,16,10,16,28]` have argmax
248046 with special and whitespace tokens (17, 18, 16, 271 = `\n\n`, 198 = `\n`)
in the top ranks — no digit token, on a prompt whose answer is `2`. hipfire's
parity against those logits is 0.99996763 correlation with identical argmax and
identical top-10, as recorded in amendment 3.

The WMMA path was cleared of suspicion directly: forcing the scalar key back on
reproduces the same degeneration (different trajectory, same failure), so the
text is not attributable to this change.

## Disposition

Accepted. `GemmPTQ1G128Wmma` is registered ahead of the scalar entry with the
`HasWmma` predicate and preferred in the dtype->key resolve; `plain_gemm_key_for`
returns it wherever the arch has WMMA. Every arch without WMMA keeps the scalar
kernel byte-for-byte, and the channel test pins both key choices.

These rows are measurement, not admission.

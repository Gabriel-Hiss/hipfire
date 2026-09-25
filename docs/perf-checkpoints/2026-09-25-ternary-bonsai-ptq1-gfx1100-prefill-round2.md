# Ternary-Bonsai-2-27B PTQ1_0 gfx1100: batched prefill 865 → 1054 tok/s at pp2048, activations at one scale per 128

**Date:** 2026-09-25
**Lifecycle:** `historical`
**Follows:** [`2026-09-25-ternary-bonsai-ptq1-gfx1100-prefill-ceiling.md`](2026-09-25-ternary-bonsai-ptq1-gfx1100-prefill-ceiling.md), which this supersedes for current numbers.
**Disposition:** **measured.** Single-stream batched prefill, no cross-request batching, no speculation.
- Prefill is 1.22x the previous checkpoint's binary measured in the same session.
- Decode is unchanged.
- The default prefill activation format moves from the fork's Q8_1 (one int8 scale per 32 values) to one scale per 128. Top-20 parity against the fork stays within the spread of the Q8_1 route.
- Prefill reaches 87.7% of the re-measured sustained ceiling. The attribution below puts the remaining 12% in the GEMM, above the format probe.

These rows are measurement, not admission.

## Fixture

| | |
|---|---|
| GPU | RX 7900 GRE (gfx1100), HIP 7.2, Windows, no other GPU load; `hipfire.exe` and `daemon.exe` at High priority (both binaries gain ~5% from it; the A/B below runs both the same way) |
| GGUF | `Ternary-Bonsai-2-27B-PTQ1_0.gguf`, md5 `e6989efc2bd2dcf94ed4233208f58c98` |
| converted model | `bonsai-2-27b.ptq1`, md5 `8abae179f984e2461cbf0fece6a8606f` |
| previous | revision `78a8a3954` (the previous checkpoint), `hipfire.exe` md5 `197d7ca854a75830ade8c23694a207b9` |
| final | revision `f7b1dcc51`, `hipfire.exe` md5 `6e3a36b0ee3ce70c9f24cc805368f34f`, `daemon.exe` md5 `6355a86232a78ee35e89f04ab4ed656e` |
| prefill bench | `hipfire bench <model> --matrix --pp 512,2048 --ctx 128 --tg 4 --runs 3 --warmups 1 --spec off --backend noslots --workload stateless --json` |
| long-prompt fixture | `benchmarks/prompts/bonsai_ptq1_prefill_parity.json`, md5 `7432739e217d9cf981760a82b5fba4a3`; `harness_long_chat` (1989 tokens) through `hipfire bench --runs 3 --warmups 1 --max-tokens 4` |
| decode bench | `hipfire bench <model> --spec off --runs 5 --warmups 3 --max-tokens 128 --backend noslots --workload stateless --json`, default prompt md5 `d94d3115a3001f08a654d91461d6bdc4` |
| ceiling probe | `benchmarks/bench_wmma_ptq1_ceiling.hip`, md5 `c8567a077b10ec79a23218c140301c35` |
| reference engine | PrismML-Eng/llama.cpp `9a9394a89`, HIP build, same GGUF |

## Result

Previous and final ran interleaved in fresh processes, three rounds:

| | pp512 tok/s | pp2048 tok/s | `harness_long_chat` prefill tok/s (TTFT) | decode tok/s |
|---|---|---|---|---|
| previous `78a8a3954` | 911.8 / 897.5 / 889.5 | 868.1 / 865.4 / 861.4 | 838.7 / 837.9 / 833.8 (2.39 s) | 67.6 / 67.5 / 67.5 |
| **final `f7b1dcc51`** | **1085.7 / 1080.4 / 1090.9** | **1055.5 / 1053.8 / 1053.4** | **1027.8 / 1020.5 / 1018.1 (1.96 s)** | **67.4 / 67.4 / 67.5** |
| median gain | 1.21x | 1.22x | 1.22x | equal |

## What landed

| commit | change | pp2048 A/B |
|---|---|---|
| `03826f537`, `34f4dab2e` | prefill activations with one scale per 128 (`HIPFIRE_PTQ1_ACT`, default `q8s128`; `q8` restores Q8_1). The GEMM chains a group's eight WMMAs in int32 and scales once; the 64x128 tile takes every shape | 810 → 928 |
| `862bb1515` | wave-pair Q8 flash prefill attention (gfx11, head_dim 256): each wave owns half of head_dim, 64 queries share a K/V tile, softmax in registers; 2.0-2.7x standalone, rel-L2 vs f64 5.5e-4 (one-wave kernel 1.1e-3) | 924 → 960 |
| `4c5f2fa3b` | token-parallel DeltaNet conv1d (the 4-tap conv has no recurrence), bit-identical; 142 → 51 µs per 512 tokens | profile 45.7 → 19.4 ms |
| `ede0956f3` | FFN gate and up in one GEMM whose epilogue writes silu(gate)·up, bit-identical | 975 → 1003 |
| `ac57ed340` | fix the opt-in gfx1151 DPP reduction's `v_permlanex16` lane selects (every partial after lane 0's first was wrong) | n/a |
| `f7b1dcc51` | top-20 parity tooling: `prefill_topk_dump`, `scripts/bonsai_ptq1_topk_parity.py` | n/a |

## The ceiling, re-measured under sustained load

A 2 s prefill runs at the board's power-limited clock. The previous checkpoint's probe timed short launches, and the kernels ran ~12% faster there than inside the model. `bench_wmma_ptq1_ceiling.hip` now launches each probe back-to-back for 1.5 s and times the last second. Four runs:

| probe (register-only, production launch shape) | sustained MAC/s |
|---|---|
| bare `v_wmma_i32_16x16x16_iu8` | 44.1-44.6e12 |
| + Q8_1 work (convert + scale FMA per 32 values) | 31.1-31.5e12 |
| + one scale per 128 (convert + FMA per group) | 37.7-38.6e12, median 37.9e12 |

The same GEMM kernel measures 33.5e12 in a short burst and 27.7-29.5e12 when launched back-to-back for 2 s. In the model it runs 32.0e12, i.e. 84.5% of the format-128 probe.

pp2048 costs 24.33e9 MAC per token, 49.83e12 in total. From the profile below:

| bound for pp2048 | ms | tok/s | final as % |
|---|---:|---:|---:|
| GEMM at the bare iu8 rate (44.3e12), rest as measured | 1507 | 1359 | 78% |
| **GEMM at the format-128 probe (37.9e12), rest as measured** | **1698** | **1206** | **87.7%** |
| measured (profiler warmup passes 1934-1942 ms) | 1937 | 1057 | |

## Where 1937 ms of pp2048 goes

`profile_prefill_qwen35 --prefill 2048 --warmup 3 --kv-mode q8`, High priority, per-kernel HIP event times:

| term | ms | status |
|---|---:|---|
| PTQ1 GEMM at the format-128 probe | 1316 | the format's arithmetic on this ISA |
| PTQ1 GEMM above the probe | 239 | LDS staging, barriers, trit decode, epilogue; attempts below |
| DeltaNet recurrence | 133 | 695 µs per 512-token launch; attempts below |
| flash attention | 58 | new kernel |
| BF16 alpha/beta GEMM | 31 | |
| fused producers (glu, rmsnorm, gated norm, sigmoid → rotate → Q8) | 68 | memory-bound |
| conv1d, qk-norm, norms, RoPE, deinterleave, lm_head, others | 40 | |
| wall minus the event sum | 52 | |

## Negative results

All of these were measured and then reverted. Timings are same-process A/B.

### GEMM

Measured with one activation scale per 128:

| attempt | result |
|---|---|
| 128x128 workgroup, each wave 32x64 (0.75 LDS loads per WMMA instead of 1) | 0.99-1.01x |
| two groups per LDS stage (half the barriers) | 0.91x: occupancy halves |
| next group's raw weights and activations prefetched into registers | 0.88x |
| token tiles fastest in the grid (weight slab reused by neighbours) | 0.89-0.96x |
| ablation without trit decode | 1.00x: the decode hides behind other waves |
| ablation without the epilogue | 1.06x |
| ablation without both | 1.09x |

### Activations

- **int4, one scale per 32:** argmax matches the fork at only 999/1024 positions, and the flips include non-tied tokens (fork margin up to 0.34). Rejected.
- **int4, one scale per 128:** 990/1024, same failure. Rejected.
- **Packing several trits per int8 WMMA operand:** does not work. The 128-product partial sums need 15 bits, so two rows cannot share one byte's accumulator.

### Attention

| attempt | result |
|---|---|
| 6-head GQA workgroup sharing the dequantized K/V tile but keeping the one-wave softmax path | 0.90-0.95x |

### DeltaNet

Sequential form:

| attempt | result |
|---|---|
| 1, 2 or 8 state rows per wave | 1.85x, 2.2x, 2.0-2.5x slower |
| one fused reduction per row, using S·q from S_{t-1} | 1.27x slower, state bit-identical |
| 2-wave workgroup sharing q/k | 2.05x slower |
| q/k of 32 tokens staged in LDS | 2-8x slower: occupancy |

Chunked UT-transform form, fp32:
- Numerically right: output rel-L2 2.5e-7, 5 of 786432 int8 state entries off by 1.
- Still 2.9-3.5x slower than the tuned sequential kernel in every configuration tried: 16/32-token chunks, 32/64-row tiles, preparation split into its own kernel, next-chunk prefetch.
- The recurrence kernel spends ~50 µs per 16-token chunk even with each stage ablated in turn, and the cause was not found.
- **Not a closed question:** it is the one algorithmic change left untried beyond a first implementation.

### Other kernels

| attempt | result |
|---|---|
| BF16 alpha/beta as one paired launch, one token per wave | 4x slower: 64 workgroups, LDS-bound, and not bit-identical |

## Parity with the fork

`scripts/bonsai_ptq1_topk_parity.py`. The fork greedily continues `harness_mid_raw` (1099 prompt tokens) for 1024 tokens with `n_probs=20`. hipfire prefills prompt plus continuation once and reports its top-20 at each of the 1024 positions. Prompt ids are identical.

| hipfire format | argmax = fork token | top-20 sets identical | top-20 order identical | overlap mean / worst | top-20 gap deviation mean / median / p99 / worst / best |
|---|---|---|---|---|---|
| `q8s128` (default) | 1021/1024 | 617 | 63 | 19.41 / 12 | 0.079 / 0.039 / 0.68 / 2.48 / 0.000 |
| `q8` (fork's Q8_1) | 1020/1024 | 633 | 84 | 19.48 / 13 | 0.070 / 0.036 / 0.60 / 2.14 / 0.000 |
| `q8`, previous binary (one-wave attention) | 1022/1024 | 651 | 97 | 19.47 / 12 | 0.071 / 0.036 / 0.65 / 1.98 / 0.000 |

Every argmax flip lands where the fork's own top-1/top-2 margin is under 0.09 logits:
- `q8s128` flips at positions 3, 4 and 461 (fork margins 0.017, 0.036, 0.088).
- `q8` adds position 563 (margin 0.026).

The per-position worst deviations come from the tail of the top 20. There, one engine's 20th entry sits near the other's 21st.

Restricting the 128-scale to the FFN (0.077) or to the other projections (0.074) did not move the mean outside the spread between formats, so the default applies everywhere.

Decode keeps Q8_1. `scripts/bonsai_ptq1_parity.py` on `bonsai_ptq1_parity.json` (48 greedy tokens) on the final binary gives the previous checkpoint's numbers exactly: 4 of 5 sequences identical, `chat_ts` diverging at token 7 on a 0.004 tie, top-5 gap mean 0.0261, p99 0.1206, max 0.2106 (n=973).

## What is left

- **GEMM above the probe (239 ms):** the tile, staging and pipeline options tried are listed above. Every one measured slower or equal.
- **DeltaNet recurrence (133 ms):** the chunked form is numerically ready but not yet faster.
- **Past the format probe:** would need a different activation format, which the int4 measurements rule out at this parity bar.

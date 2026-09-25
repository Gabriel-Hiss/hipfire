# Ternary-Bonsai-2-27B PTQ1_0 gfx1100: batched prefill 481 → 809 tok/s at pp2048, and what bounds it

**Date:** 2026-09-25
**Lifecycle:** `historical`
**Follows:** [`2026-09-24-ternary-bonsai-ptq1-gfx1100-ar-decode-ceiling.md`](2026-09-24-ternary-bonsai-ptq1-gfx1100-ar-decode-ceiling.md), unchanged.
**Disposition:** **measured.** Single-stream batched prefill. No cross-request batching and no speculation. Prefill is 1.68-1.73x faster. Parity with the PrismML llama.cpp fork is unchanged and decode did not move. The result is 75-80% of the ceiling re-measured below, so it misses the 90% target. The attribution section splits the 2.53 s of a pp2048 prefill into measured terms. These rows are measurement, not admission.

## Fixture

| | |
|---|---|
| GPU | RX 7900 GRE (gfx1100), HIP 7.2, Windows, no other GPU load |
| GGUF | `Ternary-Bonsai-2-27B-PTQ1_0.gguf`, md5 `e6989efc2bd2dcf94ed4233208f58c98` |
| converted model | `bonsai-2-27b.ptq1`, md5 `8abae179f984e2461cbf0fece6a8606f` |
| baseline | revision `4a9b8a29e` (the decode checkpoint), `hipfire.exe` md5 `27b43b7d6d6613cc1bf69ab18520e2c1` |
| final | revision `78a8a3954`, `hipfire.exe` md5 `11ae8b1603546470e667f6368f3977b3`, `daemon.exe` md5 `80de658ebed4bde27ecafbaafd188cb9` (`ac57ed340` changes only the opt-in gfx1151 DeltaNet kernel) |
| prefill bench | `hipfire bench <model> --matrix --pp 512,2048 --ctx 128 --tg 4 --runs 3 --warmups 1 --spec off --backend noslots --workload stateless --json` |
| long-prompt fixture | `benchmarks/prompts/bonsai_ptq1_prefill_parity.json`, md5 `7432739e217d9cf981760a82b5fba4a3` (`harness_mid_raw` 1099 tokens, `harness_long_chat` 1989 tokens) |
| decode bench | `hipfire bench <model> --spec off --runs 5 --warmups 3 --max-tokens 128 --backend noslots --workload stateless --json`, default prompt md5 `d94d3115a3001f08a654d91461d6bdc4` |
| reference engine | PrismML-Eng/llama.cpp `9a9394a89`, HIP build, same GGUF |

## Result

Baseline and final ran interleaved in fresh processes:

| | pp512 tok/s | pp2048 tok/s | `harness_long_chat` prefill tok/s | decode tok/s |
|---|---|---|---|---|
| baseline `4a9b8a29e` | 491.2 / 495.2 / 487.1 | 482.4 / 480.5 / 477.5 | 480.0 / 477.7 / 478.8 | 65.2, 65.3, 64.2, 65.3, 64.3 |
| **final `78a8a3954`** | **851.8 / 847.4 / 831.2** | **812.6 / 809.0 / 802.6** | **794.4 / 782.8 / 789.7** | **65.4, 65.2, 63.8, 65.3, 65.2** |
| median gain | 1.73x | 1.68x | 1.65x | 65.2 → 65.2 |

- The pp columns are medians of 3 samples each, 3 rounds.
- The fixture column is prefill of the 1989-token prompt plus the bench's chat template, with TTFT 2.52-2.55 s.
- Decode is 5 rounds, median of 5 samples each. The same baseline binary measured 67.0 tok/s in the previous record and 65.2 today. The change is in the machine, so decode is compared only within a session.

## The ceiling, re-measured

`benchmarks/bench_wmma_ptq1_ceiling.hip` uses register-only kernels at the production launch shape: 128-thread workgroups, each wave running 2x2 WMMA tiles. Median of 9 runs:

| probe | MAC/s |
|---|---|
| bare `v_wmma_i32_16x16x16_iu8`, 8 independent chains | 46.8-48.3e12 |
| the same WMMAs plus the work the Q8_1 activation format requires | **30.7-35.1e12** (66-73%) |

The format work per 32-wide activation sub-block is an int→float conversion and one FMA by that sub-block's scale per element. On top of that, each 128-weight group adds one FMA by the weight scale.

On RDNA3 the WMMA issues on the VALU, so this float work serializes with it. The t64 ablation in commit `da5423d00` showed the same thing: removing the WMMAs halved the kernel time. Any kernel that keeps the fork's per-32 activation scales does at least this work.

The PTQ1 projections cost 24.33e9 MAC per token:
- DeltaNet layers: 48 × 382.7e6
- attention layers: 16 × 372.2e6

With the non-GEMM kernels at their measured 487.9 ms (profile below):

| bound for pp2048 | ms | tok/s | final as % |
|---|---:|---:|---:|
| GEMM at the bare iu8 rate (47e12), rest as measured | 1548 | 1323 | 61% |
| GEMM at the format probe's best (35.1e12), rest as measured | 1907 | 1074 | **75%** |
| GEMM at the format probe's median (32.2e12), rest as measured | 2035 | 1006 | **80%** |
| **measured final** | **2532** | **809** | |

## Where 2.53 s of pp2048 goes

`profile_prefill_qwen35 --prefill 2048 --warmup 3 --kv-mode q8` measures per-kernel HIP event times. Its unprofiled warmup passes take 2538-2572 ms.

| term | ms | evidence |
|---|---:|---|
| PTQ1 GEMM, at the format ceiling (35.1e12) | 1419 | probe above |
| PTQ1 GEMM, above that ceiling | 488 | 1600 launches, 1906.9 ms, 26.1e12 MAC/s in-model (27-30e12 in isolation) |
| Q8 flash attention (16 layers x 4 chunks) | 160 | `prefill_attend_step` timer |
| DeltaNet recurrence (48 x 4) | 140 | 730 µs per 512-token launch |
| fused producers (rmsnorm / silu / gated norm / sigmoid → rotate → Q8_1) | 87 | bit-exact to the unfused chain |
| conv1d + BF16 alpha/beta GEMM | 76 | |
| norms, RoPE, deinterleave, lm_head, others | 23 | |
| wall minus the event sum | 137 | not a launch gap: a created stream instead of the null stream leaves the wall unchanged (603 vs 603 ms at pp512) while the event sum moves 567 → 616 ms, so this term is the event method's error |

## Negative results

Each row was measured and then reverted.

### GEMM

| attempt | result |
|---|---|
| exact f16 WMMA: trits and int8 activations are exact in f16, so no int→float conversion | 0.81-0.89x, not bit-identical |
| weights pre-expanded to int8 in memory, so no trit decode in the kernel | 0.64-1.07x: reading 128 B per group instead of 28 costs more than the decode it saves |
| register-prefetch pipelining, balanced decode, generic TI/TJ/WR/WT tilings, a 64x128 eight-wave tile | 0.90x to +9% on the largest shapes, slower on small ones (earlier session) |
| iu4 WMMA (2x the iu8 rate) | int8 activations would need two iu4 WMMAs per iu8 one, the same time. 4-bit activations would leave the fork's numerics |

### Attention

| attempt | result |
|---|---|
| one workgroup per KV head group, K/V tile dequantized once for 4 query heads | bit-identical, 0.90-0.95x |
| Phase-A d-chunk loop unrolled | bit-identical, 0.59-0.63x |
| next K chunk loaded during the current WMMA | bit-identical, 0.94-0.96x |

### DeltaNet

| attempt | result |
|---|---|
| next token's q/k/v/gate/beta prefetched | no change |
| four state rows in registers with interleaved reductions | bit-identical, 0.89x |
| DPP/permlane reduction | bit-identical after `ac57ed340` fixed its lane selects, no speed change (0.603 vs 0.606 ms) |

## What landed

| commit | change |
|---|---|
| `da5423d00` | 64x64 workgroup-tiled prefill GEMM, trits decoded once per tile with `v_perm_b32` |
| `680684e82` | four weight rows per wave in the batched BF16 GEMM (DeltaNet alpha/beta) |
| `7a30c07c0`, `f6117f483` | fused producers take a token dimension and write the GEMM's Q8_1 layout, residual adds move into the GEMM epilogue (`HIPFIRE_PTQ1_FUSED=0` restores the chain) |
| `4b71c37c6` | contiguous per-row weight staging in the tile |
| `733a11fcd` | batched PTQ1 embedding lookup plus one inverse Hadamard per chunk, replacing 3N launches |
| `75bf8d78f` | 64x128 tile for the projections with M or K ≥ 8192 |
| `78a8a3954` | ceiling probe and prefill attention timer |

The fused producers and both GEMM epilogues are gated bit-exact against the unfused chain by `test_ptq1_fused_decode` (97-token section).

## Parity with the fork

Run on the final binary with `scripts/bonsai_ptq1_parity.py`, 48 greedy tokens. Prompt ids are identical everywhere.

| fixture | greedy | top-5 logit gap mean / p99 / max | divergences |
|---|---|---|---|
| `bonsai_ptq1_prefill_parity.json` | 1 of 2 sequences identical | 0.0238 / 0.1006 / 0.1281 (n=250) | `harness_mid_raw` at token 3: fork 6090 -1.234 vs 11327 -1.251, hipfire margin 0.008 |
| `bonsai_ptq1_parity.json` | 4 of 5 identical | 0.0261 / 0.1206 / 0.2106 (n=973) | `chat_ts` at token 7: margin 0.004 on hipfire, 0.013 on the fork |

The prefill-fixture numbers match those taken before any of these changes. `ptq1_parity` last-position logits against the fork dump give corr 0.99996470, rmse 0.013558, identical top-10.

## What is left

- Most of the remaining gap comes from the activation format. Scales per 128 activations instead of per 32, or 4-bit activations, would remove the format work or halve the WMMA time. Both change the numerics away from the fork's Q8_1, so they need a new parity decision.
- Inside the format:
  - the 488 ms of GEMM above the probe ceiling (LDS staging, decode, barriers);
  - attention, the only term that grows with context;
  - the DeltaNet recurrence, a serial chain of 512 tokens per launch.
- None of the attempts above moved any of the three.

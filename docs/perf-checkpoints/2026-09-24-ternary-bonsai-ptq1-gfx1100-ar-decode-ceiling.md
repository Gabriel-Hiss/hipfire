# Ternary-Bonsai-2-27B PTQ1_0 gfx1100: AR decode 34.7 → 67.0 tok/s, and where the other 30% of the ceiling goes

**Date:** 2026-09-24
**Lifecycle:** `historical`
**Follows:** [`2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md`](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md) and its amendments, all unchanged.
**Disposition:** **measured.** Single-stream autoregressive decode, no speculation of any kind. Decode is 1.93x faster at 70.7% of the re-measured DRAM ceiling. The 90% target is out of reach on this route, and the attribution below accounts for every ms/token. These rows are measurement, not admission.

## Fixture

| | |
|---|---|
| GPU | RX 7900 GRE (gfx1100), HIP 7.2, Windows, no other GPU load (a game running during earlier sessions moved numbers by up to 3x; every row here was taken with it closed) |
| GGUF | `Ternary-Bonsai-2-27B-PTQ1_0.gguf`, md5 `e6989efc2bd2dcf94ed4233208f58c98` (the dense-trit pack; the HF repo ships no file named TQ1_0) |
| converted model | `bonsai-2-27b.ptq1`, md5 `8abae179f984e2461cbf0fece6a8606f` (`hipfire-quantize --format ternary`; a fresh conversion gives identical logits) |
| baseline | revision `3d0749c04` |
| final | revision `d05e9c48f`, `hipfire.exe` md5 `63afb21dffe20c9e196636528f6fd943`, `daemon.exe` md5 `d8ad9129d5baf6efc177798311a0c675` |
| bench | `hipfire bench <model> --spec off --runs 5 --warmups 3 --max-tokens 128 --backend noslots --workload stateless --json`, default prompt "Explain the theory of general relativity in simple terms." (md5 `d94d3115a3001f08a654d91461d6bdc4`) |
| reference engine | PrismML-Eng/llama.cpp `9a9394a89`, HIP build, same GGUF |

## Result

Three fresh-process pairs, baseline and final interleaved, 5 samples each:

| | decode tok/s (medians) | samples of pair 1 | prefill tok/s | TTFT ms |
|---|---|---|---:|---:|
| baseline `3d0749c04` | 34.9 / 34.7 / 34.7 | 35.1, 35.0, 34.9, 34.9, 34.8 | 282 | 85 |
| **final `d05e9c48f`** | **67.0 / 66.9 / 66.9** | 67.0, 67.0, 67.0, 67.0, 67.0 | 281 | 85 |
| llama.cpp fork, `llama-bench -p 0 -n 128 -r 3 -fa 1` | 30.57 ± 0.36 | | | |

## The ceiling, re-measured

Peak read bandwidth is `kernels/src/probe_dram_bw.hip` streaming 1-2 GiB (16-32x the 64 MB
Infinity Cache) with 4-8 loads in flight per thread. It reads **540-557 GB/s** across grids and
runs, so this record uses **543 GB/s**. The 551 GB/s in amendment 6 came from a 147 MB buffer,
which partly fits in the Infinity Cache; runs with the cache-resident buffer read up to 785 GB/s,
which is not DRAM.

Bytes per decoded token: every GGUF tensor except `token_embd` (one row is read) is 5.657 GB,
plus 75 MB of Q8 DeltaNet state read and written across 48 layers, for **5.735 GB/token**.

| bound | ms/token | tok/s |
|---|---:|---:|
| DRAM floor, 5.735 GB at 543 GB/s | 10.56 | 94.7 |
| 90% of the ceiling (the target) | 11.73 | 85.2 |
| **measured final** | **14.93** | **67.0 (70.7%)** |

## Where 14.93 ms/token goes

In-model decomposition: a temporary capture-time filter (not committed) kept only selected
kernel classes in the captured decode graph. Outputs are garbage when classes are dropped, so
these runs measure time only. There were 3 interleaved rounds per case:

| graph contents | ms/token (median) | rounds |
|---|---:|---|
| full decode | 14.925 | 14.903, 14.925, 14.925 |
| PTQ1 GEMVs only | 12.240 | 12.225, 12.240, 12.255 |
| empty (host round trip + sampling) | 0.313 | 0.318, 0.303, 0.313 |

Attribution of the 14.93 ms:

| term | ms/token | evidence |
|---|---:|---|
| DRAM floor for the weights | 10.42 | 5.657 GB / 543 GB/s |
| DRAM floor for DeltaNet state | 0.14 | 75 MB / 543 GB/s |
| GEMV below peak | 1.51 | GEMV-only graph 11.93 ms = 474 GB/s in model; 488 GB/s for a graph of the same 48 shapes with nothing between them |
| non-GEMV work and the GEMV restarts it forces | 2.69 | full minus GEMV-only graph |
| host round trip, sampling, token readback | 0.31 | empty graph |

The GEMV-only graph is already 12.24 ms, above the 11.73 ms that 90% allows. The target
therefore cannot be met by removing non-GEMV work alone. It needs the GEMV itself within 4% of
the probe's peak, plus overlap of everything else, and both were attempted:

**GEMV efficiency, every variant measured and rejected:**

| variant | result |
|---|---|
| lanes own 4 consecutive groups (7 aligned b128 loads, 16 lanes/row) | load-only +4-13%, with the dot products 240 GB/s against 488 in chain (bit-identical output) |
| nontemporal weight loads | 350 vs 450 GB/s |
| 128- and 256-thread workgroups | equal or slower on every shape |
| 2 row tiles per lane, 2 or 3 groups in flight per lane | slower on every shape |
| `__launch_bounds__` occupancy 8, 12, 20 | within noise of 16 |
| weights for GEMV j+1 streamed into the Infinity Cache on a side stream during GEMV j | 2.4x slower: on this Windows/HIP stack graph branches and a second stream serialise, and each fork costs ~70 µs |
| a paced persistent streamer kernel | hangs in graph replay: the two streams do not run concurrently |

**Non-GEMV overlap, measured and rejected:**

| variant | result |
|---|---|
| producer (rmsnorm/silu/gated-norm → rotate → Q8_1) and GEMV in one launch, GEMV waves spin on a counter | 10-15% slower than two launches (occupancy 16 → 12 from the producer's LDS; forcing 16 gives no gain) |
| same with a one-wave-per-block producer | 3x slower |

The non-GEMV classes by duplicate-launch cost (each class launched twice inside the graph,
second launch minus first, 2 rounds): rmsnorm+rotate+Q8 0.48, DeltaNet recurrence 0.36, attention
tile 0.23, gated norm+rotate+Q8 0.15, conv+qk-norm 0.14, others < 0.1 each, 1.6 ms total. The
rest of the 2.69 ms is DRAM idle while a small kernel runs between two GEMVs. The dependency
chain is strict (every GEMV input is the previous kernel's output), and the one stream the graph
runs on cannot hide it.

## What changed (revisions `df76a67ea`..`d05e9c48f`)

| change | effect |
|---|---|
| decode GEMV decodes five trit digits per dword with two `v_perm_b32` and dots them with `v_dot4_i32_iu8`, 8 lanes per row, plain b128+b96 weight loads | ~250 GB/s (baseline, 131072×5120 alone) → 488 GB/s (graph of the model's 48 shapes) |
| rmsnorm / silu_mul / sigmoid_mul / gated_norm fused with the Prism rotation and Q8_1 quantization; butterflies by shuffle, one LDS stage per 128-512 stride | 3 launches → 1 per GEMV input, bit-identical |
| projections sharing an input in one launch (qkv+z, q+k+v, gate+up), the BF16 beta/alpha folded in, residual adds in the GEMV epilogue | 1716 → 786 captured nodes/token before the BF16 fold, which removes 96 more; bit-identical |
| q8 flash-attention tile grid capped at 64 looping workgroups instead of spanning `max_seq` | ~9 µs per FA layer at short context, bit-identical |
| tokenizer: `\s+(?!\S)` emulated, GGUF `token_type` decides special tokens | identical ids to the fork's `llama-tokenize` on ~334K tokens of Rust, Python, C++, HIP and Markdown (files with literal `<|...|>` text compared with special parsing on, as hipfire always parses them); before, indented lines and runs of spaces differed |

`test_ptq1_fused_decode` asserts exact equality for every fused producer, the multi-segment GEMV
(plain and residual, ragged M) and the BF16 pair against the unfused chains. With the decode
kernel changes in and before the tokenizer fix (which changes prompt ids), greedy output of a
775-token prompt was token-identical to the baseline, with the attention grid capped at both 64
and 3 workgroups so the tile loop ran more than once.

## Parity with the PrismML fork

`scripts/bonsai_ptq1_parity.py` runs the prompts in
`benchmarks/prompts/bonsai_ptq1_parity.json` (md5 `479cdba3819e7e1271f0d838e58e72d9`), 256
greedy tokens each, against `llama-server` from the fork on the unmodified GGUF:

| prompt | prompt tokens hipfire / fork | first divergence | top-1 vs top-2 at the divergence, fork | same, hipfire |
|---|---|---|---:|---:|
| merge_sort_raw | 20 / 20 | none (255 tokens) | | |
| chat_lru | 31 / 31 | none (255) | | |
| rust_raw | 30 / 30 | 163 of 203 | 0.0022 | 0.0250 |
| chat_bug | 50 / 50 | none (255) | | |
| chat_ts | 40 / 40 | 7 | 0.0127 | 0.0040 |

Top-5 logit gaps to the top-1, hipfire vs fork, over every compared position (n=4656):
mean 0.0256, p99 0.130, max 0.317. The baseline build measured on the same harness: mean 0.0290,
max 4.86. Its maximum was the whitespace tokenizer bug, which made `merge_sort_raw` diverge at
token 1. The baseline matched `chat_ts` at a 0.013 tie and missed `rust_raw` at the same place as
the final build.

Both remaining divergences are near-ties. The fork's own margin is smaller than the p99 logit
disagreement between two engines that sum in different orders. Identical greedy sequences on
every prompt are not reachable while the two engines accumulate in different orders. Of the 5
prompts, 3 are identical over 255 tokens, and the other 2 match up to a top-1/top-2 tie within
the measured tolerance.

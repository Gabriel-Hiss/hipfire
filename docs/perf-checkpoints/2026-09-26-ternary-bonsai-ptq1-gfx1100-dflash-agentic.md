# Ternary-Bonsai-2-27B PTQ1_0 gfx1100: greedy DFlash on agentic tool calls, 60 → 273 tok/s

**Date:** 2026-09-26
**Lifecycle:** `historical`
**Follows:** [`2026-09-25-ternary-bonsai-ptq1-gfx1100-prefill-round2.md`](2026-09-25-ternary-bonsai-ptq1-gfx1100-prefill-round2.md), unchanged.
**Disposition:** **measured.** Single-stream greedy DFlash on agentic coding traffic (tool-call
heavy). Every DFlash output below equals the AR output byte for byte, tool calls included. AR
decode and its parity with the PrismML fork are unchanged. These rows are measurement, not
admission.

## Fixture

| | |
|---|---|
| GPU | RX 7900 GRE (gfx1100), HIP 7.2, Windows, no other GPU load |
| model | `bonsai-2-27b.ptq1`, md5 `8abae179f984e2461cbf0fece6a8606f` (from GGUF `e6989efc2bd2dcf94ed4233208f58c98`) |
| binaries | `hipfire.exe` md5 `f8c9606eb8f1f7cc4f55c457b8caf13d`, `daemon.exe` md5 `190b66a82807a7697453e37fd315e0a9` (revision `118885f03` plus the distillation commits) |
| workload | [`benchmarks/prompts/bonsai_agentic_toolcall.json`](../../benchmarks/prompts/bonsai_agentic_toolcall.json), md5 `3586564fa474ab38e6283947d7f29947`: 8 coding-agent conversations whose next turn is a tool call (whole-file rewrites in Python/Rust/TS/Go/JSON, an edit after a test failure, a `list_dir`, a four-call read) |
| harness | `serve_harness.py` driver, `--kv q8 --thinking off --sampling greedy --max-tokens 1024`, one warmup round then 2 measured rounds; aggregate = Σ gen / Σ (gen / decode tok/s) |
| drafts | 5-layer `qwen35-27b-dflash-mq4.hfq` (the previous default); 6-layer `z-lab/Qwen3.5-27B-DFlash` (safetensors md5 `ecda3d3e53652b4b4a35127ffec03d0e`) converted to MQ4, `.hfq` md5 `0c0afcd92807bb7f6648a09267c38877`; the same draft fine-tuned (below), bf16 md5 `b8af8d10d0c5d39bed7009c8ca261ca2`, MQ4 `.hfq` md5 `980b3f03d4fd32633c1e9c1354a0d0c0` |

## Result

Final binary, one session, AR and both 6-layer drafts:

| case | gen | AR tok/s | base draft | τ | fine-tuned | τ |
|---|---:|---:|---:|---:|---:|---:|
| py_rename_rewrite | 679 | 64.6 | 244.4 | 9.94 | 305.8 | 12.84 |
| rs_pool_rewrite | 735 | 64.2 | 214.2 | 8.53 | 286.0 | 11.88 |
| py_test_fix_edit | 100 | 63.8 | 89.8 | 3.21 | 214.2 | 10.00 |
| ts_router_methods | 524 | 64.9 | 209.9 | 8.38 | 290.7 | 12.08 |
| go_cache_delete | 754 | 64.6 | 181.5 | 7.01 | 229.1 | 9.18 |
| json_scripts | 298 | 65.3 | 181.5 | 7.03 | 312.6 | 13.14 |
| py_new_tests | 25 | 65.0 | 56.5 | 1.67 | 181.4 | 11.00 |
| multi_read | 109 | 65.7 | 83.4 | 2.67 | 292.9 | 12.50 |
| **aggregate** | | **64.7** | **185.8** | | **272.9** | |

τ is accepted drafts per cycle. The outputs of all three columns are identical per case.

How the aggregate moved, each row measured on the binary of that step (5-layer draft until
the draft swap):

| step | agentic tok/s |
|---|---:|
| start: DFlash with the 5-layer draft (AR 64.7) | 59.8 |
| verify GEMM for ≤16 tokens that reproduces the decode GEMV (`320146ddc`) | 76.6 |
| tool calls parsed on the DFlash path (`fdb175b8b`), no speed change | 76.6 |
| stopped window retracted in place, not re-prefilled (`9bb902ff0`) | 120.0 |
| fused producers + residual epilogue for verify batches (`f634a3bb1`) | 127.0 |
| split-key draft attention (`605654177`) | 141.1 |
| 6-layer draft, sliding layers on the split kernel (`ce8a4bb0f`) | 152.3 |
| split-key q8 verify attention + per-token BF16 rows (`c314917ef`, `1d435728e`) | 176.6 |
| causal sliding draft layers (`118885f03`) | 187.0 |
| fine-tuned draft | 272.9 |

## What each step fixed

**Verify GEMM.** A DFlash verify runs 16 tokens through the prefill GEMM, which ran its
one-wave tile at ~85 GB/s on these shapes. `gemm_ptq1g128_verify` gives each of 8 waves the
decode GEMV's residue class of K groups, runs every Q8_1 sub-block through two chained iu8
WMMAs, keeps block_dot's float order and the GEMV's xor-butterfly, so a verified token's
projections equal the decoded token's bit for bit (checked on all dense shapes at N = 1, 5,
16). The decode GEMV's block accumulate is now pinned to two roundings; the compiler had
been free to contract it. 48-shape chain: 12.2 → 3.9 ms.

**Tool calls.** The DFlash wrapper withheld the request's tools from the emitter whenever the
Hermes grammar was off, which is the default for XML-native Qwen3.5. DFlash turns came back
as raw `<tool_call>` text with `finish=stop`. Tools now always reach the emitter;
`grammar` is a separate flag.

**Window retract.** Almost every turn ends inside an accepted window (end of turn). The
unobserved tail was dropped with a recurrent reset plus a prefill of the whole history,
1.3-1.7 s per request at these context lengths, counted in decode time.
`Speculator::retract_window` rewinds in place: restore the pre-verify DeltaNet snapshot,
replay the verify's GDN tape for the kept tokens, truncate the draft's context log.

**Verify producers.** Batches of ≤16 tokens now use the fused norm/rotation/Q8_1 producers
and the GEMM residual epilogue, the path n > 32 already had.

**Draft attention.** The draft's attention ran one wave per (head, 16-query tile): 32 waves at
long context. The split-key kernel gives each workgroup one KV head and one key range, with
the GQA heads as its waves, merged by log-sum-exp: 1.2 ms → 55 µs per layer at 1.3k keys.
It also takes sliding windows, and sliding layers are causal when the draft does not say
otherwise (the upstream rule; the 6-layer draft sets no `is_causal`). Running it
non-causal had cost 12-14% of τ.

**Verify attention and BF16 rows.** The q8 flash prefill kernel serves 64 queries per
workgroup; a 16-query verify used a quarter of it on 24 workgroups. The split-key q8 kernel:
311 → 88 µs per layer at 1.3k keys. The DeltaNet β/α BF16 GEMMs (48 × 5120) ran on 24 waves;
one wave per (row, token) runs them in 14 µs and matches the decode GEMV bit for bit.

## Fine-tuning the draft

Self-distillation on the target's own outputs (`scripts/dflash_distill/`, with its
README):

1. 1600 agentic conversations built from real source files (hipfire, llama.cpp, Python
   packages): a system prompt, one of three tool sets, a `read_file` result and a request
   (rename, add a helper, docs, a local edit, tests, explain, multi-file read, test after a
   write, refactor, bug fix). None of the fixture's files is among them; tool set 0 and its
   system prompt match the fixture's.
2. The target answered 396 of them (35 min, greedy DFlash, `HIPFIRE_SPEC_TOKEN_LOG`
   recording prompt and output ids): 131k output tokens, token log md5
   `37685b87d243ddc0b7c5a6e7022ee88f`.
3. `dflash_distill_dump` replayed prompt ++ output through the target and stored the 8
   context layers the draft reads (967k positions, 80 GB f16, 19 min), plus the embedding
   rows and the effective lm_head.
4. LoRA rank 64 on every linear of the draft (40.8M trainable), CE on block positions 1-15
   weighted e^{-(k-1)/7}, 32 random blocks per sequence per step, 3 epochs (1131 steps,
   40 min on the same GPU), merged into bf16.

Offline accepted length per 16-token block on 19 held-out conversations: 5.18 before,
9.18 at step 500, 9.52 at the end.

Outside the fine-tuning distribution (different tools, system prompt or task):

| | base 6-layer draft | fine-tuned |
|---|---:|---:|
| `session_tool_call.json`, 5 turns, other tool set | — | transcript byte-identical to AR, 123.6 tok/s avg over 15-28-token turns |
| `hipfire bench` default prompt (LRU cache) | 116.4 | 133.9 |
| copy prompt, 1.3k tokens (Rust pool) | 305.1 | 320.7 |
| copy prompt, 1.4k tokens (Python module ×2) | 291.6 | 306.7 |

## AR decode and parity with the fork

- `hipfire bench --spec off` (5 runs, 3 rounds interleaved with `6c12cbf9a`): 67.0 vs 67.0 tok/s.
- `scripts/bonsai_ptq1_parity.py` (48 greedy tokens, fork `llama-server`): 4 of 5 sequences
  identical, `chat_ts` diverges at token 7 on a 0.004 tie, top-5 gap mean 0.0261, p99 0.1206,
  max 0.2106 (n=973). Same numbers as the previous checkpoint.
- On the agentic fixture the AR outputs are unchanged from the start of this work
  (same md5 per case).

## Where the cycle goes now

1.3k-token context, 6-layer draft, block 16 (`HIPFIRE_SPEC_PHASES=1`, synchronising per
phase): draft ~9 ms, verify ~30 ms, GDN tape replay 1.6 ms. The verify is dominated by the
PTQ1 verify GEMM (~25 ms per cycle for 5.66 GB of weights; the DRAM floor is 10.4 ms). Its
load-only twin on the 48-shape chain takes 2.1 of its 3.9 ms, so the remainder is the
decode and the WMMA chain, not memory. Splitting the trit decode across half-waves with
`v_permlanex16` measured 4% faster on the chain; it is not in this revision.

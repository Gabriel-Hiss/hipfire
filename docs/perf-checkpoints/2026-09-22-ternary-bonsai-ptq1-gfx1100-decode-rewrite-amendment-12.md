# Amendment 12 — Ternary-Bonsai-2-27B PTQ1_0 gfx1100: the prefix cache fires, and sampling halves tau

**Date:** 2026-09-22
**Lifecycle:** `historical`
**Amends:** [`2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md`](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md) and amendments [1](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-1.md) through [11](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-11.md), all unchanged.
**Disposition:** **measurement correction.** The first measurement of this model on a real serving session, and it moves the realistic target: the copy-heavy tau of 10.55 is a best case, not the expected acceptance on generated code. Also establishes that the prefix cache works and quantifies sampling's cost.

## Fixture

| item | value |
|---|---|
| harness | `scripts/serve_harness.py --mode session --model C:/tmp/bonsai-2-27b.ptq1 --speculation dflash --draft qwen35-27b-dflash-mq4.hfq --kv q8 --thinking off --max-tokens 128` |
| session | `benchmarks/prompts/session_coding.json` (committed 8-turn coding chain) |
| model md5 | `8abae179f984e2461cbf0fece6a8606f` |
| daemon md5 | `d16936d0b456d237fc27b13e683598f9` |
| harness reports | per-turn `cached=`, `prefill=`, `decode=`, `tau=`, `recall=` |

## The prefix cache fires

Both arms report reuse; the greedy arm:

| turn | cached | prefill | tau |
|---:|---:|---:|---:|
| 1 | 0 | 252 ms | 0.95 |
| 2 | 0 | 775 ms | 0.84 |
| 3 | 0 | 1326 ms | 0.95 |
| 4 | 256 | 1034 ms | 0.90 |
| 5 | 256 | 1533 ms | 0.92 |
| 6 | 256 | 1954 ms | 0.33 |
| 7 | **897** | **263 ms** | **2.63** |
| 8 | 256 | 2740 ms | 1.19 |

Turn 7, the one turn that branches back into earlier conversation, caches 897-914
tokens instead of 256 and its prefill drops to 263-273 ms against 775-2853 ms
for the 256-cached turns. **The mechanism works and the benefit scales with how
much of the prompt repeats.** Only 256 tokens are reused here because the eight
turns are independent coding questions; a tool-call workload that resends a
fixed system prompt and tool schemas would reuse far more, which this fixture
does not model.

Note that amendment 10's `multiturn` bench arm cannot measure this: `prefix_hits`
is written as `0` in both `ArmResult` constructions in `bench_concurrency.rs`, and
`arm_is_valid` rejects a multiturn+slots arm with zero hits as not measuring what
it claims. The session harness is the route that works.

## Sampling costs about a third of decode

Same session, same model, `--sampling greedy` vs `--sampling registry:general`
(temp 1.0, top_p 0.95, top_k 20):

| turn | tau greedy | tau sampled |
|---:|---:|---:|
| 1 | 0.95 | 0.90 |
| 2 | 0.84 | 0.57 |
| 3 | 0.95 | 0.65 |
| 4 | 0.90 | 0.41 |
| 5 | 0.92 | 0.74 |
| 7 | 2.63 | 0.92 |
| 8 | 1.19 | 0.79 |

Decode tracks it: ~21 tok/s greedy against ~15 tok/s sampled. Speculative
decoding accepts far less off the greedy path, and the loss is larger than the
sampling difference itself because tau multiplies through every cycle.

## Correction to the target

Amendments 6-8 measured tau 10.55 and 107-131 tok/s on
`code_edit_rewrite_copy.txt`, where the task is to copy a list of integers out of
the prompt. That is a best case for both drafters, not the expected acceptance
on generated text. On this session of real coding turns, tau is **0.9-2.6 at
greedy**, and the decode that follows is 15-21 tok/s.

Both numbers are correct for their fixture. The mistake would be quoting 10.55 as
"the" acceptance for this model. Any throughput claim on Ternary-Bonsai-2-27B
must name the genre, because tau spans an order of magnitude between copying and
generating.

This also reframes the 200+ tok/s targets: they are reachable on copy-heavy
traffic and not on novel generation, where tau near 1 makes throughput
cycle-bound at ~100 ms per cycle regardless of speculation.

## Not established

- **Whether a tool-call prompt shape reuses more than 256 tokens.** The session
  fixture does not have a stable system prefix across turns, so it cannot answer
  this. Measuring it needs a fixture with a fixed system prompt plus tool schemas
  plus one tool result.
- The retrieval gate failed on this run (`missing expected substrings:
  [['dedupe'], ['rayon','chunk']]`) and 7 of 8 turns hit `finish=length`. The
  throughput rows are usable; the recall rows are not.

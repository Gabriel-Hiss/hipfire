# Amendment 13 — Ternary-Bonsai-2-27B PTQ1_0 gfx1100: a tool-call prefix reuses in full, and tau on tool syntax is low

**Date:** 2026-09-22
**Lifecycle:** `historical`
**Amends:** [`2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md`](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md) and amendments [1](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-1.md) through [12](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-12.md), all unchanged.
**Disposition:** **accepted measurement, and it settles the target for agentic traffic.** On a prompt shape with a stable prefix the LCP cache reuses essentially the whole prompt and cuts prefill ~7x. Tau on tool-call syntax is low, so speculation contributes little there and TTFT is what matters.

## Fixture

New committed session fixture: [`benchmarks/prompts/session_tool_call.json`](../../benchmarks/prompts/session_tool_call.json).
Turn 1 is a fixed system block (six tool schemas plus rules); turns 2-5 are
agentic-coding requests against a `src/pool.rs` thread pool. Every later turn
re-renders turn 1 verbatim, which is what the `session_coding.json` fixture
lacks: its turn 1 is short and its eight turns are independent questions, so at
most a few hundred tokens were ever reusable.

| item | value |
|---|---|
| harness | `serve_harness.py --mode session --session benchmarks/prompts/session_tool_call.json --model C:/tmp/bonsai-2-27b.ptq1 --speculation dflash --draft qwen35-27b-dflash-mq4.hfq --kv q8 --thinking off --sampling greedy --max-tokens 96` |
| model md5 | `8abae179f984e2461cbf0fece6a8606f` |
| daemon md5 | `d16936d0b456d237fc27b13e683598f9` |

## Result

| turn | ctx | cached | prefill | tau | gen |
|---:|---:|---:|---:|---:|---:|
| 1 | 317 | 0 | 1023 ms | 3.33 | 26 |
| 2 | 371 | 343 | 140 ms | 0.87 | 28 |
| 3 | 431 | 399 | 142 ms | 1.33 | 28 |
| 4 | 485 | 459 | 151 ms | 0.67 | 15 |
| 5 | 535 | 500 | 199 ms | 1.80 | 28 |

`cached == ctx - 28` on every turn after the first: the whole prefix is reused
and only the new user turn is prefilled. Prefill falls from 1023 ms cold to
140-199 ms, about **7x**. The run was clean (`runaway=0 empty=0 attractor=0
retrieval_miss=0`) and the model emitted real tool syntax
(`<tool_call><function=read_file>...`), so the shape is representative.

Contrast amendment 12's `session_coding` run, where `cached` stalled at 256 for
five of eight turns and prefill stayed at 775-2853 ms. Same mechanism, same
binary; the difference is whether consecutive turns share a prefix. **The cache's
value is a property of the traffic shape, not of the engine.**

## Tau on tool-call syntax is low

0.67-3.33 across five turns, against 10.55 on the copy-heavy fixture. Tool-call
syntax is generated, not copied, so a drafter has little to match. For agentic
traffic this inverts the priority: output is short (15-28 tokens here), so decode
tok/s barely matters and **TTFT dominates the wall clock**. The prefix cache is
the lever that moves it, and the speculation machinery is close to irrelevant.

Caveat: five short turns is a small sample, and the per-turn decode numbers
(5.9-15.2 tok/s) include warmup. The prefill and `cached` rows are the load-bearing
ones here; the decode rows are not.

## Combined with amendment 12

For this model, in agentic serving:

| lever | measured effect |
|---|---|
| stable prompt prefix (LCP cache) | prefill 1023 -> 140 ms, ~7x |
| greedy instead of temp 1.0/top_p 0.95 | tau ~2x, decode ~+30% |
| speculative decoding | tau 0.67-3.33 on tool syntax; little |
| K above 16 | loses (cycle +45% for +11% tau) |

The first two are free and are the ones that matter. Neither is a kernel change.

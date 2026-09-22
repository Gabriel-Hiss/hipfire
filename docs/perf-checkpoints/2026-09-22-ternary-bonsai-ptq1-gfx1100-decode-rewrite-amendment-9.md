# Amendment 9 — Ternary-Bonsai-2-27B PTQ1_0 gfx1100: the batched prefill does not reproduce the reference logits

**Date:** 2026-09-22
**Lifecycle:** `historical`
**Amends:** [`2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md`](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md) and amendments [1](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-1.md), [2](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-2.md), [3](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-3.md), [4](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-4.md), [5](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-5.md), [6](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-6.md), [7](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-7.md), [8](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-8.md), all unchanged.
**Disposition:** **correctness finding, and it invalidates the prefill parity claim of this whole series.** The prefill speed numbers in amendments 3, 4, 6 and 8 are real, and they are measurements of a path that does not reproduce the reference. Read them as throughput, not as accepted work.

## Fixture

As amendment 6. New harness: `crates/saddle-lab/examples/ptq1_parity.rs` (prefills
the pinned prompt `[48,25,220,16,10,16,28]` and compares against the fork's saved
`llamacpp-Ternary-Bonsai-2-27B-PTQ1_0.bin`).

## The two paths disagree

| path | correlation | RMSE | argmax | top-10 |
|---|---:|---:|---|---|
| per-token `forward_scratch` | **0.99996763** | **0.012999** | 248046 | identical |
| batched `forward_prefill_batch` | **0.04325525** | **3.782550** | 9814 | DIFFER |

The per-token row is **exactly** the recorded baseline (0.99996763 / 0.012999 /
248046 / identical), so the model, the harness and the fork reference are all
sound. The batched row is not a small deviation.

Same measurement with the scalar PTQ1 GEMM forced instead of the WMMA one:
correlation 0.04325525 -> 0.04325525-class values (0.0433, argmax 9814). **The
matrix-unit GEMM is not the cause.**

## It diverges at the first batched step

Batched against per-token, same tokens, same model, both from a fresh
`DeltaNetState` and a fresh Q8 KV cache, `dn_state.reset` applied:

| n | correlation | RMSE | top-10 |
|---:|---:|---:|---|
| 2 | -0.03044911 | 4.695885 | DIFFER |
| 3 | NaN | 1.738856 | DIFFER |
| 4 | NaN | 1.726325 | DIFFER |
| 5 | NaN | 1.683138 | DIFFER |
| 6 | NaN | 1.685184 | DIFFER |
| 7 | NaN | 1.718822 | DIFFER |

n=2 is `MIN_BATCH`. There is no compounding-with-length story here: the batched
path is wrong on its first two tokens.

## The daemon uses it

`HIPFIRE_DEBUG_BATCH=1` on `hipfire run` for the chat-templated prompt:

```
[hipfire::batch_eligible] result=true arch=gfx1100 n=19 n>=2=true
  force_fallback=false has_dn=true moe_router_logits_present=true
  all_layers_ok=true first_reject=None
```

`force_fallback` is `!verify_decouple && !config.prefill_batched`, and
`verify_decouple` is true for `n <= 32` on gfx11. **So for prompts up to 32
tokens the batched path is taken regardless of `prefill_batched`**, which is why
setting that config to false changed the generated text by nothing at all
(`璞 (a) (a) (a) (a` both ways).

## What is ruled out

`gemm_bf16_xf32_batched` — the kernel amendment 3 added to admit the BF16 gate
projections, and the reason this model reaches the batched path at all — is
**correct**. New channel test
`crates/rdna-compute/examples/test_gemm_bf16_xf32_batched.rs` against a CPU
oracle over the bf16-rounded weights:

| shape | worst relative error |
|---|---:|
| 16x128x8 | 4.470e-7 |
| **48x5120x7** (the real w_beta shape) | 1.007e-5 |
| 257x512x17 | 2.503e-6 |
| 1024x256x33 | 1.818e-6 |

`gemm_ptq1g128_wmma` and `gemm_ptq1g128_prefill` are ruled out by amendment 6's
channel tests (7.6e-6 against the scalar at 257x512x17, 4.2e-5 at
5120x5120x64), and the batched-vs-per-token divergence survives forcing the
scalar GEMM.

## What is not yet ruled out

The batched path's remaining exclusive kernels, in the order the profile's call
counts make them interesting:

- `gated_delta_net_q8_batch_seq` (48 calls). Its own doc at `prefill.rs:305` says
  it "requant[s] the Q8 state after every token, matching the decode requant
  cadence (**distributionally equivalent to decode, not byte-identical** — the
  stochastic-rounding frame differs)". A documented non-bit-identical recurrence
  cannot satisfy a 100% parity gate, and this model has 48 DeltaNet layers. This
  is the first thing to look at, and the question is whether the divergence it
  admits can be this large at n=2 or whether something else is also wrong.
- The batched Prism-Hadamard rotation. `rotate_x_prism_hadamard` is shared with
  the decode path, but the batched path rotates `x_batch` in one call where the
  decode path rotates per GEMV, and the model declares
  `prism.hadamard.block_size = 1024` with `sign_widths = [5120, 6144, 17408]`.
  `test_prism_hadamard.rs` covers batch 2 at width 2048; the real geometry is
  batch 256 at width 5120.
- `conv1d_silu_split_f32_n`, `deinterleave_f32_batched`,
  `fused_qk_l2_norm_scale_interleave_f32_batched`,
  `rope_partial_halfsplit_batched_f32`.

`crates/hipfire-runtime/examples/bisect_forward_slots.rs` already does exactly
the bisect needed (fresh reference vs fresh candidate, diffing `pbs.x_batch`
per `max_layer` to find the first divergent layer); it compares
`forward_prefill_batch_with_pbs_opts` against `forward_batch_slots_with_max_layer`
and would need the per-token path as its candidate.

## Why this went unnoticed until now

The model was **rejected** from the batched path before commit `c9ebd40f2`
(`first_reject: "DeltaNet w_beta dtype BF16 not batchable on gfx1100"`), so it
always ran per-token and always matched. Amendment 3's parity gate measured the
per-token path (it recorded the decode as bit-identical and compared gate
variants on the pinned prompt) and never measured the batched path's own logits
against the fork. Amendment 4, 6 and 8 then measured batched throughput without
re-running the logits gate on the path they were speeding up. **The gate was
applied to the wrong path for four amendments.**

## Disposition

The batched prefill for this model does not meet the parity requirement, and the
daemon uses it for prompts up to 32 tokens. The prefill throughput rows in
amendments 3/4/6/8 (49.8 -> 231.7 -> 234.3 tok/s, 372 tok/s on the batch path
alone) are measurements of that path and must not be cited as accepted results
until the divergence is fixed.

Reverting `is_batchable_la(DType::BF16)` restores parity by returning this model
to the per-token loop (34 tok/s), and is a one-line change. That is a product
decision, not a measurement, so it is not taken here.

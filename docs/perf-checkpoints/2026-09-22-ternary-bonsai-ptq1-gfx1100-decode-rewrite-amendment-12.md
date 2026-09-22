# Amendment 12 — Ternary-Bonsai-2-27B PTQ1_0 gfx1100: the DFlash draft attaches, and what it costs

**Date:** 2026-09-22
**Lifecycle:** `historical`
**Amends:** [`2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md`](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md) and amendments [1](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-1.md) through [11](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-11.md), all unchanged.
**Disposition:** **first speculative-decode record for this model.** The draft attaches and beats AR on code. The target numbers in the SSSD brief are not reachable with this draft, and the reason is measured.

## Fixture

Target `C:/tmp/bonsai-2-27b.ptq1` (md5 `8abae179f984e2461cbf0fece6a8606f`), draft
`~/.hipfire/models/qwen35-27b-dflash-mq4.hfq` (0.919 GB, sha256
`3d428b97c1911a9ad815cc52fbee080306852c1dafad6b1b17bb70bd68010301`), revision
`d3f742b51`. Profiler: `dflash_spec_demo` with `HIPFIRE_PROFILE=1`.

## The draft fits the target

The published qwen3.5:27b DFlash draft declares hidden 5120, `num_target_layers`
64, intermediate 17408, head_dim 128 — the same shape Ternary-Bonsai-2-27B
reports. Three gates blocked it, none fundamental:

| gate | fix |
|---|---|
| loader `dflash_lm_head_quant_supported` rejected qt=43 | admit PTQ1G128H on the gfx11+gfx12 WMMA set |
| `dflash_enqueue_verify_lm_head` had no PTQ1 arm | Prism-rotate the hidden state, then `gemm_ptq1g128_wmma` |
| the noise-embedding lookup had no PTQ1 arm | `embedding_lookup_ptq1g128_prism` |

**Lossless**: the drafted sequence is identical to AR on the same prompt, and the
pinned-prompt parity is unchanged (0.99996864 / RMSE 0.013200 / argmax 248046 /
top-10 identical).

## It wins on code, by 1.4x

`def fibonacci(n):`, 128 tokens, `--backend noslots`:

| | decode | |
|---|---:|---|
| AR | 27.90 tok/s | |
| DFlash | **39.20 tok/s** | tau=5.05, 128 tokens in 21 windows |

6.1 tokens committed per window against a break-even of 4.36
(`T_verify / T_AR`). Acceptance is strongly prompt-dependent: 32% on that prompt,
**5% on the bench's own prompt** (70 windows for 128 tokens).

## Where the verify cycle goes

11138 kernel calls over 5 cycles, 1846 ms of kernel time:

| kernel | calls | total | % |
|---|---:|---:|---:|
| gemm_ptq1g128_wmma | 2000 | 503.4 ms | 27.3 |
| **quantize_q8_1_mmq_ds4** | 2147 | 235.5 ms | **12.8** |
| **rotate_x_prism_hadamard** | 1579 | 197.4 ms | **10.7** |
| gemv_ptq1g128 | 147 | 184.9 ms | 10.0 |
| rmsnorm_batched | 940 | 115.4 ms | 6.3 |
| gemm_bf16_xf32_batched | 480 | 105.7 ms | 5.7 |
| add_inplace_f32 | 640 | 73.2 ms | 4.0 |
| gated_delta_net_q8_batch_seq | 480 | 70.9 ms | 3.8 |
| conv1d_silu_split_f32_n | 480 | 54.2 ms | 2.9 |
| 10 more | 1865 | 267.5 ms | 14.5 |

**2228 launches per verify cycle.** The weight-read floor for one pass is 10.2 ms
(5.62 GB at the measured 551 GB/s); T_verify measures ~146 ms. The gap is launch
count, not bytes or math.

Two of the top three are already-known levers:

- `quantize_q8_1_mmq_ds4` at 12.8% and 429 calls/cycle is the uncached Q8_1
  activation quantization (amendment 11 lever 1). It costs more here than in AR
  decode because a verify cycle issues 17 tokens' worth of GEMMs against the
  same weights.
- `rotate_x_prism_hadamard` at 10.7% and 316 calls/cycle is 5 rotations per
  layer. After amendment 10's basis fix, `wqkv` and `wz` share an activation and
  a rotation plan, so one of those five is redundant per layer.

## The draft does not predict this target

```
seed-oracle: cycles=26 full_accept=0 mean_accept_len=1.423
             rej_match=0.000  tail_match=0.000  anypos_match=0.038
accept_rate (accepted / (cycles x (B-1))): 0.095
adaptive-b: mean_B=10.62
```

The draft's rejection prediction matches the target **0.000** of the time, and
the accepted token matches its proposal at any position **3.8%** of the time.
The draft was trained against BF16 Qwen3.5-27B; Bonsai is a ternary re-quant of
it, and the token distributions have diverged past the point where the head
predicts the target.

This is the failure mode `AGENTS.md` records for the 3.6-A3B draft ("trained on
3.5 traces; target distribution mismatch on code. tau=1.22"), and it is why the
SSSD brief's 98%-acceptance row is out of reach here.

## Disposition

The brief's blocker has two conditions. The first is resolved: a draft exists,
attaches, and beats AR by 1.4x on code, so the work is not blocked. The second is
met: measured acceptance is 9.5% overall and the rejection oracle matches 0.0%,
so the 120-180 tok/s row cannot be reached by tuning this draft.

Two independent directions, both with measured targets:

1. **Cut T_verify from 146 ms toward the 10.2 ms floor.** 2228 launches per cycle
   is the whole story, and two of the three biggest kernels are redundant work
   (the uncached Q8_1 quantization, and one of five per-layer Prism rotations).
   This multiplies whatever acceptance exists and is orthogonal to the draft.
2. **A target-aligned draft.** The published draft is for the BF16 parent. A head
   trained against Bonsai's own ternary distribution would be needed for the
   high-acceptance rows, and the project's own record says training one is out of
   scope (amendment 3's Path C note).

These rows are measurement, not admission.

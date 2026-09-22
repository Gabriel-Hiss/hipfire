# Amendment 10 — Ternary-Bonsai-2-27B PTQ1_0 gfx1100: batched prefill parity restored

**Date:** 2026-09-22
**Lifecycle:** `historical`
**Amends:** [`2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md`](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md) and amendments [1](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-1.md) through [9](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-9.md), all unchanged.
**Disposition:** **correctness fix.** Amendment 9 recorded that the batched prefill did not meet the parity gate. This closes it: two activation-basis bugs, both the same class, one in the batched QKVZA arm and one in the fused rotate helpers.

## Fixture

As amendment 6. `hipfire.exe` md5 `260f68ddbd4af27aa6ab2c2f4ff6cc91`,
`daemon.exe` md5 `ed5c023b1358610f2ff3c44d33178334`, revision `080b0e581`.

## Parity

Pinned prompt `[48,25,220,16,10,16,28]` against the fork's saved logits:

| path | correlation | RMSE | argmax | top-10 |
|---|---:|---:|---|---|
| per-token `forward_scratch` | 0.99996763 | 0.012999 | 248046 | identical |
| batched, before | 0.04258281 | 4.006916 | 99301 | DIFFER |
| **batched, after** | **0.99996864** | **0.013200** | **248046** | **identical** |

The batched path is now equivalent to the per-token reference on the same
prompt, and that reference is the recorded baseline. The daemon takes the
batched path for prompts up to 32 tokens regardless of `prefill_batched`
(`verify_decouple` overrides it, amendment 9), so this is the path a user's
short prompt actually takes.

## Bug 1: the gate projections read the rotated activation

`batch_chunk_delta_net_attn`'s QKVZA arm handed all four projections
`pbs.x_rot_batch`:

```rust
run_proj_gemm(gpu, &layer.wqkv,   &pbs.x_rot_batch, &pbs.dn_qkv_batch,   n)?;
run_proj_gemm(gpu, &layer.wz,     &pbs.x_rot_batch, &pbs.dn_z_batch,     n)?;
run_proj_gemm(gpu, &layer.w_beta, &pbs.x_rot_batch, &pbs.dn_beta_batch,  n)?;
run_proj_gemm(gpu, &layer.w_alpha,&pbs.x_rot_batch, &pbs.dn_alpha_batch, n)?;
```

Prism's checkpoints mix bases inside one layer: wqkv/wz are PTQ1G128H and carry
the checkpoint's Prism-Hadamard fold, while w_alpha/w_beta stay **BF16** and were
quantized against the plain rmsnorm. The unrotated pair was therefore dotted
against the wrong basis.

This is exactly the defect `a7f40fb56` fixed for the per-token path
(`shared_rotation_disagrees` → `GemvInput::Raw`). The batched path never got the
fix because this model was **rejected** from it until the BF16 arm was admitted
in `c9ebd40f2`.

A mixed layer now emits both buffers and each projection reads the one its dtype
needs; a uniform layer keeps the single fused launch. `mixed_basis` is computed
from the four dtypes, so checkpoints where all four are folded still take the
fused path.

## Bug 2: the fused rotate helpers used the MQ FWHT on Prism weights

`rotate_x_mq_batched_for` has had this branch all along:

```rust
if matches!(next_linear.gpu_dtype, DType::TQ2G128H | DType::PTQ1G128H) {
    return gpu.rotate_x_prism_hadamard(x, x_rot, k, batch_size);
}
```

`fused_rmsnorm_rotate_mq_batched_for` and `fused_silu_mul_rotate_mq_batched_for`
did **not**. Every activation that reached a Prism weight through them was
rotated with the MagnumQuant FWHT instead of the Prism-Hadamard:

- the FFN's gate/up preamble (the helper takes `w_gate`, which is PTQ1G128H);
- the `w_down` input (the helper takes `w_down`, also PTQ1G128H).

Both now take the Prism branch: rmsnorm/silu_mul into the destination, then
`rotate_x_prism_hadamard` in place. In place is safe because the rotation stages
each block through LDS before writing any output.

## How it was localized

`dump_hidden_localize` (an existing helper, gated behind `HIPFIRE_DUMP_HIDDEN`)
diffing `.pertoken` against `.batched` per layer, with `HIPFIRE_FORWARD_LOWERED=0`
so the per-token path takes the hand arms that carry the GDN dumps.

| checkpoint | per-token vs batched |
|---|---|
| layer-0 embedding | **bit-identical** |
| GDN input `v` | 6137 / 6144 differ |
| GDN input `alpha`, `beta` | 48 / 48 differ |
| GDN output | 0 / 6144 differ (after bug 1) |
| gated-norm output | 0 / 6144 differ (after bug 1) |
| state after `wo` + residual | 3.5e-5 relative |
| **layer-0 output** | **5067 / 5120, 56% relative** |

The post-`wo` row is what split the two bugs: everything up to the attention
output was already at accumulation noise, and the divergence was created between
there and the layer output, which is the FFN.

Ruled out along the way, each with a measurement: `gemm_bf16_xf32_batched`
(1.007e-5 against a CPU oracle at the real 48x5120x7 shape),
`gemm_ptq1g128_wmma` (7.6e-6 / 4.2e-5 against the scalar),
`rotate_x_prism_hadamard` (5.245e-6 at batch 1/2/4/8/64/256, width 5120),
the Q/K repeat-interleave convention (both kernels use `kh*ratio + r`), and
`gated_delta_net_q8_batch_seq`'s launch geometry.

## Rejected: making the batched GDN use the decode's kernel

`gated_delta_net_q8_batch_seq` and `gated_delta_net_q8_compact` are one source
compiled with different `HIPFIRE_GDN_MIN_BLOCKS` / `HIPFIRE_GDN_QK_HEAD_DIV`, so
their register allocation and FMA fusion differ. Measured with a channel test
(`test_gdn_batch_vs_pertoken`): token 0 bit-identical from a zero state, then
3457 / 5036 / 5566 of 6144 elements apart at tokens 1 / 2 / 3, because the Q8
state requant turns a 1-ULP difference into a full quantization step.

Routing the batched prefill through N per-token `_compact` launches closed that
gap (GDN output 140/6144 → 0/6144) but **made end-to-end parity slightly worse**
(0.99996734 / 0.013633 against 0.99996864 / 0.013200 without it), so it is not
taken. The GDN divergence is real but sits below the noise floor the parity gate
measures.

## Performance

| | before the fixes | after |
|---|---:|---:|
| prefill (canonical bench) | 234.3 tok/s | **234.6 tok/s** |
| TTFT | 100.8 ms | 102.3 ms |
| decode | 36.1 tok/s | 34.5 tok/s |
| prefill 64 / 256 / 1024 | 371.8 / 372.0 / 373.5 | **367.2 / 400.4 / 398.1** |

The fixes cost nothing measurable: the extra rmsnorm launch in a mixed layer and
the un-fused rotate are offset by no longer rotating activations that are then
discarded.

## Disposition

The batched prefill meets the parity gate. Amendments 3, 4, 6 and 8's throughput
rows are now measurements of an accepted path: **prefill 34 → 234.6 tok/s,
TTFT 706 → 102.3 ms**, flat across 64/256/1024, with the fork parity held.

The two ceilings of amendment 6 stand unchanged and are still the reason the
3000 and 80 tok/s targets are not reachable.

These rows are measurement, not admission.

# Amendment 3 — Ternary-Bonsai-2-27B PTQ1_0 gfx1100: prefill is ALU-bound at the vector rate

**Date:** 2026-09-22
**Lifecycle:** `historical`
**Amends:** [`2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md`](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite.md) and amendments [1](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-1.md), [2](2026-09-22-ternary-bonsai-ptq1-gfx1100-decode-rewrite-amendment-2.md), all unchanged.
**Disposition:** **measured amendment.** Records the batched-prefill unblock (a real 1.47x) and corrects the attribution of the prefill bottleneck. The parent records are all about decode; this is the first prefill measurement in the series.

## Scope

Prefill on this model was never measured in the parent records. It was, and is, the larger defect: 34 tok/s and **flat in prompt length**, against a decode of 33.7 tok/s. Prefill and decode at the same rate means prefill is not prefill.

## Fixture

Same host, device, HIP 7.2, and model as the parent records
(`C:/tmp/bonsai-2-27b.ptq1`, md5 `8abae179f984e2461cbf0fece6a8606f`).
`hipfire.exe` md5 `b8c0b6200b3e2cf072a81a6720b1ce75`, `daemon.exe` md5
`82d2dcfd5bfeeab589ef7c5abdc0a39e`, revision `c9ebd40f2`.

Prefill scaling measured with `bench_qwen35_mq4 --prefill N --warmup 1 --gen 4`;
decode and TTFT with the canonical
`hipfire bench --spec off --backend noslots --workload stateless --max-tokens 128`.

## Root cause: one BF16 tensor rejected the whole model

`HIPFIRE_DEBUG_BATCH=1` (a pre-existing debug switch) reported:

```
[hipfire::batch_eligible] result=false arch=gfx1100 n=64 n>=2=true
  force_fallback=false has_dn=true moe_router_logits_present=true
  all_layers_ok=false
  first_reject=Some("DeltaNet w_beta dtype BF16 not batchable on gfx1100")
```

The gate discards the per-layer error, so the debug line said `all_layers_ok=false`
without saying why. It now reports `first_reject` (commit `e1b06dc4a`).

Instrumenting the loaded dtypes gave the real layer shape:

| tensor | dtype | shape |
|---|---|---|
| wqkv | PTQ1G128H | 10240x5120 |
| wz | PTQ1G128H | 6144x5120 |
| **w_beta** | **BF16** | **48x5120** |
| **w_alpha** | **BF16** | **48x5120** |
| wo | PTQ1G128H | 5120x6144 |
| w_gate / w_up | PTQ1G128H | 17408x5120 |
| w_down | PTQ1G128H | 5120x17408 |

The ternary format was never the problem: `PTQ1G128H` is in `is_batchable_la`'s
`always_ok` set. **Two unquantized 48x5120 gate projections, 491 KB each, were
rejecting the whole 27B model from batched prefill.**

They cannot simply be admitted: `plain_gemm_key_for`'s `_` arm is
`GemmQ8_0BatchedChunked`, so a BF16 tensor routed through the generic path would
be read by a Q8_0 kernel — noise at full speed, no HIP error.

## Accepted — a batched BF16x F32 GEMM

New `kernels/src/gemm_bf16_xf32_batched.hip`: `Y[N x M] = X[N x K] @ W[M x K]^T`,
BF16 weights widened losslessly (bf16 is the top 16 bits of an f32), one row per
block, 8-column tile, the weight load amortised over the tile. Dispatched from
the matcher by `run_proj_gemm`, which sends BF16 straight to that kernel and
leaves every other dtype on the registry-routed path unchanged. `is_batchable_la`
admits `DType::BF16`; `plain_gemm_key_for` gains an explicit `DType::F32 =>
GemmF32Batched` arm so F32 can never fall into the Q8 catch-all.

| prompt | prefill before | prefill after |
|---:|---:|---:|
| 64 | 34.2 | 20.0 (JIT-dominated) |
| 256 | 34.1 | **51.5** |
| 1024 | 33.7 | **51.0** |

Canonical bench: prefill 34 -> **49.8 tok/s**, TTFT 706 -> **482 ms**, decode
**33.9** (unchanged).

### Decode parity is untouched, bit-for-bit

The rejected alternative was widening those two weights to F32 at load, which
also unblocks the gate and is simpler. It was rejected on measurement: it moves
the per-token GEMV onto a different accumulation order, and on the pinned prompt
`[48,25,220,16,10,16,28]` it degrades the llama.cpp parity metric.

| variant | correlation | RMSE | argmax | top-10 |
|---|---:|---:|---|---|
| accepted baseline | 0.99996763 | 0.012999 | 248046 | identical |
| **BF16 gate + batched kernel** | **0.99996763** | **0.012999** | **248046** | **identical** |
| F32 gate widening (rejected) | 0.99996453 | 0.013725 | 248046 | identical |

The BF16-gate build is additionally **bit-identical** to the previously
committed kernel on that prompt. The F32 variant is not, and RMSE moves +5.6%.

## The prefill bottleneck is ALU, not traffic

`gemm_ptq1g128_prefill` was confirmed live with a temporary call counter: 400
calls at N=256, matching the decode path's 401 calls, and zero per-token GEMV
calls during prefill.

Its geometry (one row per block, 8-column tile) reads the activation once per
output row: 557k blocks x 46 KB = 25.6 GB of activation per layer against 623 MB
of weights, 41x. That looked like the bottleneck and is not. The measurement
that settles it:

| path | MACs per token | measured rate |
|---|---:|---:|
| decode GEMV | 27e9 | 1.26e12 MAC/s |
| **batched prefill GEMM** | 27e9 | **1.38e12 MAC/s** |

Prefill runs at the same rate as decode. Tiling reduces *loads*, and the loads
were never the limit: the per-(weight, token) multiply is 27e9 x N either way.
**Tiling this kernel would buy nothing.** The bound is the vector integer
multiply rate, which amendment 2 already attributed to `V_MUL_LO_U32` at quarter
rate.

## The matrix unit is the only lever, and it is unused

The RX 7900 GRE is Navi 31 XL: 5120 lanes at 2245 MHz, 160 matrix cores, and the
spec sheet lists matrix support for **INT4, INT8, FP16, BF16** (INT1/INT2, FP8,
MXFP4/6, FP32, TF32 are not supported). Throughput: INT8 91.96 TOPS, FP16/BF16
91.96 TFLOPS, INT4 183.9 TOPS.

The tree uses exactly one of those: `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`
(and its gfx12 sibling), for flash attention and the F16 GEMM family. Nothing
INT8 or INT4 anywhere.

Feasibility was checked by compiling both builtins for gfx1100 with the project's
own flags. **Both compile.** The wave32 fragments are: A/B 4x int32 (16 int8 per
lane) for iu8, 2x int32 (16 int4 per lane) for iu4, C 8x int32 for both. Note
this is a different instruction family from `__builtin_amdgcn_sdot4`, which
hipcc still rejects on gfx1100 (`needs target feature dot1-insts`).

Ceiling for pp512, from 91.96 TOPS / 2 = 45.98e12 MAC/s and 27e9 MACs per token:

| format | MACs/s | pp512 at 100% | at 70% |
|---|---:|---:|---:|
| INT8 / FP16 / BF16 | 45.98e12 | **1702 tok/s** | 1191 |
| INT4 | 91.95e12 | 3405 tok/s | 2384 |

Against the measured 51 tok/s this is a **33x** headroom, and it is the only
headroom that exists.

### INT4 cannot be used at int8 activation precision

`v_wmma_i32_16x16x16_iu4` takes **both** operands at 4 bits. The reference
(llama.cpp's `vec_dot_ptq1_0_q8_1`) keeps the trits exact and quantizes the
**activation** to int8 with 32-element sub-blocks, so a 4-bit activation is a
quantization change, not a kernel change: 16 levels against 256 is
`254/14 = 18.1x` the RMS error. That is the same regime as the `alpha`/`beta`
basis bug this series started by fixing (correlation 0.9907).

Every exact recovery route was evaluated and every one lands back at 1702:

| scheme | result |
|---|---|
| nibble split `x = 16h + l`, two iu4 MMAs | exact, 2x MACs at 2x rate = 1702 tok/s |
| sign-plane split `w = w+ - w-`, two iu8 MMAs, 0/1 masks | exact, 2x MACs = 1702 tok/s |
| smaller int4 activation sub-blocks | does not help: the limit is the 4-bit mantissa, not the scale |
| stochastic rounding / bias correction | changes variance, not the mean; parity is a value gate |
| iterative refinement of the largest terms | the correction needs the same precision, costs the same |

**The INT4 2x is structurally cancelled by the exactness requirement.** Packing
several sub-byte values into one INT8 lane does not help either: `iu8` multiplies
the whole byte, so sub-byte fields produce cross terms that cannot be separated
out of a summed accumulator. `iu4`'s 8 independent nibbles per int32 *is* the
packing benefit, and it is already counted above.

3000 tok/s would need 8.1e13 MACs/s, **176% of the INT8 peak**. The only ways
there reduce MACs rather than raising the rate (structured sparsity, which RDNA3
WMMA has no mode for; or fewer active parameters), and neither is a kernel
change.

## Disposition

The BF16 batched GEMM and the gate admission are accepted: prefill 34 -> 49.8
tok/s and TTFT 706 -> 482 ms, with decode bit-identical and parity unchanged.

The next lever is a tiled FP16 WMMA PTQ1 prefill GEMM, and it is exact: a trit in
`{-1,0,1}` times an int8 activation in `[-127,127]` is representable in fp16
without rounding, and an fp32 accumulator over K=5120 stays under 650k, far below
the 2^24 exact-integer limit. So `wmma_f32_16x16x16_f16_w32` with an fp16 tile
loader over PTQ1 trits computes the integer dot exactly, on a kernel family the
tree already ships and has already validated on gfx11.

The 3000 tok/s prefill target is **not reachable with parity**: the proven ceiling
is 1702 tok/s at 100% matrix efficiency. That is 33x the current number and is
what the work should be measured against.

These rows are measurement, not admission.

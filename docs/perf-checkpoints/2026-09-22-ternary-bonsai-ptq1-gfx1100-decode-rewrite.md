# Ternary-Bonsai-2-27B PTQ1_0 — gfx1100 decode kernel rewrite and the bandwidth wall

**Date:** 2026-09-22
**Lifecycle:** `historical`
**Disposition:** **measured optimization evidence** for the exact fixture below. This is not a performance floor, an admission gate, a current-default baseline, or a transferable result. Two of the three attempted levers were rejected with data and are retained as rejected. Newest file != current baseline.

## Scope

Ternary-Bonsai-2-27B (PrismML, PTQ1_0 / 1.75 bpw ternary) was loadable and numerically 1:1 with the reference fork before this work, but decoded at 10.7 tok/s. This record measures where that time went, what three candidate levers actually did, and how far the format can go on this card at all.

The headline result is a negative one about the target rather than a positive one about the kernel: **the throughput ceiling for this model on this device is set by DRAM bandwidth, and the measured ceiling is below 100 tok/s.** The 2.9x kernel win is real and independent of that.

## Fixture

| field | value |
|---|---|
| Host | Windows 11 26200, rustc 1.97 |
| Device | HIP device `0`, Radeon RX 7900 GRE, `gfx1100`, 80 CU, 576 GB/s, WDDM |
| HIP / ROCm | HIP 7.2 |
| Runtime revision | `3ec62c3c4` |
| `hipfire.exe` md5 | `2f541a477375570f2f8a51954af17594` |
| `daemon.exe` md5 | `da54f054c86c0cda25dc5c2b722211eb` |
| Model artifact | `C:/tmp/bonsai-2-27b.ptq1`, md5 `8abae179f984e2461cbf0fece6a8606f`, 5.54 GiB |
| Backend / workload | `noslots` / `stateless` |
| KV | Q8, `max_seq` registry default |
| Sampling | 3 warmup, 5 measured runs, `--max-tokens 128`, `--spec off` |

Command shape for every throughput row:

```bash
hipfire bench C:/tmp/bonsai-2-27b.ptq1 \
  --runs 5 --warmups 3 --max-tokens 128 \
  --spec off --backend noslots --workload stateless --json
```

The reported prompt is the harness's built-in one; it is not a committed
fixture, so these rows compare against each other and nothing else.

## Where decode time went

`bench_qwen35_mq4 --gen 20` with `HIPFIRE_AR_GRAPH=0` and
`HIPFIRE_PROFILE_DECODE=1` (needed because the default AR hipGraph capture
replays as one opaque launch and hides every inner kernel from the internal
profiler — 20 launches reported for a 23760-launch loop).

Before any change, at 10.7 tok/s:

| kernel | calls | total | per call | share | eff. GiB/s |
|---|---:|---:|---:|---:|---:|
| `gemv_ptq1g128` | 8020 | 1844.9 ms | 230 µs | 79.9% | 56.7 |
| `rotate_x_prism_hadamard` | 5160 | 174.6 ms | 34 µs | 7.6% | 2.8 |
| `rmsnorm_batched` | 2560 | 99.5 ms | 39 µs | 4.3% | 1.5 |
| `add_inplace_f32` | 2560 | 62.0 ms | 24 µs | 2.7% | 2.4 |
| `gated_delta_net_q8_compact3_b2` | 960 | 47.2 ms | 49 µs | 2.0% | 32.6 |

401 `gemv_ptq1g128` launches per token, at 230 µs each. The kernel was
ALU-bound, not bandwidth-bound: 56.7 GiB/s is ~11% of the card.

## Levers, in the order they were tried

### Rejected — activation-requantisation cache (no effect)

`ScratchState::ensure_q8_1_mmq_x` had `let must_convert = true;` hardcoded, so
every PTQ1 GEMV relaunched `quantize_q8_1_mmq_ds4` on an activation that
`wqkv`/`wz` and `gate`/`up` already shared. Adding the standard
pointer+shape cache (plus a `prism_hadamard` write-invalidation hook, since the
rotated output is a stable pointer rewritten every layer) removed one launch
per shared pair: **10.9 → 10.9 tok/s.** Zero. The requantise was not measurable
next to the 230 µs GEMV. Retained, because it is strictly less work and the
invalidation hook closes a latent staleness bug, but it is not a perf lever.

### Rejected — 256-entry `__constant__` decode table (regression)

`ptq1_trit` decoded each trit with an iterative `v = (v*3) & 0xFF` chain. A
`__constant__ signed char PTQ1_LUT[256][5]` table verified equal to the
iterative form for all 256×5 cases replaced the chain with one constant load
and passed the channel test at the same tolerance. **10.9 → 9.3 tok/s**,
a 15% regression. The table index is the packed weight byte, so it is
data-divergent per lane, and divergent `__constant__` access serialises on the
constant cache. The lockstep ALU chain it replaced was cheaper. Reverted.

### Accepted — one whole block per lane (10.9 → 28.4 tok/s)

The original mapping gave each lane four K elements and selected the source
byte with `if (e < 80) … else if (e < 120) … else …` plus a conditional
`for (i < n)` unroll, evaluated per element. Giving each lane one complete
128-weight block instead makes the traversal straight-line and unrollable:
`qs[0..15]` at `t*16+m`, then `qs[16..23]` at `80 + t*8 + m`, then `qh[0..1]`
at `120 + t*2 + h`. Lane `L` strides `g += 32` across K. Channel test:
`max |gpu-cpu| = 1.335e-5`, unchanged from the value the kernel passed before.

### Accepted — two rows per workgroup (28.4 → 31.3 tok/s)

A PTQ1 weight block is 28 B while its Q8_1 activation block is 144 B: **the
activation is 5.1x the weight.** That ratio is the inverse of HFQ4/MQ4
(136 B weight vs 144 B activation), which is exactly why the repo's standing
"multirow does not pay on gfx1100" note is correct there and wrong here. Tiling
rows per workgroup reuses the 144 B block that dominates the traffic. Four
rows measured *worse* than two (28.4 tok/s), so the win is not monotone in the
tile:

| row tile | decode tok/s (median of 5) | measured array |
|---:|---:|---|
| 1 | 28.4 | — |
| **2** | **31.3** | 31.4/31.3/31.2/31.2/31.0 |
| 4 | 28.4 | 28.5/28.4/28.3/28.2/27.9 |

`__launch_bounds__` minimum blocks per CU moved from 8 to 24 with no effect on
the large-M rate (219.0 → 219.3 GiB/s), so occupancy was not the binding
constraint at this shape.

Cumulative: **10.7 → 31.3 tok/s decode, 2.9x**, wall 9.2 → 26.3 tok/s, TTFT
2230 → 765 ms.

### Rejected — `sdot4` packed decode (does not compile)

Porting the reference fork's `ggml_cuda_mmq_decode_ptq1_0_qs4` /
`vec_dot_ptq1_0_q8_1` form — `__byte_perm` four bytes into 16-bit lanes,
multiply by three without carry, then two `v_dot4_i32_i8` per four decoded
trits — fails on gfx1100: `'__builtin_amdgcn_sdot4' needs target feature
dot1-insts`. Confirmed by the existing note in `kernels.rs` on
`FUSED_GATE_UP_HFQ4G256_WAVE64_DP4A_SRC`. gfx1100 has no integer dot-product
instruction, so the ternary multiply cannot be moved off the VALU at all.

## The bandwidth wall

Two measurements bound the format on this card, both at `M=1048576, K=5120`
(1120 MiB of weights per call, 30 reps):

| kernel variant | weight-read rate |
|---|---:|
| with trit decode | 219 GiB/s |
| decode removed, loads kept | **343 GiB/s** |

343 GiB/s is the ceiling of this access pattern on this card (536 GiB/s
nameplate peak); the decode costs a 1.57x factor on top of it. The model is
5.54 GiB, so:

| target | required weight traffic | vs the card's 576 GB/s |
|---|---:|---|
| 100 tok/s | 554 GB/s | 96% of peak — not reachable in practice |
| 1000 tok/s | 5540 GB/s | **9.6x the entire DRAM bandwidth** |
| decode-free ceiling at 343 GiB/s | 343 GiB/s | 62 tok/s |

Single-stream decode reads every weight once per token, so no kernel change
moves the 1000 tok/s case: it is off by an order of magnitude, not by
efficiency. 100 tok/s is also above what the measured pattern ceiling allows
(62 tok/s) and above the nameplate roofline (97 tok/s) — the original 100+
target for this model on this device was unreachable by construction.

## Batched aggregate is closed for this format

`hipfire bench --concurrency 1,2,4,8` (2 runs, `--max-tokens 64`):

| k | aggregate tok/s | per-stream |
|---:|---:|---:|
| 1 | 7.95 | 7.95 |
| 2 | 18.13 | 9.07 |
| 4 | 20.54 | 5.13 |
| 8 | 21.53 | 2.69 |

The daemon refuses the batch backend outright:

```
[daemon] continuous batch requested but weight formats unsupported
         (embd=PTQ1G128H lm_head=PTQ1G128H) — fallback to sequential
```

So every row above is the sequential fallback, and aggregate throughput
degrades with concurrency rather than scaling. Batching is the only route to
high aggregate numbers on a bandwidth-bound decode (the weight read amortises
across the batch), and it is unavailable for PTQ1G128H because the batched
embedding gather and batched `lm_head` GEMM do not exist. The matmuls are not
the gap — `gemm_ptq1g128_prefill` already takes N.

## Disposition

The kernel rewrite is accepted: 2.9x decode on a format that was already
numerically 1:1, with parity re-measured after the change (final-position
logits vs `llama-debug` from the reference fork on prompt
`[48,25,220,16,10,16,28]`: correlation 0.99996763, RMSE 0.012999, identical
argmax 248046 and identical top-10 ordering; slightly closer to the reference
than the pre-optimization kernel's 0.99996598 / 0.013203).

The target revision is rejected. 100 tok/s is above both the nameplate roofline
(97 tok/s) and the measured pattern ceiling (62 tok/s) for a 5.54 GiB model on
a 576 GB/s device, and 1000 tok/s is 9.6x the device's DRAM bandwidth. The
remaining work with measured headroom is (a) the 30% of decode outside the
GEMV — `rotate_x_prism_hadamard`, `rmsnorm_batched`, `add_inplace_f32` are three
separate passes over the same activation where the MQ path already has a fused
`rmsnorm_rotate` — and (b) batched decode for PTQ1G128H, without which
aggregate throughput cannot exceed single-stream.

These rows are measurement, not admission.


# Qwen3.8-27B MQ3V2 — gfx1100 Windows tier selection against the WDDM VRAM ceiling

**Date:** 2026-09-07  
**Lifecycle:** `historical`  
**Disposition:** **measured host-selection evidence** for the exact fixture below. This is not a performance floor, an admission gate, a current-default baseline, or a transferable result. The numbers are lower than the gfx1100 Linux product rows and are retained as measured, not reconciled. Newest file != current baseline.

## Scope

First Qwen3.8-27B run on a Windows host, taken to answer which product tier a 16 GB card can serve when the desktop already holds part of the dedicated pool. WDDM overcommits video memory, so a tier that does not fit still loads and decodes: selecting by nameplate capacity costs 5x silently instead of failing. The run measures where the residency boundary sits and what crossing it costs, then checks whether speculation recovers any of it.

Two results contradicted the pre-run expectation and are kept: the cost of overflow is not proportional to the overflow, and DFlash is neutral on both prompts rather than a win on code.

## Fixture

| field | value |
|---|---|
| Host | Windows 11 26200, MSVC, rustc 1.97 |
| Device | HIP device `0`, Radeon RX 7900 GRE, `gfx1100`, 80 CU, 576 GB/s, WDDM driver model |
| HIP / ROCm | HIP 7.16, `rocm-sdk` 10.1.0a wheels, Adrenalin 32.0.31041.1004 |
| Runtime revision | `af0cbcfa94fb` |
| `hipfire.exe` md5 | `02acb5f2a0778b55d8334bcdef4ac6a3` |
| `daemon.exe` md5 | `10580db196a66fd9ce12a6c586af9d9e` |
| Backend / workload | `noslots` / `stateless` |
| KV | Q8 / VMM, `max_seq` at the registry default |
| Sampling | 1 warmup, 3 measured runs, `--max-tokens 128` |
| Prose prompt | `benchmarks/prompts/merge_sort_thinking_off.txt`, md5 `253c7ac50857fe6d0e10fb0d2c5e35c0` |
| Code prompt | `benchmarks/prompts/humaneval_3_below_zero.txt`, md5 `37c5aad9f9efe93b5c47f27256bdf149` |

Both prompt md5s are the LF values. A Windows checkout served CRLF until
`.gitattributes` pinned these fixtures, and the CRLF spellings hashed to
`46c8d9674dcc8a8f18638bbd2d42c9ae` and `06eb2ebe997c7980cccf0244a85dc2a9`. A
paired A/B on `mq3` measured no throughput difference between the two
spellings (29.1 vs 29.0 and 27.2 vs 27.2 tok/s median), so the line endings
affect fixture identity rather than token shape. Every row below was measured
after renormalization.

Command shape for every row:

```bash
hipfire bench MODEL "$(cat PROMPT)" \
  --spec {off,ngram,dflash} --runs 3 --warmups 1 --max-tokens 128 \
  --backend noslots --workload stateless --json
```

## Artifact identity

| Product | measured artifact | artifact md5 | weights |
|---|---|---|---:|
| mq3-xt | `qwen3.8-27b.mq3-xt` | `80bb9198e6a565fc006b2ae1b7c89eca` | 11.78 GB |
| mq3 | `qwen3.8-27b.mq3` | `ad20b3d3a9a7254e7a9c596fc97b411a` | 12.62 GB |
| mq3-pro | `qwen3.8-27b.mq3-pro` | `b171dd618eecf1fe6aed2c1dc5eef4dc` | 13.18 GB |
| mq4-xt | `qwen3.8-27b.mq4-xt` | `e45d15bfe0c9a87132697101d17cbed6` | 14.98 GB |

DFlash draft for the speculation rows: `qwen38-27b-dflash-mq4.hfq`, sha256
`d0a74a232a0e2166d889f823e91e0fbf778d21dd9668d7de055cdecb065401bc`, 5 layers,
block 8, windowed at W=2048 from draft metadata.

## Residency and the overflow cost

Dedicated and shared bytes read from the WDDM counters
`\GPU Process Memory(pid_<daemon>*)\{Dedicated,Shared} Usage` while the model
served a request. `hipMemGetInfo` is useless for this on Windows: it reported
17.01 of 17.16 GB free with `mq4-xt` resident, because it answers for the
calling process rather than the device.

| Product | dedicated | shared | prose decode tok/s | prose prefill tok/s |
|---|---:|---:|---:|---:|
| mq3-xt | 12.75 GB | 1.76 GB | **29.50** | **306.50** |
| mq3 | 12.72 GB | 2.63 GB | **29.00** | **301.90** |
| mq3-pro | 12.78 GB | 3.09 GB | **5.80** | **145.30** |
| mq4-xt | 12.85 GB | 4.62 GB | **4.50** | **28.70** |

Measured decode arrays: mq3-xt 28.0/29.5/29.6, mq3 28.1/29.0/29.0,
mq3-pro 5.7/5.8/5.8, mq4-xt 4.4/4.5/4.5.

The desktop session holds the rest of the pool: `dwm` alone measured 1.50 GB
and the logged-in session about 3.8 GB. Dedicated usage saturates near
12.8 GB whatever the tier, so every row above overflows and the difference is
how far. The cost of overflowing is not proportional to it: 2.63 GB of shared
memory costs 1.7% of decode, and the next half gigabyte costs 80%. Prefill
degrades harder than decode, 10.7x against 6.6x on `mq4-xt`.

What sets the cliff between 2.63 GB and 3.09 GB was not isolated here, so no
threshold is claimed. Two facts bound it. The dedicated ceiling moves with
desktop state, measured between 12.19 GB and 12.85 GB across the session, so a
row near the boundary is not reproducible without also pinning the desktop.
And committed pages are not the same as touched pages: `memory.max_seq 8192`
cut `mq3` from 2.63 GB shared to 0.97 GB while decode moved 28.9 to 29.4
tok/s, inside run-to-run spread.

## Speculation by prompt genre

| Product | prompt | spec | decode tok/s | measured array |
|---|---|---|---:|---|
| mq3-xt | code | off | **27.80** | 27.2/27.9/28.0 |
| mq3-xt | code | ngram | **35.10** | 32.6/35.1/42.5 |
| mq3-xt | code | dflash | **27.90** | 27.3/27.9/28.0 |
| mq3 | code | off | **27.00** | 26.7/27.0/27.2 |
| mq3 | code | ngram | **35.40** | 32.7/35.4/40.7 |
| mq3 | code | dflash | **27.10** | 26.4/27.1/27.2 |
| mq3 | prose | off | **29.00** | 28.1/29.0/29.0 |
| mq3 | prose | ngram | **21.40** | 19.3/21.4/39.0 |
| mq3 | prose | auto | **28.10** | 26.3/28.1/29.0 |

n-gram is worth +26% to +31% on the code prompt, where the completion copies
heavily from the prompt, and costs 26% on the prose prompt, where it does not.
Both arrays are wide, and the prose array spans 19.3 to 39.0, which is the
signature of acceptance-driven speculation rather than a tight statistical
artifact. A single-run diagnostic on the prose prompt logged the drafter at
τ=0.02, so most windows there are wasted work.

`--spec auto` measured 28.1 tok/s on prose, i.e. it correctly avoided the
n-gram loss, and 27.7 on code, i.e. it also declined the n-gram win. On this
host the +31% is only reachable by naming `--spec ngram` per genre.

DFlash loaded on every speculation row (draft logged at load) and measured
within noise of AR on both genres, so the draft is not the lever on this host.
MTP is unavailable in these artifacts: the loader refuses with "MTP head
required (mtp=on) but not found: no bundled trailer or .mtp sidecar found".

Output identity was checked rather than assumed. On the code prompt at
temperature 0 with a 1400-token budget, `off` and `ngram` produced
byte-identical text (md5 `ed8ef838a620…`, a correct `below_zero`
implementation), and on the prose prompt `off`, `ngram`, and `dflash` all
agreed byte for byte.

## Disposition

For this host the largest tier that keeps decode is `mq3` at 12.62 GB of
weights, and `mq3-xt` at 11.78 GB is 1.7% faster with a smaller overflow. The
two are within run-to-run spread of each other, so the choice between them is
a quality decision rather than a throughput one. Nothing above 12.62 GB is
usable here: the next tier up loses 80% of decode.

These rows are measurement, not admission. They do not authorize a product
claim, a default, or a comparison against the Linux gfx1100 rows, which used a
different HIP stack, a different card in the same family, and a host with no
desktop competing for the dedicated pool.

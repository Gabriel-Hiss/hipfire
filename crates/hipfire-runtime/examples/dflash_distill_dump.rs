// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! dflash_distill_dump: target-side data for fine-tuning a DFlash drafter.
//!
//! Two modes over a Qwen3.5-family target (`.hfq` / `.ptq1`):
//!
//!   tables  --target T --out DIR
//!       DIR/embed.f16    [vocab x dim]  the input embedding rows the drafter's
//!                        block tokens use (the target's own lookup, Prism
//!                        rotation undone for ternary heads)
//!       DIR/lmhead_t.f16 [dim x vocab]  the effective lm_head applied to a
//!                        drafter hidden state, transposed: row i holds the
//!                        logits of basis vector e_i (rotation + quantized
//!                        GEMM exactly as the drafter's lm_head runs; a
//!                        rotated basis vector quantizes to Q8_1 exactly)
//!
//!   hidden  --target T --tokens LOG.jsonl --layers 1,10,... --out DIR [--max-len N]
//!       For each `{"prompt":[ids],"output":[ids]}` line (HIPFIRE_SPEC_TOKEN_LOG
//!       output), prefill prompt ++ output from position 0 and write the
//!       captured hidden states of the listed target layers, in that order, as
//!       f16 `DIR/h_<line>.f16` [len x layers x dim], plus one
//!       `{"line","len","prompt_len","file"}` row per sample in DIR/index.jsonl.
//!       Existing files are skipped, so an interrupted run resumes.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::speculative::{
        seed_target_hidden_from_prompt_abortable, HiddenStateRingBuffer, ModelSlot, ModelSlotConfig,
    };
    use rdna_compute::DType;
    use std::io::{BufRead, Write};
    use std::path::{Path, PathBuf};

    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
    };
    let mode = args.get(1).cloned().unwrap_or_default();
    let target_path = arg("--target").expect("--target");
    let out_dir = PathBuf::from(arg("--out").expect("--out"));
    std::fs::create_dir_all(&out_dir).expect("create --out");
    let max_len: usize = arg("--max-len").map(|v| v.parse().unwrap()).unwrap_or(6144);

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let mut slot_cfg = ModelSlotConfig::default();
    slot_cfg.max_seq = max_len + 64;
    slot_cfg.kv_mode = hipfire_arch_qwen35::speculative::KvMode::Q8;
    let mut target =
        ModelSlot::load(&mut gpu, Path::new(&target_path), "target", slot_cfg).expect("load target");
    let dim = target.config.dim;
    let vocab = target.config.vocab_size;
    eprintln!("target: dim={dim} vocab={vocab} layers={}", target.config.n_layers);

    let write_f16 = |path: &Path, data: &[f32]| {
        let mut bytes = Vec::with_capacity(data.len() * 2);
        for &v in data {
            bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
        std::fs::write(path, bytes).expect("write f16");
    };

    match mode.as_str() {
        "tables" => {
            // Embedding rows, 1024 tokens per batched lookup.
            const CH: usize = 1024;
            let ids = gpu.alloc_tensor(&[CH], DType::F32).unwrap();
            let rows = gpu.alloc_tensor(&[CH * dim], DType::F32).unwrap();
            let mut embed = vec![0f32; vocab * dim];
            let mut t0 = 0;
            while t0 < vocab {
                let n = CH.min(vocab - t0);
                let tok: Vec<i32> = (t0..t0 + n).map(|t| t as i32).collect();
                let tok_bytes: Vec<u8> = tok.iter().flat_map(|v| v.to_le_bytes()).collect();
                gpu.hip.memcpy_htod(&ids.buf, &tok_bytes).unwrap();
                gpu.embedding_lookup_ptq1g128_prism_batched(&target.weights.token_embd, &rows, &ids, n, dim)
                    .expect("embedding lookup");
                let host = gpu.download_f32(&rows).unwrap();
                embed[t0 * dim..(t0 + n) * dim].copy_from_slice(&host[..n * dim]);
                t0 += n;
            }
            write_f16(&out_dir.join("embed.f16"), &embed);
            drop(embed);
            eprintln!("embed.f16 written");

            // Effective lm_head: logits of the identity, 16 basis vectors per GEMM.
            let w_out = &target.weights.output;
            assert_eq!(w_out.gpu_dtype, DType::PTQ1G128H, "tables mode expects a PTQ1 lm_head");
            const B: usize = 16;
            let x = gpu.alloc_tensor(&[B * dim], DType::F32).unwrap();
            let x_rot = gpu.alloc_tensor(&[B * dim], DType::F32).unwrap();
            let logits = gpu.alloc_tensor(&[B * vocab], DType::F32).unwrap();
            let mut lm_t = vec![0f32; dim * vocab];
            let mut i0 = 0;
            while i0 < dim {
                let mut basis = vec![0f32; B * dim];
                for r in 0..B {
                    basis[r * dim + i0 + r] = 1.0;
                }
                let basis_bytes: Vec<u8> = basis.iter().flat_map(|v| v.to_le_bytes()).collect();
                gpu.hip.memcpy_htod(&x.buf, &basis_bytes).unwrap();
                hipfire_runtime::llama::rotate_x_mq_batched_for(&mut gpu, w_out, &x, &x_rot, dim, B).unwrap();
                gpu.gemm_ptq1g128_wmma(&w_out.buf, &x_rot, &logits, w_out.m, w_out.k, B).unwrap();
                let host = gpu.download_f32(&logits).unwrap();
                lm_t[i0 * vocab..(i0 + B) * vocab].copy_from_slice(&host[..B * vocab]);
                i0 += B;
            }
            write_f16(&out_dir.join("lmhead_t.f16"), &lm_t);
            std::fs::write(
                out_dir.join("tables.json"),
                serde_json::json!({"dim": dim, "vocab": vocab, "embed": "embed.f16", "lmhead_t": "lmhead_t.f16"})
                    .to_string(),
            )
            .unwrap();
            eprintln!("lmhead_t.f16 written");
        }
        "hidden" => {
            let layers: Vec<usize> = arg("--layers")
                .expect("--layers")
                .split(',')
                .map(|s| s.trim().parse().unwrap())
                .collect();
            let tokens_path = arg("--tokens").expect("--tokens");
            let mut hidden_rb = HiddenStateRingBuffer::new_for_layers(
                &mut gpu,
                &layers,
                dim,
                max_len + 64,
                hipfire_arch_qwen35::qwen35::PREFILL_MAX_BATCH,
            )
            .expect("hidden ring");
            let mut index = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(out_dir.join("index.jsonl"))
                .unwrap();
            let reader = std::io::BufReader::new(std::fs::File::open(&tokens_path).expect("--tokens"));
            let t_start = std::time::Instant::now();
            let mut done_tok = 0usize;
            for (line_no, line) in reader.lines().enumerate() {
                let line = line.unwrap();
                let file = format!("h_{line_no:05}.f16");
                if out_dir.join(&file).exists() {
                    continue;
                }
                let row: serde_json::Value = serde_json::from_str(&line).unwrap();
                let prompt: Vec<u32> =
                    row["prompt"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
                let output: Vec<u32> =
                    row["output"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
                let mut seq = prompt.clone();
                seq.extend_from_slice(&output);
                if seq.len() > max_len {
                    eprintln!("line {line_no}: {} tokens > --max-len {max_len}, truncated", seq.len());
                    seq.truncate(max_len);
                }
                let mut host: Vec<f32> = Vec::new();
                seed_target_hidden_from_prompt_abortable(
                    &mut gpu,
                    &mut target,
                    &mut hidden_rb,
                    &mut host,
                    &seq,
                    &|| false,
                    None,
                    0,
                    0,
                )
                .expect("prefill");
                assert_eq!(host.len(), seq.len() * layers.len() * dim);
                write_f16(&out_dir.join(&file), &host);
                writeln!(
                    index,
                    "{}",
                    serde_json::json!({"line": line_no, "len": seq.len(), "prompt_len": prompt.len().min(seq.len()), "file": file})
                )
                .unwrap();
                done_tok += seq.len();
                if line_no % 20 == 0 {
                    eprintln!(
                        "line {line_no}: {} tokens, {:.0} tok/s overall",
                        seq.len(),
                        done_tok as f64 / t_start.elapsed().as_secs_f64()
                    );
                }
            }
        }
        _ => {
            eprintln!("usage: dflash_distill_dump (tables|hidden) --target T --out DIR [...]");
            std::process::exit(2);
        }
    }
}

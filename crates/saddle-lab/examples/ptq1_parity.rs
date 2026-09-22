// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Throwaway: re-measure Prism parity after the matrix-unit prefill GEMM.
//!
//! Prefills the pinned prompt [48,25,220,16,10,16,28] and compares the
//! last-position logits against the fork's saved dump
//! (`llamacpp-Ternary-Bonsai-2-27B-PTQ1_0.bin`, 248320 f32). Reports the same
//! metrics the baseline was recorded with: Pearson correlation, RMSE, argmax,
//! and top-10 identity.
//!
//! Usage: ptq1_parity <model.hfq> <fork_logits.bin>

use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
use std::path::Path;

const PROMPT: [u32; 7] = [48, 25, 220, 16, 10, 16, 28];

fn top10(logits: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.truncate(10);
    idx
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = &args[1];
    let fork_path = &args[2];

    let mut hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    eprintln!("vocab={} layers={} dim={}", config.vocab_size, config.n_layers, config.dim);

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("GPU: {}", gpu.arch);

    let weights = {
        let mut src = qwen35::HfqSource::new(&mut hfq, &config);
        let layout = qwen35::Layout::single(config.n_layers);
        qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
    }
    .expect("load weights");

    let kv_seq = 512usize;
    let mut kv_cache =
        KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq)
            .unwrap();
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 128, kv_seq).unwrap();

    let prompt_tokens: Vec<u32> = PROMPT.to_vec();
    if std::env::var("PARITY_SWEEP").is_ok() {
        // Does the batched path diverge from per-token, and does the divergence
        // compound with sequence length? Same model, same tokens, two paths.
        for n in 1..=prompt_tokens.len() {
            let mut rows = Vec::new();
            for batched in [false, true] {
                let mut kv =
                    KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, 512)
                        .unwrap();
                let mut dn = DeltaNetState::new(&mut gpu, &config).unwrap();
                let sc = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 128, 512).unwrap();
                dn.reset(&mut gpu);
                let toks = &prompt_tokens[..n];
                if batched {
                    qwen35::forward_prefill_batch(
                        &mut gpu, &weights, &config, toks, 0, &mut kv, &mut dn, &sc, None, None, None,
                        None,
                    )
                    .expect("batched");
                } else {
                    for (pos, &t) in toks.iter().enumerate() {
                        qwen35::forward_scratch(
                            &mut gpu, &weights, &config, t, pos, &mut kv, &mut dn, &sc,
                        )
                        .expect("per-token");
                    }
                }
                gpu.hip.device_synchronize().unwrap();
                rows.push(gpu.download_f32(&sc.logits).unwrap());
            }
            let (a, b) = (&rows[0], &rows[1]);
            let nf = a.len() as f64;
            let (mut sx, mut sy, mut sxy, mut sxx, mut syy) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
            let mut se = 0.0f64;
            for i in 0..a.len() {
                let (x, y) = (a[i] as f64, b[i] as f64);
                sx += x; sy += y; sxy += x * y; sxx += x * x; syy += y * y;
                se += (x - y) * (x - y);
            }
            let corr = (sxy - sx * sy / nf) / ((sxx - sx * sx / nf).sqrt() * (syy - sy * sy / nf).sqrt());
            let same = top10(a) == top10(b);
            println!(
                "  n={n}: per-token-vs-batched corr={corr:.8} rmse={:.6} top10 {}",
                (se / nf).sqrt(),
                if same { "identical" } else { "DIFFER" }
            );
        }
        return;
    }
    let per_token = std::env::var("PARITY_PER_TOKEN").is_ok();
    if per_token {
        eprintln!("path: per-token forward_scratch");
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            qwen35::forward_scratch(
                &mut gpu,
                &weights,
                &config,
                tok,
                pos,
                &mut kv_cache,
                &mut dn_state,
                &scratch,
            )
            .expect("forward_scratch failed");
        }
    } else {
        eprintln!("path: batched forward_prefill_batch");
        qwen35::forward_prefill_batch(
            &mut gpu,
            &weights,
            &config,
            &prompt_tokens,
            0,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
            None,
            None,
            None,
            None,
        )
        .expect("prefill failed");
    }
    gpu.hip.device_synchronize().expect("sync");

    let got = gpu.download_f32(&scratch.logits).expect("download logits");
    let raw = std::fs::read(fork_path).expect("read fork logits");
    let want: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    eprintln!("hipfire logits={} fork logits={}", got.len(), want.len());

    let n = got.len().min(want.len());
    let (mut sx, mut sy, mut sxy, mut sxx, mut syy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut se = 0.0f64;
    for i in 0..n {
        let (x, y) = (got[i] as f64, want[i] as f64);
        sx += x;
        sy += y;
        sxy += x * y;
        sxx += x * x;
        syy += y * y;
        se += (x - y) * (x - y);
    }
    let nf = n as f64;
    let cov = sxy - sx * sy / nf;
    let varx = sxx - sx * sx / nf;
    let vary = syy - sy * sy / nf;
    let corr = cov / (varx.sqrt() * vary.sqrt());
    let rmse = (se / nf).sqrt();

    let ga = top10(&got);
    let wa = top10(&want);
    println!("correlation {corr:.8}");
    println!("rmse        {rmse:.6}");
    println!("argmax      hipfire={} fork={}", ga[0], wa[0]);
    println!("top10       {}", if ga == wa { "identical" } else { "DIFFER" });
    if ga != wa {
        println!("  hipfire top10: {ga:?}");
        println!("  fork    top10: {wa:?}");
    }
}

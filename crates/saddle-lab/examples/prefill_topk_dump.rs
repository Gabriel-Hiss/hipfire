// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Top-k logits of batched prefill at every position of a fixed sequence.
//!
//! Prefills the whole token sequence once through the batched prefill path,
//! takes the post-output-norm hidden row of every position, runs the lm_head
//! on each row and records the top-k logits. Positions `prompt_len - 1 ..
//! len - 1` are written, so a reference engine's greedy continuation of the
//! prompt can be teacher-forced and compared position by position.
//!
//! Usage: prefill_topk_dump <model.ptq1> <tokens.txt> <prompt_len> <k> <out.csv>
//!   tokens.txt: whitespace-separated token ids (prompt, then continuation)
//!   out.csv:    one row per position: pos,id1,logit1,...,idk,logitk

use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::{execute_steps, GemvInput, Step};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
use rdna_compute::DType;
use std::io::Write;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 {
        eprintln!("Usage: prefill_topk_dump <model> <tokens.txt> <prompt_len> <k> <out.csv>");
        std::process::exit(1);
    }
    let tokens: Vec<u32> = std::fs::read_to_string(&args[2])
        .expect("read tokens")
        .split_whitespace()
        .map(|t| t.parse().expect("token id"))
        .collect();
    let prompt_len: usize = args[3].parse().expect("prompt_len");
    let k: usize = args[4].parse().expect("k");
    assert!(prompt_len >= 1 && prompt_len <= tokens.len(), "need 1 <= prompt_len <= len");

    let mut hfq = HfqFile::open(Path::new(&args[1])).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let weights = {
        let mut src = qwen35::HfqSource::new(&mut hfq, &config);
        let layout = qwen35::Layout::single(config.n_layers);
        qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
    }
    .expect("load weights");

    let n = tokens.len();
    let dim = config.dim;
    let kv_seq = n.next_multiple_of(256);
    let mut kv = KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq)
        .unwrap();
    let mut dn = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 128, kv_seq).unwrap();
    let hidden = gpu.alloc_tensor(&[n * dim], DType::F32).expect("hidden");

    dn.reset(&mut gpu);
    qwen35::forward_prefill_batch(
        &mut gpu, &weights, &config, &tokens, 0, &mut kv, &mut dn, &scratch, None, Some(&hidden), None, None,
    )
    .expect("prefill");

    let mut out = std::fs::File::create(&args[5]).expect("create out");
    let wr = weights.output.dispatch_ref();
    for pos in prompt_len - 1..n {
        let row = hidden.sub_offset(pos * dim, dim);
        let ctx = DispatchCtx::new(&gpu);
        execute_steps(&mut gpu, &ctx, &[Step::Gemv { w: &wr, input: GemvInput::Raw(&row), out: &scratch.logits }])
            .expect("lm_head");
        let logits = gpu.download_f32(&scratch.logits).unwrap();
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.select_nth_unstable_by(k, |&a, &b| logits[b].total_cmp(&logits[a]));
        idx.truncate(k);
        idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let cells: Vec<String> = idx.iter().map(|&i| format!("{i},{}", logits[i])).collect();
        writeln!(out, "{pos},{}", cells.join(",")).unwrap();
    }
    eprintln!("wrote {} positions", n + 1 - prompt_len);
}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Channel test: does `gated_delta_net_q8_batch_seq` (batched prefill) agree with
// N calls of `gated_delta_net_q8_compact` (per-token decode)?
//
// This is the divergence the batched prefill's parity failure was localized to:
// the embedding, the PTQ1 and BF16 GEMMs, the Prism-Hadamard rotation and the
// conv1d all compare bit-identical or verified, and the first disagreement in
// the forward pass is the GDN output at layer 0 (140 of 6144 elements, which 64
// layers then amplify to a complete decorrelation).
//
// `prefill.rs` documents the two as "distributionally equivalent to decode, not
// byte-identical — the stochastic-rounding frame differs". This measures how
// large that difference actually is.
//
// Layouts (Qwen3.5 DeltaNet, this model): 16 key heads, 48 value heads, head
// dim 128, so qk_head_div = 3. The batched kernel takes Q/K pre-expanded to 48
// heads (HIPFIRE_GDN_QK_HEAD_DIV=1, qk_head = h); the per-token kernel takes
// raw 16-head Q/K with qk_head_div=3 (qk_head = h/3). Both must read the same
// source head for value head h.
//
// cargo run --release -p rdna-compute --features deltanet,lab --example test_gdn_batch_vs_pertoken

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use rdna_compute::{DType, Gpu, GpuTensor};

    const N_TOKENS: usize = 4;
    const N_KEY_HEADS: usize = 16;
    const N_V_HEADS: usize = 48;
    const HD: usize = 128;
    const QK_DIV: usize = N_V_HEADS / N_KEY_HEADS; // 3
    const S_SIZE: usize = N_V_HEADS * HD * HD;

    let mut gpu = Gpu::init().expect("gpu init");
    println!("arch: {}", gpu.arch);

    let mut state = 0x1234_5678u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    // Raw 16-head Q/K, 48-head V, per-token gate/beta.
    let q_raw: Vec<f32> = (0..N_TOKENS * N_KEY_HEADS * HD).map(|_| next()).collect();
    let k_raw: Vec<f32> = (0..N_TOKENS * N_KEY_HEADS * HD).map(|_| next()).collect();
    let v: Vec<f32> = (0..N_TOKENS * N_V_HEADS * HD).map(|_| next()).collect();
    let gate: Vec<f32> = (0..N_TOKENS * N_V_HEADS)
        .map(|_| -0.5 + next() * 0.4)
        .collect();
    let beta: Vec<f32> = (0..N_TOKENS * N_V_HEADS).map(|_| next() * 0.5 + 0.5).collect();

    // Expand Q/K to 48 heads exactly as the batched path does: dst[(kh*ratio + r)] = src[kh].
    let mut q_exp = vec![0.0f32; N_TOKENS * N_V_HEADS * HD];
    let mut k_exp = vec![0.0f32; N_TOKENS * N_V_HEADS * HD];
    for t in 0..N_TOKENS {
        for kh in 0..N_KEY_HEADS {
            for r in 0..QK_DIV {
                let vh = kh * QK_DIV + r;
                let src = t * N_KEY_HEADS * HD + kh * HD;
                let dst = t * N_V_HEADS * HD + vh * HD;
                q_exp[dst..dst + HD].copy_from_slice(&q_raw[src..src + HD]);
                k_exp[dst..dst + HD].copy_from_slice(&k_raw[src..src + HD]);
            }
        }
    }

    let d_q_raw = gpu.upload_f32(&q_raw, &[q_raw.len()]).expect("q_raw");
    let d_k_raw = gpu.upload_f32(&k_raw, &[k_raw.len()]).expect("k_raw");
    let d_q_exp = gpu.upload_f32(&q_exp, &[q_exp.len()]).expect("q_exp");
    let d_k_exp = gpu.upload_f32(&k_exp, &[k_exp.len()]).expect("k_exp");
    let d_v = gpu.upload_f32(&v, &[v.len()]).expect("v");
    let d_gate = gpu.upload_f32(&gate, &[gate.len()]).expect("gate");
    let d_beta = gpu.upload_f32(&beta, &[beta.len()]).expect("beta");

    // One zeroed int8 state per path (the kernel takes a byte pointer; the
    // F32-typed container is how DeltaNetState allocates it).
    let mk_state = |gpu: &mut Gpu| -> GpuTensor {
        let s = gpu
            .alloc_tensor(&[S_SIZE], DType::F32)
            .expect("alloc s_matrices");
        let zeros = vec![0u8; S_SIZE];
        gpu.hip
            .memcpy_htod(&s.buf, &zeros)
            .expect("zero s_matrices");
        s
    };
    let s_batch = mk_state(&mut gpu);
    let s_pt = mk_state(&mut gpu);
    let ones = vec![1.0f32; N_V_HEADS * HD];
    let sc_batch = gpu.upload_f32(&ones, &[ones.len()]).expect("sc");
    let sc_pt = gpu.upload_f32(&ones, &[ones.len()]).expect("sc");

    let out_batch = gpu
        .alloc_tensor(&[N_TOKENS * N_V_HEADS * HD], DType::F32)
        .expect("out_batch");
    let out_pt = gpu
        .alloc_tensor(&[N_TOKENS * N_V_HEADS * HD], DType::F32)
        .expect("out_pt");

    // ── Path A: one batched call ──────────────────────────────────────────
    gpu.gated_delta_net_q8_batch_seq(
        &d_q_exp,
        &d_k_exp,
        &d_v,
        &d_gate,
        &d_beta,
        &s_batch,
        &sc_batch,
        &out_batch,
        N_TOKENS,
        N_V_HEADS,
        HD,
        None,
    )
    .expect("batch_seq");

    // ── Path B: one per-token call per step, sharing the same state ───────
    for t in 0..N_TOKENS {
        let q_t = d_q_raw.sub_offset(t * N_KEY_HEADS * HD, N_KEY_HEADS * HD);
        let k_t = d_k_raw.sub_offset(t * N_KEY_HEADS * HD, N_KEY_HEADS * HD);
        let v_t = d_v.sub_offset(t * N_V_HEADS * HD, N_V_HEADS * HD);
        let g_t = d_gate.sub_offset(t * N_V_HEADS, N_V_HEADS);
        let b_t = d_beta.sub_offset(t * N_V_HEADS, N_V_HEADS);
        let o_t = out_pt.sub_offset(t * N_V_HEADS * HD, N_V_HEADS * HD);
        gpu.gated_delta_net_q8_compact(
            &q_t, &k_t, &v_t, &g_t, &b_t, &s_pt, &sc_pt, &o_t, 1, N_V_HEADS, HD, QK_DIV, None,
        )
        .expect("compact");
    }

    gpu.hip.device_synchronize().expect("sync");
    let a = gpu.download_f32(&out_batch).expect("dl batch");
    let b = gpu.download_f32(&out_pt).expect("dl pt");

    let mut worst = 0.0f32;
    let mut n_diff = 0usize;
    let scale = a.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-9);
    for (x, y) in a.iter().zip(&b) {
        let d = (x - y).abs();
        if d > 1e-3 {
            n_diff += 1;
        }
        worst = worst.max(d);
    }
    println!(
        "batch_seq vs per-token compact: n_diff(>1e-3) = {n_diff} / {}  max|d| = {worst:.6e}  max|a| = {scale:.4}  rel = {:.3e}",
        a.len(),
        worst / scale
    );
    for t in 0..N_TOKENS {
        let o = t * N_V_HEADS * HD;
        let per_tok = (o..o + N_V_HEADS * HD)
            .filter(|&i| (a[i] - b[i]).abs() > 1e-3)
            .count();
        println!("  token {t}: {per_tok} / {} differ", N_V_HEADS * HD);
    }
    println!("PASS");
}

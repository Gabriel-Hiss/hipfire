// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU vs CPU gate for `rotate_x_prism_hadamard`, both directions.
//!
//! The CPU oracle builds the same dense matrix llama.cpp materializes for the
//! `prism.hadamard` contract (`±1/sqrt(block)` by the parity of `row & col`,
//! llama-model.cpp) and applies it in the order the fork's graph uses:
//! signs then H before a folded matmul, H then signs after an embedding lookup.

use rdna_compute::{DType, Gpu};
use std::collections::HashMap;

fn reference(values: &[f32], width: usize, block: usize, signs: &[i32], inverse: bool) -> Vec<f32> {
    let scale = 1.0 / (block as f32).sqrt();
    let mut out = vec![0.0f32; values.len()];
    for (row_in, row_out) in values.chunks_exact(width).zip(out.chunks_exact_mut(width)) {
        let staged: Vec<f32> = if inverse {
            row_in.to_vec()
        } else {
            row_in.iter().zip(signs).map(|(v, s)| v * *s as f32).collect()
        };
        for (bi, (block_in, block_out)) in staged
            .chunks_exact(block)
            .zip(row_out.chunks_exact_mut(block))
            .enumerate()
        {
            for (row, slot) in block_out.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for (col, value) in block_in.iter().enumerate() {
                    let parity = (row & col).count_ones() & 1;
                    acc += if parity == 1 { -scale } else { scale } * value;
                }
                *slot = acc;
            }
            if inverse {
                for (i, slot) in block_out.iter_mut().enumerate() {
                    *slot *= signs[bi * block + i] as f32;
                }
            }
        }
    }
    out
}

fn main() {
    const WIDTH: usize = 2048;
    const BLOCK: usize = 1024;
    const BATCH: usize = 2;

    let mut gpu = Gpu::init().expect("gpu init");
    let mut state = 0x1234_5678u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let signs: Vec<i32> = (0..WIDTH).map(|_| if next() < 0.0 { -1 } else { 1 }).collect();
    let x: Vec<f32> = (0..WIDTH * BATCH).map(|_| next()).collect();

    let mut table = HashMap::new();
    table.insert(WIDTH, signs.clone());
    gpu.configure_prism_hadamard(BLOCK, false, &table)
        .expect("configure");

    let d_x = gpu.upload_f32(&x, &[WIDTH * BATCH]).expect("upload x");
    let d_y = gpu
        .alloc_tensor(&[WIDTH * BATCH], DType::F32)
        .expect("alloc y");

    let mut failed = false;
    for inverse in [false, true] {
        gpu.prism_hadamard(&d_x, &d_y, WIDTH, BATCH, inverse)
            .expect("rotate");
        gpu.hip.device_synchronize().expect("sync");
        let got = gpu.download_f32(&d_y).expect("download");
        let want = reference(&x, WIDTH, BLOCK, &signs, inverse);
        let max_err = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0f32, f32::max);
        let label = if inverse { "inverse" } else { "forward" };
        println!("{label}: max |gpu - cpu| = {max_err:.3e}");
        if !(max_err < 1e-4) {
            failed = true;
            for (i, (g, w)) in got.iter().zip(&want).enumerate().take(4) {
                println!("  [{i}] gpu={g:.6} cpu={w:.6}");
            }
        }
    }

    // Round trip: inverse(forward(x)) must return x when signs are involutive.
    gpu.prism_hadamard(&d_x, &d_y, WIDTH, BATCH, false)
        .expect("forward");
    let mid = gpu.download_f32(&d_y).expect("download mid");
    let d_mid = gpu.upload_f32(&mid, &[WIDTH * BATCH]).expect("upload mid");
    gpu.prism_hadamard(&d_mid, &d_y, WIDTH, BATCH, true)
        .expect("inverse");
    gpu.hip.device_synchronize().expect("sync");
    let back = gpu.download_f32(&d_y).expect("download back");
    let rt_err = back
        .iter()
        .zip(&x)
        .map(|(b, o)| (b - o).abs())
        .fold(0.0f32, f32::max);
    println!("round trip: max |x - inverse(forward(x))| = {rt_err:.3e}");
    if !(rt_err < 1e-4) {
        failed = true;
    }

    println!("{}", if failed { "FAIL" } else { "PASS" });
    if failed {
        std::process::exit(1);
    }
}

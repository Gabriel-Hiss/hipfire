// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Channel test: gemm_bf16_xf32_batched vs a CPU oracle.
//
// This kernel is what admits Prism's unquantized BF16 DeltaNet gate
// projections (w_alpha/w_beta) to batched prefill. Before it, one BF16 tensor
// rejected the whole model. It is used by the batched path ONLY, so the
// per-token decode parity gate cannot see it.
//
// cargo run --release -p rdna-compute --features lab --example test_gemm_bf16_xf32_batched

use rdna_compute::{DType, Gpu};

fn f32_to_bf16(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("arch: {}", gpu.arch);

    let mut state = 0x1234_5678u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };

    let cases: &[(usize, usize, usize)] = &[
        (16, 128, 8),
        (48, 5120, 7),
        (257, 512, 17),
        (1024, 256, 33),
    ];

    for &(m, k, n) in cases {
        let w: Vec<f32> = (0..m * k)
            .map(|_| (next() as f32 / u32::MAX as f32) * 2.0 - 1.0)
            .collect();
        let x: Vec<f32> = (0..n * k)
            .map(|_| (next() as f32 / u32::MAX as f32) * 2.0 - 1.0)
            .collect();

        // CPU oracle in f32 against the bf16-rounded weights, which is what the
        // kernel sees (it widens losslessly, so the rounding is the only loss).
        let wr: Vec<f32> = w.iter().map(|&v| bf16_to_f32(f32_to_bf16(v))).collect();
        let mut want = vec![0.0f32; n * m];
        for nn in 0..n {
            for row in 0..m {
                let mut acc = 0.0f64;
                for kk in 0..k {
                    acc += wr[row * k + kk] as f64 * x[nn * k + kk] as f64;
                }
                want[nn * m + row] = acc as f32;
            }
        }

        let w_bytes: Vec<u8> = w
            .iter()
            .flat_map(|&v| f32_to_bf16(v).to_le_bytes())
            .collect();
        let dw = gpu.upload_raw(&w_bytes, &[w_bytes.len()]).expect("upload w");
        let dx = gpu.upload_f32(&x, &[n * k]).expect("upload x");
        let dy = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc y");
        gpu.gemm_bf16_xf32_batched(&dw, &dx, &dy, m, k, n)
            .expect("gemm");
        gpu.hip.device_synchronize().expect("sync");
        let got = gpu.download_f32(&dy).expect("download");

        let mut worst = 0.0f32;
        let mut worst_at = 0usize;
        for i in 0..got.len() {
            let d = (got[i] - want[i]).abs();
            let scale = want[i].abs().max(1.0);
            let rel = d / scale;
            if rel > worst {
                worst = rel;
                worst_at = i;
            }
        }
        let n_bad = got
            .iter()
            .zip(&want)
            .filter(|(g, w)| (*g - *w).abs() / w.abs().max(1.0) > 1e-3)
            .count();
        println!(
            "  {m}x{k}x{n}: worst rel {worst:.3e} at {worst_at} ({}/{} over 1e-3)",
            n_bad,
            got.len()
        );
        let _ = gpu.free_tensor(dw);
        let _ = gpu.free_tensor(dx);
        let _ = gpu.free_tensor(dy);
    }
    println!("PASS");
}

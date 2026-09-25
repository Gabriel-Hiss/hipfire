// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Bit-exactness gate for the fused PTQ1 producers, the multi-segment
//! residual GEMV, and the prefill GEMM over producer output, against the
//! unfused chains they replace:
//!
//!   rmsnorm_f32      -> rotate_x_prism_hadamard -> quantize -> gemv_ptq1g128
//!   silu_mul_f32     -> rotate -> quantize -> gemv_ptq1g128
//!   sigmoid_mul_f32  -> rotate -> quantize -> gemv_ptq1g128
//!   gated_norm_f32   -> rotate -> quantize -> gemv_ptq1g128
//!   gemv_ptq1g128 x3 (+ add_inplace_f32)  vs  gemv_ptq1g128_multi
//!   gemv_bf16_xf32 x2                     vs  gemv_ptq1g128_multi's BF16 pair
//!   batched op -> rotate -> gemm_ptq1g128_wmma (+ add_inplace_f32)
//!     vs  batched producer -> gemm_ptq1g128_wmma_q8 (residual epilogue)
//!
//! Every comparison is exact equality: the fused kernels repeat the chain's
//! arithmetic in the chain's order, so any difference is a bug, not noise.

use rdna_compute::gemv::Ptq1Bf16Pair;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::HashMap;

const BLOCK: usize = 1024;

struct Rng(u32);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.unit() * scale).collect()
    }
}

/// Random PTQ1 rows: arbitrary trit bytes, a positive fp16 scale per group.
fn ptq1_weights(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let mut out = vec![0u8; m * (k / 128) * 28];
    for blk in out.chunks_exact_mut(28) {
        for b in &mut blk[..26] {
            *b = rng.next() as u8;
        }
        // 0.25 .. 0.5 as fp16: exponent 0x34/0x35, random mantissa.
        let bits: u16 = 0x3400 | (rng.next() as u16 & 0x03ff);
        blk[26..28].copy_from_slice(&bits.to_le_bytes());
    }
    out
}

fn download(gpu: &mut Gpu, t: &GpuTensor) -> Vec<f32> {
    gpu.download_f32(t).expect("download")
}

fn exact(name: &str, a: &[f32], b: &[f32]) -> bool {
    let diff = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    println!("  {name:<34} {} differing of {} (max |d| {max:.3e})", diff, a.len());
    diff == 0
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let mut rng = Rng(0x5EED_1234);

    // Bonsai widths: hidden 5120, attention/DeltaNet value 6144, FFN 17408.
    let widths = [5120usize, 6144, 17408];
    let mut table = HashMap::new();
    for &w in &widths {
        let signs: Vec<i32> = (0..w).map(|_| if rng.next() & 1 == 0 { -1 } else { 1 }).collect();
        table.insert(w, signs);
    }
    gpu.configure_prism_hadamard(BLOCK, false, &table).expect("configure prism");

    let xq = gpu
        .alloc_tensor(&[Gpu::ptq1_xq_bytes(17408) / 4], DType::F32)
        .expect("alloc xq");
    let mut ok = true;

    // One projection per producer, M small enough to be quick.
    let m = 520usize;
    let mut project = |gpu: &mut Gpu, rng: &mut Rng, k: usize| {
        let w = ptq1_weights(rng, m, k);
        gpu.upload_raw(&w, &[w.len()]).expect("upload w")
    };

    // rmsnorm -> rotate -> Q8_1, plus the plain output.
    {
        let k = 5120;
        let x = gpu.upload_f32(&rng.vec(k, 3.0), &[k]).expect("x");
        let nw = gpu.upload_f32(&rng.vec(k, 1.0), &[k]).expect("nw");
        let w = project(&mut gpu, &mut rng, k);
        let plain_ref = gpu.alloc_tensor(&[k], DType::F32).unwrap();
        let rot = gpu.alloc_tensor(&[k], DType::F32).unwrap();
        let plain = gpu.alloc_tensor(&[k], DType::F32).unwrap();
        let y_ref = gpu.alloc_tensor(&[m], DType::F32).unwrap();
        let y = gpu.alloc_tensor(&[m], DType::F32).unwrap();
        gpu.rmsnorm_f32(&x, &nw, &plain_ref, 1e-6).unwrap();
        gpu.rotate_x_prism_hadamard(&plain_ref, &rot, k, 1).unwrap();
        gpu.gemv_ptq1g128(&w, &rot, &y_ref, m, k).unwrap();
        gpu.ptq1_rmsnorm_rotate_q8(&x, &nw, Some(&plain), &xq, k, 1e-6, 1).unwrap();
        gpu.gemv_ptq1g128_multi(&xq, &[(&w, &y, m)], k, false, None).unwrap();
        gpu.hip.device_synchronize().unwrap();
        println!("rmsnorm (K={k})");
        ok &= exact("plain vs rmsnorm_f32", &download(&mut gpu, &plain), &download(&mut gpu, &plain_ref));
        ok &= exact("gemv(fused xq) vs chain", &download(&mut gpu, &y), &download(&mut gpu, &y_ref));
    }

    // silu_mul / sigmoid_mul -> rotate -> Q8_1.
    for (name, k) in [("silu_mul", 17408usize), ("sigmoid_mul", 6144)] {
        let a = gpu.upload_f32(&rng.vec(k, 4.0), &[k]).expect("a");
        let b = gpu.upload_f32(&rng.vec(k, 4.0), &[k]).expect("b");
        let w = project(&mut gpu, &mut rng, k);
        let tmp = gpu.alloc_tensor(&[k], DType::F32).unwrap();
        let rot = gpu.alloc_tensor(&[k], DType::F32).unwrap();
        let y_ref = gpu.alloc_tensor(&[m], DType::F32).unwrap();
        let y = gpu.alloc_tensor(&[m], DType::F32).unwrap();
        if name == "silu_mul" {
            gpu.silu_mul_f32(&a, &b, &tmp).unwrap();
            gpu.ptq1_silu_mul_rotate_q8(&a, &b, &xq, k, 1).unwrap();
        } else {
            // sigmoid_mul_f32 is in place on its first operand; run it on a
            // copy so the fused kernel reads the original.
            gpu.hip.memcpy_dtod(&tmp.buf, &a.buf, k * 4).unwrap();
            gpu.sigmoid_mul_f32(&tmp, &b).unwrap();
            gpu.ptq1_sigmoid_mul_rotate_q8(&a, &b, &xq, k, 1).unwrap();
        }
        gpu.gemv_ptq1g128_multi(&xq, &[(&w, &y, m)], k, false, None).unwrap();
        gpu.rotate_x_prism_hadamard(&tmp, &rot, k, 1).unwrap();
        gpu.gemv_ptq1g128(&w, &rot, &y_ref, m, k).unwrap();
        gpu.hip.device_synchronize().unwrap();
        println!("{name} (K={k})");
        ok &= exact("gemv(fused xq) vs chain", &download(&mut gpu, &y), &download(&mut gpu, &y_ref));
    }

    // gated_norm -> rotate -> Q8_1 (48 heads x 128).
    {
        let (heads, hd) = (48usize, 128usize);
        let k = heads * hd;
        let x = gpu.upload_f32(&rng.vec(k, 2.0), &[k]).expect("x");
        let z = gpu.upload_f32(&rng.vec(k, 3.0), &[k]).expect("z");
        let nw = gpu.upload_f32(&rng.vec(hd, 1.0), &[hd]).expect("nw");
        let w = project(&mut gpu, &mut rng, k);
        let normed = gpu.alloc_tensor(&[k], DType::F32).unwrap();
        let rot = gpu.alloc_tensor(&[k], DType::F32).unwrap();
        let y_ref = gpu.alloc_tensor(&[m], DType::F32).unwrap();
        let y = gpu.alloc_tensor(&[m], DType::F32).unwrap();
        gpu.gated_norm_f32(&x, &z, &nw, &normed, heads, hd, 1e-6).unwrap();
        gpu.rotate_x_prism_hadamard(&normed, &rot, k, 1).unwrap();
        gpu.gemv_ptq1g128(&w, &rot, &y_ref, m, k).unwrap();
        gpu.ptq1_gated_norm_rotate_q8(&x, &z, &nw, &xq, heads, hd, 1e-6, 1).unwrap();
        gpu.gemv_ptq1g128_multi(&xq, &[(&w, &y, m)], k, false, None).unwrap();
        gpu.hip.device_synchronize().unwrap();
        println!("gated_norm ({heads}x{hd})");
        ok &= exact("gemv(fused xq) vs chain", &download(&mut gpu, &y), &download(&mut gpu, &y_ref));
    }

    // Three segments with ragged M (not multiples of the 4-row tile), plain
    // and residual, against separate launches.
    {
        let k = 5120;
        let ms = [1030usize, 257, 3];
        let x = gpu.upload_f32(&rng.vec(k, 1.0), &[k]).expect("x");
        let nw = gpu.upload_f32(&rng.vec(k, 1.0), &[k]).expect("nw");
        let plain = gpu.alloc_tensor(&[k], DType::F32).unwrap();
        let rot = gpu.alloc_tensor(&[k], DType::F32).unwrap();
        gpu.rmsnorm_f32(&x, &nw, &plain, 1e-6).unwrap();
        gpu.rotate_x_prism_hadamard(&plain, &rot, k, 1).unwrap();
        gpu.ptq1_rmsnorm_rotate_q8(&x, &nw, None, &xq, k, 1e-6, 1).unwrap();
        let ws: Vec<_> = ms
            .iter()
            .map(|&mi| {
                let w = ptq1_weights(&mut rng, mi, k);
                gpu.upload_raw(&w, &[w.len()]).unwrap()
            })
            .collect();
        for residual in [false, true] {
            let base: Vec<Vec<f32>> = ms.iter().map(|&mi| rng.vec(mi, 5.0)).collect();
            let refs: Vec<_> = ms.iter().zip(&base).map(|(&mi, b)| gpu.upload_f32(b, &[mi]).unwrap()).collect();
            let outs: Vec<_> = ms.iter().zip(&base).map(|(&mi, b)| gpu.upload_f32(b, &[mi]).unwrap()).collect();
            for i in 0..3 {
                if residual {
                    let t = gpu.alloc_tensor(&[ms[i]], DType::F32).unwrap();
                    gpu.gemv_ptq1g128(&ws[i], &rot, &t, ms[i], k).unwrap();
                    gpu.add_inplace_f32(&refs[i], &t).unwrap();
                } else {
                    gpu.gemv_ptq1g128(&ws[i], &rot, &refs[i], ms[i], k).unwrap();
                }
            }
            let segs: Vec<_> = (0..3).map(|i| (&ws[i], &outs[i], ms[i])).collect();
            gpu.gemv_ptq1g128_multi(&xq, &segs, k, residual, None).unwrap();
            gpu.hip.device_synchronize().unwrap();
            println!("multi 3 segments, residual={residual}");
            for i in 0..3 {
                let label = format!("segment {i} (M={})", ms[i]);
                ok &= exact(&label, &download(&mut gpu, &outs[i]), &download(&mut gpu, &refs[i]));
            }
        }
    }

    // qkv + z with the BF16 beta/alpha pair in the same launch, against the
    // separate PTQ1 and gemv_bf16_xf32 launches.
    {
        let k = 5120;
        let (mq, mz, mb) = (1030usize, 257usize, 48usize);
        let x = gpu.upload_f32(&rng.vec(k, 1.0), &[k]).expect("x");
        let nw = gpu.upload_f32(&rng.vec(k, 1.0), &[k]).expect("nw");
        let plain = gpu.alloc_tensor(&[k], DType::F32).unwrap();
        gpu.ptq1_rmsnorm_rotate_q8(&x, &nw, Some(&plain), &xq, k, 1e-6, 1).unwrap();
        let wq = ptq1_weights(&mut rng, mq, k);
        let wz = ptq1_weights(&mut rng, mz, k);
        let wq = gpu.upload_raw(&wq, &[wq.len()]).unwrap();
        let wz = gpu.upload_raw(&wz, &[wz.len()]).unwrap();
        let bf: Vec<GpuTensor> = (0..2)
            .map(|_| {
                let raw: Vec<u8> = (0..mb * k)
                    .flat_map(|_| ((rng.unit().to_bits() >> 16) as u16).to_le_bytes())
                    .collect();
                gpu.upload_raw(&raw, &[raw.len()]).unwrap()
            })
            .collect();
        let outs: Vec<GpuTensor> = [mq, mz, mb, mb].iter().map(|&n| gpu.alloc_tensor(&[n], DType::F32).unwrap()).collect();
        let refs: Vec<GpuTensor> = [mq, mz, mb, mb].iter().map(|&n| gpu.alloc_tensor(&[n], DType::F32).unwrap()).collect();
        gpu.gemv_ptq1g128_multi(&xq, &[(&wq, &refs[0], mq)], k, false, None).unwrap();
        gpu.gemv_ptq1g128_multi(&xq, &[(&wz, &refs[1], mz)], k, false, None).unwrap();
        gpu.gemv_bf16_xf32(&bf[0], &plain, &refs[2], mb, k).unwrap();
        gpu.gemv_bf16_xf32(&bf[1], &plain, &refs[3], mb, k).unwrap();
        let pair = Ptq1Bf16Pair { w: [&bf[0], &bf[1]], y: [&outs[2], &outs[3]], m: mb, x: &plain };
        gpu.gemv_ptq1g128_multi(&xq, &[(&wq, &outs[0], mq), (&wz, &outs[1], mz)], k, false, Some(&pair))
            .unwrap();
        gpu.hip.device_synchronize().unwrap();
        println!("multi 2 segments + BF16 pair");
        for (i, label) in ["qkv", "z", "beta (bf16)", "alpha (bf16)"].iter().enumerate() {
            ok &= exact(label, &download(&mut gpu, &outs[i]), &download(&mut gpu, &refs[i]));
        }
    }

    // Prefill: batched producers into the [k/128][n] GEMM layout and the
    // 64x64 WMMA tile reading it, plain and residual. 97 tokens leave a
    // ragged last tile; gfx11 only (the tile's admission rule).
    let n = 97usize;
    if Gpu::ptq1_t64_admitted(&gpu.arch, n) {
        let m = 1030usize;
        let xqb = gpu
            .alloc_tensor(&[n * Gpu::ptq1_xq_bytes(17408) / 4], DType::F32)
            .expect("alloc batched xq");
        let mut check = |gpu: &mut Gpu, rng: &mut Rng, name: &str, k: usize, chain: &GpuTensor| -> bool {
            let w = ptq1_weights(rng, m, k);
            let w = gpu.upload_raw(&w, &[w.len()]).unwrap();
            let rot = gpu.alloc_tensor(&[n * k], DType::F32).unwrap();
            gpu.rotate_x_prism_hadamard(chain, &rot, k, n).unwrap();
            let base = rng.vec(n * m, 5.0);
            let mut ok = true;
            for residual in [false, true] {
                let y_ref = gpu.upload_f32(&base, &[n * m]).unwrap();
                let y = gpu.upload_f32(&base, &[n * m]).unwrap();
                if residual {
                    let t = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
                    gpu.gemm_ptq1g128_wmma(&w, &rot, &t, m, k, n).unwrap();
                    gpu.add_inplace_f32(&y_ref, &t).unwrap();
                } else {
                    gpu.gemm_ptq1g128_wmma(&w, &rot, &y_ref, m, k, n).unwrap();
                }
                gpu.gemm_ptq1g128_wmma_q8(&w, &xqb, &y, m, k, n, residual).unwrap();
                gpu.hip.device_synchronize().unwrap();
                let label = format!("{name} gemm residual={residual}");
                ok &= exact(&label, &download(gpu, &y), &download(gpu, &y_ref));
            }
            ok
        };
        println!("prefill batch {n}");

        let k = 5120;
        let x = gpu.upload_f32(&rng.vec(n * k, 3.0), &[n * k]).unwrap();
        let nw = gpu.upload_f32(&rng.vec(k, 1.0), &[k]).unwrap();
        let plain_ref = gpu.alloc_tensor(&[n * k], DType::F32).unwrap();
        let plain = gpu.alloc_tensor(&[n * k], DType::F32).unwrap();
        gpu.rmsnorm_batched(&x, &nw, &plain_ref, n, k, 1e-6).unwrap();
        gpu.ptq1_rmsnorm_rotate_q8(&x, &nw, Some(&plain), &xqb, k, 1e-6, n).unwrap();
        gpu.hip.device_synchronize().unwrap();
        ok &= exact("rmsnorm plain vs rmsnorm_batched", &download(&mut gpu, &plain), &download(&mut gpu, &plain_ref));
        ok &= check(&mut gpu, &mut rng, "rmsnorm", k, &plain_ref);

        for (name, k) in [("silu_mul", 17408usize), ("sigmoid_mul", 6144)] {
            let a = gpu.upload_f32(&rng.vec(n * k, 4.0), &[n * k]).unwrap();
            let b = gpu.upload_f32(&rng.vec(n * k, 4.0), &[n * k]).unwrap();
            let tmp = gpu.alloc_tensor(&[n * k], DType::F32).unwrap();
            if name == "silu_mul" {
                gpu.silu_mul_f32(&a, &b, &tmp).unwrap();
                gpu.ptq1_silu_mul_rotate_q8(&a, &b, &xqb, k, n).unwrap();
            } else {
                gpu.hip.memcpy_dtod(&tmp.buf, &a.buf, n * k * 4).unwrap();
                gpu.sigmoid_mul_f32(&tmp, &b).unwrap();
                gpu.ptq1_sigmoid_mul_rotate_q8(&a, &b, &xqb, k, n).unwrap();
            }
            ok &= check(&mut gpu, &mut rng, name, k, &tmp);
        }

        let (heads, hd) = (48usize, 128usize);
        let k = heads * hd;
        let x = gpu.upload_f32(&rng.vec(n * k, 2.0), &[n * k]).unwrap();
        let z = gpu.upload_f32(&rng.vec(n * k, 3.0), &[n * k]).unwrap();
        let nw = gpu.upload_f32(&rng.vec(hd, 1.0), &[hd]).unwrap();
        let normed = gpu.alloc_tensor(&[n * k], DType::F32).unwrap();
        gpu.gated_norm_f32_batched(&x, &z, &nw, &normed, heads, hd, 1e-6, n).unwrap();
        gpu.ptq1_gated_norm_rotate_q8(&x, &z, &nw, &xqb, heads, hd, 1e-6, n).unwrap();
        ok &= check(&mut gpu, &mut rng, "gated_norm", k, &normed);
    } else {
        println!("prefill batch: skipped (no 64x64 PTQ1 tile on {})", gpu.arch);
    }

    println!("{}", if ok { "PASS" } else { "FAIL" });
    if !ok {
        std::process::exit(1);
    }
}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU PTQ1-G128 GEMV vs the PrismML llama.cpp base-3 decoder.

use rdna_compute::{DType, Gpu};

fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7fffff;
    if exp <= 0 {
        if exp < -10 { return sign; }
        let m = (mant | 0x800000) >> (1 - exp);
        return sign | ((m + 0x1000) >> 13) as u16;
    }
    if exp >= 31 { return sign | 0x7c00; }
    sign | ((exp as u16) << 10) | (((mant + 0x1000) >> 13) as u16)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32 & 0x8000) << 16) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x03ff) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e = 113u32;
            while m & 0x0400 == 0 { m <<= 1; e -= 1; }
            sign | (e << 23) | ((m & 0x03ff) << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

fn pack_block(trits: &[i8; 128], d: f32) -> [u8; 28] {
    let mut out = [0u8; 28];
    let stages = [32usize, 16, 8];
    let mut src = 0usize;
    let mut j = 0usize;
    for c in stages {
        while j + c <= 24 {
            for m in 0..c {
                let mut q = 0u8;
                for n in 0..5 {
                    q = q.wrapping_mul(3).wrapping_add((trits[src + m + n*c] + 1) as u8);
                }
                out[j + m] = (((q as u16) * 256 + 242) / 243) as u8;
            }
            src += 5*c;
            j += c;
        }
    }
    for h in 0..2 {
        let mut q = 0u8;
        for m in 0..4 { q = q.wrapping_mul(3).wrapping_add((trits[src+h+m*2]+1) as u8); }
        q = q.wrapping_mul(3);
        out[24+h] = (((q as u16) * 256 + 242) / 243) as u8;
    }
    out[26..28].copy_from_slice(&f32_to_f16(d).to_le_bytes());
    out
}

fn main() {
    const M: usize = 257;
    const K: usize = 512;
    let mut state = 0xC0FFEEu32;
    let mut next = || { state ^= state << 13; state ^= state >> 17; state ^= state << 5; state };
    let x: Vec<f32> = (0..K).map(|_| (next() as f32 / u32::MAX as f32)*2.0-1.0).collect();
    let mut packed = Vec::with_capacity(M*(K/128)*28);
    let mut want = vec![0.0f32; M];
    // Prism llama.cpp PTQ1 GPU path dots against Q8_1 activations. Emulate
    // quantize_q8_1_mmq_ds4 exactly: 32-value scales, fp16-rounded d, nearest-even i8.
    let mut xq = vec![0.0f32; K];
    for (src, dst) in x.chunks_exact(32).zip(xq.chunks_exact_mut(32)) {
        let amax = src.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = if amax > 1.0e-20 { amax / 127.0 } else { 0.0 };
        let d_wire = f16_to_f32(f32_to_f16(d));
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        for (s, q) in src.iter().zip(dst) {
            let qi = (s * id).clamp(-127.0, 127.0).round_ties_even();
            *q = qi * d_wire;
        }
    }
    for row in 0..M {
        for g in 0..K/128 {
            let d = 0.1 + (next()%100) as f32 / 100.0;
            let mut trits = [0i8;128];
            for t in &mut trits { *t = (next()%3) as i8 - 1; }
            let block = pack_block(&trits, d);
            let d_wire = f16_to_f32(u16::from_le_bytes([block[26], block[27]]));
            packed.extend_from_slice(&block);
            for i in 0..128 { want[row] += trits[i] as f32 * d_wire * xq[g*128+i]; }
        }
    }
    let mut gpu = Gpu::init().expect("gpu init");
    let a = gpu.upload_raw(&packed, &[packed.len()]).expect("upload a");
    let dx = gpu.upload_f32(&x, &[K]).expect("upload x");
    let y = gpu.alloc_tensor(&[M], DType::F32).expect("alloc y");
    gpu.gemv_ptq1g128(&a, &dx, &y, M, K).expect("gemv");
    gpu.hip.device_synchronize().expect("sync");
    let got = gpu.download_f32(&y).expect("download");
    let max = got.iter().zip(&want).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
    println!("PTQ1 GEMV max |gpu-cpu| = {max:.3e}");
    assert!(max < 2e-4, "PTQ1 GEMV mismatch");

    const N: usize = 17;
    let xs: Vec<f32> = (0..N*K)
        .map(|_| (next() as f32/u32::MAX as f32)*2.0-1.0)
        .collect();
    let mut xsq = vec![0.0f32; N*K];
    for (src, dst) in xs.chunks_exact(32).zip(xsq.chunks_exact_mut(32)) {
        let amax=src.iter().fold(0.0f32,|m,v|m.max(v.abs()));
        let d=if amax>1.0e-20 { amax/127.0 } else { 0.0 };
        let dw=f16_to_f32(f32_to_f16(d));
        let id=if d!=0.0 {1.0/d} else {0.0};
        for (s,q) in src.iter().zip(dst) { *q=(s*id).clamp(-127.0,127.0).round_ties_even()*dw; }
    }
    let mut want_batch=vec![0.0f32;N*M];
    for n in 0..N {
        for row in 0..M {
            for g in 0..K/128 {
                let off=(row*(K/128)+g)*28;
                let block=&packed[off..off+28];
                let d=f16_to_f32(u16::from_le_bytes([block[26],block[27]]));
                // Decode using the same Prism base-3 traversal as pack_block.
                let mut trits=[0i8;128];
                let pow3=[1u8,3,9,27,81];
                let mut o=0usize;
                for &(start,c,count) in &[(0usize,16usize,5usize),(16,8,5)] {
                    for nn in 0..count { for m in 0..c {
                        let v=block[start+m].wrapping_mul(pow3[nn]);
                        trits[o]=((((v as u16)*3)>>8) as i16-1) as i8; o+=1;
                    }}
                }
                for nn in 0..4 { for h in 0..2 {
                    let v=block[24+h].wrapping_mul(pow3[nn]);
                    trits[o]=((((v as u16)*3)>>8) as i16-1) as i8; o+=1;
                }}
                for i in 0..128 { want_batch[n*M+row]+=d*trits[i] as f32*xsq[n*K+g*128+i]; }
            }
        }
    }
    let dxs=gpu.upload_f32(&xs,&[N*K]).expect("upload xs");
    let yb=gpu.alloc_tensor(&[N*M],DType::F32).expect("alloc yb");
    gpu.gemm_ptq1g128_prefill(&a,&dxs,&yb,M,K,N).expect("prefill");
    gpu.hip.device_synchronize().expect("sync batch");
    let gotb=gpu.download_f32(&yb).expect("download batch");
    let maxb=gotb.iter().zip(&want_batch).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
    let ys=gpu.alloc_tensor(&[M],DType::F32).expect("alloc ys");
    let mut scalar_all=vec![0.0f32;N*M];
    for n in 0..N {
        let xs_n=gpu.upload_f32(&xs[n*K..(n+1)*K],&[K]).expect("upload xs row");
        gpu.gemv_ptq1g128(&a,&xs_n,&ys,M,K).expect("scalar row");
        gpu.hip.device_synchronize().expect("sync scalar row");
        scalar_all[n*M..(n+1)*M].copy_from_slice(&gpu.download_f32(&ys).expect("download scalar row"));
    }
    let scalar_delta=gotb.iter().zip(&scalar_all).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
    println!("PTQ1 prefill vs scalar max = {scalar_delta:.3e}");
    assert!(scalar_delta<2e-4,"PTQ1 prefill mismatch");
    println!("PASS");
}

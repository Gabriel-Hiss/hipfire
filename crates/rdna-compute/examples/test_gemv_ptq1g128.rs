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
    // Hypothesis: every passing WMMA test above used ONE dw for all rows (the
    // mix test a single scale, the identity/pair tests 1.0), so a per-row dw
    // indexing bug would be invisible in all of them. Vary dw per row with
    // everything else controlled, and pair it with an equal-dw control.
    {
        const DM: usize = 16;
        const DK: usize = 128;
        const DN: usize = 16;
        for &(name, vary_dw) in &[("dw equal   ", false), ("dw per-row ", true)] {
            let mut pk = Vec::new();
            for r in 0..DM {
                let mut trits = [0i8; 128];
                for (e, t) in trits.iter_mut().enumerate() {
                    *t = match e % 3 {
                        0 => 1,
                        1 => -1,
                        _ => 0,
                    };
                }
                let d = if vary_dw { 0.1 + r as f32 * 0.05 } else { 0.5 };
                pk.extend_from_slice(&pack_block(&trits, d));
            }
            let a = gpu.upload_raw(&pk, &[pk.len()]).expect("upload dw a");
            let x: Vec<f32> = (0..DN * DK)
                .map(|_| (next() as f32 / u32::MAX as f32) * 2.0 - 1.0)
                .collect();
            let dx = gpu.upload_f32(&x, &[DN * DK]).expect("upload dw x");
            let yw = gpu.alloc_tensor(&[DN * DM], DType::F32).expect("alloc dw yw");
            let ys = gpu.alloc_tensor(&[DN * DM], DType::F32).expect("alloc dw ys");
            gpu.gemm_ptq1g128_wmma(&a, &dx, &yw, DM, DK, DN).expect("dw wmma");
            gpu.gemm_ptq1g128_prefill(&a, &dx, &ys, DM, DK, DN).expect("dw scalar");
            gpu.hip.device_synchronize().expect("sync dw");
            let gw = gpu.download_f32(&yw).expect("dl dw wmma");
            let gs = gpu.download_f32(&ys).expect("dl dw scalar");
            let d = gw
                .iter()
                .zip(&gs)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("  {name}: wmma-vs-scalar max = {d:.3e}");
            if vary_dw && d >= 2e-4 {
                // C[m][n] = dw[m] * (column-only factor), so W/S per output row
                // reveals which row's dw the kernel actually applied.
                println!("    out_row : W[0]      S[0]      ratio   expected dw");
                for m in 0..DM {
                    let w = gw[m];
                    let s = gs[m];
                    let ratio = if s.abs() > 1e-9 { w / s } else { f32::NAN };
                    println!(
                        "      {m:2}    : {w:8.4}  {s:8.4}  {ratio:6.3}   {:.3}",
                        0.1 + m as f32 * 0.05
                    );
                }
            }
        }
    }

    // End-to-end WMMA vs scalar at realistic shapes, with random trits and
    // random per-row/per-group scales -- the configuration that exposed the
    // output-column dw bug.
    // N > 32 takes the 64x64 workgroup tile; M and N off the tile grid cover
    // its row and token edges.
    for &(name, m, k, n) in &[
        ("wmma small", 16usize, 128usize, 16usize),
        ("wmma wide ", 257, 512, 17),
        ("wmma t64  ", 257, 512, 97),
        ("wmma t64 w", 1030, 5120, 200),
    ] {
        let mut pk = Vec::with_capacity(m * (k / 128) * 28);
        for _r in 0..m {
            for _g in 0..k / 128 {
                let d = 0.05 + (next() % 100) as f32 / 100.0;
                let mut trits = [0i8; 128];
                for t in &mut trits {
                    *t = (next() % 3) as i8 - 1;
                }
                pk.extend_from_slice(&pack_block(&trits, d));
            }
        }
        let a = gpu.upload_raw(&pk, &[pk.len()]).expect("upload e2e a");
        let x: Vec<f32> = (0..n * k)
            .map(|_| (next() as f32 / u32::MAX as f32) * 2.0 - 1.0)
            .collect();
        let dx = gpu.upload_f32(&x, &[n * k]).expect("upload e2e x");
        let yw = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc e2e yw");
        let ys = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc e2e ys");
        gpu.gemm_ptq1g128_wmma(&a, &dx, &yw, m, k, n)
            .expect("e2e wmma");
        gpu.gemm_ptq1g128_prefill(&a, &dx, &ys, m, k, n)
            .expect("e2e scalar");
        gpu.hip.device_synchronize().expect("sync e2e");
        let gw = gpu.download_f32(&yw).expect("dl e2e wmma");
        let gs = gpu.download_f32(&ys).expect("dl e2e scalar");
        let d = gw
            .iter()
            .zip(&gs)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("  {name} {m}x{k}x{n}: wmma-vs-scalar max = {d:.3e}");
        assert!(d < 2e-4, "{name} end-to-end mismatch {d:.3e}");
    }

    // Kernel-level timing at a realistic prefill shape. If the WMMA only
    // replaces the multiply, and the multiply is not the bottleneck, the
    // kernel-level speedup will be ~1x even though the matrix unit is 33x
    // faster than the vector integer multiply.
    {
        const BM: usize = 5120;
        const BK: usize = 5120;
        const BN: usize = 64;
        let mut pk = Vec::with_capacity(BM * (BK / 128) * 28);
        for _r in 0..BM {
            for _g in 0..BK / 128 {
                let mut trits = [0i8; 128];
                for t in &mut trits {
                    *t = (next() % 3) as i8 - 1;
                }
                pk.extend_from_slice(&pack_block(&trits, 0.5));
            }
        }
        let a = gpu.upload_raw(&pk, &[pk.len()]).expect("upload bench a");
        let x: Vec<f32> = (0..BN * BK)
            .map(|_| (next() as f32 / u32::MAX as f32) * 2.0 - 1.0)
            .collect();
        let dx = gpu.upload_f32(&x, &[BN * BK]).expect("upload bench x");
        let yw = gpu.alloc_tensor(&[BN * BM], DType::F32).expect("alloc bench yw");
        let ys = gpu.alloc_tensor(&[BN * BM], DType::F32).expect("alloc bench ys");
        const ITERS: usize = 20;
        for _ in 0..3 {
            gpu.gemm_ptq1g128_wmma(&a, &dx, &yw, BM, BK, BN).unwrap();
            gpu.gemm_ptq1g128_prefill(&a, &dx, &ys, BM, BK, BN).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            gpu.gemm_ptq1g128_wmma(&a, &dx, &yw, BM, BK, BN).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let tw = t0.elapsed().as_secs_f64() / ITERS as f64;
        let t1 = std::time::Instant::now();
        for _ in 0..ITERS {
            gpu.gemm_ptq1g128_prefill(&a, &dx, &ys, BM, BK, BN).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let ts = t1.elapsed().as_secs_f64() / ITERS as f64;
        let macs = (BM * BK * BN) as f64;
        let gwb = gpu.download_f32(&yw).expect("dl bench wmma");
        let gsb = gpu.download_f32(&ys).expect("dl bench scalar");
        let dmax = gwb
            .iter()
            .zip(&gsb)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("  bench-shape correctness: wmma-vs-scalar max = {dmax:.3e}");
        println!(
            "  kernel {BM}x{BK}x{BN}: scalar {:.3} ms ({:.2}e12 MAC/s)  wmma {:.3} ms ({:.2}e12 MAC/s)  speedup {:.2}x",
            ts * 1e3,
            macs / ts / 1e12,
            tw * 1e3,
            macs / tw / 1e12,
            ts / tw
        );
    }

    // Achieved DRAM bandwidth of the decode GEMV at the model's real shapes
    // (Ternary-Bonsai-2-27B: 5120 hidden, 17408 FFN, 10240 qkv, 6144 z/out,
    // 248320 lm_head), plus a 131072-row shape large enough that the
    // per-launch activation quantization is a small fraction of the time.
    // The model streams 5.6 GB per token, so no weight is ever resident in the
    // 64 MB Infinity Cache; the loop cycles through enough copies (>= 512 MB)
    // that a small shape cannot be served from it either.
    for &(gm, gk) in &[
        (131072usize, 5120usize),
        (17408, 5120),
        (5120, 17408),
        (10240, 5120),
        (6144, 5120),
        (5120, 6144),
        (248320, 5120),
    ] {
        let groups = gk / 128;
        let mut pk = vec![0u8; gm * groups * 28];
        for blk in pk.chunks_exact_mut(28) {
            let mut trits = [0i8; 128];
            for t in &mut trits {
                *t = (next() % 3) as i8 - 1;
            }
            blk.copy_from_slice(&pack_block(&trits, 0.5));
        }
        let copies = (512usize << 20).div_ceil(pk.len()).max(1);
        let a: Vec<_> = (0..copies)
            .map(|_| gpu.upload_raw(&pk, &[pk.len()]).expect("upload bw a"))
            .collect();
        let x: Vec<f32> = (0..gk)
            .map(|_| (next() as f32 / u32::MAX as f32) * 2.0 - 1.0)
            .collect();
        let dx = gpu.upload_f32(&x, &[gk]).expect("upload bw x");
        let y1 = gpu.alloc_tensor(&[gm], DType::F32).expect("alloc bw y");
        const GI: usize = 20;
        for c in 0..3 {
            gpu.gemv_ptq1g128(&a[c % copies], &dx, &y1, gm, gk).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for c in 0..GI {
            gpu.gemv_ptq1g128(&a[c % copies], &dx, &y1, gm, gk).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let ms = t0.elapsed().as_secs_f64() / GI as f64 * 1e3;
        let wbytes = (gm * groups * 28) as f64;
        println!(
            "  decode gemv bw {gm}x{gk}: {ms:.3} ms for {:.1} MB weights = {:.1} GB/s",
            wbytes / 1e6,
            wbytes / (ms * 1e-3) / 1e9
        );
        for t in a {
            gpu.free_tensor(t).unwrap();
        }
        gpu.free_tensor(dx).unwrap();
        gpu.free_tensor(y1).unwrap();
    }
    {
        // Reference: what this card can actually read. 1 GiB is 16x the
        // Infinity Cache, so this is DRAM, not MALL. The decode ceiling is
        // stated as a fraction of this, not of the spec sheet.
        const GI: usize = 20;
        let n4 = (1usize << 30) / 16;
        let src: Vec<f32> = vec![0.0f32; n4 * 4];
        let dsrc = gpu.upload_f32(&src, &[n4 * 4]).expect("upload bw src");
        let dout = gpu.alloc_tensor(&[4], DType::F32).expect("alloc bw out");
        for _ in 0..3 {
            gpu.probe_dram_bw(&dsrc, &dout, n4).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let t2 = std::time::Instant::now();
        for _ in 0..GI {
            gpu.probe_dram_bw(&dsrc, &dout, n4).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let ms2 = t2.elapsed().as_secs_f64() / GI as f64 * 1e3;
        let b2 = (n4 * 16) as f64;
        println!(
            "  peak read bw ({:.1} MB buffer): {ms2:.3} ms = {:.1} GB/s",
            b2 / 1e6,
            b2 / (ms2 * 1e-3) / 1e9
        );
    }

    // iu4 vs iu8 WMMA throughput. The ternary INT4 route only pays if iu4 is
    // more than 2x iu8, because an 8-bit activation needs two iu4 MMAs to
    // reproduce one iu8 dot exactly.
    {
        let out = gpu.alloc_tensor(&[8], DType::F32).expect("alloc rate out");
        const REPS: usize = 200;
        let mut rates = Vec::new();
        for which in ["probe_wmma_iu8_rate", "probe_wmma_iu4_rate"] {
            for _ in 0..3 {
                gpu.probe_wmma_rate(which, &out, 3).unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..REPS {
                gpu.probe_wmma_rate(which, &out, 3).unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            let ms = t0.elapsed().as_secs_f64() / REPS as f64 * 1e3;
            // 8 independent chains x 2048 iterations x 4096 MACs, one wave.
            let macs = 8.0 * 2048.0 * 4096.0;
            rates.push(macs / (ms * 1e-3));
            println!("  {which}: {ms:.4} ms  ->  {:.2}e9 MMA-MAC/s per wave", macs / (ms * 1e-3) / 1e9);
        }
        println!(
            "  iu4 / iu8 throughput ratio: {:.3}x",
            rates[1] / rates[0]
        );
    }

    println!("PASS");
}

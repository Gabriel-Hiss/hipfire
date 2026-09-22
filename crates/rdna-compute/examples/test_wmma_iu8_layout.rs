// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Throwaway channel test: verify the wave32 `v_wmma_i32_16x16x16_iu8` fragment
// layout before building a prefill GEMM on it. Compares the kernel's output
// against the CPU int8 product of the same two 16x16 matrices.
//
// cargo run --release -p rdna-compute --features lab --example test_wmma_iu8_layout

use rdna_compute::{DType, Gpu};

const N: usize = 16;

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("arch: {}", gpu.arch);

    // Small signed values so the int32 accumulator cannot overflow: |sum| <= 16*9*9.
    let mut state = 0x1234_5678u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state % 19) as i32 - 9
    };
    let a: Vec<i8> = (0..N * N).map(|_| next() as i8).collect();
    let b: Vec<i8> = (0..N * N).map(|_| next() as i8).collect();

    // CPU reference: C[m][n] = sum_k A[m][k] * B[k][n].
    let mut want = vec![0i32; N * N];
    for m in 0..N {
        for n in 0..N {
            let mut acc = 0i32;
            for k in 0..N {
                acc += a[m * N + k] as i32 * b[k * N + n] as i32;
            }
            want[m * N + n] = acc;
        }
    }

    // upload_raw takes bytes; the int32 output is carried in an F32-sized
    // allocation and reinterpreted through to_bits().
    let a_bytes: Vec<u8> = a.iter().map(|&v| v as u8).collect();
    let b_bytes: Vec<u8> = b.iter().map(|&v| v as u8).collect();
    let da = gpu.upload_raw(&a_bytes, &[N * N]).expect("upload a");
    let db = gpu.upload_raw(&b_bytes, &[N * N]).expect("upload b");
    let dc = gpu.alloc_tensor(&[N * N], DType::F32).expect("alloc c");
    gpu.probe_wmma_iu8(&da, &db, &dc).expect("probe");
    gpu.hip.device_synchronize().expect("sync");

    let got_f = gpu.download_f32(&dc).expect("download c");
    let got: Vec<i32> = got_f.iter().map(|f| f.to_bits() as i32).collect();

    let mismatches = got.iter().zip(&want).filter(|(g, w)| g != w).count();
    println!("mismatches: {mismatches} / {}", N * N);
    if mismatches > 0 {
        println!("got  [0..8] = {:?}", &got[0..8]);
        println!("want [0..8] = {:?}", &want[0..8]);
        println!("got  [8..16]= {:?}", &got[8..16]);
        println!("want [8..16]= {:?}", &want[8..16]);
    }
    assert_eq!(mismatches, 0, "iu8 WMMA fragment layout assumption is wrong");

    let _ = gpu.free_tensor(da);
    let _ = gpu.free_tensor(db);
    let _ = gpu.free_tensor(dc);
    println!("PASS");
}

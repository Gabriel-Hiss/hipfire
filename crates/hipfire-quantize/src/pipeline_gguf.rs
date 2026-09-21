// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    non_snake_case,
    clippy::all
)]

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use crate::calibration::*;
use crate::cli::{guard_qwen3_arch_override, QuantizeArgs};
use crate::dequant::*;
use crate::e8;
use crate::e8_gptq;
use crate::gguf_input;
use crate::hfq::*;
use crate::model_filter::*;
use crate::quant_e8::*;
use crate::quant_fwht::*;
use crate::quant_hfp4::*;
use crate::quant_mq::*;
use crate::quant_q4::*;
use crate::reap_overlay;
use clap::Parser;
use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32, f32_to_f16};
use hipfire_quantize::hessian_io;
use hipfire_quantize::safetensors_file::{SafetensorsFile, TensorMeta};

/// 2D-weight quantization target chosen at the per-tensor level. The choice
/// per format flag:
///
/// | --format | 2D weights      | embedding | comment                          |
/// |----------|-----------------|-----------|----------------------------------|
/// | hfq4     | HFQ4G256        | Q8F16     | dense default — no FWHT, plain   |
/// | hfq6     | HFQ6G256        | Q8F16     | dense + higher quality           |
/// | mq4      | MQ4G256         | Q8F16     | Qwen3.5+ (DeltaNet) — FWHT-rot   |
/// | mq5      | MQ5G256         | Q8F16     | 5-bit FWHT (5.25 bpw, 168 B/grp) |
/// | mq6      | MQ6G256         | Q8F16     | Qwen3.5+ (DeltaNet) + higher q   |
/// | mq3      | MQ3G256         | Q8F16     | Sub-4-bit FWHT (3.25 bpw)        |
/// | mq2      | MQ2G256         | Q8F16     | Sub-4-bit FWHT (2.25 bpw)        |
///
/// **MQ4/MQ6 for non-Qwen3.5 dense produces correct output on the Llama path
/// (the rotation cancels via `gemv_mq4g256_with_rotate`) but adds per-layer
/// `rotate_x_mq` overhead with no quality benefit — those rotations were
/// calibrated for Qwen3.5+ training.** Default is HFQ4 for dense GGUFs;
/// pass `--format mq4` only when the source is a Qwen3.5+ family model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GgufFormat {
    Hfq4,
    Hfq6,
    Mq4,
    Mq4V2,
    Mq4C,
    Mq5,
    Mq6,
    Mq3,
    Mq2,
    Mq2Lloyd,
    Mq2LloydAnchored,
    Mq3Lloyd,
    Mq4Lloyd,
    Mq6V2,
    Mq5V2,
    Mq3V2,
    Mq2V2,
    Hfp4,      // HFP4G32 — RDNA-optimal FP4 (E2M1 + UE8M0 g32 + FP16 row scale)
    Mfp4,      // MFP4G32 — HFP4G32 + offline FWHT rotation (drop-in MQ4 replacement)
    Mfp4Lloyd, // mfp4 + per-tensor 16-entry Lloyd codebook
    Mfp4P,     // mfp4+P — mfp4 with E4M3 (non-power-of-2) per-block scale
    Mfp4E8, // mfp4-E8 — mfp4+P container with E8-lattice vector quantization (4 codewords/32 weights)
    Mfp4E8Soa, // mfp4-E8 SoA — same E8 data in structure-of-arrays layout for coalesced GEMV
    Mfp3E8, // mfp3-E8 — mfp4-E8 frame with 3-bit lattice (13 B/blk, 3.25 bpw; drop-in for MQ3-Lloyd cold)
    Mfp2E8, // mfp2-E8 — mfp4-E8 frame with 2-bit lattice (9 B/blk, 2.25 bpw; drop-in for MQ2-Lloyd cold)
    /// Ternary — PrismML Bonsai family. Source GGUF is already mixed-precision
    /// (Q2_0 ternary matmuls + Q8_0/F16 embeddings + F32/F16 norms); the
    /// per-tensor precision PrismML chose is authoritative, so 2D matmul
    /// tensors already in Q2_0 pass through byte-verbatim to TQ2G128
    /// (see the dedicated arm in `run_gguf_pipeline`) rather than going
    /// through the kmap/Q8 rules. `quantize_tq2g128` is only the re-quant
    /// fallback for the rare non-Q2_0 matmul tensor under this format.
    Ternary,
    /// Binary — PrismML Bonsai family, 1-bit variant. Source GGUF is already
    /// mixed-precision (Q1_0 binary matmuls + Q8_0/F16 embeddings + F32/F16
    /// norms); the per-tensor precision PrismML chose is authoritative, so 2D
    /// matmul tensors already in Q1_0 pass through byte-verbatim to BQ1G128
    /// (see the dedicated arm in `run_gguf_pipeline`) rather than going
    /// through the kmap/Q8 rules. There is no re-quant fallback for this
    /// format — binary weights always take the byte-verbatim passthrough.
    Binary,
}
impl GgufFormat {
    pub(crate) fn from_flag(flag: &str) -> Option<Self> {
        match flag {
            "hfq4" | "hfq4g256" | "hf4" => Some(Self::Hfq4),
            "hfq6" | "hfq6g256" | "hf6" => Some(Self::Hfq6),
            "mq4v1" | "mq4g256" | "magnum" => Some(Self::Mq4),
            "mq4v2" | "mq4" | "mq4g256v2" => Some(Self::Mq4V2),
            "mq4c" | "mq4cg256" | "mq4g256c" => Some(Self::Mq4C),
            "mq5" | "mq5g256" => Some(Self::Mq5),
            "mq6" | "mq6g256" => Some(Self::Mq6),
            "mq3" | "mq3g256" => Some(Self::Mq3),
            "mq2" | "mq2g256" => Some(Self::Mq2),
            "mq6v2" | "mq6g256v2" => Some(Self::Mq6V2),
            "mq5v2" | "mq5g256v2" => Some(Self::Mq5V2),
            "mq3v2" | "mq3g256v2" => Some(Self::Mq3V2),
            "mq2v2" | "mq2g256v2" => Some(Self::Mq2V2),
            "mq2-lloyd" | "mq2g256-lloyd" | "mq2lloyd" => Some(Self::Mq2Lloyd),
            "mq2lloyd-anchored"
            | "mq2lloyd_anchored"
            | "mq2-lloyd-anchored"
            | "mq2-lloyd_anchored"
            | "mq2g256-lloyd-anchored"
            | "mq2g256-lloyd_anchored" => Some(Self::Mq2LloydAnchored),
            "mq3-lloyd" | "mq3g256-lloyd" | "mq3lloyd" => Some(Self::Mq3Lloyd),
            "mq4-lloyd" | "mq4g256-lloyd" | "mq4lloyd" => Some(Self::Mq4Lloyd),
            "hfp4" | "hfp4g32" | "hf4p" | "fp4" => Some(Self::Hfp4),
            "mfp4" | "mfp4g32" | "mf4p" => Some(Self::Mfp4),
            "mfp4l" | "mfp4-lloyd" | "mfp4g32-lloyd" | "mfp4lloyd" => Some(Self::Mfp4Lloyd),
            "mfp4p" | "mfp4+p" | "mfp4-p" => Some(Self::Mfp4P),
            "mfp4e8" | "mfp4-e8" | "mfp4l8" => Some(Self::Mfp4E8),
            "mfp4e8soa" | "mfp4-e8-soa" | "mfp4e8-soa" => Some(Self::Mfp4E8Soa),
            "mfp3e8" | "mfp3-e8" => Some(Self::Mfp3E8),
            "mfp2e8" | "mfp2-e8" => Some(Self::Mfp2E8),
            "ternary" | "tq2" | "tq2g128" => Some(Self::Ternary),
            "binary" | "bq1" | "bq1g128" => Some(Self::Binary),
            _ => None,
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Hfq4 => "HFQ4G256",
            Self::Hfq6 => "HFQ6G256",
            Self::Mq4 => "MQ4G256",
            Self::Mq4V2 => "MQ4G256V2",
            Self::Mq4C => "MQ4CG256",
            Self::Mq5 => "MQ5G256",
            Self::Mq6 => "MQ6G256",
            Self::Mq3 => "MQ3G256",
            Self::Mq2 => "MQ2G256",
            Self::Mq6V2 => "MQ6G256V2",
            Self::Mq5V2 => "MQ5G256V2",
            Self::Mq3V2 => "MQ3G256V2",
            Self::Mq2V2 => "MQ2G256V2",
            Self::Mq2Lloyd => "MQ2G256Lloyd",
            Self::Mq2LloydAnchored => "MQ2G256Lloyd",
            Self::Mq3Lloyd => "MQ3G256Lloyd",
            Self::Mq4Lloyd => "MQ4G256Lloyd",
            Self::Hfp4 => "HFP4G32",
            Self::Mfp4 => "MFP4G32",
            Self::Mfp4Lloyd => "MFP4G32Lloyd",
            Self::Mfp4P => "MFP4G32P",
            Self::Mfp4E8 => "MFP4G32E8",
            Self::Mfp4E8Soa => "MFP4G32E8SOA",
            Self::Mfp3E8 => "MFP3G32E8",
            Self::Mfp2E8 => "MFP2G32E8",
            Self::Ternary => "TQ2G128",
            Self::Binary => "BQ1G128",
        }
    }
}
/// True for MQ{2,3,5,6}V2 — dense-only formats (gfx1201 Qwen3.8).
/// MoE GGUFs must not select these: expert packs are often non-2D and would
/// silently fall through the GGUF `!is_2d` arm to F16.
pub(crate) fn gguf_format_is_dense_only_mq_v2(format: GgufFormat) -> bool {
    matches!(
        format,
        GgufFormat::Mq6V2 | GgufFormat::Mq5V2 | GgufFormat::Mq3V2 | GgufFormat::Mq2V2
    )
}

/// MoE / MoE-like arch ids that reject dense-only MQ V2 formats on the GGUF path.
/// Mirrors `is_moe_like` in `pipeline.rs` (qwen MoE, deepseek4, minimax, lfm2,
/// cohere2moe, gemma4).
pub(crate) fn gguf_arch_is_moe_like(arch_id: u32) -> bool {
    matches!(arch_id, 6 | 9 | 10 | 11 | 12 | 13)
}

/// Q1_0 on-disk layout == hipfire BQ1G128 layout: copy verbatim.
pub(crate) fn convert_binary_tensor(
    src: &[u8],
    dtype: gguf_input::GgmlType,
) -> (Vec<u8>, crate::hfq::QuantType, u32) {
    debug_assert_eq!(dtype, gguf_input::GgmlType::Q1_0);
    (src.to_vec(), crate::hfq::QuantType::BQ1G128, 128)
}

fn meta_string_array(gguf: &gguf_input::GgufFile, key: &str) -> Vec<String> {
    match gguf.metadata.get(key) {
        Some(gguf_input::MetaValue::Array(values)) => values
            .iter()
            .filter_map(|value| value.as_str().map(str::to_owned))
            .collect(),
        _ => Vec::new(),
    }
}

fn meta_i32_array(gguf: &gguf_input::GgufFile, key: &str) -> Vec<i32> {
    match gguf.metadata.get(key) {
        Some(gguf_input::MetaValue::Array(values)) => values
            .iter()
            .filter_map(|value| match value {
                gguf_input::MetaValue::I32(v) => Some(*v),
                gguf_input::MetaValue::U32(v) => i32::try_from(*v).ok(),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn repack_ptq1_to_tq2(src: &[u8], n: usize) -> Vec<u8> {
    let values = gguf_input::tensor_to_f32(
        &gguf_input::TensorInfo {
            name: "ptq1-repack".into(),
            shape: vec![n],
            dtype: gguf_input::GgmlType::PTQ1_0,
            offset: 0,
        },
        src,
    );
    quantize_tq2g128(&values)
}

/// Rows of a Gated-DeltaNet tensor that carry V heads, as
/// `(first_row, rows_per_v_head)`, or `None` when the tensor has no V rows.
/// `ssm_out` is deliberately absent: its V axis is the input dimension and a
/// folded checkpoint leaves it in training order.
fn deltanet_v_row_span(
    gguf_name: &str,
    n_k: usize,
    n_v: usize,
    head_v: usize,
    head_k: usize,
    rows: usize,
) -> Option<(usize, usize)> {
    if n_k == 0 || n_v == 0 || n_v % n_k != 0 || n_v == n_k {
        return None;
    }
    let value_rows = n_v * head_v;
    let qk_rows = 2 * n_k * head_k;
    let span = if gguf_name.ends_with(".attn_qkv.weight") && rows == qk_rows + value_rows {
        (qk_rows, head_v)
    } else if gguf_name.ends_with(".attn_gate.weight") && rows == value_rows {
        (0, head_v)
    } else if gguf_name.ends_with(".ssm_conv1d.weight") && rows == qk_rows + value_rows {
        (qk_rows, head_v)
    } else if (gguf_name.ends_with(".ssm_alpha.weight")
        || gguf_name.ends_with(".ssm_beta.weight")
        || gguf_name.ends_with(".ssm_a")
        || gguf_name.ends_with(".ssm_dt.bias"))
        && rows == n_v
    {
        (0, 1)
    } else {
        return None;
    };
    Some(span)
}

/// Permute `[rep][k_head][head_rows]` back to `[k_head][rep][head_rows]` over
/// a row span. Rows are fixed-size records (a quantized weight row, an F32
/// conv1d channel, one scalar), so this is a byte move and the payload size is
/// unchanged.
fn untile_v_rows(
    data: Vec<u8>,
    rows: usize,
    first_row: usize,
    n_k: usize,
    rep: usize,
    head_rows: usize,
) -> std::io::Result<Vec<u8>> {
    if rows == 0 || data.len() % rows != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("DeltaNet V reorder: {} bytes do not divide {rows} rows", data.len()),
        ));
    }
    let row_bytes = data.len() / rows;
    let mut out = data.clone();
    for k in 0..n_k {
        for r in 0..rep {
            for h in 0..head_rows {
                let tiled = first_row + (r * n_k + k) * head_rows + h;
                let grouped = first_row + (k * rep + r) * head_rows + h;
                let src = tiled * row_bytes;
                let dst = grouped * row_bytes;
                out[dst..dst + row_bytes].copy_from_slice(&data[src..src + row_bytes]);
            }
        }
    }
    Ok(out)
}

/// Convert a GGUF file to a hipfire `.hfq`. Per-format quantization target
/// applies to 2D weight matrices; the embedding table is always Q8F16
/// (Q4-grade is too lossy for embeddings) and 1D norms stay F16. Tensor
/// names are translated GGUF → safetensors style so the engine's existing
/// `load_weights_hfq` can consume the output.
pub(crate) fn run_gguf_pipeline(
    input: &Path,
    output: &Path,
    format: GgufFormat,
    no_kmap: bool,
    kmap_dense: bool,
    kmap_mode: u8,
    arch_id_override: Option<u32>,
    force_arch_id: bool,
) -> std::io::Result<()> {
    eprintln!("=== GGUF → {} conversion ===", format.label());
    eprintln!("Input:  {}", input.display());
    eprintln!("Output: {}", output.display());

    let gguf = gguf_input::GgufFile::open(input)?;
    eprintln!("GGUF version: {}", gguf.version);
    eprintln!("Tensors: {}", gguf.tensors.len());

    let arch_str = gguf
        .meta_str("general.architecture")
        .unwrap_or("llama")
        .to_string();
    // Single source of truth lives in hipfire-runtime::arch_mapping. Fail-closed
    // on unknown architecture: error listing supported types, unless an explicit
    // --arch-id override is supplied. The old qwen3* pillar guard is subsumed
    // by this uniform check — any unknown qwen3* now fails with the same clear
    // error instead of a bespoke branch.
    let auto_arch_id: u32 = match hipfire_runtime::arch_mapping::lookup_model_type(
        arch_str.as_str(),
    ) {
        Some(id) => id,
        None => {
            if let Some(ov) = arch_id_override {
                eprintln!(
                    "warning: unknown GGUF architecture '{}' but --arch-id {} override supplied; proceeding with override",
                    arch_str, ov
                );
                ov
            } else {
                let supported = hipfire_runtime::arch_mapping::supported_model_types_display();
                eprintln!(
                    "error: unknown GGUF architecture '{}'; supported architectures are: [{}]. Hint: pass --arch-id <id> to override for this model",
                    arch_str, supported
                );
                std::process::exit(1);
            }
        }
    };
    // --arch-id <u32> overrides the auto-detected id. Use when the
    // model's family maps to a different crate than the default
    // (e.g. plain Qwen2 → arch_id=7 for the hipfire-arch-qwen2 crate
    // instead of the LLaMA-family default 1, which silently drops
    // Q/K/V bias on the LLaMA loader path). See docs/plans/
    // dots-ocr-devlog.md §7 (R1) for the bring-up context.
    let arch_id: u32 = arch_id_override.unwrap_or(auto_arch_id);
    guard_qwen3_arch_override(auto_arch_id, arch_id, force_arch_id);
    if arch_id != auto_arch_id {
        eprintln!(
            "Architecture: {arch_str} (auto id={auto_arch_id}, overridden via --arch-id to {arch_id})"
        );
    } else {
        eprintln!("Architecture: {arch_str} (id={arch_id})");
    }

    // Metadata JSON: must populate `config.*` so engine's `config_from_hfq`
    // can reconstruct LlamaConfig at load time. Also keep the raw GGUF
    // metadata tree under `gguf_meta` for any consumer that wants original
    // values (chat template, vocab, scores, merges, etc.).
    // Strict validation before worker threads — same parser as CLI.
    crate::model_filter::validate_env_fixed_tier_or_exit();
    let config_json = config_json_from_gguf(&gguf, &arch_str, arch_id);
    let metadata = serde_json::json!({
        "architecture": arch_str,
        "source": "gguf",
        "config": config_json,
        "gguf_meta": gguf_meta_to_json(&gguf.metadata),
    });
    let metadata_json = serde_json::to_string(&metadata)?;
    let prism_version = gguf.meta_u32("prism.hadamard.version").unwrap_or(0);
    let prism_block_size = gguf.meta_u32("prism.hadamard.block_size").unwrap_or(0) as usize;
    let prism_weights: HashSet<String> = meta_string_array(&gguf, "prism.hadamard.weight_names")
        .into_iter()
        .collect();
    let prism_inverse: HashSet<String> =
        meta_string_array(&gguf, "prism.hadamard.inverse_weight_names")
            .into_iter()
            .collect();
    let mut prism_signs = HashMap::<usize, Vec<i32>>::new();
    if prism_version != 0 {
        if prism_version != 1 || prism_block_size == 0 || !prism_block_size.is_power_of_two() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "unsupported Prism Hadamard contract: version={prism_version}, block_size={prism_block_size}"
                ),
            ));
        }
        let widths = meta_i32_array(&gguf, "prism.hadamard.sign_widths");
        let values = meta_i32_array(&gguf, "prism.hadamard.sign_values");
        let mut offset = 0usize;
        for width in widths {
            let width = usize::try_from(width).map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "negative Prism sign width")
            })?;
            let end = offset.checked_add(width).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "Prism sign width overflow")
            })?;
            if width % prism_block_size != 0 || end > values.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("invalid Prism sign vector width {width}"),
                ));
            }
            prism_signs.insert(width, values[offset..end].to_vec());
            offset = end;
        }
        if offset != values.len() || prism_weights.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "incomplete Prism Hadamard metadata",
            ));
        }
    }
    if prism_version != 0 && matches!(arch_id, 5 | 6) {
        // A folded Gated-DeltaNet checkpoint keeps `ssm_out` in the training
        // (grouped) V-head order only when the converter says so; without the
        // flag the columns are in llama.cpp's tiled order, which hipfire's
        // DeltaNet output does not produce and a packed ternary weight cannot
        // be re-permuted into.
        let grouped = matches!(
            gguf.metadata.get("prism.hadamard.gdn_v_grouped"),
            Some(gguf_input::MetaValue::Bool(true))
        );
        if !grouped {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "prism.hadamard.gdn_v_grouped is not set: ssm_out columns are in llama.cpp's tiled V-head order and cannot be repacked",
            ));
        }
    }
    if matches!(format, GgufFormat::Ternary)
        && gguf
            .metadata
            .keys()
            .any(|key| key.ends_with(".expert_count"))
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "MoE GGUFs are not supported on the ternary path: the per-expert tensor layout has no hipfire mapping",
        ));
    }
    // safetensors path so the engine's runtime FWHT inverse stays identical.
    let needs_signs = matches!(
        format,
        GgufFormat::Mq4
            | GgufFormat::Mq4V2
            | GgufFormat::Mq4C
            | GgufFormat::Mq6
            | GgufFormat::Mq6V2
            | GgufFormat::Mq5
            | GgufFormat::Mq5V2
            | GgufFormat::Mq3
            | GgufFormat::Mq3V2
            | GgufFormat::Mq2
            | GgufFormat::Mq2V2
            | GgufFormat::Mq2Lloyd
            | GgufFormat::Mq2LloydAnchored
            | GgufFormat::Mq3Lloyd
            | GgufFormat::Mq4Lloyd
            | GgufFormat::Mfp4
            | GgufFormat::Mfp4P
            | GgufFormat::Mfp4E8
            | GgufFormat::Mfp3E8
            | GgufFormat::Mfp2E8
    );
    let signs1 = if needs_signs {
        gen_fwht_signs(42, 256)
    } else {
        Vec::new()
    };
    let signs2 = if needs_signs {
        gen_fwht_signs(1042, 256)
    } else {
        Vec::new()
    };

    // K-map setup for GGUF path
    let is_moe = arch_id == 6;
    // MQ{2,3,5,6}V2 are dense-only. Reject every MoE GGUF before any tensor
    // work or output write so non-2D expert packs cannot silently become F16.
    if gguf_format_is_dense_only_mq_v2(format) && gguf_arch_is_moe_like(arch_id) {
        eprintln!(
            "error: --format mq{{2,3,5,6}}v2 is dense-only (gfx1201 Qwen3.8); MoE model (arch_id={arch_id}) is not supported with this format. Use legacy mq{{2,3,5,6}} or mq4/mq4v2/mq4c for MoE, or run on a dense checkpoint."
        );
        std::process::exit(2);
    }
    if format == GgufFormat::Mq2LloydAnchored && gguf_arch_is_moe_like(arch_id) {
        eprintln!(
            "error: --format mq2lloyd-anchored is dense-only (Qwen 3.8 Lloyd rescue); MoE model (arch_id={arch_id}) is not supported with this format. Use --format mq2lloyd for MoE routed experts or a dense checkpoint."
        );
        std::process::exit(2);
    }

    let is_gemma4 = arch_id == 13;
    let n_layers: usize = config_json
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    // Build K-map using translated (safetensors-style) names where available,
    // falling back to raw GGUF names for untranslated tensors.
    //
    // K-map is gated to MoE models only. On dense models the author's own
    // bench shows a mixed picture (PPL +1.5% to +2.5% at 2K context on 4B
    // and 27B; PPL -4.8% on 27B at 8K context — crossover at ~3K). The
    // ship-default is the conservative shape per maintainer directive
    // (2026-05-08): never silently change dense quantization. Users who
    // want K-map on dense pass `--kmap-dense` (see flag parsing below).
    // K-map is enabled for: MoE models (default), gemma4 (arch_id 13,
    // default mode=2), or any dense model with --kmap-dense.
    // Suppress with --no-kmap / --uniform.
    let kmap: HashMap<String, QuantLevel> = if no_kmap || (!is_moe && !is_gemma4 && !kmap_dense) {
        HashMap::new()
    } else {
        let mut map = HashMap::new();
        let mut counts = [0u32; 4];
        for info in &gguf.tensors {
            let out_name = gguf_to_safetensors_name(&info.name, arch_id)
                .unwrap_or_else(|| info.name.clone());
            let level = kmap_resolve_mode(&out_name, n_layers, is_moe, kmap_mode);
            match level {
                QuantLevel::F16 => counts[0] += 1,
                QuantLevel::Q8 => counts[1] += 1,
                QuantLevel::Promote6 => counts[2] += 1,
                QuantLevel::Override(_) => counts[3] += 1,
                QuantLevel::Base => counts[3] += 1,
            }
            map.insert(out_name, level);
        }
        if !map.is_empty() {
            let mode_label = match kmap_mode {
                0 => "full",
                1 => "alternating",
                2 => "typed",
                _ => "?",
            };
            eprintln!(
                "K-map plan ({} base, {n_layers} layers{}, mode={mode_label}):",
                format.label(),
                if is_moe { ", MoE" } else { "" }
            );
            eprintln!("  F16:       {:>4} tensors", counts[0]);
            eprintln!("  Q8:        {:>4} tensors", counts[1]);
            eprintln!("  Promote6:  {:>4} tensors", counts[2]);
            eprintln!("  Base:      {:>4} tensors", counts[3]);
        }
        map
    };

    let mut hfq_tensors: Vec<HfqTensor> = Vec::with_capacity(gguf.tensors.len());
    let mut total_params: u64 = 0;
    let mut quant_params: u64 = 0;
    let mut total_bytes_in: u64 = 0;
    let mut total_bytes_out: u64 = 0;

    for info in &gguf.tensors {
        let raw = gguf.tensor_data(info);
        let n_elements = info.numel();
        total_params += n_elements as u64;
        total_bytes_in += raw.len() as u64;

        let shape: Vec<u32> = info.shape.iter().map(|&s| s as u32).collect();

        // Tensor classification (uses the original GGUF name).
        let is_norm = gguf_is_norm_tensor(&info.name);
        let is_embed = gguf_is_embed_tensor(&info.name);
        let is_2d = info.shape.len() == 2;
        let k_dim = if is_2d { info.shape[0] } else { n_elements };

        // Translate to the safetensors-style name `hipfire_runtime::hfq::load_weights_hfq`
        // expects. If we don't have a translation, keep the original name —
        // the future loader can ignore unknown tensors.
        let out_name =
            gguf_to_safetensors_name(&info.name, arch_id).unwrap_or_else(|| info.name.clone());

        let kmap_level = kmap.get(&out_name).copied().unwrap_or(QuantLevel::Base);

        let source_passthrough = match info.dtype {
            gguf_input::GgmlType::F32 => Some((QuantType::F32, "F32 (passthrough)")),
            gguf_input::GgmlType::F16 => Some((QuantType::F16, "F16 (passthrough)")),
            gguf_input::GgmlType::BF16 => Some((QuantType::BF16, "BF16 (passthrough)")),
            _ => None,
        };
        // Two Qwen3.5 tensors are stored in llama.cpp's convention, not
        // HuggingFace's, and hipfire reads the HuggingFace one. Left alone they
        // are silent wrong-math, not load errors:
        //   * every `*_norm.weight` except `linear_attn.norm` ships as `w + 1`
        //     (conversion/qwen.py:394), and hipfire's norm loader adds 1 again;
        //   * `ssm_a` ships as `-exp(A_log)` (qwen35.cpp:459 multiplies it
        //     straight in), while hipfire's gate kernel computes `-exp(a_log)`
        //     itself.
        // Both stay F32, so undoing the convention costs no bytes.
        let convention_fix: Option<fn(f32) -> f32> = if matches!(arch_id, 5 | 6) {
            if info.name.ends_with(".ssm_a") {
                Some(|v: f32| (-v).max(f32::MIN_POSITIVE).ln())
            } else if is_norm && !info.name.contains("ssm_norm") {
                Some(|v: f32| v - 1.0)
            } else {
                None
            }
        } else {
            None
        };
        let (data, quant_type, group_size, label) = if let Some(fix) = convention_fix {
            let values: Vec<f32> = gguf_input::tensor_to_f32(info, raw)
                .into_iter()
                .map(fix)
                .collect();
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            (bytes, QuantType::F32, 0u32, "F32 (convention-adjusted)")
        } else if let (GgufFormat::Ternary, Some((qt, label))) = (format, source_passthrough) {
            // Bonsai ships mixed precision on purpose: the conv1d bank and the
            // DeltaNet alpha/beta gates are the precision the publisher chose.
            // Re-encoding them would make this a different model than the one
            // the GGUF describes.
            (raw.to_vec(), qt, 0u32, label)
        } else if is_norm || !is_2d || info.name.contains("conv1d") {
            // Norms, 1D tensors and the DeltaNet conv1d bank (K=4, no group
            // divides it) always F16.
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            let f16_bytes: Vec<u8> = f32_data
                .iter()
                .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                .collect();
            (f16_bytes, QuantType::F16, 0u32, "F16")
        } else if matches!(format, GgufFormat::Ternary)
            && matches!(
                info.dtype,
                gguf_input::GgmlType::Q2_0
                    | gguf_input::GgmlType::PQ2_0
                    | gguf_input::GgmlType::PTQ1_0
            )
        {
            let expected_block_bytes = if info.dtype == gguf_input::GgmlType::PTQ1_0 {
                28
            } else {
                34
            };
            let expected = n_elements.div_ceil(128) * expected_block_bytes;
            if raw.len() != expected {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "{}: ternary payload has {} bytes, expected {expected}",
                        info.name,
                        raw.len()
                    ),
                ));
            }
            let bytes = raw.to_vec();
            // The embedding table is folded too, but as `inverse-after-lookup`:
            // its rows stay latent on disk and the runtime un-folds the row it
            // reads. PTQ1 stays in the publisher's dense base-3 packing; it is
            // never expanded to the 2-bit-slot TQ2 container.
            let folded =
                prism_weights.contains(&info.name) || prism_inverse.contains(&info.name);
            if prism_version != 0 && !folded {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "{}: ternary tensor is in neither prism.hadamard.weight_names nor inverse_weight_names",
                        info.name
                    ),
                ));
            }
            let ptq1 = info.dtype == gguf_input::GgmlType::PTQ1_0;
            (
                bytes,
                if ptq1 {
                    QuantType::PTQ1G128H
                } else if folded {
                    QuantType::TQ2G128H
                } else {
                    QuantType::TQ2G128
                },
                128u32,
                if ptq1 {
                    "PTQ1G128H (Prism Hadamard)"
                } else if folded {
                    "TQ2G128H (Prism Hadamard)"
                } else {
                    "TQ2G128 (passthrough)"
                },
            )
        } else if kmap_level == QuantLevel::Q8 || is_embed {
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            let q = quantize_q8f16(&f32_data);
            quant_params += n_elements as u64;
            (q, QuantType::Q8F16, 32u32, "Q8_F16")
        } else if matches!(format, GgufFormat::Ternary) {
            // A Bonsai GGUF is mixed precision by construction: the matmuls the
            // publisher chose to ternarize are already ternary, and everything
            // else (DeltaNet alpha/beta gates here) is the precision they kept.
            // Re-quantizing those to ternary would be our loss, not theirs.
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            let q = quantize_q8f16(&f32_data);
            quant_params += n_elements as u64;
            (q, QuantType::Q8F16, 32u32, "Q8_F16")
        } else if crate::model_filter::product_tier_cli().is_some_and(|t| {
            crate::model_filter::q8_class_of(&out_name).is_some_and(|cls| t.lifts(cls))
        }) || crate::model_filter::fixed_tier_override_applies(&out_name)
        {
            // Product tier lift and/or explicit --fixed-tier / HIPFIRE_FIXED_TIER
            // entry. Codec overrides (e.g. attn_full:mq6v2) apply even when the
            // ProductTier does not lift that class; missing override => Q8.
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            quant_params += n_elements as u64;
            if let Some(dt) = crate::model_filter::fixed_tier_dtype_for(&out_name) {
                let m = info.shape[0] as usize;
                let k = info.shape[1] as usize;
                if k % 256 != 0 && matches!(dt, "mq2v2" | "mq3v2" | "mq4v2" | "mq5v2" | "mq6v2") {
                    eprintln!(
                        "error: fixed-tier dtype {dt} requires K%256==0 for {out_name} (K={k})"
                    );
                    std::process::exit(2);
                }
                match dt {
                    "mq6v2" => {
                        let q = quantize_mq6g256v2(&f32_data, m, k, &signs1, &signs2);
                        (q, QuantType::MQ6G256V2, 256u32, "MQ6G256V2")
                    }
                    "mq5v2" => {
                        let q = quantize_mq5g256v2(&f32_data, m, k, &signs1, &signs2);
                        (q, QuantType::MQ5G256V2, 256u32, "MQ5G256V2")
                    }
                    "mq4v2" => {
                        let q = quantize_mq4g256v2(&f32_data, m, k, &signs1, &signs2);
                        (q, QuantType::MQ4G256V2, 256u32, "MQ4G256V2")
                    }
                    "mq3v2" => {
                        let q = quantize_mq3g256v2(&f32_data, m, k, &signs1, &signs2);
                        (q, QuantType::MQ3G256V2, 256u32, "MQ3G256V2")
                    }
                    "mq2v2" => {
                        let q = quantize_mq2g256v2(&f32_data, m, k, &signs1, &signs2);
                        (q, QuantType::MQ2G256V2, 256u32, "MQ2G256V2")
                    }
                    "mq4" => {
                        let q = quantize_mq4g256(&f32_data, &signs1, &signs2);
                        (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                    }
                    "mq3l" => {
                        let q = quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2);
                        (q, QuantType::MQ3G256Lloyd, 256u32, "MQ3G256Lloyd")
                    }
                    "mfp4e8" => {
                        let q = quantize_mfp4g32_e8_2d(&f32_data, m, k, &signs1, &signs2);
                        (q, QuantType::MFP4G32E8, 32u32, "MFP4G32E8")
                    }
                    "mfp4e8soa" => {
                        let q = quantize_mfp4g32_e8_soa_2d(&f32_data, m, k, &signs1, &signs2);
                        (q, QuantType::MFP4G32E8SOA, 32u32, "MFP4G32E8SOA")
                    }
                    _ => {
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    }
                }
            } else {
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_F16")
            }
        } else if matches!(format, GgufFormat::Ternary) && info.dtype == gguf_input::GgmlType::Q2_0
        {
            // TQ2G128 byte-verbatim passthrough wins over the kmap/Q8 rules
            // when the source is already Q2_0 — PrismML's per-tensor format
            // layout is byte-identical between the GGUF Q2_0 block and hipfire TQ2G128
            let expected = (n_elements.div_ceil(128)) * 34;
            // Use exact check: n_elements already includes padding? For Q2_0 n_elements is logical count (raw.len()/34*128)
            // but info.numel() is logical; raw.len() should be n_blocks*34.
            if raw.len() != (n_elements.div_ceil(128) * 34) && raw.len() != (n_elements / 128 * 34)
            {
                // Fallback permissive check - just record
            }
            if raw.len() != expected && raw.len() != (n_elements / 128 * 34) {
                eprintln!(
                    "Q2_0 size mismatch for {}: got {} bytes, expected {}",
                    info.name,
                    raw.len(),
                    expected
                );
                std::process::exit(1);
            }
            (
                raw.to_vec(),
                QuantType::TQ2G128,
                128u32,
                "TQ2G128 (passthrough)",
            )
        } else if matches!(format, GgufFormat::Binary) && info.dtype == gguf_input::GgmlType::Q1_0 {
            // BQ1G128 byte-verbatim passthrough — mirrors the TQ2G128/Q2_0 arm
            // Q1_0 on-disk layout == hipfire BQ1G128 layout: copy verbatim.
            let expected = (n_elements.div_ceil(128)) * 18;
            if raw.len() != expected && raw.len() != (n_elements / 128 * 18) {
                eprintln!(
                    "Q1_0 size mismatch for {}: got {} bytes, expected {}",
                    info.name,
                    raw.len(),
                    expected
                );
                std::process::exit(1);
            }
            let (bytes, quant_type, group_size) =
                convert_binary_tensor(raw, gguf_input::GgmlType::Q1_0);
            (bytes, quant_type, group_size, "BQ1G128 (passthrough)")
        } else if kmap_level == QuantLevel::Promote6 && k_dim % 256 == 0 {
            // K-map promote to 6-bit
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            quant_params += n_elements as u64;
            match format {
                GgufFormat::Mq4
                | GgufFormat::Mq4V2
                | GgufFormat::Mq4C
                | GgufFormat::Mq3
                | GgufFormat::Mq2
                | GgufFormat::Mq2Lloyd
                | GgufFormat::Mq2LloydAnchored
                | GgufFormat::Mq3Lloyd
                | GgufFormat::Mq4Lloyd
                | GgufFormat::Mq5
                | GgufFormat::Mq6 => {
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                }
                GgufFormat::Mq6V2 | GgufFormat::Mq5V2 | GgufFormat::Mq3V2 | GgufFormat::Mq2V2 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mq6g256v2(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MQ6G256V2, 256u32, "MQ6G256V2")
                }
                GgufFormat::Hfq4 | GgufFormat::Hfq6 => {
                    let q = quantize_hfq6g256(&f32_data);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                }
                GgufFormat::Hfp4 => {
                    // No HFP6 variant in v1. Promote6 for HFP4 stays at HFP4G32 (4.25 bpw).
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_hfp4g32_2d(&f32_data, m, k);
                    (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                }
                GgufFormat::Mfp4 => {
                    // No MFP6 variant. Promote6 for MFP4 stays at MFP4G32 (4.25 bpw).
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp4g32_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                }
                GgufFormat::Mfp4Lloyd => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp4g32_lloyd_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32Lloyd, 32u32, "MFP4G32Lloyd")
                }
                GgufFormat::Mfp4P => {
                    // No MFP6 variant. Promote6 for mfp4+P stays at MFP4G32P (4.25 bpw).
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp4g32_p_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32P, 32u32, "MFP4G32P")
                }
                GgufFormat::Mfp4E8 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp4g32_e8_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32E8, 32u32, "MFP4G32E8")
                }
                GgufFormat::Mfp4E8Soa => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp4g32_e8_soa_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32E8SOA, 32u32, "MFP4G32E8SOA")
                }
                GgufFormat::Mfp3E8 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp3g32_e8_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP3G32E8, 32u32, "MFP3G32E8")
                }
                GgufFormat::Mfp2E8 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp2g32_e8_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP2G32E8, 32u32, "MFP2G32E8")
                }
                GgufFormat::Ternary => {
                    // Terminal low-bit format — promotion is a no-op.
                    // Direct TQ2G128 semantics: scale-only ternary g128, 34 B/blk (2.125 bpw).
                    let q = quantize_tq2g128(&f32_data);
                    (q, QuantType::TQ2G128, 128u32, "TQ2G128")
                }
                GgufFormat::Binary => {
                    // Terminal low-bit format — promotion is a no-op.
                    // Direct BQ1G128 semantics: scale-only binary g128, 18 B/blk (1.14 bpw).
                    let q = quantize_bq1g128(&f32_data);
                    (q, QuantType::BQ1G128, 128u32, "BQ1G128")
                }
            }
        } else if let (QuantLevel::Override(override_fmt), true) = (kmap_level, k_dim % 256 == 0) {
            // K-map says override (lm_head when --lm-head-format set).
            // GGUF pipeline has no AWQ wiring (AWQ is safetensors-only today),
            // so this is a plain quantize on the carried target format.
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            quant_params += n_elements as u64;
            match override_fmt {
                GgufFormat::Mq6 => {
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                }
                GgufFormat::Hfq6 => {
                    let q = quantize_hfq6g256(&f32_data);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                }
                GgufFormat::Mq4 => {
                    let q = quantize_mq4g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                }
                GgufFormat::Mq4V2 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mq4g256v2(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MQ4G256V2, 256u32, "MQ4G256V2")
                }
                GgufFormat::Mq4C => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mq4cg256(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MQ4CG256, 256u32, "MQ4CG256")
                }
                GgufFormat::Mq5 => {
                    let q = quantize_mq5g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ5G256, 256u32, "MQ5G256")
                }
                GgufFormat::Mq6V2 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mq6g256v2(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MQ6G256V2, 256u32, "MQ6G256V2")
                }
                GgufFormat::Mq5V2 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mq5g256v2(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MQ5G256V2, 256u32, "MQ5G256V2")
                }
                GgufFormat::Mq3V2 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mq3g256v2(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MQ3G256V2, 256u32, "MQ3G256V2")
                }
                GgufFormat::Mq2V2 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mq2g256v2(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MQ2G256V2, 256u32, "MQ2G256V2")
                }
                GgufFormat::Hfq4 => {
                    let q = quantize_hfq4g256(&f32_data);
                    (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                }
                GgufFormat::Mq3 => {
                    let q = quantize_mq3g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ3G256, 256u32, "MQ3G256")
                }
                GgufFormat::Mq2 => {
                    let q = quantize_mq2g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ2G256, 256u32, "MQ2G256")
                }
                GgufFormat::Mq2Lloyd => {
                    let q = quantize_mq2g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ2G256Lloyd, 256u32, "MQ2G256Lloyd")
                }
                GgufFormat::Mq2LloydAnchored => {
                    let q = quantize_mq2g256_lloyd_anchored(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ2G256Lloyd, 256u32, "MQ2G256Lloyd")
                }
                GgufFormat::Mq3Lloyd => {
                    let q = quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ3G256Lloyd, 256u32, "MQ3G256Lloyd")
                }
                GgufFormat::Mq4Lloyd => {
                    let q = quantize_mq4g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ4G256Lloyd, 256u32, "MQ4G256Lloyd")
                }
                GgufFormat::Hfp4 => {
                    let m = info.shape[0] as usize;
                    let q = quantize_hfp4g32_2d(&f32_data, m, k_dim);
                    (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                }
                GgufFormat::Mfp4 => {
                    let m = info.shape[0] as usize;
                    let q = quantize_mfp4g32_2d(&f32_data, m, k_dim, &signs1, &signs2);
                    (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                }
                GgufFormat::Mfp4Lloyd => {
                    let m = info.shape[0] as usize;
                    let q = quantize_mfp4g32_lloyd_2d(&f32_data, m, k_dim, &signs1, &signs2);
                    (q, QuantType::MFP4G32Lloyd, 32u32, "MFP4G32Lloyd")
                }
                GgufFormat::Mfp4P => {
                    let m = info.shape[0] as usize;
                    let q = quantize_mfp4g32_p_2d(&f32_data, m, k_dim, &signs1, &signs2);
                    (q, QuantType::MFP4G32P, 32u32, "MFP4G32P")
                }
                GgufFormat::Mfp4E8 => {
                    let m = info.shape[0] as usize;
                    let q = quantize_mfp4g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2);
                    (q, QuantType::MFP4G32E8, 32u32, "MFP4G32E8")
                }
                GgufFormat::Mfp4E8Soa => {
                    let m = info.shape[0] as usize;
                    let q = quantize_mfp4g32_e8_soa_2d(&f32_data, m, k_dim, &signs1, &signs2);
                    (q, QuantType::MFP4G32E8SOA, 32u32, "MFP4G32E8SOA")
                }
                GgufFormat::Mfp3E8 => {
                    let m = info.shape[0] as usize;
                    let q = quantize_mfp3g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2);
                    (q, QuantType::MFP3G32E8, 32u32, "MFP3G32E8")
                }
                GgufFormat::Mfp2E8 => {
                    let m = info.shape[0] as usize;
                    let q = quantize_mfp2g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2);
                    (q, QuantType::MFP2G32E8, 32u32, "MFP2G32E8")
                }
                GgufFormat::Ternary => {
                    // Terminal low-bit: direct TQ2G128 semantics (plain, no rotation).
                    let q = quantize_tq2g128(&f32_data);
                    (q, QuantType::TQ2G128, 128u32, "TQ2G128")
                }
                GgufFormat::Binary => {
                    // Terminal low-bit: direct BQ1G128 semantics (plain, no rotation).
                    let q = quantize_bq1g128(&f32_data);
                    (q, QuantType::BQ1G128, 128u32, "BQ1G128")
                }
            }
        } else if k_dim % 256 == 0 {
            // 256-aligned 2D weight — quantize per the chosen format (Base level).
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            quant_params += n_elements as u64;
            match format {
                GgufFormat::Hfq4 => {
                    let q = quantize_hfq4g256(&f32_data);
                    (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                }
                GgufFormat::Hfq6 => {
                    let q = quantize_hfq6g256(&f32_data);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                }
                GgufFormat::Mq4 => {
                    let q = quantize_mq4g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                }
                GgufFormat::Mq4V2 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mq4g256v2(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MQ4G256V2, 256u32, "MQ4G256V2")
                }
                GgufFormat::Mq4C => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mq4cg256(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MQ4CG256, 256u32, "MQ4CG256")
                }
                GgufFormat::Mq5 => {
                    let q = quantize_mq5g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ5G256, 256u32, "MQ5G256")
                }
                GgufFormat::Mq6 => {
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                }
                GgufFormat::Mq6V2 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mq6g256v2(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MQ6G256V2, 256u32, "MQ6G256V2")
                }
                GgufFormat::Mq5V2 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mq5g256v2(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MQ5G256V2, 256u32, "MQ5G256V2")
                }
                GgufFormat::Mq3V2 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mq3g256v2(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MQ3G256V2, 256u32, "MQ3G256V2")
                }
                GgufFormat::Mq2V2 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mq2g256v2(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MQ2G256V2, 256u32, "MQ2G256V2")
                }
                GgufFormat::Mq3 => {
                    let q = quantize_mq3g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ3G256, 256u32, "MQ3G256")
                }
                GgufFormat::Mq2 => {
                    let q = quantize_mq2g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ2G256, 256u32, "MQ2G256")
                }
                GgufFormat::Mq2Lloyd => {
                    let q = quantize_mq2g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ2G256Lloyd, 256u32, "MQ2G256Lloyd")
                }
                GgufFormat::Mq2LloydAnchored => {
                    let q = quantize_mq2g256_lloyd_anchored(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ2G256Lloyd, 256u32, "MQ2G256Lloyd")
                }
                GgufFormat::Mq3Lloyd => {
                    let q = quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ3G256Lloyd, 256u32, "MQ3G256Lloyd")
                }
                GgufFormat::Mq4Lloyd => {
                    let q = quantize_mq4g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ4G256Lloyd, 256u32, "MQ4G256Lloyd")
                }
                GgufFormat::Hfp4 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_hfp4g32_2d(&f32_data, m, k);
                    (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                }
                GgufFormat::Mfp4 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp4g32_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                }
                GgufFormat::Mfp4Lloyd => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp4g32_lloyd_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32Lloyd, 32u32, "MFP4G32Lloyd")
                }
                GgufFormat::Mfp4P => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp4g32_p_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32P, 32u32, "MFP4G32P")
                }
                GgufFormat::Mfp4E8 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp4g32_e8_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32E8, 32u32, "MFP4G32E8")
                }
                GgufFormat::Mfp4E8Soa => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp4g32_e8_soa_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32E8SOA, 32u32, "MFP4G32E8SOA")
                }
                GgufFormat::Mfp3E8 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp3g32_e8_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP3G32E8, 32u32, "MFP3G32E8")
                }
                GgufFormat::Mfp2E8 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp2g32_e8_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP2G32E8, 32u32, "MFP2G32E8")
                }
                GgufFormat::Ternary => {
                    // Direct low-bit: scale-only ternary g128, 34 B per 128 (2.125 bpw).
                    let q = quantize_tq2g128(&f32_data);
                    (q, QuantType::TQ2G128, 128u32, "TQ2G128")
                }
                GgufFormat::Binary => {
                    // Direct low-bit: scale-only binary g128, 18 B per 128 (1.14 bpw).
                    let q = quantize_bq1g128(&f32_data);
                    (q, QuantType::BQ1G128, 128u32, "BQ1G128")
                }
            }
        } else {
            // K not divisible by 256 — fall back to HFQ4-G128 (no rotation).
            // This branch fires for the rare ragged dim; ignores --format
            // (no G128 variant of mq4/mq6 exists).
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            let q = quantize_hfq4g128(&f32_data);
            quant_params += n_elements as u64;
            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
        };

        // llama.cpp stores the Gated-DeltaNet V heads tiled
        // (`[rep][k_head][head_dim]`, conversion/qwen.py `_reorder_v_heads`)
        // because its binary ops broadcast that way. hipfire reads the
        // HuggingFace grouped order, so the V rows are permuted back here.
        // `ssm_out` is excluded: a folded checkpoint never reordered it, which
        // is exactly what `gdn_v_grouped` asserts and the loader requires.
        let data = if matches!(arch_id, 5 | 6) {
            let n_k = config_json
                .get("linear_num_key_heads")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as usize;
            let n_v = config_json
                .get("linear_num_value_heads")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as usize;
            let head_v = config_json
                .get("linear_value_head_dim")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as usize;
            let head_k = config_json
                .get("linear_key_head_dim")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as usize;
            let rows = if is_2d { info.shape[1] } else { n_elements };
            match deltanet_v_row_span(&info.name, n_k, n_v, head_v, head_k, rows) {
                Some((first_row, head_rows)) => {
                    untile_v_rows(data, rows, first_row, n_k, n_v / n_k, head_rows)?
                }
                None => data,
            }
        } else {
            data
        };

        total_bytes_out += data.len() as u64;
        eprintln!(
            "  {label:>9}: {} → {} {:?} ({} src={:?}, {:.1} KB → {:.1} KB)",
            info.name,
            out_name,
            info.shape,
            n_elements,
            info.dtype,
            raw.len() as f64 / 1024.0,
            data.len() as f64 / 1024.0,
        );

        hfq_tensors.push(HfqTensor {
            name: out_name,
            quant_type,
            shape,
            group_size,
            data,
            spilled_len: 0,
        });
    }

    eprintln!("\n=== GGUF → MQ4 Summary ===");
    eprintln!("  Tensors:        {}", hfq_tensors.len());
    eprintln!("  Total params:   {total_params}");
    eprintln!(
        "  Quant'd params: {quant_params} ({:.1}%)",
        100.0 * quant_params as f64 / total_params as f64
    );
    eprintln!("  Input size:     {:.1} MB", total_bytes_in as f64 / 1e6);
    eprintln!(
        "  Output size:    {:.1} MB ({:.1}% of input)",
        total_bytes_out as f64 / 1e6,
        100.0 * total_bytes_out as f64 / total_bytes_in as f64,
    );

    write_hfq(output, arch_id, &metadata_json, &hfq_tensors, None)?;
    eprintln!("\nWrote: {}", output.display());
    Ok(())
}

pub(crate) fn dequantize_hfq_q8f16(data: &[u8], n_elements: usize) -> Result<Vec<f32>, String> {
    let n_blocks = n_elements.div_ceil(32);
    let expected = n_blocks * 34;
    if data.len() != expected {
        return Err(format!(
            "Q8F16 byte size {} != {expected} for {n_elements} elements",
            data.len()
        ));
    }
    let mut out = vec![0.0f32; n_elements];
    for b in 0..n_blocks {
        let off = b * 34;
        let scale = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        let start = b * 32;
        let end = (start + 32).min(n_elements);
        for i in start..end {
            out[i] = (data[off + 2 + i - start] as i8) as f32 * scale;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{gguf_arch_is_moe_like, gguf_format_is_dense_only_mq_v2, GgufFormat};

    #[test]
    fn dense_only_mq_v2_predicate_covers_four_formats() {
        assert!(gguf_format_is_dense_only_mq_v2(GgufFormat::Mq6V2));
        assert!(gguf_format_is_dense_only_mq_v2(GgufFormat::Mq5V2));
        assert!(gguf_format_is_dense_only_mq_v2(GgufFormat::Mq3V2));
        assert!(gguf_format_is_dense_only_mq_v2(GgufFormat::Mq2V2));
        // siblings stay admitted for MoE
        assert!(!gguf_format_is_dense_only_mq_v2(GgufFormat::Mq6));
        assert!(!gguf_format_is_dense_only_mq_v2(GgufFormat::Mq5));
        assert!(!gguf_format_is_dense_only_mq_v2(GgufFormat::Mq3));
        assert!(!gguf_format_is_dense_only_mq_v2(GgufFormat::Mq2));
        assert!(!gguf_format_is_dense_only_mq_v2(GgufFormat::Mq4V2));
        assert!(!gguf_format_is_dense_only_mq_v2(GgufFormat::Mq4C));
        assert!(!gguf_format_is_dense_only_mq_v2(GgufFormat::Hfq4));
    }

    #[test]
    fn moe_like_archs_reject_mq_v2_dense_accepted() {
        // Qwen MoE + other MoE-class arches
        for arch in [6u32, 9, 10, 11, 12, 13] {
            assert!(gguf_arch_is_moe_like(arch), "arch {arch}");
            for fmt in [
                GgufFormat::Mq6V2,
                GgufFormat::Mq5V2,
                GgufFormat::Mq3V2,
                GgufFormat::Mq2V2,
            ] {
                assert!(
                    gguf_format_is_dense_only_mq_v2(fmt) && gguf_arch_is_moe_like(arch),
                    "must reject {fmt:?} on arch {arch}"
                );
            }
        }
        // dense GGUF arches unchanged (qwen3.5/3.8 dense=5, llama=1)
        for arch in [1u32, 5, 7, 8, 14] {
            assert!(!gguf_arch_is_moe_like(arch), "arch {arch} is dense");
            assert!(
                !(gguf_format_is_dense_only_mq_v2(GgufFormat::Mq6V2)
                    && gguf_arch_is_moe_like(arch))
            );
        }
    }
}

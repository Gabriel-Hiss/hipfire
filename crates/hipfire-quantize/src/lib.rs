//! Shared model-format helpers used by the quantizer binaries.

pub mod float16;
pub mod gptq;
pub mod hessian_io;
pub mod hfhs_diag;
pub mod hfqm;
pub mod safetensors_file;

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// Root of the local Hugging Face hub cache, resolved the way `huggingface_hub`
/// itself resolves it.
pub fn hf_hub_cache_root() -> Option<PathBuf> {
    let home = hipfire_config::home_dir();
    hf_hub_cache_root_from(
        std::env::var_os("HF_HUB_CACHE").as_deref(),
        std::env::var_os("HUGGINGFACE_HUB_CACHE").as_deref(),
        std::env::var_os("HF_HOME").as_deref(),
        std::env::var_os("XDG_CACHE_HOME").as_deref(),
        home.as_deref(),
    )
}

fn hf_hub_cache_root_from(
    hf_hub_cache: Option<&OsStr>,
    huggingface_hub_cache: Option<&OsStr>,
    hf_home: Option<&OsStr>,
    xdg_cache_home: Option<&OsStr>,
    home: Option<&Path>,
) -> Option<PathBuf> {
    if let Some(path) = hf_hub_cache.filter(|path| !path.is_empty()) {
        return Some(PathBuf::from(path));
    }
    if let Some(path) = huggingface_hub_cache.filter(|path| !path.is_empty()) {
        return Some(PathBuf::from(path));
    }
    if let Some(path) = hf_home.filter(|path| !path.is_empty()) {
        return Some(PathBuf::from(path).join("hub"));
    }
    if let Some(path) = xdg_cache_home.filter(|path| !path.is_empty()) {
        return Some(PathBuf::from(path).join("huggingface").join("hub"));
    }
    home.map(|path| path.join(".cache").join("huggingface").join("hub"))
}

#[cfg(test)]
mod hf_hub_cache_root_tests {
    use super::hf_hub_cache_root_from;
    use std::ffi::OsStr;
    use std::path::{Path, PathBuf};

    fn value(value: &str) -> Option<&OsStr> {
        Some(OsStr::new(value))
    }

    #[test]
    fn prefers_explicit_cache_locations_in_documented_order() {
        let home = Path::new("home");
        assert_eq!(
            hf_hub_cache_root_from(
                value("hf-hub-cache"),
                value("legacy-hub-cache"),
                value("hf-home"),
                value("xdg-cache"),
                Some(home),
            ),
            Some(PathBuf::from("hf-hub-cache"))
        );
        assert_eq!(
            hf_hub_cache_root_from(
                None,
                value("legacy-hub-cache"),
                value("hf-home"),
                value("xdg-cache"),
                Some(home),
            ),
            Some(PathBuf::from("legacy-hub-cache"))
        );
        assert_eq!(
            hf_hub_cache_root_from(None, None, value("hf-home"), value("xdg-cache"), Some(home)),
            Some(PathBuf::from("hf-home").join("hub"))
        );
        assert_eq!(
            hf_hub_cache_root_from(None, None, None, value("xdg-cache"), Some(home)),
            Some(PathBuf::from("xdg-cache").join("huggingface").join("hub"))
        );
        assert_eq!(
            hf_hub_cache_root_from(None, None, None, None, Some(home)),
            Some(
                PathBuf::from("home")
                    .join(".cache")
                    .join("huggingface")
                    .join("hub")
            )
        );
    }

    #[test]
    fn skips_empty_environment_values() {
        assert_eq!(
            hf_hub_cache_root_from(
                value(""),
                value(""),
                value(""),
                value(""),
                Some(Path::new("home")),
            ),
            Some(
                PathBuf::from("home")
                    .join(".cache")
                    .join("huggingface")
                    .join("hub")
            )
        );
    }

    #[test]
    fn returns_none_without_a_cache_source() {
        assert_eq!(hf_hub_cache_root_from(None, None, None, None, None), None);
    }
}

use std::sync::OnceLock;

static MQ_CLIPSEARCH: OnceLock<bool> = OnceLock::new();

/// Whether the `mqN+` clip-search variant is active for MQ codecs.
pub fn mq_clipsearch_enabled() -> bool {
    MQ_CLIPSEARCH.get().copied().unwrap_or(false)
}

/// Arm the `mqN+` clip-search variant (idempotent; first set wins).
pub fn set_mq_clipsearch(enabled: bool) {
    let _ = MQ_CLIPSEARCH.set(enabled);
}

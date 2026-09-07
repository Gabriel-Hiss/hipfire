// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! One shared-library open policy for every ROCm dlopen site in this crate.
//!
//! `libloading::Library::new` calls `LoadLibraryExW(path, NULL, 0)` on Windows.
//! Flags of zero exclude the loaded DLL's own directory from the dependency
//! search, so an absolute candidate inside a self-contained ROCm tree
//! (`<root>\bin\amdhip64_7.dll`) fails on its siblings — `amd_comgr_*.dll`,
//! the `rocm_sysdeps` runtime — with a bare "could not find module" that names
//! the DLL we *did* find. Absolute candidates therefore open with
//! `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS`,
//! which resolves dependencies from the same install.
//!
//! Bare sonames keep the default search: `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` is
//! defined only for an absolute path, and a relative one makes `LoadLibraryExW`
//! fail outright. Unix is unchanged — `dlopen` already searches the object's
//! own `RUNPATH`.

use libloading::Library;

/// `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` — search the directory of the DLL being
/// loaded when resolving its dependencies.
#[cfg(windows)]
const LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR: u32 = 0x0000_0100;
/// `LOAD_LIBRARY_SEARCH_DEFAULT_DIRS` — application dir, `System32`, and the
/// process's user directories. Keeps driver-provided DLLs reachable.
#[cfg(windows)]
const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x0000_1000;

/// Open one candidate soname or absolute library path.
///
/// # Safety
/// Same contract as [`Library::new`]: the library's initializers run in this
/// process.
pub(crate) unsafe fn open(candidate: &str) -> Result<Library, libloading::Error> {
    #[cfg(windows)]
    if std::path::Path::new(candidate).is_absolute() {
        return libloading::os::windows::Library::load_with_flags(
            candidate,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
        )
        .map(Library::from);
    }
    Library::new(candidate)
}

/// Full text of a load failure, including the OS error.
///
/// `libloading::Error`'s own `Display` for the Windows arm is the bare string
/// `"LoadLibraryExW failed"` — the `io::Error` carrying the actual reason
/// (missing dependency vs. missing file vs. arch mismatch) hangs off `source()`
/// and is otherwise invisible in a user-facing diagnostic.
pub(crate) fn describe(err: &libloading::Error) -> String {
    let mut out = err.to_string();
    let mut source = std::error::Error::source(err);
    while let Some(cause) = source {
        out.push_str(": ");
        out.push_str(&cause.to_string());
        source = cause.source();
    }
    out
}

# Windows / Linux parity

Where hipfire's Windows behavior matches Linux, where it differs, and why.

| Field | Value |
|---|---|
| Page state | current (update with the change that moves a row) |
| Verified on | Windows 11 26200, MSVC, rustc 1.97, RX 7900 GRE (`gfx1100`) |
| HIP stack | `rocm-sdk` 10.1.0a wheels, HIP 7.16, Adrenalin 32.0.31041.1004 |
| Model exercised | `lfm2.5:1.2b` (arch 11), `qwen3.8:27b-mq3` (arch 20) |

Windows needs a HIP **development** stack, not just the driver: hipfire
JIT-compiles its kernels. Install routes are in
[GETTING_STARTED.md](GETTING_STARTED.md).

## At parity

Verified by running the thing, not by reading the code.

| Surface | Evidence |
|---|---|
| `cargo build --release`, `cargo check --all-targets` | clean |
| Workspace library tests | 2504 passed, 0 failed |
| `hipfire-cli` binary tests | 201 passed, 0 failed |
| Kernel JIT through `hipcc` | 1032-kernel tree compiles; cache at `%USERPROFILE%\.hipfire_kernels\<arch>` |
| HIP runtime load from an absolute SDK path | `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` resolves sibling DLLs |
| AMDGCN device bitcode discovery | explicit `--rocm-device-lib-path` for bundled-LLVM layouts |
| `hipfire pull` / `list` / `rm` | writes under `%USERPROFILE%\.hipfire\models` |
| `hipfire run`, AR decode | 365 tok/s median on `lfm2.5:1.2b` |
| Speculative decode (n-gram) | engaged; greedy output byte-identical to AR across 8 fixtures |
| `hipfire serve` foreground and `--detach` | survives the caller exiting; composes with a pipe |
| OpenAI HTTP surface | `/v1/chat/completions` streaming and non-streaming |
| `hipfire ps`, `stop`, `stop --force`, `restart` | PID identity, start-time, TCP listener ownership, tree reap |
| Daemon single-instance lock | `LockFileEx` on a sentinel byte; contention names the PID |
| `hipfire diag` | reports `driver model: WDDM` and real `GPU targets` from HIP |
| `hipfire profile` roofline | 80 CU, 1927 MHz, 576 GB/s read from HIP device attributes |
| `hipfire setup`, `hipfire update` | `install.ps1` bootstrap; rename-then-write replaces a running `.exe` |
| `scripts/uninstall.ps1` | mirrors `uninstall.sh` including `-DryRun` and exact PATH removal |
| `scripts/serve_harness.py` | `battery` and `chain` on the real model: 0 empty, 0 attractor |
| `scripts/redline_daemon_harness.py` | runs, measures the HIP route, reports retained replay unavailable |
| `hipfire-tui` | `--check` and the doctor view, with WDDM rows marked not applicable |
| `hipfire quantize` HF cache lookup | honors `HF_HUB_CACHE`, `HF_HOME`, `XDG_CACHE_HOME`, then the user home |
| Kernel Atlas task evaluation | POSIX task commands run through Git for Windows `bash.exe` |

## Structurally unavailable on Windows

These are not unfinished work. AMD's Windows compute stack is WDDM/PAL, and
neither component below exists for it. Every one fails closed to ordinary HIP
dispatch and says so.

| Subsystem | What it needs | Windows behavior |
|---|---|---|
| Redline retained AQL replay | ROCr (`libhsa-runtime64`) user-mode queues | refuses immediately, falls back to HIP |
| Redline retained PM4 replay | `/dev/dri/renderD*` + libdrm_amdgpu ioctls | `device`/`kfd`/`queue`/`dispatch` are `cfg(target_os = "linux")` |
| `hsa-bridge` AQL launch bypass | ROCr | refuses immediately |
| RCCL collectives / multi-GPU peer | `librccl` | AMD ships no Windows build; candidate list is empty and the error says so |
| `amdgpu` module, `/dev/kfd`, KFD topology, `/sys/class/drm` | the Linux kernel driver | `diag` and the TUI report `n/a (WDDM)` instead of a false failure |

`hsa-runtime64.dll` is absent from the Windows HIP SDK; this was checked across
the installed SDK and the whole filesystem.

## Degraded, not broken

| Behavior | Linux | Windows |
|---|---|---|
| Page-cache residency for the zero-copy weight path | `mincore` | `QueryWorkingSetEx`, same sampling policy and threshold |
| Weight warmup | threaded `pread` sweep | one synchronous `PrefetchVirtualMemory` |
| Sequential / random access intent | `posix_fadvise` | `FILE_FLAG_SEQUENTIAL_SCAN` / `FILE_FLAG_RANDOM_ACCESS` at open |
| Dropping a file from the page cache after load | `posix_fadvise(DONTNEED)` | no unprivileged API exists; a Windows run holds more standby memory after load |
| Graceful daemon shutdown on `stop` | `SIGTERM`, the daemon runs its shutdown path | `TerminateProcess`; Windows has no cooperative termination signal, so shutdown code does not run |
| Process identity for the serve PID guard | `/proc/<pid>/cmdline`, full argv | ToolHelp image name only, so the guard matches `hipfire.exe` rather than argv containing `serve` |
| Pidfile permissions | `0o600` | inherits the `%USERPROFILE%` ACL; no std-level `chmod` equivalent |

## VRAM ceiling under WDDM

WDDM virtualizes video memory per process and overcommits it. An allocation
that does not fit in dedicated VRAM still succeeds, backed by system memory
over PCIe, so a model too large for the card loads and decodes instead of
failing. Sizing a model by the card's nameplate capacity therefore produces a
silent 5x throughput loss rather than an out-of-memory error.

Two consequences for measurement:

- `hipMemGetInfo` is not a residency check here. With a 15 GB model resident it
  reported 17.01 GB of 17.16 GB free, because it answers for the calling
  process, not the device.
- The desktop competes for the same dedicated pool. On the verified host `dwm`
  alone held 1.50 GB and the logged-in session about 3.8 GB, leaving roughly
  12.4 GB for a compute process on a 16 GB card.

Read the real split from the WDDM counters:

```powershell
$d = (Get-CimInstance Win32_Process -Filter "Name='daemon.exe'").ProcessId
Get-Counter "\GPU Process Memory(pid_${d}*)\Dedicated Usage",
            "\GPU Process Memory(pid_${d}*)\Shared Usage"
```

Measured on `gfx1100` (16 GB) with Qwen3.8-27B, `--spec off`, 3 runs,
`benchmarks/prompts/merge_sort_thinking_off.txt`
(md5 `253c7ac50857fe6d0e10fb0d2c5e35c0`), `max_seq` at the registry default,
counters sampled while the model served a request:

| Tier | Weights | Dedicated | Shared | Decode |
|---|---|---|---|---|
| `mq3-xt` | 11.78 GB | 12.75 GB | 1.76 GB | 29.5 tok/s |
| `mq3` | 12.62 GB | 12.72 GB | 2.63 GB | 29.0 tok/s |
| `mq3-pro` | 13.18 GB | 12.78 GB | 3.09 GB | 5.8 tok/s |
| `mq4-xt` | 14.98 GB | 12.85 GB | 4.62 GB | 4.5 tok/s |

Dedicated usage saturates near 12.8 GB whatever the tier, and the overflow
lands in shared memory. The cost of that overflow is not proportional: 2.63 GB
shared costs 1.7% of decode, and the next half gigabyte costs 80%. Prefill
falls harder than decode, 306 tok/s down to 29 on `mq4-xt`.

Capping the context shrinks the shared footprint without buying throughput.
`memory.max_seq 8192` took `mq3` from 2.63 GB shared to 0.97 GB and moved
decode from 28.9 to 29.4 tok/s, within run-to-run spread. Untouched committed
pages are not what costs the throughput, so size the tier, not the context.

Pick a tier by measuring, not by subtracting weights from the nameplate
capacity.

Full arrays, artifact digests, and the speculation rows are in
[`docs/perf-checkpoints/2026-09-07-qwen38-mq3-gfx1100-windows-vram-ceiling.md`](perf-checkpoints/2026-09-07-qwen38-mq3-gfx1100-windows-vram-ceiling.md)
(lifecycle `historical`, measurement only, not an admission or a default).

## Environment note for isolated runs

`scripts/serve_harness.py` runs the daemon under its own `--home`, which by
design does not read the user's `developer.rocm_path`. On Linux the default
`/opt/rocm` discovery still finds a toolchain; Windows has no conventional
default path, so export the root first:

```powershell
$env:HIPFIRE_ROCM_PATH = "$env:USERPROFILE\rocmenv\Lib\site-packages\_rocm_sdk_devel"
```

Without it the run fails closed with
`no usable cached kernel image exists, but hipcc is unavailable`.

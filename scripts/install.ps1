# hipfire installer for Windows.
# Usage: irm https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/install.ps1 | iex
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Ref,
    [string]$Branch,
    [string]$Tag,
    [string]$Commit,
    [Alias("rocm-root")]
    [string]$RocmRoot,
    [string]$Hipcc,
    [Alias("strict-rocm")]
    [switch]$StrictRocm,
    [Alias("gpu-arch")]
    [string]$GpuArch,
    [string]$Profile,
    [Alias("y", "non-interactive")]
    [switch]$Yes,
    [Alias("no-path")]
    [switch]$NoPath,
    [Alias("h")]
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"

function Show-Usage {
    Write-Host "Usage: install.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  --branch NAME, -Branch NAME        Install from branch NAME"
    Write-Host "  --ref REF, -Ref REF                Install from git ref REF"
    Write-Host "  --tag TAG, -Tag TAG                Install from tag TAG"
    Write-Host "  --commit SHA, -Commit SHA          Install from commit SHA"
    Write-Host "  --rocm-root PATH, -RocmRoot PATH   ROCm installation root (forwarded to setup)"
    Write-Host "  --hipcc PATH, -Hipcc PATH          ROCm device compiler in a different prefix"
    Write-Host "  --strict-rocm, -StrictRocm         Disable cross-root compiler fallback"
    Write-Host "  --gpu-arch ARCH, -GpuArch ARCH     GPU architecture (forwarded to setup)"
    Write-Host "  --profile VALUE, -Profile VALUE    Replay profile: auto, hip, or redline"
    Write-Host "  --yes, -y, --non-interactive, -Yes Non-interactive mode"
    Write-Host "  --help, -Help                       Show this help"
    Write-Host ""
    Write-Host "Bootstrap only initializes source, stages the Windows HIP runtime, and builds"
    Write-Host "hipfire-cli, then runs:"
    Write-Host "  hipfire.exe setup --source <repo> [forwarded options]"
}

# PowerShell 5.1 treats GNU-style --names as positional values rather than
# parameter names. Capture and bind that documented spelling explicitly.
for ($index = 0; $index -lt $RemainingArgs.Count; $index++) {
    $option = $RemainingArgs[$index]
    switch ($option) {
        { $_ -in @("--ref", "--branch", "--tag", "--commit", "--rocm-root", "--hipcc", "--gpu-arch", "--profile") } {
            if ($index + 1 -ge $RemainingArgs.Count) {
                Write-Host "ERROR: $option requires a value" -ForegroundColor Red
                exit 1
            }
            $index++
            $value = $RemainingArgs[$index]
            $name = switch ($option) {
                "--ref" { "Ref" }
                "--branch" { "Branch" }
                "--tag" { "Tag" }
                "--commit" { "Commit" }
                "--rocm-root" { "RocmRoot" }
                "--hipcc" { "Hipcc" }
                "--gpu-arch" { "GpuArch" }
                "--profile" { "Profile" }
            }
            if ($name -in @("Ref", "Branch", "Tag", "Commit") -and $PSBoundParameters.ContainsKey($name)) {
                throw "Choose only one -Ref, -Branch, -Tag, or -Commit."
            }
            Set-Variable -Name $name -Value $value
            $PSBoundParameters[$name] = $value
        }
        "--strict-rocm" { $StrictRocm = $true; $PSBoundParameters["StrictRocm"] = $true }
        { $_ -in @("--yes", "--non-interactive", "-y") } { $Yes = $true; $PSBoundParameters["Yes"] = $true }
        "--no-path" { $NoPath = $true; $PSBoundParameters["NoPath"] = $true }
        { $_ -in @("--help", "-h") } { $Help = $true; $PSBoundParameters["Help"] = $true }
        default { throw "Unknown installer option: $option" }
    }
}

if ($Help) {
    Show-Usage
    return
}

# Preserve the loose spelling accepted by older wrappers that populate $args.
if ($args -contains "--no-path") { $NoPath = $true }

if ($PSBoundParameters.ContainsKey("Profile") -and $Profile -cnotin @("auto", "hip", "redline")) {
    Write-Host "ERROR: --profile must be exactly auto, hip, or redline (got: $Profile)" -ForegroundColor Red
    exit 1
}

# Windows PowerShell 5.1 turns native stderr into a terminating error under
# EAP=Stop, and git prints normal fetch progress on stderr. Keep the relaxation
# local to git and check $LASTEXITCODE at each call site.
function Invoke-Git {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & git @args
    } finally {
        $ErrorActionPreference = $previous
    }
}

function Test-RemoteRef([string]$Repo, [string]$RemoteRef) {
    Invoke-Git -C $Repo ls-remote --exit-code origin $RemoteRef 2>&1 | Out-Null
    return $LASTEXITCODE -eq 0
}

function Checkout-InstallRef([string]$Repo) {
    $kind = $script:InstallRefKind
    if ($kind -eq "auto") {
        if (Test-RemoteRef $Repo "refs/heads/$script:InstallRef") {
            $kind = "branch"
        } elseif (Test-RemoteRef $Repo "refs/tags/$script:InstallRef") {
            $kind = "tag"
        } else {
            $kind = "commit"
        }
    }

    switch ($kind) {
        "branch" {
            if (-not (Test-RemoteRef $Repo "refs/heads/$script:InstallRef")) {
                throw "Origin has no branch '$script:InstallRef'."
            }
            Invoke-Git -C $Repo fetch --depth 1 origin "+refs/heads/$script:InstallRef`:refs/remotes/origin/$script:InstallRef"
            if ($LASTEXITCODE -ne 0) { throw "git fetch branch failed." }
            Invoke-Git -C $Repo checkout -B $script:InstallRef "refs/remotes/origin/$script:InstallRef"
            if ($LASTEXITCODE -ne 0) { throw "git checkout branch failed." }
        }
        "tag" {
            if (-not (Test-RemoteRef $Repo "refs/tags/$script:InstallRef")) {
                throw "Origin has no tag '$script:InstallRef'."
            }
            Invoke-Git -C $Repo fetch --depth 1 origin "refs/tags/$script:InstallRef"
            if ($LASTEXITCODE -ne 0) { throw "git fetch tag failed." }
            Invoke-Git -C $Repo checkout --detach "FETCH_HEAD^{commit}"
            if ($LASTEXITCODE -ne 0) { throw "git checkout tag failed." }
        }
        "commit" {
            Invoke-Git -C $Repo fetch --depth 1 origin $script:InstallRef
            if ($LASTEXITCODE -ne 0) { throw "git fetch commit failed." }
            Invoke-Git -C $Repo checkout --detach "FETCH_HEAD^{commit}"
            if ($LASTEXITCODE -ne 0) { throw "git checkout commit failed." }
        }
        default { throw "Unsupported revision kind '$kind'." }
    }
    $script:InstallRefKind = $kind
}

function Install-Binary([string]$Source, [string]$Destination) {
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Build artifact missing: $Source"
    }

    $parent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $name = Split-Path -Leaf $Destination
    $nonce = [Guid]::NewGuid().ToString("N")
    $temporary = Join-Path $parent ".$name.install-$PID-$nonce.tmp"
    $backup = Join-Path $parent ".$name.prev-$PID-$nonce"
    $backupCreated = $false

    try {
        Copy-Item -LiteralPath $Source -Destination $temporary
        if (Test-Path -LiteralPath $Destination -PathType Leaf) {
            # Copy-Item -Force cannot overwrite a running image on Windows, but
            # Windows permits renaming it. Keep every move on the same volume.
            Move-Item -LiteralPath $Destination -Destination $backup
            $backupCreated = $true
        }
        Move-Item -LiteralPath $temporary -Destination $Destination
        if ($backupCreated) {
            Remove-Item -LiteralPath $backup -Force
            $backupCreated = $false
        }
    } catch {
        $failure = $_
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
        if ($backupCreated -and (Test-Path -LiteralPath $backup -PathType Leaf)) {
            Remove-Item -LiteralPath $Destination -Force -ErrorAction SilentlyContinue
            Move-Item -LiteralPath $backup -Destination $Destination -ErrorAction SilentlyContinue
        }
        throw $failure
    } finally {
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
    }
}

function Test-PathEntry([string]$PathValue, [string]$Entry) {
    if ($null -eq $PathValue) { return $false }
    foreach ($part in $PathValue.Split([char]';', [System.StringSplitOptions]::None)) {
        if ([string]::Equals($part, $Entry, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    return $false
}

# Paths
$HipfireDir = Join-Path $env:USERPROFILE ".hipfire"
$BinDir = Join-Path $HipfireDir "bin"
$RuntimeDir = Join-Path $HipfireDir "runtime"
$SrcDir = Join-Path $HipfireDir "src"
$GithubRepo = "warpfront/hipfire"

# Keep the existing selector rules. A command-line selector overrides the
# automation environment variable, and only one command-line selector is legal.
$InstallRef = if ($env:HIPFIRE_INSTALL_REF) { $env:HIPFIRE_INSTALL_REF } else { "master" }
$InstallRefKind = if ($env:HIPFIRE_INSTALL_REF) { "auto" } else { "branch" }
$Selectors = @(
    if ($PSBoundParameters.ContainsKey("Ref")) { @{ Value = $Ref; Kind = "auto" } }
    if ($PSBoundParameters.ContainsKey("Branch")) { @{ Value = $Branch; Kind = "branch" } }
    if ($PSBoundParameters.ContainsKey("Tag")) { @{ Value = $Tag; Kind = "tag" } }
    if ($PSBoundParameters.ContainsKey("Commit")) { @{ Value = $Commit; Kind = "commit" } }
)
if ($Selectors.Count -gt 1) {
    throw "Choose only one -Ref, -Branch, -Tag, or -Commit."
}
if ($Selectors.Count -eq 1) {
    $InstallRef = [string]$Selectors[0].Value
    $InstallRefKind = [string]$Selectors[0].Kind
}
$InstallRef = $InstallRef.Trim().TrimStart("@")
if ($InstallRef.StartsWith("refs/heads/")) {
    $InstallRef = $InstallRef.Substring(11)
    $InstallRefKind = "branch"
} elseif ($InstallRef.StartsWith("refs/tags/")) {
    $InstallRef = $InstallRef.Substring(10)
    $InstallRefKind = "tag"
} elseif ($InstallRef.StartsWith("origin/")) {
    $InstallRef = $InstallRef.Substring(7)
    if ($InstallRefKind -eq "auto") { $InstallRefKind = "branch" }
}
if (
    [string]::IsNullOrWhiteSpace($InstallRef) -or
    $InstallRef -match '^[./-]|[./]$|\.\.|@\{|//|[\s\\:\?\*\[\]\^~]'
) {
    throw "Unsafe or invalid git revision '$InstallRef'."
}
if ($InstallRefKind -eq "commit" -and $InstallRef -notmatch '^[0-9a-fA-F]{7,40}$') {
    throw "-Commit requires a 7-40 character hexadecimal git commit."
}

Write-Host "=== hipfire installer ===" -ForegroundColor Cyan
Write-Host "Requested source: $InstallRefKind '$InstallRef'"
Write-Host ""

Write-Host "=== prerequisites ===" -ForegroundColor Cyan
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "  ERROR: git is required. Install from https://git-scm.com and re-run." -ForegroundColor Red
    exit 1
}
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Host "  ERROR: cargo is required. Install Rust from https://rustup.rs/ and re-run." -ForegroundColor Red
    exit 1
}
Write-Host "  git and cargo found ✓" -ForegroundColor Green

Write-Host ""
Write-Host "=== source checkout ===" -ForegroundColor Cyan
if (-not (Test-Path -LiteralPath (Join-Path $SrcDir ".git") -PathType Container)) {
    if ((Test-Path -LiteralPath $SrcDir) -and (Get-ChildItem -LiteralPath $SrcDir -Force -ErrorAction SilentlyContinue | Select-Object -First 1)) {
        throw "$SrcDir exists but is not a git checkout; move it aside and retry."
    }
    New-Item -ItemType Directory -Force -Path $SrcDir | Out-Null
    Invoke-Git -C $SrcDir init --quiet
    if ($LASTEXITCODE -ne 0) { throw "git init failed." }
    Invoke-Git -C $SrcDir remote add origin "https://github.com/$GithubRepo.git"
    if ($LASTEXITCODE -ne 0) { throw "git remote add failed." }
    Checkout-InstallRef $SrcDir
    Write-Host "  Checked out $InstallRefKind '$InstallRef' ✓" -ForegroundColor Green
} else {
    $status = Invoke-Git -C $SrcDir status --porcelain --untracked-files=normal 2>&1 | Out-String
    if ($status.Trim()) {
        throw "Managed source at $SrcDir is dirty; commit, stash, or clean it before re-running the installer."
    }
    $origin = (Invoke-Git -C $SrcDir remote get-url origin 2>$null | Out-String).Trim()
    if ($origin -ne "https://github.com/$GithubRepo.git") {
        Invoke-Git -C $SrcDir remote set-url origin "https://github.com/$GithubRepo.git"
        if ($LASTEXITCODE -ne 0) { throw "git remote set-url failed." }
    }
    Checkout-InstallRef $SrcDir
    Write-Host "  Updated to $InstallRefKind '$InstallRef' ✓" -ForegroundColor Green
}
$RepoDir = $SrcDir

# The Rust setup command resolves ROCm but cannot start until Windows can load
# amdhip64.dll. Stage that DLL first and add its directory to this process PATH.
Write-Host ""
Write-Host "=== Windows HIP runtime ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
$HipDllDest = Join-Path $RuntimeDir "amdhip64.dll"
$HipDllFound = Test-Path -LiteralPath $HipDllDest -PathType Leaf
$RocmCandidates = @()
if ($PSBoundParameters.ContainsKey("RocmRoot")) { $RocmCandidates += $RocmRoot }
if ($env:HIP_PATH) { $RocmCandidates += $env:HIP_PATH }
$RocmCandidates += "C:\Program Files\AMD\ROCm"

if (-not $HipDllFound) {
    foreach ($root in $RocmCandidates) {
        if ([string]::IsNullOrWhiteSpace($root) -or -not (Test-Path -LiteralPath $root -PathType Container)) { continue }
        $roots = @($root)
        $roots += @(Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | ForEach-Object { $_.FullName })
        foreach ($candidateRoot in $roots) {
            foreach ($dllName in @("amdhip64.dll", "amdhip64_7.dll", "amdhip64_6.dll")) {
                $candidate = Join-Path $candidateRoot "bin\$dllName"
                if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                    Copy-Item -LiteralPath $candidate -Destination $HipDllDest -Force
                    Write-Host "  $dllName staged from $candidate ✓" -ForegroundColor Green
                    $HipDllFound = $true
                    break
                }
            }
            if ($HipDllFound) { break }
        }
        if ($HipDllFound) { break }
    }
}

if (-not $HipDllFound) {
    $DllUrl = "https://github.com/$GithubRepo/releases/download/hip-runtime/amdhip64.dll"
    try {
        Invoke-WebRequest -Uri $DllUrl -OutFile $HipDllDest -UseBasicParsing
        Write-Host "  amdhip64.dll downloaded ✓" -ForegroundColor Green
        $HipDllFound = $true
    } catch {
        Write-Host "  ERROR: amdhip64.dll was not found and the fallback download failed: $_" -ForegroundColor Red
        Write-Host "  Install ROCm for Windows or place amdhip64.dll in $RuntimeDir" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "  amdhip64.dll ready ✓" -ForegroundColor Green
}
$env:PATH = "$RuntimeDir;$env:PATH"

Write-Host ""
Write-Host "=== CLI bootstrap ===" -ForegroundColor Cyan
$env:CARGO_TARGET_DIR = Join-Path $RepoDir "target"
Push-Location $RepoDir
try {
    cargo build --release -p hipfire-cli
    if ($LASTEXITCODE -ne 0) { throw "cargo build hipfire-cli failed." }
} finally {
    Pop-Location
}
$CliExe = Join-Path $RepoDir "target\release\hipfire.exe"
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
Install-Binary $CliExe (Join-Path $BinDir "hipfire.exe")
Write-Host "  hipfire.exe built and staged ✓" -ForegroundColor Green

$Forwarded = @("setup", "--source", $RepoDir)
if ($PSBoundParameters.ContainsKey("RocmRoot")) { $Forwarded += @("--rocm-root", $RocmRoot) }
if ($PSBoundParameters.ContainsKey("Hipcc")) { $Forwarded += @("--hipcc", $Hipcc) }
if ($StrictRocm) { $Forwarded += "--strict-rocm" }
if ($PSBoundParameters.ContainsKey("GpuArch")) { $Forwarded += @("--gpu-arch", $GpuArch) }
if ($PSBoundParameters.ContainsKey("Profile")) { $Forwarded += @("--profile", $Profile) }
if ($Yes) { $Forwarded += "--yes" }
if ($Selectors.Count -eq 1) {
    $selectorOption = switch ([string]$Selectors[0].Kind) {
        "auto" { "--ref" }
        "branch" { "--branch" }
        "tag" { "--tag" }
        "commit" { "--commit" }
    }
    $Forwarded += @($selectorOption, [string]$Selectors[0].Value)
} elseif ($env:HIPFIRE_INSTALL_REF) {
    $Forwarded += @("--ref", $env:HIPFIRE_INSTALL_REF)
}

Write-Host ""
Write-Host "=== runtime setup ===" -ForegroundColor Cyan
& (Join-Path $BinDir "hipfire.exe") @Forwarded
if ($LASTEXITCODE -ne 0) { throw "hipfire setup failed with exit code $LASTEXITCODE." }
Write-Host "  Runtime setup complete ✓" -ForegroundColor Green

Write-Host ""
Write-Host "=== user PATH ===" -ForegroundColor Cyan
$CurrentUserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($null -eq $CurrentUserPath) { $CurrentUserPath = "" }
if ($NoPath) {
    Write-Host "  Skipping PATH modification (--no-path)" -ForegroundColor Yellow
    Write-Host "  Add manually to user PATH: $BinDir" -ForegroundColor Yellow
} elseif (Test-PathEntry $CurrentUserPath $BinDir) {
    Write-Host "  hipfire already in PATH ✓" -ForegroundColor Green
} else {
    $addPath = $Yes
    if (-not $Yes) {
        $reply = Read-Host "  Add $BinDir to user PATH permanently? [Y/n]"
        $addPath = $reply -notmatch "^[Nn]$"
    }
    if ($addPath) {
        $NewPath = if ($CurrentUserPath.Length -eq 0) { $BinDir } else { "$BinDir;$CurrentUserPath" }
        if ($NewPath.Length -gt 2040) {
            Write-Host "  WARNING: User PATH would be $($NewPath.Length) chars; add $BinDir manually." -ForegroundColor Red
        } else {
            [Environment]::SetEnvironmentVariable("PATH", $NewPath, "User")
            if (-not (Test-PathEntry $env:PATH $BinDir)) { $env:PATH = "$BinDir;$env:PATH" }
            Write-Host "  PATH updated ✓ (restart your shell to apply)" -ForegroundColor Green
        }
    } else {
        Write-Host "  Add manually to user PATH: $BinDir" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=== hipfire installed ===" -ForegroundColor Cyan
Write-Host "  Binaries: $BinDir" -ForegroundColor Green
Write-Host "  Models:   $(Join-Path $HipfireDir 'models')" -ForegroundColor Green

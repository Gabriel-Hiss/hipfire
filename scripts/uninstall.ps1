# hipfire uninstaller for installs created by scripts/install.ps1.
# By default it removes the installed program and preserves models and settings.
[CmdletBinding(PositionalBinding = $false)]
param(
    [Alias("dry-run")]
    [switch]$DryRun,
    [switch]$Purge,
    [Alias("y")]
    [switch]$Yes,
    [Alias("h")]
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"

function Show-Usage {
    Write-Host "hipfire uninstaller"
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  uninstall.ps1 [-DryRun] [-Purge] [-Yes]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -DryRun   Show what would be removed without changing anything"
    Write-Host "  -Purge    Also delete models, configuration, and all other ~/.hipfire data"
    Write-Host "  -Yes      Skip the interactive confirmation required by -Purge"
    Write-Host "  -Help     Show this help"
    Write-Host ""
    Write-Host "GNU-style --dry-run, --purge, --yes, and --help spellings are also accepted."
    Write-Host ""
    Write-Host "Default behavior:"
    Write-Host "  Removes installed binaries, kernels, managed source, runtime PID/log files,"
    Write-Host "  and the user PATH entry added by install.ps1. Models and settings are preserved."
}

# PowerShell 5.1 treats GNU-style --names as positional values. Capture them
# without allowing them to bind accidentally to unrelated declared switches.
foreach ($option in $RemainingArgs) {
    switch ($option) {
        "--dry-run" { $DryRun = $true }
        "--purge" { $Purge = $true }
        { $_ -in @("--yes", "-y") } { $Yes = $true }
        { $_ -in @("--help", "-h") } { $Help = $true }
        default {
            Write-Host "ERROR: unknown uninstaller option: $option" -ForegroundColor Red
            Show-Usage
            exit 2
        }
    }
}

if ($Help) {
    Show-Usage
    return
}

# Preserve loose arguments passed through wrappers that populate $args.
if ($args -contains "--dry-run") { $DryRun = $true }
if ($args -contains "--purge") { $Purge = $true }
if ($args -contains "--yes") { $Yes = $true }

if ([string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
    throw "USERPROFILE is not set; refusing to choose an uninstall target."
}
if (-not [System.IO.Path]::IsPathRooted($env:USERPROFILE)) {
    throw "Unsafe USERPROFILE value '$env:USERPROFILE'; refusing to uninstall."
}

$UserProfile = [System.IO.Path]::GetFullPath($env:USERPROFILE)
$ProfileRoot = [System.IO.Path]::GetPathRoot($UserProfile)
$TrimmedProfile = $UserProfile.TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
$TrimmedRoot = $ProfileRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
if ([string]::IsNullOrWhiteSpace($TrimmedProfile) -or [string]::Equals($TrimmedProfile, $TrimmedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Unsafe USERPROFILE value '$env:USERPROFILE'; refusing to uninstall."
}

$HipfireDir = [System.IO.Path]::GetFullPath((Join-Path $UserProfile ".hipfire"))
$BinDir = Join-Path $HipfireDir "bin"
$RuntimeDir = Join-Path $HipfireDir "runtime"
$SrcDir = Join-Path $HipfireDir "src"
$LegacyCliDir = Join-Path $HipfireDir "cli"

# Prove the recursive-removal root is exactly a .hipfire child of the validated
# profile. Do not weaken this to a prefix comparison.
$HipfireParent = [System.IO.Path]::GetFullPath((Split-Path -Parent $HipfireDir))
$HipfireLeaf = Split-Path -Leaf $HipfireDir
if (
    -not [string]::Equals($HipfireParent.TrimEnd('\'), $UserProfile.TrimEnd('\'), [System.StringComparison]::OrdinalIgnoreCase) -or
    -not [string]::Equals($HipfireLeaf, ".hipfire", [System.StringComparison]::OrdinalIgnoreCase)
) {
    throw "Unsafe uninstall target '$HipfireDir'."
}

function Invoke-Git {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & git @args
    } finally {
        $ErrorActionPreference = $previous
    }
}

function Test-AllowedTree([string]$Target) {
    $full = [System.IO.Path]::GetFullPath($Target)
    foreach ($allowed in @($BinDir, $RuntimeDir, $SrcDir, $LegacyCliDir, $HipfireDir)) {
        if ([string]::Equals($full, [System.IO.Path]::GetFullPath($allowed), [System.StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    return $false
}

function Remove-SafeTree([string]$Target) {
    if (-not (Test-AllowedTree $Target)) {
        throw "Refusing unexpected recursive removal target '$Target'."
    }
    if (-not (Test-Path -LiteralPath $Target)) { return }

    $item = Get-Item -LiteralPath $Target -Force
    if (-not $item.PSIsContainer) {
        throw "Refusing recursive removal because '$Target' is not a directory."
    }
    if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing recursive removal through reparse point '$Target'."
    }

    if ($DryRun) {
        Write-Host "Would remove: $Target"
    } else {
        Remove-Item -LiteralPath $Target -Recurse -Force
        Write-Host "Removed: $Target" -ForegroundColor Green
    }
}

function Remove-SafeFile([string]$Target) {
    $allowed = @(
        (Join-Path $HipfireDir "serve.pid"),
        (Join-Path $HipfireDir "daemon.pid"),
        (Join-Path $HipfireDir "serve.log")
    )
    $isAllowed = $false
    foreach ($path in $allowed) {
        if ([string]::Equals([System.IO.Path]::GetFullPath($Target), [System.IO.Path]::GetFullPath($path), [System.StringComparison]::OrdinalIgnoreCase)) {
            $isAllowed = $true
            break
        }
    }
    if (-not $isAllowed) { throw "Refusing unexpected file removal target '$Target'." }
    if (-not (Test-Path -LiteralPath $Target)) { return }

    if ($DryRun) {
        Write-Host "Would remove: $Target"
    } else {
        Remove-Item -LiteralPath $Target -Force
        Write-Host "Removed: $Target" -ForegroundColor Green
    }
}

function Stop-InstalledProcesses {
    $images = @(
        [System.IO.Path]::GetFullPath((Join-Path $BinDir "hipfire.exe")),
        [System.IO.Path]::GetFullPath((Join-Path $BinDir "daemon.exe"))
    )
    $processes = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        if ([string]::IsNullOrWhiteSpace($_.ExecutablePath)) { return $false }
        $candidate = [System.IO.Path]::GetFullPath($_.ExecutablePath)
        foreach ($image in $images) {
            if ([string]::Equals($candidate, $image, [System.StringComparison]::OrdinalIgnoreCase)) { return $true }
        }
        return $false
    })

    foreach ($process in $processes) {
        if ($DryRun) {
            Write-Host "Would stop: $($process.ExecutablePath) (PID $($process.ProcessId))"
            continue
        }
        try {
            Stop-Process -Id $process.ProcessId -ErrorAction Stop
            Write-Host "Stopped: $($process.Name) (PID $($process.ProcessId))" -ForegroundColor Green
        } catch {
            Write-Host "WARNING: could not stop PID $($process.ProcessId) at $($process.ExecutablePath): $_" -ForegroundColor Yellow
        }
    }
}

function Remove-InstallPath {
    $current = [Environment]::GetEnvironmentVariable("PATH", "User")
    if ($null -eq $current) { return }

    $parts = $current.Split(@([char]';'), [System.StringSplitOptions]::None)
    $kept = New-Object System.Collections.Generic.List[string]
    $removed = 0
    foreach ($part in $parts) {
        if ([string]::Equals($part, $BinDir, [System.StringComparison]::OrdinalIgnoreCase)) {
            $removed++
        } else {
            $kept.Add($part)
        }
    }
    if ($removed -eq 0) { return }

    if ($DryRun) {
        Write-Host "Would remove exact user PATH entry: $BinDir"
    } else {
        # Split with StringSplitOptions.None and join the untouched entries so
        # empty segments and every unrelated PATH entry survive byte-for-byte.
        $updated = [string]::Join(";", $kept.ToArray())
        [Environment]::SetEnvironmentVariable("PATH", $updated, "User")
        Write-Host "Updated user PATH: removed $BinDir ✓" -ForegroundColor Green
    }
}

function Test-ManagedSource {
    if (-not (Test-Path -LiteralPath (Join-Path $SrcDir ".git"))) { return $false }
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { return $false }
    $origin = (Invoke-Git -C $SrcDir remote get-url origin 2>$null | Out-String).Trim()
    return $origin -in @(
        "https://github.com/warpfront/hipfire",
        "https://github.com/warpfront/hipfire.git",
        "git@github.com:warpfront/hipfire",
        "git@github.com:warpfront/hipfire.git",
        "ssh://git@github.com/warpfront/hipfire",
        "ssh://git@github.com/warpfront/hipfire.git",
        "https://github.com/Kaden-Schutt/hipfire",
        "https://github.com/Kaden-Schutt/hipfire.git",
        "git@github.com:Kaden-Schutt/hipfire",
        "git@github.com:Kaden-Schutt/hipfire.git",
        "ssh://git@github.com/Kaden-Schutt/hipfire",
        "ssh://git@github.com/Kaden-Schutt/hipfire.git"
    )
}

function Test-SourceHasLocalWork {
    $status = (Invoke-Git -C $SrcDir status --porcelain --untracked-files=normal 2>$null | Out-String)
    if ($status.Trim()) { return $true }

    Invoke-Git -C $SrcDir rev-parse --quiet --verify refs/stash 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { return $true }

    $branches = @(Invoke-Git -C $SrcDir for-each-ref "--format=%(refname:short)%09%(upstream:short)" refs/heads 2>$null)
    foreach ($row in $branches) {
        if ([string]::IsNullOrWhiteSpace($row)) { continue }
        $fields = $row -split "`t", 2
        $branch = $fields[0]
        $upstream = if ($fields.Count -gt 1) { $fields[1] } else { "" }
        if ([string]::IsNullOrWhiteSpace($upstream)) {
            $contained = (Invoke-Git -C $SrcDir for-each-ref "--contains=$branch" "--format=%(refname)" refs/remotes refs/tags 2>$null | Out-String).Trim()
            if (-not $contained) { return $true }
            continue
        }

        Invoke-Git -C $SrcDir rev-parse --verify $upstream 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) { return $true }
        $ahead = (Invoke-Git -C $SrcDir rev-list --count "$upstream..$branch" 2>$null | Out-String).Trim()
        if ($LASTEXITCODE -ne 0 -or $ahead -ne "0") { return $true }
    }
    return $false
}

function Remove-ManagedSource {
    if (-not (Test-Path -LiteralPath $SrcDir)) { return }
    if (-not (Test-ManagedSource)) {
        Write-Host "Preserved: $SrcDir (not a recognized managed hipfire checkout)" -ForegroundColor Yellow
        return
    }
    if (Test-SourceHasLocalWork) {
        Write-Host "Preserved: $SrcDir (contains local work or git stashes)" -ForegroundColor Yellow
        Write-Host "  Review it manually, or use -Purge to delete all hipfire data."
        return
    }
    Remove-SafeTree $SrcDir
}

function Confirm-Purge {
    if (-not $Purge -or $DryRun -or $Yes) { return }
    Write-Host "PURGE will permanently delete models, settings, and every file under:" -ForegroundColor Red
    Write-Host "  $HipfireDir"
    $reply = Read-Host "Type 'delete' to continue"
    if ($reply -cne "delete") {
        Write-Host "Purge cancelled." -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "=== hipfire uninstaller ===" -ForegroundColor Cyan
Write-Host "Install root: $HipfireDir"
if ($DryRun) {
    Write-Host "Mode: dry run"
} elseif ($Purge) {
    Write-Host "Mode: purge"
} else {
    Write-Host "Mode: preserve models and settings"
}
Write-Host ""

Confirm-Purge
Stop-InstalledProcesses
Remove-InstallPath

if ($Purge) {
    Remove-SafeTree $HipfireDir
} else {
    Remove-SafeTree $BinDir
    Remove-SafeTree $RuntimeDir
    Remove-SafeTree $LegacyCliDir
    Remove-ManagedSource
    Remove-SafeFile (Join-Path $HipfireDir "serve.pid")
    Remove-SafeFile (Join-Path $HipfireDir "daemon.pid")
    Remove-SafeFile (Join-Path $HipfireDir "serve.log")
}

Write-Host ""
if ($DryRun) {
    Write-Host "Dry run complete; nothing was changed."
} elseif ($Purge) {
    Write-Host "hipfire and all ~/.hipfire data were removed."
} else {
    Write-Host "hipfire was uninstalled."
    Write-Host "Models and settings were preserved under $HipfireDir."
    Write-Host "Run this script with --purge to remove that data too."
}
Write-Host "ROCm, Rust, and other shared system dependencies were not removed."

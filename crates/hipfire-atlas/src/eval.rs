// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::task::TaskBundle;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The POSIX shell Atlas task commands are written for, and the argument that
/// makes it read a command string.
#[derive(Debug, Clone)]
struct EvalShell {
    program: PathBuf,
    args: Vec<String>,
}

#[derive(Debug, Clone)]
struct ShellCandidate {
    program: PathBuf,
    available: bool,
}

impl ShellCandidate {
    fn available(program: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            available: true,
        }
    }

    fn probe(program: impl Into<PathBuf>) -> Self {
        let program = program.into();
        Self {
            available: program.is_file(),
            program,
        }
    }
}

const POSIX_SHELL_ERROR: &str = "Kernel Atlas task commands are POSIX shell scripts. Windows \
needs a POSIX shell to run them; install Git for Windows, or set HIPFIRE_ATLAS_SHELL to another \
POSIX shell.";

fn select_eval_shell(
    shell_override: Option<&str>,
    candidates: &[ShellCandidate],
) -> Result<EvalShell, String> {
    if let Some(shell_override) = shell_override.filter(|value| !value.trim().is_empty()) {
        return Ok(EvalShell {
            program: PathBuf::from(shell_override),
            // Keep the existing login-shell behavior: these bundles rely on it
            // to place their toolchain on PATH, unlike a plain `-c` shell.
            args: vec!["-lc".to_string()],
        });
    }

    candidates
        .iter()
        .find(|candidate| candidate.available)
        .map(|candidate| EvalShell {
            program: candidate.program.clone(),
            // Keep the existing login-shell behavior: these bundles rely on it
            // to place their toolchain on PATH, unlike a plain `-c` shell.
            args: vec!["-lc".to_string()],
        })
        .ok_or_else(|| POSIX_SHELL_ERROR.to_string())
}

/// Resolves HIPFIRE_ATLAS_SHELL as an escape hatch for hosts whose POSIX shell
/// is not discoverable through the standard platform locations.
fn resolve_eval_shell() -> Result<EvalShell, String> {
    let shell_override = env::var("HIPFIRE_ATLAS_SHELL").ok();
    if !cfg!(windows) {
        return select_eval_shell(
            shell_override.as_deref(),
            &[ShellCandidate::available("sh")],
        );
    }

    select_eval_shell(shell_override.as_deref(), &windows_shell_candidates())
}

fn windows_shell_candidates() -> Vec<ShellCandidate> {
    let mut candidates = vec![ShellCandidate {
        program: PathBuf::from("bash.exe"),
        // Let Windows resolve PATH instead of duplicating its search semantics.
        available: Command::new("bash.exe").arg("--version").output().is_ok(),
    }];

    for git in git_executable_paths() {
        if let Some(git_dir) = git.parent() {
            if let Some(git_root) = git_dir.parent() {
                candidates.push(ShellCandidate::probe(git_root.join("bin").join("bash.exe")));
                candidates.push(ShellCandidate::probe(
                    git_root.join("usr").join("bin").join("bash.exe"),
                ));
            }
        }
    }

    for variable in ["ProgramFiles", "ProgramFiles(x86)"] {
        if let Some(root) = env::var_os(variable) {
            candidates.push(ShellCandidate::probe(
                PathBuf::from(&root)
                    .join("Git")
                    .join("bin")
                    .join("bash.exe"),
            ));
            candidates.push(ShellCandidate::probe(
                PathBuf::from(root)
                    .join("Git")
                    .join("usr")
                    .join("bin")
                    .join("bash.exe"),
            ));
        }
    }

    candidates
}

fn git_executable_paths() -> Vec<PathBuf> {
    let Ok(output) = Command::new("where.exe").arg("git.exe").output() else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandResult {
    pub command: String,
    pub status: i32,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    pub schema: String,
    pub task_id: String,
    pub status: String,
    pub commands: Vec<CommandResult>,
}

pub fn eval_task_file(path: impl AsRef<Path>, cwd: Option<&str>) -> Result<EvalResult, String> {
    let text = fs::read_to_string(path.as_ref())
        .map_err(|e| format!("read task {}: {e}", path.as_ref().display()))?;
    let task: TaskBundle = serde_json::from_str(&text)
        .map_err(|e| format!("parse task {}: {e}", path.as_ref().display()))?;
    eval_task(&task, cwd)
}

pub fn eval_task(task: &TaskBundle, cwd: Option<&str>) -> Result<EvalResult, String> {
    let mut commands = Vec::new();
    let mut shell = None;
    for command in task
        .correctness_commands
        .iter()
        .chain(task.eval_commands.iter())
    {
        if shell.is_none() {
            shell = Some(
                resolve_eval_shell()
                    .map_err(|error| format!("run command {command:?}: {error}"))?,
            );
        }
        commands.push(run_shell(
            command,
            cwd,
            shell
                .as_ref()
                .expect("shell is resolved before running commands"),
        )?);
    }
    let pass = commands.iter().all(|result| result.status == 0);
    Ok(EvalResult {
        schema: "hipfire.kernel_atlas.eval.v0".to_string(),
        task_id: task.task_id.clone(),
        status: if pass { "pass" } else { "fail" }.to_string(),
        commands,
    })
}

fn run_shell(command: &str, cwd: Option<&str>, shell: &EvalShell) -> Result<CommandResult, String> {
    let mut cmd = Command::new(&shell.program);
    cmd.args(&shell.args).arg(command);
    if let Some(cwd) = cwd {
        cmd.current_dir(cwd);
    }
    let output = cmd
        .output()
        .map_err(|e| format!("run command {command:?}: {e}"))?;
    Ok(CommandResult {
        command: command.to_string(),
        status: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_override_wins_over_candidates() {
        let candidates = [ShellCandidate::available("bash.exe")];

        let shell = select_eval_shell(Some("custom-posix-shell"), &candidates).unwrap();

        assert_eq!(shell.program, PathBuf::from("custom-posix-shell"));
        assert_eq!(shell.args, vec!["-lc".to_string()]);
    }

    #[test]
    fn shell_policy_chooses_first_available_windows_candidate() {
        let candidates = [
            ShellCandidate {
                program: PathBuf::from(r"C:\first\bash.exe"),
                available: false,
            },
            ShellCandidate::available(r"C:\Program Files\Git\bin\bash.exe"),
            ShellCandidate::available(r"C:\Program Files\Git\usr\bin\bash.exe"),
        ];

        let shell = select_eval_shell(None, &candidates).unwrap();

        if cfg!(windows) {
            assert_eq!(
                shell.program,
                PathBuf::from(r"C:\Program Files\Git\bin\bash.exe")
            );
        } else {
            assert_eq!(
                shell.program,
                PathBuf::from(r"C:\Program Files\Git\bin\bash.exe")
            );
        }
        assert_eq!(shell.args, vec!["-lc".to_string()]);
    }

    #[test]
    fn shell_policy_reports_missing_posix_shell() {
        let error = select_eval_shell(None, &[]).unwrap_err();

        assert!(error.contains("POSIX shell scripts"));
        assert!(error.contains("HIPFIRE_ATLAS_SHELL"));
        assert!(error.contains("Git for Windows"));
    }
}

//! Layer 2 — Sandboxed process launcher (bubblewrap / firejail).
//!
//! `lamu agent <command...>` runs the command inside a strict
//! container:
//! - cwd bound rw, everything else either read-only or hidden
//! - $HOME, ~/.ssh, ~/.aws, ~/.gnupg invisible
//! - /etc/sudoers, /etc/shadow invisible
//! - network namespace dropped by default (no internet)
//! - tmpfs /tmp
//! - new pid namespace, no setuid
//!
//! Bubblewrap (`bwrap`) is the preferred backend — small, on Arch core,
//! used by Flatpak. Firejail is the fallback. If neither exists, refuse
//! to launch and print install instructions.

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug, Clone)]
pub struct SandboxOpts {
    /// Project / working directory the agent is allowed to mutate.
    pub workdir: PathBuf,
    /// Allow network access. Off by default — agents shouldn't need it
    /// unless explicitly asked.
    pub allow_net: bool,
    /// Read-only bind mounts beyond the defaults (e.g. /opt/local-models).
    pub extra_ro: Vec<PathBuf>,
}

impl SandboxOpts {
    pub fn new(workdir: PathBuf) -> Self {
        Self { workdir, allow_net: false, extra_ro: Vec::new() }
    }

    pub fn with_net(mut self) -> Self { self.allow_net = true; self }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SandboxBackend {
    Bwrap,
    Firejail,
    None,
}

pub fn detect_backend() -> SandboxBackend {
    if which("bwrap").is_some() { return SandboxBackend::Bwrap; }
    if which("firejail").is_some() { return SandboxBackend::Firejail; }
    SandboxBackend::None
}

fn which(bin: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|p| p.join(bin))
            .find(|p| p.is_file())
    })
}

/// Build the full argv to run `cmd_argv` inside the chosen sandbox.
/// Returns `(program, args)` ready to pass to `Command::new`.
pub fn build_launch(
    backend: SandboxBackend,
    opts: &SandboxOpts,
    cmd_argv: &[String],
) -> Result<(String, Vec<String>)> {
    if cmd_argv.is_empty() {
        anyhow::bail!("empty command");
    }
    match backend {
        SandboxBackend::Bwrap => Ok(("bwrap".into(), bwrap_args(opts, cmd_argv))),
        SandboxBackend::Firejail => Ok(("firejail".into(), firejail_args(opts, cmd_argv))),
        SandboxBackend::None => anyhow::bail!(
            "no sandbox backend available. Install bubblewrap (`pacman -S bubblewrap` / `apt install bubblewrap`) or firejail."
        ),
    }
}

fn bwrap_args(opts: &SandboxOpts, cmd_argv: &[String]) -> Vec<String> {
    let mut a: Vec<String> = vec![
        "--die-with-parent".into(),
        "--unshare-user".into(),
        "--unshare-pid".into(),
        "--unshare-ipc".into(),
        "--unshare-uts".into(),
        "--new-session".into(),
        // Read-only essentials.
        "--ro-bind".into(), "/usr".into(), "/usr".into(),
        "--ro-bind".into(), "/etc/ssl".into(), "/etc/ssl".into(),
        "--ro-bind".into(), "/etc/ca-certificates".into(), "/etc/ca-certificates".into(),
        "--ro-bind-try".into(), "/etc/resolv.conf".into(), "/etc/resolv.conf".into(),
        "--symlink".into(), "usr/bin".into(), "/bin".into(),
        "--symlink".into(), "usr/lib".into(), "/lib".into(),
        "--symlink".into(), "usr/lib".into(), "/lib64".into(),
        "--symlink".into(), "usr/sbin".into(), "/sbin".into(),
        "--proc".into(), "/proc".into(),
        "--dev".into(), "/dev".into(),
        "--tmpfs".into(), "/tmp".into(),
        "--tmpfs".into(), "/run".into(),
        // Workdir: the only writable path.
        "--bind".into(), opts.workdir.display().to_string(),
        opts.workdir.display().to_string(),
        "--chdir".into(), opts.workdir.display().to_string(),
    ];
    if !opts.allow_net {
        a.push("--unshare-net".into());
    }
    for ro in &opts.extra_ro {
        a.push("--ro-bind".into());
        a.push(ro.display().to_string());
        a.push(ro.display().to_string());
    }
    // Minimal env. Agent should re-export anything else explicitly.
    a.push("--clearenv".into());
    a.push("--setenv".into()); a.push("PATH".into()); a.push("/usr/bin:/usr/local/bin".into());
    a.push("--setenv".into()); a.push("HOME".into()); a.push(opts.workdir.display().to_string());
    a.push("--setenv".into()); a.push("TERM".into());
    a.push(std::env::var("TERM").unwrap_or_else(|_| "xterm-256color".into()));
    a.push("--".into());
    a.extend(cmd_argv.iter().cloned());
    a
}

fn firejail_args(opts: &SandboxOpts, cmd_argv: &[String]) -> Vec<String> {
    let mut a: Vec<String> = vec![
        "--quiet".into(),
        "--noprofile".into(),
        "--private-tmp".into(),
        format!("--private={}", opts.workdir.display()),
        "--no3d".into(),
        "--nogroups".into(),
        "--nonewprivs".into(),
        "--noroot".into(),
        "--seccomp".into(),
        "--blacklist=/root".into(),
        "--blacklist=/etc/sudoers".into(),
        "--blacklist=/etc/shadow".into(),
        "--blacklist=/home/*/.ssh".into(),
        "--blacklist=/home/*/.gnupg".into(),
        "--blacklist=/home/*/.aws".into(),
    ];
    if !opts.allow_net {
        a.push("--net=none".into());
    }
    for ro in &opts.extra_ro {
        a.push(format!("--read-only={}", ro.display()));
    }
    a.push("--".into());
    a.extend(cmd_argv.iter().cloned());
    a
}

pub fn run(opts: &SandboxOpts, cmd_argv: &[String]) -> Result<std::process::ExitStatus> {
    let backend = detect_backend();
    let (prog, args) = build_launch(backend.clone(), opts, cmd_argv)?;
    eprintln!("[sandbox: {:?}] running {:?}", backend, cmd_argv);
    let status = Command::new(&prog).args(&args).status()
        .with_context(|| format!("spawn {}", prog))?;
    Ok(status)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bwrap_args_block_net_by_default() {
        let opts = SandboxOpts::new(PathBuf::from("/tmp/proj"));
        let args = bwrap_args(&opts, &["echo".into(), "hi".into()]);
        assert!(args.iter().any(|a| a == "--unshare-net"));
        assert!(args.iter().any(|a| a == "--clearenv"));
    }

    #[test]
    fn bwrap_args_drop_net_flag_when_allowed() {
        let opts = SandboxOpts::new(PathBuf::from("/tmp/proj")).with_net();
        let args = bwrap_args(&opts, &["echo".into(), "hi".into()]);
        assert!(!args.iter().any(|a| a == "--unshare-net"));
    }

    #[test]
    fn build_launch_errors_when_no_backend() {
        let opts = SandboxOpts::new(PathBuf::from("/tmp/proj"));
        let r = build_launch(SandboxBackend::None, &opts, &["x".into()]);
        assert!(r.is_err());
    }

    #[test]
    fn firejail_args_blacklist_secrets() {
        let opts = SandboxOpts::new(PathBuf::from("/tmp/proj"));
        let args = firejail_args(&opts, &["echo".into()]);
        assert!(args.iter().any(|a| a.contains(".ssh")));
        assert!(args.iter().any(|a| a.contains("sudoers")));
        assert!(args.iter().any(|a| a == "--net=none"));
    }
}

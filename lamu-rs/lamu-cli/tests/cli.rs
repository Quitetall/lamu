//! CLI parsing tests.
//! Drives the `lamu` binary with `--help` / unknown subcommand arguments.

use std::process::Command;

fn lamu() -> Command {
    Command::new(env!("CARGO_BIN_EXE_lamu"))
}

#[test]
fn help_succeeds() {
    let out = lamu().arg("--help").output().expect("run");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("scan"));
    assert!(stdout.contains("status"));
    assert!(stdout.contains("start"));
    assert!(stdout.contains("serve"));
}

#[test]
fn version_succeeds() {
    let out = lamu().arg("--version").output().expect("run");
    assert!(out.status.success());
}

#[test]
fn unknown_command_fails() {
    let out = lamu().arg("totally-bogus").output().expect("run");
    assert!(!out.status.success());
}

#[test]
fn serve_help_lists_port_flag() {
    let out = lamu().args(["serve", "--help"]).output().expect("run");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("--port") || stdout.contains("-p"));
}

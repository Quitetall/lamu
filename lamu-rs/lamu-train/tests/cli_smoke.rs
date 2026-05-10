//! End-to-end smoke for the lamu-train CLI binary.
//!
//! Invokes the compiled binary as a subprocess so flag parsing,
//! subcommand dispatch, and exit codes match what users see.
//! Uses `--help` and `jobs` (no-op when no jobs) to avoid spawning
//! the trainer.

use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn binary() -> PathBuf {
    // Cargo puts the binary at target/<profile>/<name> when running
    // integration tests; CARGO_BIN_EXE_<name> is the env var Cargo
    // sets so tests don't have to guess the profile dir.
    PathBuf::from(env!("CARGO_BIN_EXE_lamu-train"))
}

#[test]
fn help_prints_top_level() {
    let out = Command::new(binary())
        .arg("--help")
        .output()
        .expect("spawn lamu-train");
    assert!(out.status.success(), "--help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("lamu-train"), "missing tool name in help");
    assert!(stdout.contains("Local fine-tuning"), "missing about line");
}

#[test]
fn train_help_lists_critical_flags() {
    let out = Command::new(binary())
        .arg("train")
        .arg("--help")
        .output()
        .expect("spawn");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    for needle in ["--base", "--dataset", "--method", "--optim", "--allow-evict", "--background"] {
        assert!(stdout.contains(needle), "train help missing {needle}");
    }
}

#[test]
fn jobs_subcommand_handles_empty() {
    let _g = ENV_LOCK.lock().unwrap();
    // Point at an empty jobs dir; subcommand must succeed + print
    // the no-jobs message rather than erroring.
    let td = tempfile::tempdir().unwrap();
    let out = Command::new(binary())
        .env("LAMU_TRAIN_JOBS_DIR", td.path())
        .arg("jobs")
        .output()
        .expect("spawn");
    assert!(out.status.success(), "jobs on empty dir must succeed");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("no jobs"), "expected no-jobs message, got: {stdout}");
}

#[test]
fn cancel_unknown_id_errors() {
    let _g = ENV_LOCK.lock().unwrap();
    let td = tempfile::tempdir().unwrap();
    let out = Command::new(binary())
        .env("LAMU_TRAIN_JOBS_DIR", td.path())
        .arg("cancel")
        .arg("nonexistent-job-id")
        .output()
        .expect("spawn");
    assert!(!out.status.success(), "cancel of unknown id must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("no job matches") || stderr.contains("nonexistent"),
        "stderr: {stderr}"
    );
}

#[test]
fn log_subcommand_renders_status() {
    use lamu_train::jobs::{self, JobState};
    use lamu_train::protocol::StatusUpdate;
    use std::path::PathBuf;

    let _g = ENV_LOCK.lock().unwrap();
    let td = tempfile::tempdir().unwrap();

    // Set the env var for both the test setup AND the spawned binary.
    let prev = std::env::var("LAMU_TRAIN_JOBS_DIR").ok();
    // SAFETY: ENV_LOCK serializes mutations across this test binary.
    unsafe {
        std::env::set_var("LAMU_TRAIN_JOBS_DIR", td.path());
    }

    let id = "20260510-120000-000000001";
    jobs::write_state(id, JobState::Done).unwrap();
    jobs::append_status(
        id,
        &StatusUpdate::Step {
            step: 1,
            total: 10,
            loss: 1.5,
            lr: 2e-4,
            vram_mb: 8000,
        },
    )
    .unwrap();
    jobs::append_status(
        id,
        &StatusUpdate::Done {
            final_loss: 0.5,
            checkpoint_dir: PathBuf::from("/tmp/ckpt"),
        },
    )
    .unwrap();

    let out = Command::new(binary())
        .env("LAMU_TRAIN_JOBS_DIR", td.path())
        .arg("log")
        .arg(id)
        .output()
        .expect("spawn");
    assert!(out.status.success(), "log subcommand must succeed");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("step 1/10"), "log must render step lines: {stdout}");
    assert!(stdout.contains("done"), "log must render done line: {stdout}");

    unsafe {
        match prev {
            Some(v) => std::env::set_var("LAMU_TRAIN_JOBS_DIR", v),
            None => std::env::remove_var("LAMU_TRAIN_JOBS_DIR"),
        }
    }
}

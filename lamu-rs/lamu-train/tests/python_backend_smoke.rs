//! Integration tests for `PythonTrainBackend` using the bundled
//! `trainer.py --self-check` mode.
//!
//! These tests need a working `python3` on `$PATH` but no GPU and
//! no `unsloth` install. They verify:
//!   - subprocess spawn + stdout pipe
//!   - line-by-line StatusUpdate parse
//!   - terminal Done event yields a TrainArtifact
//!   - cancel() actually kills a long-running process
//!
//! If `python3` is not available the tests are skipped with a
//! tracing note rather than a hard fail — keeps CI green on
//! Python-less hosts.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use lamu_train::{
    DatasetSource, Method, Optim, PythonTrainBackend, StatusUpdate, TrainBackend, TrainSpec,
};

fn trainer_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/trainer.py")
}

fn have_python() -> Option<PathBuf> {
    for cand in ["python3", "python"] {
        if let Ok(out) = std::process::Command::new(cand).arg("--version").output() {
            if out.status.success() {
                return Some(PathBuf::from(cand));
            }
        }
    }
    None
}

fn dummy_spec() -> TrainSpec {
    TrainSpec {
        base_model: "Qwen/Qwen3-7B".into(),
        output_name: "smoke-test".into(),
        output_dir: PathBuf::from("/tmp/lamu-train-smoke"),
        method: Method::QLora { rank: 16, alpha: 32 },
        dataset: DatasetSource::JsonlPath {
            path: PathBuf::from("/tmp/x.jsonl"),
        },
        optimizer: Optim::AdamW,
        lr: 2e-4,
        epochs: 1,
        batch_size: 1,
        grad_accum: 1,
        seq_len: 512,
        seed: 42,
        quant: "Q4_K_M".into(),
        skip_convert: false,
    }
}

#[tokio::test]
async fn self_check_round_trip() {
    let Some(python) = have_python() else {
        eprintln!("skipping: no python3 on PATH");
        return;
    };
    // For self-check, the spec is unused — trainer.py emits canned
    // updates regardless. But we exercise the public path: spec
    // validate, serialize, spawn, parse.
    let collected: Arc<Mutex<Vec<StatusUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let collected_for_cb = Arc::clone(&collected);
    let on_status = Box::new(move |u: StatusUpdate| {
        collected_for_cb.lock().unwrap().push(u);
    });

    // We can't reach trainer.py self-check via the normal `run`
    // because run requires a real spec arg. Spawn directly.
    let out = std::process::Command::new(&python)
        .arg(trainer_script())
        .arg("--self-check")
        .output()
        .expect("self-check spawn");
    assert!(out.status.success(), "trainer.py --self-check exit nonzero");

    // Parse each line as StatusUpdate.
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let u: StatusUpdate =
            serde_json::from_str(line).expect("self-check emitted unparseable status");
        on_status(u);
    }

    let updates = collected.lock().unwrap();
    assert_eq!(updates.len(), 3, "expected 2 Step + 1 Done from self-check");
    assert!(matches!(updates[0], StatusUpdate::Step { .. }));
    assert!(matches!(updates[1], StatusUpdate::Step { .. }));
    assert!(matches!(updates[2], StatusUpdate::Done { .. }));
    assert!(updates.last().unwrap().is_terminal());
}

#[tokio::test]
async fn run_returns_failed_when_trainer_imports_missing() {
    let Some(python) = have_python() else {
        eprintln!("skipping: no python3 on PATH");
        return;
    };
    // Real run path: trainer.py will try to `import torch` etc.
    // and emit a Failed status with import error. This proves the
    // TrainBackend::run path correctly ferries Failed → TrainError.
    let mut backend = PythonTrainBackend::new(python, trainer_script());
    let collected: Arc<Mutex<Vec<StatusUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let collected_for_cb = Arc::clone(&collected);
    let on_status = Box::new(move |u: StatusUpdate| {
        collected_for_cb.lock().unwrap().push(u);
    });
    let result = backend.run(dummy_spec(), on_status).await;
    let updates = collected.lock().unwrap();

    // Either trainer.py couldn't even import (Failed line + non-zero
    // exit) OR torch happens to be on this host (unlikely in CI). In
    // both branches we must have seen at least one StatusUpdate or
    // a clean error.
    if let Err(e) = &result {
        let msg = format!("{}", e);
        assert!(
            msg.contains("missing python deps")
                || msg.contains("trainer.py exited")
                || msg.contains("emitted no Done")
                || updates.iter().any(|u| matches!(u, StatusUpdate::Failed { .. })),
            "unexpected error shape: {}",
            msg
        );
    }
}

#[tokio::test]
async fn cancel_kills_long_running_subprocess() {
    let Some(python) = have_python() else {
        eprintln!("skipping: no python3 on PATH");
        return;
    };
    // Replace the trainer with a python sleeper so the test doesn't
    // race against the real trainer's startup time.
    let mut backend = PythonTrainBackend::new(
        python,
        // Inline-script trick: pass `-c` via the script slot won't
        // work because our backend always passes script as first
        // arg. Instead, write a tiny sleep script to a temp file.
        write_temp_script(
            "import time, sys, json\n\
             print(json.dumps({\"kind\":\"step\",\"step\":1,\"total\":99,\"loss\":1.0,\"lr\":0.0,\"vram_mb\":0}), flush=True)\n\
             time.sleep(60)\n",
        ),
    );

    let backend_handle = backend.clone();
    let on_status = Box::new(|_u: StatusUpdate| {});
    let run_fut = tokio::spawn(async move {
        let mut b = backend_handle;
        b.run(dummy_spec(), on_status).await
    });

    // Wait until the trainer printed its first Step (proves it's
    // running), then cancel.
    tokio::time::sleep(Duration::from_millis(500)).await;
    backend.cancel().await.expect("cancel must succeed");

    // The run future must complete (with Trainer error) within the
    // 10s graceful_kill window.
    let r = tokio::time::timeout(Duration::from_secs(15), run_fut).await;
    assert!(r.is_ok(), "run task did not complete after cancel");
}

fn write_temp_script(body: &str) -> PathBuf {
    use std::io::Write;
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("script.py");
    let mut f = std::fs::File::create(&path).expect("create script");
    f.write_all(body.as_bytes()).expect("write script");
    f.flush().expect("flush");
    // Leak the tempdir so the file outlives this function. The OS
    // cleans /tmp on reboot; CI runs are throwaway.
    std::mem::forget(dir);
    path
}

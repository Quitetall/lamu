//! `lamu-train` — local fine-tuning subcommand binary.
//!
//! Top-level subcommands:
//!
//!   train (default)        Run a single fine-tune to completion (or
//!                          background if --background).
//!   jobs                   List jobs (running + completed).
//!   cancel <id>            SIGTERM the trainer subprocess.
//!   log <id>               Print rendered status.jsonl tail.
//!
//! Acquires the cross-process scheduler lock for the duration of a
//! training run; inference paths (lamu-mcp, lamu-api) refuse during
//! that window. Pass `--allow-evict` to wait if the lock is already
//! held by an inference exclusive instead of erroring.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use lamu_core::scheduler_lock::{self, LockKind};
use lamu_train::{
    backend::{StatusFn, TrainBackend},
    convert,
    jobs::{self, JobState},
    paths,
    protocol::StatusUpdate,
    python_backend::PythonTrainBackend,
    spec::{DatasetSource, Method, Optim, TrainSpec},
};

#[derive(Parser, Debug)]
#[command(name = "lamu-train", version, about = "Local fine-tuning")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    #[command(flatten)]
    train_args: TrainArgs,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run a fine-tune (also the default when invoked without a subcommand).
    Train(TrainArgs),
    /// List training jobs (running + completed).
    Jobs,
    /// SIGTERM a running training job.
    Cancel {
        /// Job id (or unique prefix).
        id: String,
        /// Grace period before escalating to SIGKILL.
        #[arg(long, default_value = "10s", value_parser = parse_duration)]
        grace: Duration,
    },
    /// Print rendered training log for a job.
    Log {
        /// Job id (or unique prefix).
        id: String,
        /// How many lines to tail. 0 = all.
        #[arg(long, default_value_t = 0)]
        tail: usize,
    },
}

#[derive(Args, Debug)]
struct TrainArgs {
    /// Registry name for the trained model. Required for actual runs;
    /// missing → help text.
    output_name: Option<String>,

    /// HuggingFace base model (org/name).
    #[arg(long, default_value = "Qwen/Qwen3-7B")]
    base: String,

    /// JSONL chat dataset path.
    #[arg(long)]
    dataset: Option<PathBuf>,

    /// Pull conversations from lamu-mcp memory (overrides --dataset).
    /// Materialization happens in step 7; flag accepted now for parity.
    #[arg(long, default_value_t = false)]
    from_conversations: bool,

    /// Window for --from-conversations.
    #[arg(long, default_value = "30d", value_parser = parse_duration)]
    since: Duration,

    /// Fine-tuning method.
    #[arg(long, value_enum, default_value_t = MethodArg::Qlora)]
    method: MethodArg,

    /// LoRA rank.
    #[arg(long, default_value_t = 16)]
    rank: u32,

    /// LoRA alpha.
    #[arg(long, default_value_t = 32)]
    alpha: u32,

    /// Optimizer.
    #[arg(long, value_enum)]
    optim: Option<OptimArg>,

    #[arg(long, default_value_t = 2e-4)]
    lr: f32,

    #[arg(long, default_value_t = 3)]
    epochs: u32,

    #[arg(long, default_value_t = 1)]
    batch_size: u32,

    #[arg(long, default_value_t = 8)]
    grad_accum: u32,

    #[arg(long, default_value_t = 4096)]
    seq_len: u32,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Final GGUF quant.
    #[arg(long, default_value = "Q4_K_M")]
    quant: String,

    /// Skip GGUF convert + registry register (HF checkpoint only).
    #[arg(long, default_value_t = false)]
    no_convert: bool,

    /// Detach: write to ~/.local/share/lamu/train-jobs/<id>/, return
    /// the job id immediately. Use `lamu-train jobs` + `log <id>`.
    /// (Implementation: spawns a child of itself with --foreground.
    ///  v1 placeholder — wires to real detach in step 10 hardening.)
    #[arg(long, default_value_t = false)]
    background: bool,

    /// Wait for the GPU lock to release instead of erroring on hold.
    /// Polling interval is 500 ms; default timeout 1 h.
    #[arg(long, default_value_t = false)]
    allow_evict: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum MethodArg {
    Qlora,
    Lora,
    Full,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OptimArg {
    Adamw,
    Adamw8bit,
    Apollo,
    ApolloMini,
}

fn parse_duration(s: &str) -> Result<Duration, String> {
    humantime::parse_duration(s).map_err(|e| format!("{e}"))
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Train(args)) => run_train(args).await,
        Some(Command::Jobs) => run_jobs(),
        Some(Command::Cancel { id, grace }) => run_cancel(&id, grace).await,
        Some(Command::Log { id, tail }) => run_log(&id, tail),
        None => run_train(cli.train_args).await,
    }
}

fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let _ = fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,lamu_train=info,hyper=warn,reqwest=warn")),
        )
        .with_target(false)
        .with_writer(std::io::stderr)
        .try_init();
}

async fn run_train(args: TrainArgs) -> Result<()> {
    let output_name = args
        .output_name
        .clone()
        .ok_or_else(|| anyhow!("output-name is required (positional). See `lamu-train --help`."))?;

    let dataset_src = build_dataset(&args)?;
    let optimizer = pick_optimizer(args.optim, args.method);
    let method = build_method(args.method, args.rank, args.alpha);

    let job_id = jobs::new_job_id();
    let job_dir = paths::job_dir(&job_id)
        .with_context(|| format!("create job dir for {job_id}"))?;
    let output_dir = job_dir.join("checkpoint");
    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("create checkpoint dir {}", output_dir.display()))?;

    // trainer.py only accepts JsonlPath at runtime. Materialize
    // Conversations sources to a JSONL file under paths::data_dir
    // before spec construction so the file path lands in the
    // committed spec.json on disk for audit.
    let dataset = match dataset_src {
        DatasetSource::Conversations { .. } => {
            let data_dir = paths::data_dir().context("resolve train-data dir")?;
            std::fs::create_dir_all(&data_dir)
                .with_context(|| format!("create {}", data_dir.display()))?;
            let out_path = data_dir.join(format!("{job_id}.jsonl"));
            let stats = lamu_train::conversations::dump_to_jsonl(
                args.since,
                &out_path,
            )
            .context("dump conversations to JSONL")?;
            eprintln!(
                "dataset materialized: {} conversations, {} turns → {}",
                stats.n_conversations,
                stats.n_turns,
                stats.path.display()
            );
            if stats.n_conversations == 0 {
                return Err(anyhow!(
                    "no usable conversations in window (--since {:?}). \
                     {} short raw, {} gutted by filters, \
                     {} error messages, {} oversize messages.",
                    args.since,
                    stats.n_dropped_short,
                    stats.n_dropped_filtered_below_min,
                    stats.n_dropped_errors,
                    stats.n_dropped_oversize
                ));
            }
            DatasetSource::JsonlPath { path: out_path }
        }
        other => other,
    };

    let spec = TrainSpec {
        base_model: args.base.clone(),
        output_name: output_name.clone(),
        output_dir: output_dir.clone(),
        method,
        dataset,
        optimizer,
        lr: args.lr,
        epochs: args.epochs,
        batch_size: args.batch_size,
        grad_accum: args.grad_accum,
        seq_len: args.seq_len,
        seed: args.seed,
        quant: args.quant.clone(),
        skip_convert: args.no_convert,
    };
    spec.validate().context("TrainSpec validation")?;
    jobs::write_spec(&job_id, &spec)?;
    jobs::write_state(&job_id, JobState::Running)?;

    eprintln!("job  {job_id}");
    eprintln!("dir  {}", job_dir.display());

    if args.background {
        // Background scaffold — spawn ourselves with the same args
        // minus --background. v1 implementation: print the job id +
        // a hint; the actual detach is wired in a follow-up commit
        // since clean nohup-style detach + log redirection deserves
        // its own review pass.
        eprintln!(
            "background mode is recognised but real detach lands in a follow-up.\n\
             For now, run without --background and use `lamu-train cancel {job_id}`\n\
             from another terminal if you need to stop early."
        );
        return Ok(());
    }

    // Resolve subprocess paths BEFORE acquiring the GPU lock so a
    // path-resolution failure doesn't hold the lock. Cheap (a few
    // env reads + stat calls); failure here means the user's setup
    // is wrong and they need a clear error, not a held lock.
    let python = paths::resolve_python().context("resolve python")?;
    let trainer_script = paths::resolve_trainer_script().context("resolve trainer.py")?;
    eprintln!("python {}", python.display());
    eprintln!("trainer {}", trainer_script.display());

    // Acquire the GPU lock. --allow-evict waits for an existing
    // inference exclusive to release; otherwise hard error.
    let lock = if args.allow_evict {
        eprintln!("lock waiting for GPU release (--allow-evict, up to 1h)...");
        scheduler_lock::await_unlock(Duration::from_secs(3600))
            .await
            .context("await_unlock")?;
        scheduler_lock::acquire_exclusive(format!("lamu-train:{job_id}"), LockKind::Training)
            .context("acquire_exclusive after wait")?
    } else {
        scheduler_lock::acquire_exclusive(format!("lamu-train:{job_id}"), LockKind::Training)
            .context("acquire_exclusive (use --allow-evict to wait)")?
    };
    eprintln!("lock acquired ({})", lock.path().display());

    let mut backend = PythonTrainBackend::new(python, trainer_script);

    let job_id_for_cb = job_id.clone();
    let on_status: StatusFn = Box::new(move |u: StatusUpdate| {
        // Persist to status.jsonl + render to stderr so the user
        // sees progress live in foreground mode. A persist failure
        // (full disk, permissions, etc.) is logged but doesn't stop
        // the run — losing status history is bad but losing the
        // training job mid-flight is worse.
        if let Err(e) = jobs::append_status(&job_id_for_cb, &u) {
            tracing::warn!(
                "failed to persist status to {}: {}",
                job_id_for_cb,
                e
            );
        }
        match &u {
            StatusUpdate::Step {
                step, total, loss, lr, vram_mb,
            } => eprintln!("step {step}/{total}  loss={loss:.4}  lr={lr:.2e}  vram={vram_mb}MB"),
            StatusUpdate::Eval { step, eval_loss } => {
                eprintln!("eval @{step}  loss={eval_loss:.4}")
            }
            StatusUpdate::Saved { path } => eprintln!("saved {}", path.display()),
            StatusUpdate::Done {
                final_loss,
                checkpoint_dir,
            } => eprintln!(
                "done  final_loss={final_loss:.4}  ckpt={}",
                checkpoint_dir.display()
            ),
            StatusUpdate::Failed { error } => eprintln!("FAILED: {error}"),
        }
    });

    let result = backend.run(spec.clone(), on_status).await;

    drop(lock); // release GPU before convert + register; convert is
                // CPU-bound and llama.cpp tools don't need the card.

    match result {
        Ok(artifact) => {
            jobs::write_state(&job_id, JobState::Done)?;
            eprintln!(
                "trained in {:?}, final_loss={:.4}, ckpt={}",
                artifact.elapsed,
                artifact.final_loss,
                artifact.checkpoint_dir.display()
            );

            if !args.no_convert {
                eprintln!("converting to GGUF ({})...", args.quant);
                let gguf = convert::convert_to_gguf(
                    &artifact.checkpoint_dir,
                    &output_name,
                    &args.quant,
                )
                .await
                .context("convert_to_gguf")?;
                eprintln!("gguf  {}", gguf.display());
                register_in_registry(&output_name, &gguf, &spec)?;
                eprintln!("registry updated; `mcp__local-llm__query model={output_name}` should work.");
            } else {
                eprintln!("--no-convert: HF checkpoint left at {}", artifact.checkpoint_dir.display());
            }
        }
        Err(e) => {
            jobs::write_state(&job_id, JobState::Failed)?;
            return Err(anyhow!(e));
        }
    }
    Ok(())
}

fn run_jobs() -> Result<()> {
    let jobs = jobs::list_jobs()?;
    if jobs.is_empty() {
        println!("no jobs.");
        return Ok(());
    }
    println!("{:<24} {:<10} {:<6} {:<24} {}", "id", "state", "pid", "output", "last");
    for j in jobs {
        let last = match (j.last_step, j.last_loss, j.final_loss) {
            (_, _, Some(fl)) => format!("final_loss={fl:.4}"),
            (Some(step), Some(loss), _) => format!("step={step} loss={loss:.4}"),
            _ => "-".into(),
        };
        let pid = j.pid.map(|p| p.to_string()).unwrap_or_else(|| "-".into());
        let output = j.output_name.unwrap_or_else(|| "-".into());
        println!(
            "{:<24} {:<10} {:<6} {:<24} {}",
            j.id,
            j.state.as_str(),
            pid,
            output,
            last
        );
    }
    Ok(())
}

async fn run_cancel(id_query: &str, grace: Duration) -> Result<()> {
    let id = jobs::resolve_job_id(id_query)?;
    eprintln!("cancelling {id} (grace {grace:?})...");
    jobs::cancel_job(&id, grace).await?;
    eprintln!("cancelled.");
    Ok(())
}

fn run_log(id_query: &str, tail: usize) -> Result<()> {
    let id = jobs::resolve_job_id(id_query)?;
    let updates = jobs::read_status(&id)?;
    let rendered = jobs::render_log(&updates);
    if tail == 0 {
        print!("{rendered}");
    } else {
        let lines: Vec<&str> = rendered.lines().collect();
        let start = lines.len().saturating_sub(tail);
        for l in &lines[start..] {
            println!("{l}");
        }
    }
    Ok(())
}

fn build_dataset(args: &TrainArgs) -> Result<DatasetSource> {
    if args.from_conversations {
        // Step 7 materializes this to a JsonlPath; for v1 the CLI
        // accepts the flag and constructs the variant so the spec
        // round-trips cleanly. Use checked_sub so a since-window
        // larger than time-since-epoch (~55 years) saturates at 0
        // instead of panicking on SystemTime underflow.
        let cutoff = std::time::SystemTime::now()
            .checked_sub(args.since)
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        Ok(DatasetSource::Conversations { since_ts: cutoff })
    } else {
        let path = args.dataset.clone().ok_or_else(|| {
            anyhow!("--dataset is required unless --from-conversations is set")
        })?;
        Ok(DatasetSource::JsonlPath { path })
    }
}

fn build_method(method: MethodArg, rank: u32, alpha: u32) -> Method {
    match method {
        MethodArg::Qlora => Method::QLora { rank, alpha },
        MethodArg::Lora => Method::Lora { rank, alpha },
        MethodArg::Full => Method::Full,
    }
}

fn pick_optimizer(opt: Option<OptimArg>, method: MethodArg) -> Optim {
    if let Some(o) = opt {
        return match o {
            OptimArg::Adamw => Optim::AdamW,
            OptimArg::Adamw8bit => Optim::AdamW8bit,
            OptimArg::Apollo => Optim::ApolloRank4,
            OptimArg::ApolloMini => Optim::ApolloMini,
        };
    }
    // Defaults pegged to memory profile of each method.
    match method {
        MethodArg::Qlora => Optim::ApolloMini,
        MethodArg::Lora => Optim::AdamW8bit,
        MethodArg::Full => Optim::AdamW,
    }
}

fn register_in_registry(
    name: &str,
    gguf_path: &PathBuf,
    spec: &TrainSpec,
) -> Result<()> {
    use lamu_core::registry;
    use lamu_core::types::{
        BackendType, Capability, ModelEntry, ModelFormat, ModelStatus,
    };
    let registry_path = lamu_core::config::registry_path();
    let entry = ModelEntry {
        name: name.into(),
        path: gguf_path.clone(),
        format: ModelFormat::Gguf,
        backend: BackendType::LlamaCpp,
        arch: "trained".into(), // refined post-conversion in a future step
        params_b: 0.0,           // unknown until we parse GGUF
        quant: spec.quant.clone(),
        vram_mb: 0,
        context_max: spec.seq_len,
        capabilities: vec![Capability::Chat],
        reasoning_marker: None,
        speculative: None,
        pinned: false,
        notes: format!("trained from {} via lamu-train", spec.base_model),
        status: ModelStatus::default(),
    };
    registry::add_entry(entry, &registry_path, true)
        .map_err(|e| anyhow!("registry update failed: {e}"))
}


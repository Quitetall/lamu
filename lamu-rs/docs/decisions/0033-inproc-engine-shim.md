# ADR 0033: In-process engine backends behind a loopback HTTP shim (`lamu-inproc`)

## Status

Accepted 2026-06-12

## Context

New runtimes (ONNX via `ort`, candle next) are Rust LIBRARIES — no
subprocess to spawn. But LAMU's load architecture is port-anchored: the
loader drops the `Box<dyn Backend>` right after load and every consumer
proxies by port (`/v1/embeddings`, chat, health-confirm A6, the
port-anchored kill path). An embed-direct trait method would be
unreachable after that drop without re-architecting loader, health,
queue, and api proxying.

## Decision

In-process engines present a PORT: `lamu-inproc` provides an
`EmbedEngine` trait (sync `embed`, executed via `spawn_blocking`) and
`spawn_embed_server(port, name, engine)` — a tokio-task axum server
serving `/health` + `/v1/health` (`{"status":"ok"}`, matching both
existing health-poll conventions) and OpenAI-shape `/v1/embeddings`.
`Backend::load()` spawns the task and returns `std::process::id()` as
the pid; `unload()` aborts the task. `ChatEngine` (for candle, ADR 0035)
is reserved scope. Heavy engine crates are workspace members but
OPTIONAL deps of lamu-cli behind cargo features (`onnx`, later
`hf-candle`, umbrella `full`); workspace `default-members` excludes them
so a bare `cargo build` never compiles ort; the module-registry
"not registered" error now hints at the feature flag.

Safety consequence handled now: `kill_pid_and_verify` — the no-handle
unload path — would have signalled LAMU'S OWN process group for an
in-process backend (pid == self, port bound). It fails closed on
`pid == std::process::id()` with an explicit in-process error; v1
teardown is the owning handle's `unload()` or process exit (vram 0 makes
that a non-event for ONNX; candle revisits this in ADR 0035).

## Rationale

- Zero changes to loader/health/api/queue: the shim satisfies the
  port-anchored contracts exactly as a llama-server child would.
- Documented in-process tradeoffs vs subprocess: a panicking engine is
  inside lamu's process (mitigated by spawn_blocking + JoinError
  handling, never unwrap in the forward pass); PDEATHSIG hardening
  doesn't apply (nothing to leak); CUDA-context residency arrives only
  with candle (ADR 0035 records the margin).
- Feature-gating keeps the default build lean (ort downloads ONNX
  Runtime binaries; candle compiles CUDA kernels) while CI builds
  `--features full`.

## Alternatives Considered

- **Embed-direct trait method** — unreachable post-load (the Box is
  dropped); would fork every consumer into port-vs-direct paths.
- **Wrap engines in real subprocesses** — ops weight (a second binary,
  IPC) purely to imitate a constraint the shim satisfies in-process.
- **Always-compiled engines** — multi-minute default builds and binary
  downloads for users who never load an .onnx file.

## Related Decisions

ADR 0023/0026 (module/backend registry), ADR 0034 (first consumer),
ADR 0035 (candle, second consumer), A6 (health identity-confirm the
shim must satisfy).

## Validation

7 lamu-inproc tests (health both paths, embeddings string/array/error
shapes); self-pid guard test; composition_root covers feature-on
resolution AND feature-off not-registered-with-hint. Default `cargo
build` verified ort-free.

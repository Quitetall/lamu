# ADR 0004: Manage model backends as hardened child processes with unconditional PDEATHSIG

## Status
Accepted 2026-05-31

## Context
LAMU runs inference on external model servers — `llama-server` (the
BeeLlama fork for DFlash), plus Python servers for megakernel, DFlash,
fish-speech TTS, and ComfyUI image generation. Each loads weights into
GPU VRAM and holds them for the process lifetime. On a single 4090 (24 GB),
VRAM is the scarce resource the whole scheduler (ADR 0003) exists to
manage. A backend process that survives the LAMU process that spawned it
is a hard VRAM leak: its weights stay resident, nothing tracks it, and the
scheduler's accounting silently diverges from reality until the GPU is
reset.

Two facts about the runtime make leaks easy to hit. First, LAMU has two
spawn paths with different ownership models. The MCP path
(`handle_load_model`) retains the `Box<dyn Backend>` so it can call
`unload()` later. The HTTP path (`lamu serve`) deliberately drops the
trait object the instant load succeeds and thereafter proxies requests to
the still-running server by its port — `ensure_loaded_with` keeps only the
port in `scheduler.get_loaded(name)` and lets `_backend` fall out of scope
(loader.rs:153, loader.rs:170-178; documented loader.rs:6-14, mod.rs:103-120).
Second, LAMU itself can die abnormally: a hard crash, an external
`SIGKILL`, or its own orphan-watchdog calling `std::process::exit(0)`
(lifecycle.rs:66-100) — none of which run Rust destructors. Any teardown
that depends on a handle's `Drop` is therefore unreliable precisely in the
cases that produce orphans.

## Decision
LAMU spawns and owns every model server as a child process via
`tokio::process::Command`, and every spawn passes through
`harden_child_command` (mod.rs:129-157) which, on Linux, calls
`PR_SET_PDEATHSIG(SIGKILL)` in `pre_exec` so the kernel SIGKILLs the child
the moment its parent (LAMU) dies by any means. This is wired into all five
backends before `.spawn()` (llamacpp.rs:352, dflash.rs:90, megakernel.rs:75,
fish_speech.rs:130, comfyui.rs:91). It is unconditional — unlike LAMU's own
self-applied PDEATHSIG (lifecycle.rs), which uses SIGTERM and opts out under
`nohup`. We deliberately do NOT set `kill_on_drop(true)` on the child,
because the HTTP path drops the handle while the server must keep running;
PDEATHSIG is a property of the child process, independent of the handle, so
dropping the handle never disarms it. Normal teardown goes through each
backend's `unload()`, which calls `graceful_kill` — SIGTERM, wait 10s, then
SIGKILL (mod.rs:159-185).

## Rationale
- The VRAM-leak-on-death hole is the core threat: a leaked backend holds
  GPU memory with no owner and no scheduler entry, defeating ADR 0003's
  accounting until manual GPU reset. PDEATHSIG with SIGKILL closes it for
  every abnormal LAMU exit, including the ones that skip destructors
  (crash, external SIGKILL, watchdog `exit(0)`).
- `kill_on_drop(true)` is incompatible with the handle-drop proxy pattern:
  the HTTP path drops `_backend` right after load (loader.rs:153) and then
  serves requests against the still-live port. `kill_on_drop` would SIGKILL
  the server on that drop, leaving a phantom `Loaded` entry in the scheduler
  pointing at a dead port (mod.rs:112-116). PDEATHSIG gives crash-safety
  without coupling process lifetime to handle lifetime.
- The `getppid() == 1` re-check inside `pre_exec` (mod.rs:144-149) closes
  the fork-vs-parent-death race: if LAMU died between fork and the prctl
  call, the death signal would be relative to init (pid 1) and never fire.
  Detecting reparenting-to-init and failing the exec means the system never
  spawns an immortal orphan; it returns a spawn error instead.
- SIGKILL rather than SIGTERM for the backstop is correct because PDEATHSIG
  only fires when LAMU is already gone — there is no LAMU left to perform an
  orderly shutdown, so the only goal is guaranteed VRAM reclamation, and
  SIGKILL cannot be ignored. Graceful, flush-respecting teardown is handled
  separately by `graceful_kill` during normal `unload()` (mod.rs:159-185),
  which gives Python servers their 2-5s to flush before escalating.
- LAMU owns the lifecycle directly (spawn + register in the scheduler +
  unload) so it already has the PIDs, ports, and VRAM figures it needs for
  scheduling (loader.rs:162-178). An external supervisor would put that
  state in another process and another sync problem.

## Alternatives Considered
- **In-process inference (candle / llama-rs bindings)** — link the
  inference engine into the LAMU process so there is no child to orphan.
  Rejected: it forecloses the heterogeneous backend set LAMU actually runs.
  `make_backend` dispatches to five distinct servers including Python
  processes for megakernel, fish-speech, and ComfyUI (mod.rs:23-33); none
  of those are Rust libraries, and the DFlash path specifically needs the
  BeeLlama `llama-server` fork binary (mod.rs:17-22). In-process also means
  one OOM or segfault in native CUDA code takes down the whole orchestrator
  and every other loaded model, instead of a single isolable subprocess.
- **`kill_on_drop(true)`** — let Tokio SIGKILL the child when the handle
  drops. Rejected for a concrete correctness bug, not just style: the HTTP
  path drops the handle immediately after load and keeps serving by port
  (loader.rs:153, loader.rs:170-178), so `kill_on_drop` would kill the
  server on that drop and leave a phantom `Loaded` scheduler entry pointing
  at a dead port (mod.rs:112-116). It also still wouldn't cover the
  destructor-skipping exits (crash, external SIGKILL, watchdog `exit(0)`),
  which is exactly what PDEATHSIG is for.
- **External supervisor (systemd / supervisord)** — register each backend
  as a managed unit and let the supervisor handle lifetime. Rejected: it
  splits ownership of state the scheduler must hold inline — PID, port,
  measured VRAM, load/loading/unloaded transitions
  (loader.rs:132-178) — across a process boundary, creating a sync problem
  on every load and evict. It also doesn't actually solve the leak for free:
  systemd ties a child's life to a unit, not to the specific LAMU process
  instance, so a crashed-and-restarted LAMU would inherit or race against
  orphaned units. PDEATHSIG binds the child to the exact parent PID with no
  external daemon and no extra config surface.

## Consequences
- The kernel guarantees no backend outlives its LAMU: every abnormal exit
  path reclaims VRAM with no manual intervention. This is the property the
  scheduler in ADR 0003 relies on to keep its accounting trustworthy.
- The handle-drop proxy pattern (loader.rs) is now load-bearing and subtle:
  a future change that adds `kill_on_drop(true)` "for safety" would silently
  break the HTTP serve path. The long doc comment at mod.rs:103-120 exists
  to defend against exactly that, and must be kept in sync.
- The protection is Linux-only. `harden_child_command` is a no-op on other
  platforms including macOS (mod.rs:154-156, documented mod.rs:126-128).
  There is no portable PDEATHSIG equivalent; on non-Linux a LAMU crash can
  leak a backend. We accept this because the backends are CUDA/Linux-only in
  practice, so the gap is theoretical for the supported deployment.
- SIGKILL on parent death means backends get no flush-on-crash: a server
  that buffers its KV cache or logs loses them when LAMU dies abnormally.
  This is intended — PDEATHSIG only fires when no orderly shutdown is
  possible anyway — but it does mean post-crash backend logs may be
  truncated. Orderly shutdowns keep their flush window via `graceful_kill`.
- Backends are intentionally NOT detachable. Unlike LAMU itself, which
  honors `nohup` to survive a parent terminal (lifecycle.rs:34-45,
  lifecycle.rs:66-75), a backend has no legitimate reason to outlive its
  LAMU, so `harden_child_command` has no nohup escape hatch (mod.rs:122-124).
  This forecloses any future "keep the model warm across LAMU restarts"
  feature built on detached backends; such a feature would need a different
  mechanism (e.g. handoff via a stable socket), not a flag here.

## Related Decisions
ADR 0003 (the VRAM scheduler whose accounting integrity this leak-proofing
protects), ADR 0001 and ADR 0006 (the MCP-vs-HTTP split that produces the
two ownership models — MCP retains the handle and can `unload`/evict, HTTP
drops it and proxies by port and never auto-evicts).

## Validation
- Right if: after killing LAMU with `kill -9` while a model is loaded,
  `nvidia-smi` shows the backend process gone and its VRAM reclaimed within
  one scheduling tick, with no phantom `Loaded` entries. The
  `ensure_loaded_rolls_back_on_spawn_failure` test (loader.rs:432-453)
  already asserts no phantom entry survives a failed spawn; a parallel
  integration check should confirm no phantom survives a LAMU crash.
- Wrong / revisit if: orphaned backend processes appear in `S<l+` holding
  VRAM after LAMU exits (the symptom lifecycle.rs:10 documents for the
  pre-watchdog era), or if the fork-vs-death race produces immortal orphans
  despite the `getppid() == 1` guard, or if a non-Linux deployment becomes a
  real target and the no-op stub starts leaking in practice.
- Also revisit if a warm-across-restart requirement emerges: that directly
  contradicts the "a backend must never survive its LAMU" invariant and
  would force re-opening this decision rather than patching around it.


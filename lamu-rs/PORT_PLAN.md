# Rust Port Plan

Skeleton in place. Port mechanically from Python — no redesign.

## Crate ↔ Python Package Mapping

| Crate | Python | Status |
|-------|--------|--------|
| `lamu-core` | `lamu/core/` | skeleton (types stubbed, rest `todo!()`) |
| `lamu-mcp` | `lamu/mcp/` | skeleton |
| `lamu-api` | `lamu/api/` | skeleton |
| `lamu-cli` | `lamu/daemon.py` + `__main__.py` | skeleton |
| (in `lamu-core`) | `lamu/backends/` | not yet stubbed |

## Port Order

1. **lamu-core/types.rs** — already direct port. Validate.
2. **lamu-core/registry.rs** — GGUF parsing + YAML I/O.
3. **lamu-core/scheduler.rs** — nvidia-ml-rs replaces nvidia-smi subprocess.
4. **lamu-core/router.rs** — capability matching, ranking.
5. **lamu-core/reasoning.rs** — split + streaming filter.
6. **backends/llamacpp.rs** — subprocess management. tokio::process.
7. **lamu-mcp** — pick MCP crate (rmcp recommended) or hand-roll JSON-RPC.
8. **lamu-api** — axum router, SSE streaming with reasoning strip.
9. **lamu-cli** — wire commands.

## Translation Rules

| Python | Rust |
|--------|------|
| `@dataclass(frozen=True)` | `#[derive(Debug, Clone, Serialize, Deserialize)]` struct |
| `enum.Enum` | `enum` with `#[serde(rename_all = "snake_case")]` |
| `Optional[T]` | `Option<T>` |
| `tuple[A, B]` | `(A, B)` |
| `dict[K, V]` | `HashMap<K, V>` |
| `bare except` | `Result<T, Error>` with `?` |
| `subprocess.Popen` | `tokio::process::Command` |
| `urllib.request` | `reqwest` (sync via `blocking::`) |
| `asyncio.run` | `#[tokio::main]` |

## Dependencies to Pick Later

- MCP transport: `rmcp` vs hand-rolled
- HTTP framework: `axum` (likely)
- GPU queries: `nvml-wrapper` (nvidia-ml-rs)
- HTTP client to backends: `reqwest`

## Build

```bash
cd lamu-rs
cargo build --workspace
cargo test --workspace
```

Today: every `cargo build` will fail with `todo!()` panics (intentional).
Each port phase removes `todo!()` macros and adds tests.

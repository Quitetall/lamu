//! lamu-api — OpenAI-compatible HTTP layer.
//!
//! Port of `lamu/api/openai_compat.py`.
//! Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions
//! Streaming SSE with reasoning strip in token stream.

pub mod openai_compat;

pub async fn serve(_port: u16) -> anyhow::Result<()> {
    todo!("port serve() — axum router + reasoning-aware streaming")
}

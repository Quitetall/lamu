//! lamu-api — OpenAI-compatible HTTP layer.
//! Direct port of `lamu/api/openai_compat.py`.

pub mod metrics;
pub mod openai_compat;

use lamu_core::config::registry_path;

pub async fn serve(port: u16) -> anyhow::Result<()> {
    let state = openai_compat::build_state(&registry_path())?;
    openai_compat::auto_register(&state).await;
    let app = openai_compat::build_app(state);

    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("LAMU OpenAI-compat listening on {}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}

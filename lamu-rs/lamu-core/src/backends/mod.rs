//! Backends — model lifecycle management.
//! Direct port of `lamu/backends/`.

pub mod llamacpp;

use crate::types::ModelEntry;
use crate::Result;
use async_trait::async_trait;
use futures_util::stream::Stream;
use std::pin::Pin;

#[async_trait]
pub trait Backend: Send + Sync {
    /// Load model. Returns PID.
    async fn load(&mut self, entry: &ModelEntry, port: u16) -> Result<u32>;

    /// Stop process and free VRAM.
    async fn unload(&mut self) -> Result<()>;

    /// Health check.
    async fn is_healthy(&self) -> bool;

    /// Generate non-streaming. Returns raw text (think blocks included).
    async fn generate(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<String>;

    /// Generate streaming. Yields tokens.
    async fn stream(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>>;

    fn port(&self) -> u16;
    fn model_name(&self) -> &str;
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

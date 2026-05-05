//! Reasoning extractor — per-family think-block detection.
//!
//! Port of `lamu/core/reasoning.py`.
//! Streaming buffer-then-emit pattern.

use crate::types::{ReasoningMarker, StreamChunk};

pub trait ReasoningExtractor: Send + Sync {
    /// Split full response into (reasoning, content). Non-streaming.
    fn split(&self, text: &str) -> (String, String);

    /// Strip reasoning, return only content.
    fn strip(&self, text: &str) -> String {
        let (_, content) = self.split(text);
        content
    }

    // TODO: stream_filter via async Stream when async_trait or AFIT settles.
}

pub struct MarkerExtractor {
    marker: ReasoningMarker,
}

impl MarkerExtractor {
    pub fn new(marker: ReasoningMarker) -> Self {
        Self { marker }
    }
}

impl ReasoningExtractor for MarkerExtractor {
    fn split(&self, _text: &str) -> (String, String) {
        todo!("port MarkerExtractor.split — find open_tag/close_tag in text")
    }
}

/// For models without think-blocks. Passes through everything.
pub struct NullExtractor;

impl ReasoningExtractor for NullExtractor {
    fn split(&self, text: &str) -> (String, String) {
        (String::new(), text.to_string())
    }
}

pub fn get_extractor(marker: Option<ReasoningMarker>) -> Box<dyn ReasoningExtractor> {
    match marker {
        Some(m) => Box::new(MarkerExtractor::new(m)),
        None => Box::new(NullExtractor),
    }
}

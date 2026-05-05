//! Reasoning extractor — per-family think-block detection.
//! Direct port of `lamu/core/reasoning.py`.

use crate::types::{ReasoningMarker, StreamChunk};

pub trait ReasoningExtractor: Send + Sync {
    /// Split into (reasoning, content). Non-streaming.
    fn split(&self, text: &str) -> (String, String);

    fn strip(&self, text: &str) -> String {
        let (_, content) = self.split(text);
        content
    }

    /// Streaming filter: takes iterator of token chunks, yields StreamChunks.
    /// Default impl handles buffering and emit logic.
    fn stream_filter(
        &self,
        tokens: Box<dyn Iterator<Item = String> + Send>,
        include_reasoning: bool,
    ) -> Box<dyn Iterator<Item = StreamChunk> + Send>;
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
    fn split(&self, text: &str) -> (String, String) {
        let open = &self.marker.open_tag;
        let close = &self.marker.close_tag;

        let Some(open_idx) = text.find(open.as_str()) else {
            return (String::new(), text.to_string());
        };

        let after_open = open_idx + open.len();
        let Some(close_rel) = text[after_open..].find(close.as_str()) else {
            // Open found but no close — all text after open is reasoning (truncated)
            let reasoning = text[after_open..].trim().to_string();
            return (reasoning, String::new());
        };

        let close_idx = after_open + close_rel;
        let reasoning = text[after_open..close_idx].trim().to_string();
        let content = text[close_idx + close.len()..].trim().to_string();
        (reasoning, content)
    }

    fn stream_filter(
        &self,
        tokens: Box<dyn Iterator<Item = String> + Send>,
        include_reasoning: bool,
    ) -> Box<dyn Iterator<Item = StreamChunk> + Send> {
        Box::new(StreamFilter {
            tokens,
            open_tag: self.marker.open_tag.clone(),
            close_tag: self.marker.close_tag.clone(),
            include_reasoning,
            buffer: String::new(),
            in_reasoning: false,
            reasoning_done: false,
            pending_emit: Vec::new(),
            done: false,
        })
    }
}

pub struct NullExtractor;

impl ReasoningExtractor for NullExtractor {
    fn split(&self, text: &str) -> (String, String) {
        (String::new(), text.to_string())
    }

    fn stream_filter(
        &self,
        tokens: Box<dyn Iterator<Item = String> + Send>,
        _include_reasoning: bool,
    ) -> Box<dyn Iterator<Item = StreamChunk> + Send> {
        Box::new(tokens.map(StreamChunk::Content))
    }
}

pub fn get_extractor(marker: Option<ReasoningMarker>) -> Box<dyn ReasoningExtractor> {
    match marker {
        Some(m) => Box::new(MarkerExtractor::new(m)),
        None => Box::new(NullExtractor),
    }
}

struct StreamFilter {
    tokens: Box<dyn Iterator<Item = String> + Send>,
    open_tag: String,
    close_tag: String,
    include_reasoning: bool,
    buffer: String,
    in_reasoning: bool,
    reasoning_done: bool,
    pending_emit: Vec<StreamChunk>,
    done: bool,
}

impl Iterator for StreamFilter {
    type Item = StreamChunk;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(chunk) = self.pending_emit.pop() {
                return Some(chunk);
            }
            if self.done {
                return None;
            }

            let Some(tok) = self.tokens.next() else {
                self.done = true;
                if self.buffer.trim().is_empty() {
                    return None;
                }
                let buf = std::mem::take(&mut self.buffer);
                if self.in_reasoning && !self.reasoning_done {
                    if self.include_reasoning {
                        return Some(StreamChunk::Reasoning(buf));
                    }
                    return None;
                }
                return Some(StreamChunk::Content(buf));
            };
            self.buffer.push_str(&tok);

            if !self.in_reasoning && !self.reasoning_done {
                if let Some(idx) = self.buffer.find(self.open_tag.as_str()) {
                    self.in_reasoning = true;
                    let pre = self.buffer[..idx].to_string();
                    let after = self.buffer[idx + self.open_tag.len()..].to_string();
                    self.buffer = after;
                    if !pre.trim().is_empty() {
                        return Some(StreamChunk::Content(pre));
                    }
                } else if self.buffer.len() > self.open_tag.len() * 2 {
                    self.reasoning_done = true;
                    let buf = std::mem::take(&mut self.buffer);
                    return Some(StreamChunk::Content(buf));
                }
            } else if self.in_reasoning && !self.reasoning_done {
                if let Some(idx) = self.buffer.find(self.close_tag.as_str()) {
                    let reasoning_text = self.buffer[..idx].to_string();
                    let after = self.buffer[idx + self.close_tag.len()..].to_string();
                    self.buffer = String::new();
                    self.in_reasoning = false;
                    self.reasoning_done = true;

                    // Emit content first (after pop), then reasoning
                    if !after.trim().is_empty() {
                        self.pending_emit.push(StreamChunk::Content(after));
                    }
                    if self.include_reasoning && !reasoning_text.trim().is_empty() {
                        return Some(StreamChunk::Reasoning(reasoning_text));
                    }
                    if let Some(chunk) = self.pending_emit.pop() {
                        return Some(chunk);
                    }
                } else if self.include_reasoning && self.buffer.len() > 100 {
                    let buf = std::mem::take(&mut self.buffer);
                    return Some(StreamChunk::Reasoning(buf));
                } else if !self.include_reasoning {
                    self.buffer.clear();
                }
            } else if self.reasoning_done {
                let buf = std::mem::take(&mut self.buffer);
                return Some(StreamChunk::Content(buf));
            }
        }
    }
}

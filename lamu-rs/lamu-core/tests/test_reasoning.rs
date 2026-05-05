use lamu_core::reasoning::{get_extractor, MarkerExtractor, NullExtractor, ReasoningExtractor};
use lamu_core::types::{ReasoningMarker, StreamChunk};

fn marker() -> ReasoningMarker {
    ReasoningMarker {
        open_tag: "<think>".to_string(),
        close_tag: "</think>".to_string(),
        family: "qwen35".to_string(),
    }
}

#[test]
fn split_basic() {
    let ext = MarkerExtractor::new(marker());
    let (r, c) = ext.split("<think>\nhmm let me think\n</think>\nThe answer is 42.");
    assert_eq!(r, "hmm let me think");
    assert_eq!(c, "The answer is 42.");
}

#[test]
fn split_no_marker() {
    let ext = MarkerExtractor::new(marker());
    let (r, c) = ext.split("just plain text");
    assert_eq!(r, "");
    assert_eq!(c, "just plain text");
}

#[test]
fn split_truncated() {
    let ext = MarkerExtractor::new(marker());
    let (r, c) = ext.split("<think>\nstill thinking...");
    assert_eq!(r, "still thinking...");
    assert_eq!(c, "");
}

#[test]
fn null_passthrough() {
    let ext = NullExtractor;
    let (r, c) = ext.split("anything");
    assert_eq!(r, "");
    assert_eq!(c, "anything");
}

#[test]
fn factory() {
    let ext = get_extractor(Some(marker()));
    let (r, _) = ext.split("<think>x</think>y");
    assert_eq!(r, "x");

    let ext = get_extractor(None);
    let (r, c) = ext.split("anything");
    assert_eq!(r, "");
    assert_eq!(c, "anything");
}

#[test]
fn stream_strip() {
    let ext = MarkerExtractor::new(marker());
    let tokens: Vec<String> = vec!["<think>".into(), "thinking ".into(), "tokens".into(),
        "</think>".into(), "Hi".into(), " there".into()];
    let chunks: Vec<StreamChunk> = ext.stream_filter(Box::new(tokens.into_iter()), false).collect();
    let content: String = chunks.iter().filter_map(|c| match c {
        StreamChunk::Content(s) => Some(s.as_str()),
        _ => None,
    }).collect();
    assert!(content.contains("Hi"));
    assert!(!content.contains("thinking"));
}

#[test]
fn stream_include_reasoning() {
    let ext = MarkerExtractor::new(marker());
    let tokens: Vec<String> = vec!["<think>".into(), "abc".into(), "</think>".into(), "ok".into()];
    let chunks: Vec<StreamChunk> = ext.stream_filter(Box::new(tokens.into_iter()), true).collect();
    let has_reasoning = chunks.iter().any(|c| matches!(c, StreamChunk::Reasoning(_)));
    assert!(has_reasoning, "expected reasoning chunk");
}

//! lamu-tts — text-to-speech MODULE (ADR 0023).
//!
//! Expands the backend with TTS. Owns the fish-speech backend (an engine seam
//! for dots.tts will follow) and — after the tool move — the `text_to_speech`
//! tool. Depends on `lamu-core`; `lamu-core` does NOT depend on this crate.
//!
//! At the composition root each frontend calls `lamu_tts::register()` to
//! install this module's backend factory (kind `fish_speech`).

mod fish_speech;
mod tts;
pub use fish_speech::FishSpeechBackend;

/// Register this module's backend + MCP tool into lamu-core (ADR 0023). Call
/// ONCE at the composition root before serving. Idempotent.
pub fn register() {
    lamu_core::backends::register_backend("fish_speech", |_entry| {
        Ok(Box::new(fish_speech::FishSpeechBackend::new()?) as Box<dyn lamu_core::backends::Backend>)
    });
    // Tool: text_to_speech now lives here, dispatched over &dyn ToolCtx. Flagged
    // `cloud` so the local-only routing gate still blocks the Fish Audio path.
    lamu_core::tools_ext::register_tool(lamu_core::tools_ext::ModuleTool {
        name: "text_to_speech",
        description: "Synthesize speech from text. Routes by the model's registry modality: a `modality: tts` entry (e.g. 'local-fish-s2pro') is served LOCALLY (spawns the managed fish-speech S2-Pro server, evicting LLMs as needed); any other model goes to the Fish Audio CLOUD API ('s2-pro'/'s1', needs FISH_AUDIO_API_KEY). Writes an audio file under <data_dir>/lamu/tts and returns its path. Pass VERBALIZED prose — raw LaTeX/markup is spoken literally.",
        schema_fn: tts::schema_text_to_speech,
        handler: tts::dispatch_text_to_speech,
        cloud: true,
    });
}

#[cfg(test)]
mod tests {
    #[test]
    fn register_installs_text_to_speech_tool() {
        super::register();
        let found = lamu_core::tools_ext::find_handler("text_to_speech");
        assert!(found.is_some(), "register() must install the text_to_speech tool");
        assert!(found.unwrap().1, "text_to_speech must be flagged cloud (local-only gate)");
        assert!(
            lamu_core::tools_ext::list_entries().iter().any(|e| e["name"] == "text_to_speech"),
            "text_to_speech must appear in tools/list"
        );
    }
}

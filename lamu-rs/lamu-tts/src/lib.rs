//! lamu-tts — text-to-speech MODULE (ADR 0023).
//!
//! Expands the backend with TTS. Owns the fish-speech backend (an engine seam
//! for dots.tts will follow) and — after the tool move — the `text_to_speech`
//! tool. Depends on `lamu-core`; `lamu-core` does NOT depend on this crate.
//!
//! At the composition root each frontend calls `lamu_tts::register(&mut …)` to
//! install this module's backend factory (kind `fish_speech`).

mod fish_speech;
pub use fish_speech::FishSpeechBackend;

/// Register this module's backend(s) into lamu-core (ADR 0023). Call ONCE at the
/// composition root before serving. Idempotent.
pub fn register() {
    lamu_core::backends::register_backend("fish_speech", |_entry| {
        Ok(Box::new(fish_speech::FishSpeechBackend::new()?) as Box<dyn lamu_core::backends::Backend>)
    });
}

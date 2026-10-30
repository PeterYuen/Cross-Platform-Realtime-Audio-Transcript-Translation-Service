use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use rodio::buffer::SamplesBuffer;
use tracing::{debug, info};

use crate::config::SttConfig;
use crate::pipeline::whisper_server::WhisperServer;

/// Speech-to-text engine. Either candle/rwhisper (portable, no external binary) or whisper.cpp
/// via whisper-server (about 15x faster per call).
pub struct SttEngine {
    backend: SttBackend,
}

enum SttBackend {
    Server(WhisperServer),
    Candle(kalosm::sound::Whisper),
}

impl SttEngine {
    /// Load a Whisper model selected by name. Default is the multilingual `small` model,
    /// which balances accuracy and CPU speed. Higher tiers trade speed for quality; lower
    /// tiers (e.g. `tiny_q`, `base`) are faster but less accurate.
    ///
    /// Prefers whisper.cpp because candle pays ~8 s per call regardless of chunk length, which
    /// makes the pipeline fall behind; falls back to candle when whisper.cpp is unavailable.
    pub async fn load(model: &str, cfg: &SttConfig) -> Result<Self> {
        if cfg.engine != "candle" {
            match WhisperServer::start(
                Some(&cfg.server),
                Some(&cfg.model),
                cfg.port,
                cfg.threads,
                &cfg.language,
            )
            .await
            {
                Ok(server) => {
                    info!("stt backend: whisper.cpp ({} threads)", cfg.threads);
                    return Ok(Self {
                        backend: SttBackend::Server(server),
                    });
                }
                Err(e) => {
                    if cfg.engine == "whisper" {
                        return Err(anyhow!("whisper.cpp requested but unavailable: {}", e));
                    }
                    info!("stt backend: whisper.cpp unavailable ({}), using candle", e);
                }
            }
        }
        let source = parse_source(model);
        info!("Loading Whisper model '{}' ({:?})...", model, source);
        let whisper = kalosm::sound::Whisper::builder()
            .with_source(source)
            .with_language(None) // auto-detect language
            .build()
            .await
            .map_err(|e| anyhow!("Failed to load Whisper: {}", e))?;
        info!("Whisper model loaded successfully");
        Ok(Self {
            backend: SttBackend::Candle(whisper),
        })
    }

    /// Transcribe audio samples (f32 mono, 16kHz) to text
    pub async fn transcribe(&self, samples: &[f32]) -> Result<String> {
        self.transcribe_with(samples, |_| {}).await
    }

    /// Like `transcribe`, but invokes `on_segment` for every segment as soon as Whisper finishes
    /// it. A long chunk contains several sentences, and forwarding them one at a time means the
    /// translator can start on sentence 1 while sentence 2 is still being decoded.
    pub async fn transcribe_with(
        &self,
        samples: &[f32],
        mut on_segment: impl FnMut(&str),
    ) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        let whisper = match &self.backend {
            SttBackend::Server(server) => {
                // whisper-server returns the whole chunk in one response, so there is exactly one
                // segment to hand over.
                let raw = server.transcribe(samples).await?;
                let text = clean_output(raw.trim());
                if !text.is_empty() {
                    on_segment(&text);
                }
                return Ok(text);
            }
            SttBackend::Candle(w) => w,
        };

        // Create a rodio SamplesBuffer from our raw f32 data
        let source = SamplesBuffer::new(1, crate::audio::TARGET_SAMPLE_RATE, samples.to_vec());

        // Transcribe returns a stream of Segments
        let mut task = whisper.transcribe(source);

        let mut text = String::new();
        while let Some(segment) = task.next().await {
            let seg_text = segment.text().to_string();
            if seg_text.trim().is_empty() {
                continue;
            }
            // Whisper's own silence detector is the cheapest honest filter we have. Measured with
            // tiny_q: real speech 0.005–0.35, non-speech/noise 0.87. Without this gate the model
            // happily prints invented text for audio that contains no speech at all.
            if segment.probability_of_no_speech() > MAX_NO_SPEECH_PROB {
                info!(
                    "suppressed hallucination (no_speech={:.2}): {:?}",
                    segment.probability_of_no_speech(),
                    seg_text.trim()
                );
                continue;
            }
            if !is_meaningful(seg_text.trim()) {
                debug!("dropped fragment ({:?})", seg_text.trim());
                continue;
            }
            // Surfaced at INFO on purpose: without Whisper's own numbers in the log there is no
            // way to tell a good caption from an invented one after the fact.
            info!(
                "whisper: no_speech={:.2} conf={:.2} text={:?}",
                segment.probability_of_no_speech(),
                segment.confidence(),
                seg_text.trim()
            );
            let cleaned = strip_leading_artifact(&seg_text);
            if !is_meaningful(&cleaned) {
                continue;
            }
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(&cleaned);
            on_segment(&cleaned);
        }

        Ok(text)
    }
}

/// Above this, Whisper itself is saying "there is no speech here".
const MAX_NO_SPEECH_PROB: f64 = 0.6;

/// Same filters the candle path applies, applied to a whole-chunk response from whisper.cpp.
fn clean_output(text: &str) -> String {
    let t = strip_leading_artifact(text);
    if is_meaningful(&t) {
        t
    } else {
        String::new()
    }
}

/// Whisper-tiny almost always opens a clean utterance with a stray "a" (an artifact of the
/// language-token prefix, not something that was said). Strip it only in the unambiguous shape
/// `"a <lowercase word> …"` with several words, so real sentences like "A man walked in" survive.
fn strip_leading_artifact(text: &str) -> String {
    let t = text.trim();
    if let Some(rest) = t.strip_prefix("a ") {
        let words = rest.split_whitespace().count();
        if words >= 4 && rest.chars().next().is_some_and(|c| c.is_ascii_lowercase()) {
            return rest.trim().to_string();
        }
    }
    t.to_string()
}

/// Whisper-tiny frequently answers a silent or mid-word-cut fragment with a stray "a", "??" or a
/// lone punctuation mark. None of those are captions, and feeding them to the translator wastes an
/// entire generation slot, so reject anything that is not a real word.
/// Drop non-speech markers Whisper emits for gaps: "[Blank Audio]", "[Silence]", "(no speech)".
/// Left in place they travel all the way through translation and show up as a caption.
fn strip_markers(text: &str) -> String {
    let mut out = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '[' || c == '(' {
            let close = if c == '[' { ']' } else { ')' };
            let mut inner = String::new();
            let mut ok = true;
            while let Some(&n) = chars.peek() {
                if n == close {
                    chars.next();
                    break;
                }
                if inner.len() > 40 {
                    ok = false;
                    break;
                }
                inner.push(n);
                chars.next();
            }
            // Keep real parenthetical speech, drop the noise labels.
            let looks_like_label = inner.starts_with(|c: char| c.is_ascii_alphabetic())
                && !inner.contains(' ')
                || inner.to_lowercase().contains("blank")
                || inner.to_lowercase().contains("silence")
                || inner.to_lowercase().contains("noise")
                || inner.to_lowercase().contains("no speech")
                || inner.to_lowercase().contains("inaudible");
            if !ok {
                out.push(c);
                out.push_str(&inner);
            } else if !looks_like_label {
                out.push(c);
                out.push_str(&inner);
                out.push(close);
            }
            continue;
        }
        out.push(c);
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_meaningful(text: &str) -> bool {
    let t = strip_markers(text.trim());
    if t.chars().count() < 3 {
        return false;
    }
    // At least one alphanumeric character and not entirely replacement/punctuation junk.
    t.chars().filter(|c| c.is_alphanumeric()).count() >= 2
        && !t.contains('\u{fffd}')
}

/// Map a user-facing model name to a `WhisperSource`. Unknown names fall back to `Small`.
fn parse_source(model: &str) -> kalosm::sound::WhisperSource {
    use kalosm::sound::WhisperSource::*;
    match model.trim().to_ascii_lowercase().as_str() {
        "tiny_q" | "tiny-quantized" | "quantized_tiny" | "quantized-tiny" => QuantizedTiny,
        "tiny" => Tiny,
        "base" => Base,
        "small" => Small,
        "medium" => Medium,
        "large_v2" | "large-v2" => LargeV2,
        "large_q" | "large-quantized" | "large_v3_turbo" | "quantized_large_v3_turbo" => {
            QuantizedLargeV3Turbo
        }
        _ => Small,
    }
}

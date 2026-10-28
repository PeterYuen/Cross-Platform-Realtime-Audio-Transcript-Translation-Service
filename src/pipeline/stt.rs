use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use rodio::buffer::SamplesBuffer;
use tracing::info;

/// Speech-to-text engine using Whisper via rwhisper (exposed through kalosm-sound)
pub struct SttEngine {
    whisper: kalosm::sound::Whisper,
}

impl SttEngine {
    /// Load a Whisper model selected by name. Default is the multilingual `small` model,
    /// which balances accuracy and CPU speed. Higher tiers trade speed for quality; lower
    /// tiers (e.g. `tiny_q`, `base`) are faster but less accurate.
    pub async fn load(model: &str) -> Result<Self> {
        let source = parse_source(model);
        info!("Loading Whisper model '{}' ({:?})...", model, source);
        let whisper = kalosm::sound::Whisper::builder()
            .with_source(source)
            .with_language(None) // auto-detect language
            .build()
            .await
            .map_err(|e| anyhow!("Failed to load Whisper: {}", e))?;
        info!("Whisper model loaded successfully");
        Ok(Self { whisper })
    }

    /// Transcribe audio samples (f32 mono, 16kHz) to text
    pub async fn transcribe(&self, samples: &[f32]) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        // Create a rodio SamplesBuffer from our raw f32 data
        let source = SamplesBuffer::new(1, crate::audio::TARGET_SAMPLE_RATE, samples.to_vec());

        // Transcribe returns a stream of Segments
        let mut task = self.whisper.transcribe(source);

        let mut text = String::new();
        while let Some(segment) = task.next().await {
            let seg_text = segment.text().to_string();
            if !seg_text.trim().is_empty() {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(&seg_text);
            }
        }

        Ok(text)
    }
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

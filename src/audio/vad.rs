use tracing::{debug, info};

/// Voice Activity Detector using energy-based detection.
/// Buffers audio and emits chunks when speech is detected to end.
pub struct Vad {
    threshold: f32,
    hangover_frames: usize,
    silence_count: usize,
    in_speech: bool,
    buffer: Vec<f32>,
    max_chunk_samples: usize,
}

impl Vad {
    pub fn new(threshold: f32, hangover_frames: usize, max_chunk_seconds: f32) -> Self {
        Self {
            threshold,
            hangover_frames,
            silence_count: 0,
            in_speech: false,
            buffer: Vec::new(),
            max_chunk_samples: (max_chunk_seconds * crate::audio::TARGET_SAMPLE_RATE as f32) as usize,
        }
    }

    /// Process a frame of audio samples. Returns Some(chunk) when a speech segment ends.
    pub fn process_frame(&mut self, samples: &[f32]) -> Option<Vec<f32>> {
        let rms = compute_rms(samples);
        let is_speech = rms > self.threshold;

        if is_speech {
            self.silence_count = 0;
            self.in_speech = true;
        } else if self.in_speech {
            self.silence_count += 1;
        }

        // Buffer the samples regardless (we want to include leading silence for context)
        if self.in_speech {
            self.buffer.extend_from_slice(samples);
        }

        // End of speech: hangover expired or max chunk reached
        let should_emit = (self.in_speech && self.silence_count >= self.hangover_frames)
            || self.buffer.len() >= self.max_chunk_samples;

        if should_emit && !self.buffer.is_empty() {
            let chunk = std::mem::take(&mut self.buffer);
            self.in_speech = false;
            self.silence_count = 0;
            info!("VAD emitted chunk: {} samples ({:.2}s)", chunk.len(), chunk.len() as f32 / crate::audio::TARGET_SAMPLE_RATE as f32);
            return Some(chunk);
        }

        None
    }

    /// True if there is buffered audio waiting to be flushed.
    pub fn has_pending(&self) -> bool {
        !self.buffer.is_empty()
    }

    /// Force-flush any buffered audio (e.g. after a wall-clock silence gap, which is
    /// required for WASAPI loopback where silence produces no audio callbacks).
    pub fn flush(&mut self) -> Option<Vec<f32>> {
        if self.buffer.is_empty() {
            return None;
        }
        let chunk = std::mem::take(&mut self.buffer);
        self.in_speech = false;
        self.silence_count = 0;
        debug!(
            "VAD flushed pending chunk by timer: {} samples ({:.2}s)",
            chunk.len(),
            chunk.len() as f32 / crate::audio::TARGET_SAMPLE_RATE as f32
        );
        Some(chunk)
    }

    /// Discard buffered audio and speech state. Called when the audio backlog is trimmed so the
    /// next emitted chunk is contiguous real audio, never stitched across the dropped gap (which
    /// otherwise makes Whisper return empty/garbage text).
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.in_speech = false;
        self.silence_count = 0;
    }
}

fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

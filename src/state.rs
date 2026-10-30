use anyhow::Result;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::info;

use crate::config::Config;
use crate::pipeline::events::PipelineEvent;
use crate::pipeline::Pipeline;

/// Live capture + pipeline counters, updated from the audio callback / worker threads
/// and readable over HTTP (non-buffered, unlike stdout).
#[derive(Debug, Default)]
pub struct AudioStats {
    pub frames: AtomicU64,
    pub samples: AtomicU64,
    pub last_rms: AtomicU32,
    pub worker_frames: AtomicU64,
    pub worker_rms: AtomicU32,
    pub chunks_emitted: AtomicU64,
    pub transcribe_nonempty: AtomicU64,
    pub transcribe_empty: AtomicU64,
    pub transcribe_errors: AtomicU64,
    pub translations: AtomicU64,
    /// Captions that reached the translator but came back with no usable output.
    pub empty_translations: AtomicU64,
}

impl AudioStats {
    pub fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "frames": self.frames.load(Ordering::Relaxed),
            "samples": self.samples.load(Ordering::Relaxed),
            "last_rms": f32::from_bits(self.last_rms.load(Ordering::Relaxed)),
            "worker_frames": self.worker_frames.load(Ordering::Relaxed),
            "worker_rms": f32::from_bits(self.worker_rms.load(Ordering::Relaxed)),
            "chunks_emitted": self.chunks_emitted.load(Ordering::Relaxed),
            "transcribe_nonempty": self.transcribe_nonempty.load(Ordering::Relaxed),
            "transcribe_empty": self.transcribe_empty.load(Ordering::Relaxed),
            "transcribe_errors": self.transcribe_errors.load(Ordering::Relaxed),
            "translations": self.translations.load(Ordering::Relaxed),
            "empty_translations": self.empty_translations.load(Ordering::Relaxed),

        })
    }

    pub fn record(&self, mono_len: usize, rms: f32) {
        self.frames.fetch_add(1, Ordering::Relaxed);
        self.samples.fetch_add(mono_len as u64, Ordering::Relaxed);
        self.last_rms.store(rms.to_bits(), Ordering::Relaxed);
    }

    pub fn reset(&self) {
        self.frames.store(0, Ordering::Relaxed);
        self.samples.store(0, Ordering::Relaxed);
        self.last_rms.store(0.0f32.to_bits(), Ordering::Relaxed);
        self.worker_frames.store(0, Ordering::Relaxed);
        self.worker_rms.store(0.0f32.to_bits(), Ordering::Relaxed);
        self.chunks_emitted.store(0, Ordering::Relaxed);
        self.transcribe_nonempty.store(0, Ordering::Relaxed);
        self.transcribe_empty.store(0, Ordering::Relaxed);
        self.transcribe_errors.store(0, Ordering::Relaxed);
        self.translations.store(0, Ordering::Relaxed);
    }

    pub fn rms(&self) -> f32 {
        f32::from_bits(self.last_rms.load(Ordering::Relaxed))
    }
}

pub struct AppState {
    pub config: Mutex<Config>,
    pub event_tx: broadcast::Sender<PipelineEvent>,
    pipeline: Mutex<Option<Pipeline>>,
    alive: Arc<AtomicBool>,
    ready: Arc<AtomicBool>,
    capture_error: Arc<AtomicBool>,
    pub audio_stats: Arc<AudioStats>,
}

impl AppState {
    pub fn new(config: Config) -> Arc<Self> {
        let (event_tx, _) = broadcast::channel(256);
        Arc::new(Self {
            config: Mutex::new(config),
            event_tx,
            pipeline: Mutex::new(None),
            alive: Arc::new(AtomicBool::new(false)),
            ready: Arc::new(AtomicBool::new(false)),
            capture_error: Arc::new(AtomicBool::new(false)),
            audio_stats: Arc::new(AudioStats::default()),
        })
    }

    pub fn is_running(&self) -> bool {
        self.pipeline.lock().is_some() && self.alive.load(Ordering::Relaxed)
    }

    /// True once the worker has loaded its models and entered the processing loop.
    pub fn is_ready(&self) -> bool {
        self.is_running() && self.ready.load(Ordering::Relaxed)
    }

    /// True when the capture stream reported that its device disappeared.
    pub fn device_error(&self) -> bool {
        self.capture_error.load(Ordering::Relaxed)
    }

    pub async fn start(self: &Arc<Self>) -> Result<()> {
        if self.is_running() {
            return Ok(());
        }

        // If a previous pipeline is still stored but its worker is dead, stop and drop it
        // cleanly before starting fresh, so its capture stream is fully released.
        if let Some(mut stale) = self.pipeline.lock().take() {
            stale.stop();
            info!("Stopped stale pipeline before restart");
        }

        info!("Starting pipeline...");
        let config = self.config.lock().clone();
        let event_tx = self.event_tx.clone();

        self.audio_stats.reset();
        self.alive.store(true, Ordering::Relaxed);
        self.ready.store(false, Ordering::Relaxed);
        self.capture_error.store(false, Ordering::Relaxed);
        let pipeline = Pipeline::start(
            config,
            event_tx,
            self.alive.clone(),
            self.ready.clone(),
            self.audio_stats.clone(),
            self.capture_error.clone(),
        )
        .await?;
        *self.pipeline.lock() = Some(pipeline);
        info!("Pipeline started");

        Ok(())
    }

    pub fn stop(&self) {
        self.alive.store(false, Ordering::Relaxed);
        self.ready.store(false, Ordering::Relaxed);
        if let Some(mut pipeline) = self.pipeline.lock().take() {
            pipeline.stop();
            info!("Pipeline stopped");
        }
    }
}

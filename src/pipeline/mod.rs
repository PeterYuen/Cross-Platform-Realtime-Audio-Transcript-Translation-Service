pub mod events;
pub mod stt;
pub mod translate;

use anyhow::Result;
use parking_lot::Mutex;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

use crate::audio::{self, vad::Vad};
use crate::config::Config;
use crate::pipeline::events::PipelineEvent;
use crate::pipeline::stt::SttEngine;
use crate::pipeline::translate::TranslateEngine;

pub struct Pipeline {
    worker_handle: Option<JoinHandle<()>>,
    // Shared between `stop()` and the monitor so capture is torn down exactly once,
    // regardless of whether the worker ended via user Stop or died on its own.
    audio: Arc<Mutex<Option<audio::AudioStreamHandle>>>,
    _monitor: Option<JoinHandle<()>>,
}

impl Pipeline {
    pub async fn start(
        config: Config,
        event_tx: tokio::sync::broadcast::Sender<PipelineEvent>,
        alive: Arc<std::sync::atomic::AtomicBool>,
        ready: Arc<std::sync::atomic::AtomicBool>,
        stats: Arc<crate::state::AudioStats>,
        capture_error: Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<Self> {
        // Single unbounded tokio channel: the cpal callback sends directly into it, so no
        // blocking bridge task sits between the audio thread and the async worker.
        let (audio_tx, audio_rx) = mpsc::unbounded_channel::<Vec<f32>>();

        // Start audio capture (spawns its own thread). It sets `capture_error` if the device
        // disappears, so the worker surfaces the failure instead of idling silently.
        let audio_handle = audio::start_capture(
            config.audio_device_name.as_deref(),
            config.audio_device_is_loopback,
            audio_tx,
            stats.clone(),
            capture_error.clone(),
        )?;
        let audio = Arc::new(Mutex::new(Some(audio_handle)));

        // Spawn worker task (does NOT own the stream; teardown is deterministic below).
        let worker_handle = tokio::spawn(run_pipeline_worker(
            audio_rx,
            config,
            event_tx,
            alive.clone(),
            ready.clone(),
            stats,
            capture_error,
        ));

        // Monitor: when the worker dies for any reason (model-load failure, audio closed,
        // or user Stop which flips `alive` first), tear down capture exactly once. This
        // avoids relying on Drop during task abort, which can leave a zombie callback.
        let audio_mon = audio.clone();
        let alive_mon = alive.clone();
        let monitor = tokio::spawn(async move {
            loop {
                if !alive_mon.load(Ordering::Relaxed) {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
            if let Some(mut handle) = audio_mon.lock().take() {
                handle.stop();
            }
        });

        Ok(Self {
            worker_handle: Some(worker_handle),
            audio,
            _monitor: Some(monitor),
        })
    }

    pub fn stop(&mut self) {
        // Stop capture first: this closes the audio channel, so the worker finishes its
        // current chunk and exits through the closed-channel branch instead of being
        // aborted mid-transcription (which emits rwhisper "receiver is gone").
        if let Some(mut handle) = self.audio.lock().take() {
            handle.stop();
        }
        if let Some(worker) = self.worker_handle.take() {
            // Grace period for the worker to notice the closed channel and exit cleanly.
            // If it is stuck (e.g. a stalled model call), hard-abort so it cannot leak CPU.
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                if !worker.is_finished() {
                    worker.abort();
                }
            });
        }
    }
}

async fn run_pipeline_worker(
    mut audio_rx: mpsc::UnboundedReceiver<Vec<f32>>,
    config: Config,
    event_tx: tokio::sync::broadcast::Sender<PipelineEvent>,
    alive: Arc<std::sync::atomic::AtomicBool>,
    ready: Arc<std::sync::atomic::AtomicBool>,
    stats: Arc<crate::state::AudioStats>,
    capture_error: Arc<std::sync::atomic::AtomicBool>,
) {
    use std::time::{Duration, Instant};

    // Marks the pipeline dead on any exit path so /api/status reflects reality.
    let mark_dead = |event_tx: &tokio::sync::broadcast::Sender<PipelineEvent>| {
        alive.store(false, Ordering::Relaxed);
        ready.store(false, Ordering::Relaxed);
        let _ = event_tx.send(PipelineEvent::Status { running: false });
    };

    // Load models
    let stt = match SttEngine::load(&config.whisper_model).await {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to load STT model: {}", e);
            let _ = event_tx.send(PipelineEvent::Error {
                message: format!("STT model load failed: {}", e),
            });
            mark_dead(&event_tx);
            return;
        }
    };

    let translator = match TranslateEngine::load().await {
        Ok(t) => t,
        Err(e) => {
            error!("Failed to load translation model: {}", e);
            let _ = event_tx.send(PipelineEvent::Error {
                message: format!("LLM model load failed: {}", e),
            });
            mark_dead(&event_tx);
            return;
        }
    };

    let target_language = config.target_language.clone();
    let mut vad = Vad::new(
        config.vad_threshold,
        config.vad_hangover_frames,
        config.max_chunk_seconds,
    );

    // Translation runs on its own task fed by a queue, so a slow generation can never block
    // capture or transcription. The worker only enqueues finished transcripts here.
    let (transcript_tx, transcript_rx) = mpsc::unbounded_channel::<(String, String)>();
    tokio::spawn(run_translator_task(
        translator,
        transcript_rx,
        event_tx.clone(),
        alive.clone(),
        stats.clone(),
    ));

    info!("Pipeline worker ready, target_language={}", target_language);
    ready.store(true, Ordering::Relaxed);
    let _ = event_tx.send(PipelineEvent::Status { running: true });

    // Wall-clock silence gap: needed because WASAPI loopback emits no callbacks during
    // silence, so hangover-frame detection alone would never flush a buffered utterance.
    let silence_gap = Duration::from_millis(250);
    let mut ticker = tokio::time::interval(Duration::from_millis(100));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut last_frame = Instant::now();
    let mut frames_in: u64 = 0;

    // If more frames queue up than this, we are falling behind real time; drop the oldest so
    // we always transcribe current audio instead of a stale backlog. ~3 s at ~10 ms/frame.
    const MAX_QUEUED_FRAMES: usize = 300;

    loop {
        // Surface device loss: the capture callback sets this flag when the endpoint vanishes.
        // Tear down so the UI can prompt a Start, instead of idling with a dead stream.
        if capture_error.load(Ordering::Relaxed) {
            let _ = event_tx.send(PipelineEvent::Error {
                message: "Capture device lost — check your output device and press Start."
                    .to_string(),
            });
            error!("Capture device lost; stopping pipeline");
            break;
        }
        let mut pending_flush: Option<Vec<f32>> = None;
        let mut drained: Vec<Vec<f32>> = Vec::new();
        let kind = tokio::select! {
            f = audio_rx.recv() => {
                match f {
                    Some(first) => {
                        drained.push(first);
                        // Grab everything already queued so we can jump to the newest audio.
                        while let Ok(next) = audio_rx.try_recv() {
                            drained.push(next);
                        }
                        0u8
                    }
                    None => 2u8,
                }
            }
            _ = ticker.tick() => {
                // Heartbeat so the meter proves the stream is alive even during silence.
                let _ = event_tx.send(PipelineEvent::AudioLevel { rms: stats.rms() });
                if vad.has_pending() && last_frame.elapsed() >= silence_gap {
                    pending_flush = vad.flush();
                }
                1u8
            }
        };

        if let Some(chunk) = pending_flush.take() {
            process_chunk(chunk, &stt, &transcript_tx, &target_language, &event_tx, &stats).await;
            continue;
        }

        match kind {
            2 => break,
            0 => {
                // Trim stale backlog, keeping only the most recent window.
                if drained.len() > MAX_QUEUED_FRAMES {
                    let skip = drained.len() - MAX_QUEUED_FRAMES;
                    drained.drain(..skip);
                    // We dropped audio we could not process in time. Clear VAD so the next chunk
                    // is contiguous with *this* window instead of stitched across the gap, which
                    // otherwise produces empty/garbage transcriptions and stalls the output.
                    vad.reset();
                }
                last_frame = Instant::now();

                for frame in drained {
                    frames_in += 1;
                    let rms = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len().max(1) as f32).sqrt();
                    stats.worker_frames.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    stats.worker_rms.store(rms.to_bits(), std::sync::atomic::Ordering::Relaxed);
                    if frames_in % 200 == 1 {
                        info!("worker received frame #{} (len={}, rms={:.6})", frames_in, frame.len(), rms);
                    }
                    if let Some(chunk) = vad.process_frame(&frame) {
                        process_chunk(chunk, &stt, &transcript_tx, &target_language, &event_tx, &stats).await;
                    }
                }
            }
            _ => {} // ticker tick, handled above
        }
    }

    // Audio source closed (capture stopped): shut down cleanly.
    mark_dead(&event_tx);
}

async fn process_chunk(
    chunk: Vec<f32>,
    stt: &SttEngine,
    transcript_tx: &mpsc::UnboundedSender<(String, String)>,
    target_language: &str,
    event_tx: &tokio::sync::broadcast::Sender<PipelineEvent>,
    stats: &crate::state::AudioStats,
) {
    if chunk.is_empty() {
        return;
    }
    stats.chunks_emitted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Ignore sub-half-second fragments: they are usually VAD noise and transcribe empty.
    const MIN_CHUNK_SAMPLES: usize = crate::audio::TARGET_SAMPLE_RATE as usize / 2;
    if chunk.len() < MIN_CHUNK_SAMPLES {
        info!("process_chunk: skipping short chunk ({} samples)", chunk.len());
        return;
    }

    info!("process_chunk: {} samples -> transcribing", chunk.len());
    match stt.transcribe(&chunk).await {
        Ok(text) => {
            if text.trim().is_empty() {
                stats.transcribe_empty.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                info!("transcribe returned empty");
                return;
            }
            stats.transcribe_nonempty.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            info!("Transcript: {}", text);
            let _ = event_tx.send(PipelineEvent::Transcript { text: text.clone() });
            // Hand off to the translator task; never await it here so capture/transcription
            // keep running while translation is in flight.
            let _ = transcript_tx.send((text, target_language.to_string()));
        }
        Err(e) => {
            stats.transcribe_errors.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            error!("STT error: {}", e);
            let _ = event_tx.send(PipelineEvent::Error {
                message: e.to_string(),
            });
        }
    }
}

/// Dedicated translation task: drains the transcript queue independently of the audio loop,
/// streaming tokens and a completion event. Exits when the queue closes or `alive` goes false.
async fn run_translator_task(
    translator: TranslateEngine,
    mut rx: mpsc::UnboundedReceiver<(String, String)>,
    event_tx: tokio::sync::broadcast::Sender<PipelineEvent>,
    alive: Arc<std::sync::atomic::AtomicBool>,
    stats: Arc<crate::state::AudioStats>,
) {
    use std::time::Duration;
    loop {
        if !alive.load(Ordering::Relaxed) {
            break;
        }
        let item = tokio::select! {
            i = rx.recv() => i,
            _ = tokio::time::sleep(Duration::from_millis(200)) => continue,
        };
        let (text, lang) = match item {
            Some(v) => v,
            None => break,
        };

        let (token_tx, mut token_rx) = mpsc::unbounded_channel::<String>();
        let etx = event_tx.clone();
        let broadcast_task = tokio::spawn(async move {
            while let Some(token) = token_rx.recv().await {
                let _ = etx.send(PipelineEvent::TranslationToken { token });
            }
        });

        match translator.translate_stream(&text, &lang, token_tx).await {
            Ok(full_text) => {
                let _ = broadcast_task.await;
                if full_text.trim().is_empty() {
                    warn!("Translation produced empty output, skipping");
                } else {
                    stats.translations.fetch_add(1, Ordering::Relaxed);
                    let _ = event_tx.send(PipelineEvent::TranslationComplete { text: full_text });
                }
            }
            Err(e) => {
                let _ = broadcast_task.await;
                error!("Translation error: {}", e);
                let _ = event_tx.send(PipelineEvent::Error {
                    message: e.to_string(),
                });
            }
        }
    }
}

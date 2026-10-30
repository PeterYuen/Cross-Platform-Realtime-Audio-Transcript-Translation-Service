pub mod events;
pub mod llama_server;
pub mod stt;
pub mod translate;
pub mod whisper_server;

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
    let stt = match SttEngine::load(&config.whisper_model, &config.stt).await {
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

    // Transcription runs on its own task too. Keeping `stt.transcribe()` inside this loop made
    // the whole audio path stop for the 20-45 s each call takes: the level bar froze, frames
    // piled up and were then thrown away. The worker now only segments and enqueues.
    let (chunk_tx, chunk_rx) = mpsc::unbounded_channel::<Vec<f32>>();
    tokio::spawn(run_stt_task(
        stt,
        chunk_rx,
        transcript_tx.clone(),
        target_language.clone(),
        event_tx.clone(),
        alive.clone(),
        stats.clone(),
    ));

    // The level bar must tick even while STT or translation are grinding. Emitting it from the
    // worker's own select meant it went silent whenever the worker was busy inside a model call.
    tokio::spawn(run_level_ticker(
        event_tx.clone(),
        stats.clone(),
        alive.clone(),
    ));

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
    let mut last_frames_at = Instant::now();
    let mut warned_dead = false;
    let mut ever_had_frames = false;

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
                // Watchdog. WASAPI loopback delivers no callbacks while nothing is rendering, so
                // "no frames lately" is normal during silence — treating that as a fault produced
                // 16 false alarms in 40 minutes of otherwise healthy audio. Only a stream that has
                // never produced a single frame is actually broken (wrong or vanished device).
                if stats.frames.load(Ordering::Relaxed) > 0 {
                    ever_had_frames = true;
                    last_frames_at = Instant::now();
                } else if !ever_had_frames
                    && last_frames_at.elapsed() > Duration::from_secs(20)
                    && !warned_dead
                {
                    warned_dead = true;
                    let msg = "Capture produced no audio frames since Start — wrong or vanished "
                        .to_string()
                        + "device. Pick another output device and press Start.";
                    let _ = event_tx.send(PipelineEvent::Error { message: msg.clone() });
                    error!("{}", msg);
                }
                if vad.has_pending() && last_frame.elapsed() >= silence_gap {
                    pending_flush = vad.flush();
                }
                1u8
            }
        };

        if let Some(chunk) = pending_flush.take() {
            enqueue_chunk(chunk, &chunk_tx, &event_tx, &stats);
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
                        enqueue_chunk(chunk, &chunk_tx, &event_tx, &stats);
                    }
                }
            }
            _ => {} // ticker tick, handled above
        }
    }

    // Audio source closed (capture stopped): shut down cleanly.
    mark_dead(&event_tx);
}

/// Hand a VAD-completed chunk to the transcription task. Never blocks: the audio loop must stay
/// responsive so the level bar keeps moving and frames keep flowing.
fn enqueue_chunk(
    chunk: Vec<f32>,
    chunk_tx: &mpsc::UnboundedSender<Vec<f32>>,
    event_tx: &tokio::sync::broadcast::Sender<PipelineEvent>,
    stats: &crate::state::AudioStats,
) {
    if chunk.is_empty() {
        return;
    }
    // Ignore sub-half-second fragments: they are usually VAD noise and transcribe empty.
    const MIN_CHUNK_SAMPLES: usize = crate::audio::TARGET_SAMPLE_RATE as usize / 2;
    if chunk.len() < MIN_CHUNK_SAMPLES {
        info!("worker: skipping short chunk ({} samples)", chunk.len());
        return;
    }
    stats
        .chunks_emitted
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if chunk_tx.send(chunk).is_err() {
        let _ = event_tx.send(PipelineEvent::Error {
            message: "Transcription task is not running".to_string(),
        });
    }
}

/// Transcription task. Kept out of the audio loop so a 20-45 s Whisper call cannot freeze the
/// meter or throw away audio we already captured.
async fn run_stt_task(
    stt: SttEngine,
    mut rx: mpsc::UnboundedReceiver<Vec<f32>>,
    transcript_tx: mpsc::UnboundedSender<(String, String)>,
    target_language: String,
    event_tx: tokio::sync::broadcast::Sender<PipelineEvent>,
    alive: Arc<std::sync::atomic::AtomicBool>,
    stats: Arc<crate::state::AudioStats>,
) {
    loop {
        if !alive.load(Ordering::Relaxed) {
            break;
        }
        let Some(first) = rx.recv().await else {
            break;
        };
        // Everything that queued while we were inside a model call gets processed in order.
        // The old policy kept only the newest chunk and threw the rest away, which is exactly why
        // continuous speech looked frozen: at ~1.2x real time the backlog grows, every skipped
        // chunk is speech that never appears, and captions stall and then jump.
        let mut pending = vec![first];
        while let Ok(next) = rx.try_recv() {
            pending.push(next);
        }
        let queued_secs: f32 = pending.iter().map(|c| c.len() as f32 / 16000.0).sum();
        // Process at most this much backlog. Beyond that we are genuinely past catching up, so
        // drop the oldest and say so loudly instead of silently losing it.
        const MAX_BACKLOG_SECS: f32 = 25.0;
        let mut budget = MAX_BACKLOG_SECS;
        let mut start = 0usize;
        for (i, c) in pending.iter().enumerate() {
            budget -= c.len() as f32 / 16000.0;
            start = i;
            if budget <= 0.0 {
                break;
            }
        }
        if start > 0 {
            warn!(
                "stt: {:.1}s of audio queued, dropping {:.1}s of oldest backlog",
                queued_secs,
                pending[..start].iter().map(|c| c.len() as f32 / 16000.0).sum::<f32>()
            );
        } else if pending.len() > 1 {
            info!(
                "stt: {:.1}s queued across {} chunks, working through in order",
                queued_secs,
                pending.len()
            );
        }
        for chunk in pending.drain(start..) {
            process_chunk(
                chunk,
                &stt,
                &transcript_tx,
                &target_language,
                &event_tx,
                &stats,
            )
            .await;
        }
    }
}

/// Emits the input level on its own timer. Doing it from the worker's select loop made the bar
/// freeze every time the worker went inside a model call.
async fn run_level_ticker(
    event_tx: tokio::sync::broadcast::Sender<PipelineEvent>,
    stats: Arc<crate::state::AudioStats>,
    alive: Arc<std::sync::atomic::AtomicBool>,
) {
    let mut ticker = tokio::time::interval(std::time::Duration::from_millis(100));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        ticker.tick().await;
        if !alive.load(Ordering::Relaxed) {
            break;
        }
        let _ = event_tx.send(PipelineEvent::AudioLevel { rms: stats.rms() });
    }
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

    // Ignore sub-half-second fragments: they are usually VAD noise and transcribe empty.
    const MIN_CHUNK_SAMPLES: usize = crate::audio::TARGET_SAMPLE_RATE as usize / 2;
    if chunk.len() < MIN_CHUNK_SAMPLES {
        info!("process_chunk: skipping short chunk ({} samples)", chunk.len());
        return;
    }

    info!("process_chunk: {} samples -> transcribing", chunk.len());
    // Segments are forwarded the moment Whisper finishes them, so a 20-second chunk that contains
    // three sentences reaches the translator three times instead of once at the end.
    let mut segments = 0usize;
    let result = stt
        .transcribe_with(&chunk, |segment| {
            segments += 1;
            let _ = event_tx.send(PipelineEvent::Transcript {
                text: segment.to_string(),
            });
            let _ = transcript_tx.send((segment.to_string(), target_language.to_string()));
        })
        .await;
    match result {
        Ok(text) => {
            if segments == 0 {
                stats.transcribe_empty.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                info!("transcribe returned empty");
                return;
            }
            stats.transcribe_nonempty.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let _ = &text; // the joined text already went out segment by segment
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
    // A transcript whose target language differs from the batch in flight; parked here so it is
    // not lost while we finish translating the current batch.
    let mut carry: Option<(String, String)> = None;
    loop {
        if !alive.load(Ordering::Relaxed) {
            break;
        }
        let item = match carry.take() {
            Some(v) => Some(v),
            None => {
                tokio::select! {
                    i = rx.recv() => i,
                    _ = tokio::time::sleep(Duration::from_millis(200)) => continue,
                }
            }
        };
        let (mut text, lang) = match item {
            Some(v) => v,
            None => break,
        };

        // Drain whatever else is already queued, but translate each item on its own.
        // Joining them into a single generation used to be worth it (candle cost ~20s of fixed
        // overhead per call) but llama.cpp answers in ~0.5s, and merging made one bad generation
        // take four captions down with it.
        const MAX_DRAINED: usize = 4;
        let mut batch: Vec<(String, String)> = vec![(text, lang)];
        while batch.len() < MAX_DRAINED {
            match rx.try_recv() {
                Ok((next, next_lang)) => {
                    if next_lang != batch[0].1 {
                        carry = Some((next, next_lang));
                        break;
                    }
                    batch.push((next, next_lang));
                }
                Err(_) => break,
            }
        }

        for (text, lang) in batch {
            let (token_tx, mut token_rx) = mpsc::unbounded_channel::<String>();
            let etx = event_tx.clone();
            let broadcast_task = tokio::spawn(async move {
                while let Some(token) = token_rx.recv().await {
                    let _ = etx.send(PipelineEvent::TranslationToken { token });
                }
            });

            let outcome = match translator.translate_stream(&text, &lang, token_tx).await {
                Ok(t) => t,
                Err(e) => {
                    error!("Translation error: {}; retrying once", e);
                    match translator.translate_stream(&text, &lang, mpsc::unbounded_channel().0).await {
                        Ok(t) => t,
                        Err(e2) => {
                            error!("Translation failed twice: {}", e2);
                            let _ = event_tx.send(PipelineEvent::Error {
                                message: e2.to_string(),
                            });
                            continue;
                        }
                    }
                }
            };
            let _ = broadcast_task.await;
            let full_text = outcome.trim().to_string();
            if full_text.is_empty() {
                // Empty output is usually the small model stalling on a long or unusual input,
                // not a genuine "nothing to say". One retry recovers most of those.
                warn!("Translation produced empty output, retrying once");
                let again = match translator
                    .translate_stream(&text, &lang, mpsc::unbounded_channel().0)
                    .await
                {
                    Ok(t) => t,
                    Err(e) => {
                        warn!("Translation retry failed: {}", e);
                        stats.empty_translations.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                };
                let again = again.trim().to_string();
                if again.is_empty() {
                    stats.empty_translations.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                stats.translations.fetch_add(1, Ordering::Relaxed);
                let _ = event_tx.send(PipelineEvent::TranslationComplete { text: again });
                continue;
            }
            stats.translations.fetch_add(1, Ordering::Relaxed);
            let _ = event_tx.send(PipelineEvent::TranslationComplete { text: full_text });
        }
    }
}

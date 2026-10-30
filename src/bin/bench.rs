//! Throughput benchmark: how far from real time is each stage?
//!   cargo run --release --bin bench -- stt
//!   cargo run --release --bin bench -- llm
//! Set RAYON_NUM_THREADS to change candle's CPU parallelism.
use std::time::Instant;

use futures_util::StreamExt;
use kalosm::language::*;
use rodio::buffer::SamplesBuffer;

fn read_wav_mono_16k(path: &str) -> (Vec<f32>, f32) {
    let file = std::fs::File::open(path).unwrap();
    let mut rdr = hound::WavReader::new(file).unwrap();
    let spec = rdr.spec();
    let sr = spec.sample_rate as f64;
    let ch = spec.channels.max(1) as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = 1f32 / ((1i32 << (spec.bits_per_sample - 1)) as f32);
            rdr.samples::<i32>()
                .map(|s| s.unwrap_or(0) as f32 * max)
                .collect()
        }
        hound::SampleFormat::Float => rdr.samples::<f32>().map(|s| s.unwrap()).collect(),
    };
    // downmix to mono
    let mono: Vec<f32> = samples
        .chunks(ch)
        .map(|c| c.iter().sum::<f32>() / ch as f32)
        .collect();
    // linear resample to 16 kHz
    let target = 16000.0f32;
    let ratio = target / sr as f32;
    let n_out = (mono.len() as f32 * ratio) as usize;
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let pos = i as f32 / ratio;
        let i0 = pos.floor() as usize;
        let i1 = (i0 + 1).min(mono.len() - 1);
        let f = pos - i0 as f32;
        out.push(mono[i0] * (1.0 - f) + mono[i1] * f);
    }
    let dur = n_out as f32 / target;
    (out, dur)
}

/// Which clip to measure against. Deliberately has no default: a baked-in path only ever worked
/// on the machine that wrote it, and silently reading the wrong file produces confident numbers.
fn test_wav() -> &'static str {
    static PATH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let p = PATH.get_or_init(|| match std::env::var("RT_TEST_WAV") {
        Ok(v) => v,
        Err(_) => {
            eprintln!(
                "bench: RT_TEST_WAV is not set. Point it at a 16 kHz mono WAV, for example: \
                 RT_TEST_WAV=/path/to/clip.wav cargo run --release --bin bench -- lat"
            );
            std::process::exit(2);
        }
    });
    // Leak is fine here; this is a benchmark binary that runs once.
    Box::leak(p.clone().into_boxed_str())
}

fn main() -> anyhow::Result<()> {
    // Bench used to run with logging off, which hid which backend was actually in use — and
    // "which model is this" is the first thing you need when numbers come out slow.
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    let mode = std::env::args().nth(1).unwrap_or_else(|| "stt".to_string());
    println!(
        "bench mode={} RAYON_NUM_THREADS={} OMP_NUM_THREADS={}",
        mode,
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".into()),
        std::env::var("OMP_NUM_THREADS").unwrap_or_else(|_| "unset".into())
    );

    if mode == "stt" {
        let (samples, dur) = read_wav_mono_16k(test_wav());
        println!("audio: {:.2}s of speech, {} samples @16k", dur, samples.len());
        let chunk_secs: f32 = std::env::args()
            .nth(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async move {
            let w = kalosm::sound::Whisper::builder()
                .with_source(kalosm::sound::WhisperSource::QuantizedTiny)
                .with_language(None)
                .build()
                .await?;
            let step = if chunk_secs > 0.0 {
                (chunk_secs * 16000.0) as usize
            } else {
                samples.len().max(1)
            };
            for rep in 0..if chunk_secs > 0.0 { 1 } else { 3 } {
                let t = Instant::now();
                let mut text = String::new();
                let mut n_chunks = 0usize;
                for win in samples.chunks(step) {
                    n_chunks += 1;
                    let mut task = w.transcribe(SamplesBuffer::new(1, 16000, win.to_vec()));
                    while let Some(seg) = task.next().await {
                        text.push_str(seg.text());
                        text.push(' ');
                    }
                }
                let wall = t.elapsed().as_secs_f32();
                println!(
                    "STT rep{}: chunks={} chunk={:.1}s wall={:.2}s audio={:.2}s RTF={:.2}x words={}",
                    rep,
                    n_chunks,
                    chunk_secs,
                    wall,
                    dur,
                    wall / dur,
                    text.split_whitespace().count()
                );
            }
            Ok::<(), anyhow::Error>(())
        })?;
    } else if mode == "lat" {
        // End-to-end latency through the real components: feed a clip in real time, let the actual
        // VAD segment it, and measure how long after each utterance ENDS before the transcript and
        // the translation both exist. Everything is measured against the wall-clock moment the
        // audio would have been captured, so falling behind shows up as a growing lag.
        let (samples, dur) = read_wav_mono_16k(test_wav());
        let target = std::env::var("RT_TARGET").unwrap_or_else(|_| "Chinese (Traditional)".into());
        let chunk_secs: f32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(10.0);
        println!("clip {:.1}s, max_chunk {}s, target {}", dur, chunk_secs, target);
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async move {
            let mut stt_cfg = realtime_translator::config::SttConfig::default();
            if let Ok(p) = std::env::var("RT_STT_PORT") {
                stt_cfg.port = p.parse().unwrap_or(stt_cfg.port);
            }
            if let Ok(e) = std::env::var("RT_STT_ENGINE") {
                stt_cfg.engine = e;
            }
            if let Ok(l) = std::env::var("RT_STT_LANG") {
                stt_cfg.language = l;
            }
            let stt = realtime_translator::pipeline::stt::SttEngine::load("tiny_q", &stt_cfg).await?;
            let mut cfg = realtime_translator::config::TranslationConfig::default();
            if let Ok(m) = std::env::var("RT_LLM_MODEL") {
                cfg.model = m;
            }
            if let Ok(p) = std::env::var("RT_PORT") {
                cfg.port = p.parse().unwrap_or(cfg.port);
            }
            let tr = realtime_translator::pipeline::translate::TranslateEngine::load(&cfg).await?;
            let hangover: usize = std::env::var("RT_HANGOVER")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(15);
            let frame = 160usize;
            let t0 = Instant::now();
            // Feed and process run in SEPARATE tasks. Doing both in one loop means the clock keeps
            // advancing while we sit inside a model call, which made every earlier latency number
            // read worse than reality.
            let (chunk_tx, mut chunk_rx) = tokio::sync::mpsc::unbounded_channel::<(Vec<f32>, f32)>();
            let samples_c = samples.clone();
            let feed = tokio::spawn(async move {
                let base = tokio::time::Instant::now();
                let mut vad = realtime_translator::audio::vad::Vad::new(0.003, hangover, chunk_secs);
                let mut idx = 0usize;
                while idx < samples_c.len() {
                    let t_start = t0.elapsed();
                    let end = (idx + frame).min(samples_c.len());
                    if let Some(buf) = vad.process_frame(&samples_c[idx..end]) {
                        let audio_end = end as f32 / 16000.0;
                        let _ = chunk_tx.send((buf, audio_end));
                    }
                    idx = end;
                    // Absolute deadline, not a relative sleep: relative sleeps drift whenever the
                    // runtime is busy, and that drift is what made the feed fall behind.
                    let target = std::time::Duration::from_secs_f64(idx as f64 / 16000.0);
                    tokio::time::sleep_until(base + target).await;
                    let _ = t_start;
                }
                if let Some(buf) = vad.flush() {
                    let n = samples_c.len();
                    let _ = chunk_tx.send((buf, n as f32 / 16000.0));
                }
                t0
            });
            let mut best = Vec::new();
            while let Some((buf, audio_end)) = chunk_rx.recv().await {
                let wall_of_capture = t0.elapsed().as_secs_f32();
                let backlog = wall_of_capture - audio_end;
                let mut first_slot: Option<Instant> = None;
                let mut first_seen = 0usize;
                let seg_start = Instant::now();
                let text = stt
                    .transcribe_with(&buf, |seg| {
                        if first_seen == 0 {
                            first_slot = Some(Instant::now());
                        }
                        first_seen += 1;
                        let _ = seg;
                    })
                    .await
                    .unwrap_or_default();
                let t_stt = t0.elapsed().as_secs_f32();
                println!(
                    "   [stt call took {:.2}s for {:.2}s of audio]",
                    seg_start.elapsed().as_secs_f32(),
                    buf.len() as f32 / 16000.0
                );
                let first_seg_at = first_slot.map(|i| (i - t0).as_secs_f64() as f32 - audio_end);
                let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();
                let tracer = tokio::spawn(async move {
                    let mut first = None;
                    let mut last = None;
                    let mut acc = String::new();
                    let mut rx = rx;
                    while let Some(tok) = rx.recv().await {
                        if first.is_none() {
                            first = Some(Instant::now());
                        }
                        last = Some(Instant::now());
                        acc.push_str(&tok);
                    }
                    (first, last, acc)
                });
                let tr_start = Instant::now();
                let out = if std::env::var("RT_NO_TR").is_ok() {
                    // Close the channel so the tracer task finishes; otherwise it waits on a
                    // sender that is still alive.
                    drop(tx);
                    String::new()
                } else {
                    tr.translate_stream(&text, &target, tx).await.unwrap_or_default()
                };
                let t_tr = t0.elapsed().as_secs_f32();
                let (first, _last, streamed) = tracer.await.unwrap_or((None, None, String::new()));
                let first_ms = first.map(|i| (i - tr_start).as_secs_f64() * 1000.0);
                {
                    let want = match target.to_lowercase().as_str() {
                        t if t.contains("korean") => 'h',
                        t if t.contains("chinese") => 'n',
                        _ => 'l',
                    };
                    let foreign = streamed
                        .chars()
                        .filter(|c| {
                            let n = *c as u32;
                            match want {
                                'h' => (0x3040..=0x30ff).contains(&n) || (0x4e00..=0x9fff).contains(&n),
                                'n' => (0x3040..=0x30ff).contains(&n) || (0xac00..=0xd7a3).contains(&n),
                                _ => (0x3040..=0x30ff).contains(&n) || (0xac00..=0xd7a3).contains(&n) || (0x4e00..=0x9fff).contains(&n),
                            }
                        })
                        .count();
                    if foreign > 0 {
                        println!("   !! STREAM contained {} foreign-script chars: {}", foreign, streamed);
                    }
                }
                let lag_s = t_stt - audio_end;
                let lag_t = t_tr - audio_end;
                best.push((lag_s, lag_t));
                println!(
                    "audio_end={:5.1}s  backlog_at_entry={:5.2}s  first_text={:6.2}s  full_text={:6.2}s  translate_done={:6.2}s  first_tr_token={}ms  {} chars | {} | TR: {}",
                    audio_end,
                    backlog,
                    first_seg_at.unwrap_or(-1.0),
                    lag_s,
                    lag_t,
                    first_ms.map(|m| m as i32).unwrap_or(-1),
                    out.chars().count(),
                    text.chars().take(40).collect::<String>(),
                    out.chars().take(60).collect::<String>()
                );
            }
            let _ = feed.await;
            if !best.is_empty() {
                let n = best.len() as f32;
                println!(
                    "MEAN transcript_lag={:.2}s translate_lag={:.2}s over {} utterances",
                    best.iter().map(|b| b.0).sum::<f32>() / n,
                    best.iter().map(|b| b.1).sum::<f32>() / n,
                    best.len()
                );
            }
            Ok::<(), anyhow::Error>(())
        })?;
    } else if mode == "win" {
        // What does Whisper actually return for realistic VAD-sized windows? Production chunks are
        // 1-10 s; the whole-clip benchmark hides what happens on short fragments.
        let (samples, dur) = read_wav_mono_16k(test_wav());
        println!("clip: {:.1}s", dur);
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async move {
            let w = kalosm::sound::Whisper::builder()
                .with_source(kalosm::sound::WhisperSource::QuantizedTiny)
                .with_language(None)
                .build()
                .await?;
            let sizes: Vec<f32> = std::env::args()
                .nth(2)
                .map(|a| {
                    a.split(',')
                        .filter_map(|s| s.trim().parse::<f32>().ok())
                        .collect()
                })
                .unwrap_or_else(|| vec![2.0, 5.0, 10.0]);
            for secs in sizes {
                let step = (secs * 16000.0) as usize;
                println!("--- window={:.0}s ---", secs);
                for (i, win) in samples.chunks(step).enumerate() {
                    let mut task = w.transcribe(SamplesBuffer::new(1, 16000, win.to_vec()));
                    let mut text = String::new();
                    let mut ns = -1.0;
                    while let Some(seg) = task.next().await {
                        text.push_str(seg.text());
                        ns = seg.probability_of_no_speech();
                    }
                    println!(
                        "  win{} [{:.1}-{:.1}] no_speech={:.3} text={:?}",
                        i,
                        i as f32 * secs,
                        (i + 1) as f32 * secs,
                        ns,
                        text.trim()
                    );
                }
            }
            Ok::<(), anyhow::Error>(())
        })?;
    } else if mode == "halluc" {
        // Does Whisper invent text when there is nothing there? Feed it silence and noise and ask
        // what it says, plus its own probability_of_no_speech so we can pick a rejection threshold.
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async move {
            let (speech, _dur) = read_wav_mono_16k(test_wav());
            let n = 16000 * 5;
            let silence = vec![0.0f32; n];
            let mut st: u32 = 0x1234_5678;
            let mut noise = Vec::with_capacity(n);
            for _ in 0..n {
                st ^= st << 13;
                st ^= st >> 17;
                st ^= st << 5;
                noise.push(((st as f32 / u32::MAX as f32) - 0.5) * 0.004);
            }
            let (silence, noise) = (silence, noise);

            for lang_opt in [false, true] {
                let w = kalosm::sound::Whisper::builder()
                    .with_source(kalosm::sound::WhisperSource::QuantizedTiny)
                    .with_language(if lang_opt {
                        Some(kalosm::sound::WhisperLanguage::English)
                    } else {
                        None
                    })
                    .build()
                    .await?;
                println!("--- language={}", if lang_opt { "English" } else { "auto" });
                for (label, buf) in [
                    ("silence", silence.clone()),
                    ("noise", noise.clone()),
                    ("speech", speech[..n.min(speech.len())].to_vec()),
                ] {
                    let t = Instant::now();
                    let mut task = w.transcribe(SamplesBuffer::new(1, 16000, buf));
                    while let Some(seg) = task.next().await {
                        println!(
                            "  {}: text={:?} no_speech={:.3} conf={:.3} ({:.1}s)",
                            label,
                            seg.text().trim(),
                            seg.probability_of_no_speech(),
                            seg.confidence(),
                            t.elapsed().as_secs_f32()
                        );
                    }
                }
            }
            Ok::<(), anyhow::Error>(())
        })?;
    } else if mode == "sess" {
        // Is the ~21s prefill paid once per *generation*, or once per *session*? If a reused
        // chat session skips it, holding one session in TranslateEngine is a free 21s/caption.
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async move {
            let model = Llama::phi_3().await?;
            let mut chat = model.chat().with_system_prompt(
                "You are a professional translator. Translate the user's text into Chinese (Traditional). Output ONLY the translated text.",
            );
            let msgs = [
                "The pipeline captures system audio and transcribes it with Whisper.",
                "Translation is decoupled so it never blocks capture.",
                "VAD segments speech by energy and hangover frames.",
            ];
            for (i, m) in msgs.iter().enumerate() {
                let t = Instant::now();
                let mut n = 0usize;
                let mut first = 0f32;
                let mut s = chat.add_message(*m);
                while s.next().await.is_some() {
                    if n == 0 {
                        first = t.elapsed().as_secs_f32();
                    }
                    n += 1;
                    if n >= 30 {
                        break;
                    }
                }
                println!(
                    "SESS msg{}: ttft={:.2}s total={:.2}s tokens={} tok/s={:.2}",
                    i,
                    first,
                    t.elapsed().as_secs_f32(),
                    n,
                    n as f32 / t.elapsed().as_secs_f32()
                );
            }
            Ok::<(), anyhow::Error>(())
        })?;
    } else if mode == "pos" {
        // Does candle reuse its KV cache? If per-token cost grows with position it does not,
        // and no amount of sampling tuning will make the translator fast.
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async move {
            use kalosm::language::TextCompletionModelExt;
            let model = Llama::phi_3().await?;
            let prompt = "Translate to Chinese (Traditional): The pipeline captures system audio and transcribes it with Whisper.\nTranslation: ";
            let t = Instant::now();
            let mut s = model.complete(prompt);
            let mut last = t.elapsed();
            let mut n = 0usize;
            while s.next().await.is_some() {
                n += 1;
                let now = t.elapsed();
                if n <= 3 || n.is_multiple_of(10) {
                    println!(
                        "POS token#{:<3} since_start={:>6.2}s delta={:>5.2}s",
                        n,
                        now.as_secs_f32(),
                        (now - last).as_secs_f32()
                    );
                }
                last = now;
                if n >= 60 {
                    break;
                }
            }
            Ok::<(), anyhow::Error>(())
        })?;
    } else if mode == "raw" {
        // Raw completion: bypass kalosm's Chat machinery entirely, to see whether the per-token
        // cost is the model or the chat/template plumbing around it.
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async move {
            use kalosm::language::TextCompletionModelExt;
            let model = Llama::phi_3().await?;
            let prompt = "Translate to Chinese (Traditional): The pipeline captures system audio and transcribes it with Whisper.\nTranslation: ";
            for rep in 0..2 {
                let t = Instant::now();
                let mut n = 0usize;
                let mut first = None;
                let mut text = String::new();
                let mut s = model.complete(prompt);
                while let Some(tok) = s.next().await {
                    if first.is_none() { first = Some(t.elapsed().as_secs_f32()); }
                    text.push_str(&tok);
                    n += 1;
                    if n >= 40 { break; }
                }
                let wall = t.elapsed().as_secs_f32();
                println!("RAW rep{}: tokens={} wall={:.2}s tok/s={:.2} ttft={:.2}s text={:?}",
                    rep, n, wall, n as f32 / wall, first.unwrap_or(0.0),
                    text.chars().take(120).collect::<String>());
            }
            Ok::<(), anyhow::Error>(())
        })?;
    } else {
        let which = std::env::args().nth(2).unwrap_or_else(|| "phi3".to_string());
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async move {
            let model = match which.as_str() {
                "qwen05" => Llama::builder()
                    .with_source(LlamaSource::qwen_2_5_0_5b_instruct())
                    .build()
                    .await?,
                "qwen15" => Llama::builder()
                    .with_source(LlamaSource::qwen_2_5_1_5b_instruct())
                    .build()
                    .await?,
                _ => Llama::phi_3().await?,
            };
            let mut chat = model.chat().with_system_prompt(
                "You are a professional translator. Translate the user's text into Chinese (Traditional). Output ONLY the translated text.",
            );
            for rep in 0..2 {
                let t = Instant::now();
                let mut n = 0usize;
                let mut s = chat.add_message("The pipeline captures system audio and transcribes it with Whisper. Translation is decoupled so it never blocks capture.");
                let mut first = None;
                let mut text = String::new();
                while let Some(tok) = s.next().await {
                    if first.is_none() {
                        first = Some(t.elapsed().as_secs_f32());
                    }
                    text.push_str(&tok);
                    n += 1;
                    if n >= 50 {
                        break;
                    }
                }
                let wall = t.elapsed().as_secs_f32();
                println!(
                    "LLM rep{}: tokens={} wall={:.2}s tok/s={:.2} ttft={:.2}s text={:?}",
                    rep,
                    n,
                    wall,
                    n as f32 / wall,
                    first.unwrap_or(0.0),
                    text.chars().take(140).collect::<String>()
                );
            }
            Ok::<(), anyhow::Error>(())
        })?;
    }
    Ok(())
}

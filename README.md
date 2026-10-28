# Cross-Platform Realtime Audio Transcript & Translation Service

Self-hosted, local-first audio captioning: it captures **system audio** (WASAPI loopback on
Windows), transcribes speech with **Whisper**, translates it with a small **LLM (Phi-3-mini)**,
and streams both to a browser control panel over WebSockets. Everything runs locally — no cloud.

> ⚠️ **Performance reality:** On a typical 2-core CPU, Whisper-tiny runs ~5x slower than real
> time and Phi-3 translation takes minutes per utterance. This is a functional POC, not a
> real-time product. See [Limitations](#limitations).

## Features
- System-audio loopback capture (no mic needed) via `cpal`
- Energy-based VAD segmentation
- Whisper STT (`tiny_q` … `large_v3_turbo`, selectable)
- LLM translation (Phi-3-mini) streamed token-by-token
- Browser control panel + live WebSocket captions
- Local model cache (offline after first download)
- Graceful start/stop, watchdog, and device-loss detection

## Requirements
- Windows 10/11 (WASAPI loopback). Linux/macOS use mic input until loopback parity lands.
- Rust (stable) + MSVC build tools.
- Internet on first run to fetch models from Hugging Face.

## Build & Run
```bash
cargo build --release
target\release\realtime-translator.exe
```
Open **http://localhost:3000**, pick a loopback output device, choose language + STT model,
then **Start**. First launch downloads models to `%LOCALAPPDATA%\realtime-translator\models`.

## Configuration
Persisted at `%APPDATA%\realtime-translator\config.toml`:

| Key | Meaning |
|-----|---------|
| `audio_device_name` / `audio_device_is_loopback` | capture source |
| `target_language` | translation target |
| `whisper_model` | `tiny_q`, `base`, `small`, `medium`, `large_v2`, `large_q` |
| `vad_threshold`, `vad_hangover_frames`, `max_chunk_seconds` | segmentation tuning |

Settings can also be changed in the UI (**Apply** saves; **Start** applies immediately).

## How it works
`capture -> VAD -> STT (worker) -> queue -> translation (separate task) -> WebSocket`

Translation is decoupled so it never blocks capture/transcription.

## HTTP / WS API
- `GET /api/status`, `GET /api/config`, `POST /api/config`
- `POST /api/start`, `POST /api/stop`, `GET /api/devices`
- `GET /api/debug/audio` (pipeline counters)
- `GET /ws` (transcript / translation events)

## Limitations
- **Not real-time on low-core CPUs.** Concurrency can't add throughput when compute-bound;
  real-time needs GPU acceleration.
- GPU (CUDA) is **not yet exposed** as a build feature — enabling candle/ort CUDA is a roadmap item.
- Debug builds are extremely slow; always use `--release`.
- Translation latency dominates; ASR output is not yet streamed per-segment.

## Project layout
```
src/
  audio/       capture (cpal) + VAD
  pipeline/    worker loop, STT, translation, events
  web/         Axum routes + WebSocket
  config.rs    state.rs    main.rs
static/        browser UI
```

## Roadmap
- Per-segment streaming ASR, shorter VAD windows
- CUDA/GPU backend for real-time
- Faster/optional translation backend
- Floating always-on-top caption overlay
- Cross-platform loopback parity

## Third-party components & attributions
This project is built on, and depends on, the following third-party components. All are
permissively licensed (MIT and/or Apache-2.0) and compatible with the Apache-2.0 project license.

**Machine-learning models** (downloaded to the local model cache on first run; not bundled in
this repository):

| Component | Used for | License |
|-----------|----------|---------|
| [Whisper](https://github.com/openai/whisper) (OpenAI) | speech-to-text | MIT |
| [Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) (Microsoft) | translation | MIT |

**Core libraries:**

| Crate | License |
|-------|---------|
| [kalosm](https://github.com/LostDok/kalosm) | MIT / Apache-2.0 |
| [cpal](https://github.com/RustAudio/cpal) | Apache-2.0 |
| [axum](https://github.com/tokio-rs/axum), [tokio](https://github.com/tokio-rs/tokio), [tower-http](https://github.com/tower-rs/tower-http) | MIT |
| [ort](https://github.com/pykeio/ort), [candle](https://github.com/huggingface/candle), [tokenizers](https://github.com/huggingface/tokenizers) | MIT and/or Apache-2.0 |
| [rodio](https://github.com/RustAudio/rodio) | MIT / Apache-2.0 |

All other transitive dependencies are likewise MIT and/or Apache-2.0. To regenerate a full,
exact list of every crate and license, run `cargo install cargo-about` and generate a notice file.

## License
Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full text.
Copyright 2026 PeterYuen.

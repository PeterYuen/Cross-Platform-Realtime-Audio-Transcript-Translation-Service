# Cross-Platform Realtime Audio Transcript & Translation Service

Self-hosted, local-first audio captioning: it captures **system audio** (WASAPI loopback on
Windows), transcribes speech with **Whisper**, translates it with a small local **LLM**, and streams
both to a browser control panel over WebSockets. Everything runs locally — no cloud.

> ⚠️ **Performance reality:** this is CPU-bound and depends heavily on core count. With whisper.cpp
> and llama.cpp it keeps up with continuous speech on a 12-thread laptop; on a 2-core machine
> Whisper-tiny runs ~5x slower than real time and translation takes minutes per utterance. See
> [Performance](#performance) and [Limitations](#limitations).

## Features
- System-audio loopback capture (no mic needed) via `cpal`
- Energy-based VAD segmentation
- Whisper STT (`tiny_q` … `large_v3_turbo`, selectable)
- LLM translation streamed token-by-token (llama.cpp + Qwen instruct; candle Phi-3 as fallback)
- Browser control panel + live WebSocket captions
- Local model cache (offline after first download)
- Graceful start/stop, watchdog, and device-loss detection

## Requirements
- Windows 10/11 (WASAPI loopback). Linux/macOS use mic input until loopback parity lands.
- Rust (stable), plus MSVC build tools **or** the MinGW-w64 toolchain described below.
- Internet on first run to fetch models from Hugging Face.
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) and
  [llama.cpp](https://github.com/ggml-org/llama.cpp) built locally — see
  [Third-party binaries](#third-party-binaries).

## Build & Run
```bash
cargo build --release
target\release\realtime-translator.exe
```
Open **http://localhost:3000**, pick a loopback output device, choose language + STT model,
then **Start**. First launch downloads models (Whisper GGUFs + the Qwen instruct GGUF) into
`%APPDATA%\kalosm\cache`; the `HF_HOME` cache set by `main.rs` is `%LOCALAPPDATA%\realtime-translator\models`.

### Run it locally

The binary starts either way, but with nothing configured it quietly runs the candle fallback and
every number in [Performance](#performance) is 20-40x worse. Two things it cannot detect for you:

1. **Both server binaries must be reachable** — see
   [Third-party binaries](#third-party-binaries). A healthy server already listening on 9183/8183 is
   *adopted* rather than restarted, so leftover processes get reused.
2. **The weights you want must be pinned.** When whisper.cpp is active, `whisper_model` is ignored
   (it selects the candle model only); the weights come from `stt.model`. Auto-discovery ranks
   `tiny` ahead of `base`, so name the file explicitly if you want base or small.

```toml
[stt]
server = "C:\\whisper.cpp\\build\\bin\\whisper-server.exe"
model  = "C:\\whisper.cpp\\models\\ggml-base.bin"   # tiny wins discovery unless pinned

[translation]
llama_server = "C:\\llama.cpp\\build\\bin\\llama-server.exe"
```

Then run **from the repo root** — `static/app.js` and `static/style.css` are served from disk, so
the wrong working directory looks exactly like a broken UI change:

```powershell
$env:RUST_LOG = "info"
target\release\realtime-translator.exe
```

Read the first log lines: `stt backend: whisper.cpp` and `translation backend: llama.cpp` mean you
are on the fast path. Anything that says candle means that backend was not found.

#### Verify without a microphone

Live WASAPI loopback can go silent without raising an error, so "no captions appeared" is not
evidence of a bug. Drive the real VAD + STT + translation path from a clip instead:

```powershell
$env:RT_TEST_WAV = "C:\path\to\clip.wav"    # 16 kHz mono; bench exits 2 if unset
cargo run --release --bin bench -- lat
```

Modes are `stt | lat | win | halluc | sess | pos | raw`, and `lat` is the end-to-end gate. Pass =
non-empty transcripts *and* translations, no `STREAM contained ... foreign-script chars`, and a
backlog that does not grow without bound.

### Building without Visual Studio (x86_64-pc-windows-gnu)

The `stable-x86_64-pc-windows-gnu` toolchain works and needs no MSVC, but two things are required:

1. **A C/C++ compiler on PATH** — `ring` and `onig_sys` compile C/C++ (winlibs MinGW-w64 is fine).
2. **A local ONNX Runtime** — `ort-sys` ships no prebuilt binaries for the GNU target, so point it at a
   directory containing `onnxruntime.dll` plus a GNU import library:

```powershell
# 1. MinGW-w64 (winlibs) — provides gcc/g++/dlltool
curl.exe -Lo mingw.zip https://github.com/brechtsanders/winlibs_mingw/releases/download/16.2.0posix-23.1.3-14.0.0-ucrt-r3/winlibs-x86_64-posix-seh-gcc-16.2.0-mingw-w64ucrt-14.0.0-r3.zip
Expand-Archive mingw.zip -DestinationPath tools

# 2. ONNX Runtime 1.18.1 + a dlltool-generated import lib (ort-sys expects 1.18.1)
curl.exe -Lo ort.zip https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-win-x64-1.18.1.zip
Expand-Archive ort.zip -DestinationPath tools
mkdir tools\ort-lib
copy tools\onnxruntime-win-x64-1.18.1\lib\*.dll tools\ort-lib\
gendef.exe tools\ort-lib\onnxruntime.dll > tools\ort-lib\onnxruntime.def
dlltool.exe -d tools\ort-lib\onnxruntime.def -l tools\ort-lib\libonnxruntime.dll.a -D tools\ort-lib\onnxruntime.dll

# 3. Build with MinGW + the rustlib dlltool on PATH
cargo build --release
```

Then copy `onnxruntime.dll` and `onnxruntime_providers_shared.dll` next to
`target\release\realtime-translator.exe` — the binary has a load-time import for them, so it will not
start without them.

## Configuration
Persisted at `%APPDATA%\realtime-translator\config.toml`:

| Key | Meaning |
|-----|---------|
| `audio_device_name` / `audio_device_is_loopback` | capture source |
| `target_language` | translation target |
| `whisper_model` | candle STT weights: `tiny_q`, `base`, `small`, `medium`, `large_v2`, `large_q`. Ignored when whisper.cpp is active — use `stt.model` |
| `vad_threshold`, `vad_hangover_frames`, `max_chunk_seconds` | segmentation tuning |
| `translation.engine` | `auto` (default), `llama`, or `candle` |
| `stt.engine` | `auto` (default), `whisper` (whisper.cpp), or `candle` |
| `stt.server` / `stt.model` | path to `whisper-server.exe` / ggml model; empty = auto-discover |
| `stt.language` | source language, e.g. `en`. `auto` costs an extra encoder pass (~0.5 s) |
| `stt.port`, `stt.threads` | whisper server port (9183) and threads (6) |
| `translation.llama_server` | path to `llama-server.exe`; empty = search PATH and build output |
| `translation.model` | path to a `.gguf`; empty = search the model cache for Qwen instruct |
| `translation.port`, `translation.threads` | llama server port (default 8183) and CPU threads (default 4 — more measured *worse*, it steals cores from Whisper) |
| env: `RT_WHISPER_SERVER`, `RT_WHISPER_MODEL`, `RT_LLAMA_SERVER`, `RT_LLAMA_MODEL`, `WHISPER_CPP_HOME`, `LLAMA_CPP_HOME` | locate the binaries/models without editing config |

### Third-party binaries

Both models run as child processes this app starts and supervises — `whisper-server.exe` on
`127.0.0.1:9183` and `llama-server.exe` on `127.0.0.1:8183`. They are not bundled, so you build
them once:

```powershell
git clone https://github.com/ggml-org/whisper.cpp && cd whisper.cpp
cmake -B build -S . ; cmake --build build --config Release   # produces whisper-server.exe

git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -S . -DLLAMA_BUILD_SERVER=ON ; cmake --build build --target llama-server
```

Then tell the app where they are, by whichever of these you prefer (first match wins):

1. `[stt].server` / `[translation].llama_server` in `config.toml`
2. `RT_WHISPER_SERVER` / `RT_LLAMA_SERVER` environment variables
3. `WHISPER_CPP_HOME` / `LLAMA_CPP_HOME` pointing at the build tree
4. Placed next to `realtime-translator.exe` — makes a release zip self-contained
5. Anything on `PATH`

Model files resolve the same way via `[stt].model`, `RT_WHISPER_MODEL`, `WHISPER_CPP_HOME\models`, or
a `models` folder next to the executable. Translation weights are discovered in the model cache and
prefer the largest Qwen instruct GGUF found — quality tracking accuracy depends on it.

`llama-server.exe` and `whisper-server.exe` each need the MinGW runtime DLLs beside them
(`libgomp-1.dll`, `libstdc++-6.dll`, `libgcc_s_seh-1.dll`, `libwinpthread-1.dll`, `libdl.dll`).
Without them Windows fails the load before the app can report anything, which looks like a hang in
`loading_models`.

If a backend is missing the app logs a warning and falls back to candle Phi-3 (~20 s per caption)
so nothing breaks silently.

### Translation backends
By default the app starts `llama-server.exe` as a child process and translates through it
(~0.5 s per caption). If llama.cpp is missing it logs a warning and falls back to candle Phi-3
(~20 s per caption) so nothing breaks.

Two things to know:
- llama.cpp must be built first (`cmake --build . --target llama-server`).
- `llama-server.exe` needs the MinGW runtime DLLs next to it (`libgomp-1.dll`, `libstdc++-6.dll`,
  `libgcc_s_seh-1.dll`, `libwinpthread-1.dll`, `libdl.dll`). Without them Windows fails the load
  before the app can report anything. The same applies to `whisper-server.exe`.

### Terminology (glossary)
Small models invent literal equivalents for business jargon — "budgeted positions" came out as
"next quarter's egg budget". Add your own terms so they stay fixed:

```toml
[[translation.glossary]]
language = "Korean"
source   = "middleware"
target   = "미들웨어"
```

A term is only mentioned to the model when it actually appears in the text being translated;
listing one unconditionally made the model insert it into sentences that never mentioned it.

### Source languages
The STT model is multilingual, so Japanese, Mandarin, Cantonese and Korean all work. Two rules:
- The request must carry a `language` field. Omitting it makes whisper default to English and it
  will transcribe CJK audio as invented English — the code always sends `auto` now.
- Cantonese maps to `zh`, **not** `yue`; `yue` triggers a long repetition loop.

Accuracy on CJK improves markedly with `stt.model` set to `ggml-base.bin` (~4x slower, still
faster than real time).

Settings can also be changed in the UI. **Apply** saves and restarts the pipeline so the new
settings take effect; **Start** applies them immediately.

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
- **No speaker diarisation.** Attribution is only as good as utterance ordering, so multi-speaker
  meetings are not separated.
- **Translation quality is bounded by the model.** Qwen2.5-1.5B keeps an occasional stray
  source-language character, and Whisper-tiny mishears domain terms ("headcount" → "headcom"),
  which no glossary can repair because the term never reaches it. `stt.model = ggml-base.bin` fixes
  most of that at roughly 4x the cost.
- Live WASAPI loopback capture can go silent without raising an error; the watchdog reports it, but
  the underlying cause is environmental. If `/api/debug/audio` shows `frames` stuck at 0 while
  `device_error` is false, nothing is being captured — re-check with `bench lat` before assuming a
  latency bug.
- **A pipeline stuck in `loading_models` is a load failure, not a hang.** The MinGW runtime DLLs
  must sit next to `whisper-server.exe` / `llama-server.exe`; without them Windows refuses the load
  before the app can report anything.
- **UI changes that do not appear:** `static/app.js` and `static/style.css` are served from disk and
  take effect on reload (hard-refresh, a stale `app.js` looks identical to a broken change), but
  `static/index.html` is compiled into the binary and needs a rebuild.

## Performance

Measured on a 12-thread i7-1355U, CPU only, `bench lat` driving the real pipeline. Full history and
methodology in [BENCHMARKS.md](BENCHMARKS.md).

| Stage | candle (original) | whisper.cpp + llama.cpp |
|---|---|---|
| Transcribe a 10 s chunk | ~8 s | ~0.7 s |
| One translated caption | 15–35 s | 0.5–1.1 s |
| First translated token | after full generation | ~150–250 ms |
| End-to-end, audio in → translation out | ~31 s | ~1.2 s |

Translation is the bottleneck, not capture or ASR, and the gap is throughput rather than tuning:
the same prompt on a 0.5B model answers several Korean sentences in Chinese outright, which is why
model size is a correctness decision here, not a latency one.

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
| [Whisper](https://github.com/openai/whisper) (OpenAI) | speech-to-text weights | MIT |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (ggerganov / ggml-org) | transcription runtime | MIT |
| [Qwen2.5-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) (Alibaba) | translation | Apache-2.0 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) (ggml-org) | translation runtime | MIT |
| [Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) (Microsoft) | fallback translation | MIT |

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

# AGENTS.md

Rust single-binary app: WASAPI loopback capture → VAD → Whisper STT → LLM translation → browser UI.
Windows-first. `README.md` owns usage, HTTP/WS API and build prerequisites; `BENCHMARKS.md` is the
measurement log. This file is only the non-obvious, agent-relevant context.

## Commands

- `cargo build --release` — **always release.** Debug ML (candle/ort/tokenizers) is ~10x slower.
- `cargo check` — fast typecheck once deps are cached.
- `cargo test --lib` — the only automated tests: output-filter unit tests in
  `src/pipeline/translate.rs`. Run this before touching translation logic.
- Run: `target\release\realtime-translator.exe`, then open `http://localhost:3000`.
- Bench harness: `cargo run --release --bin bench -- <mode>` where mode is
  `stt | lat | win | halluc | sess | pos | raw`. `lat` is the end-to-end latency harness — it drives
  the real VAD + STT + translation path from a WAV fed in real time, so it works without a microphone.
  Env overrides: `RT_TEST_WAV`, `RT_TARGET`, `RT_HANGOVER`, `RT_NO_TR`, `RT_LLM_MODEL`,
  `RT_STT_ENGINE`, `RT_STT_LANG`, `RT_STT_PORT`, `RT_PORT`. Set `RUST_LOG=info` to see which backend
  is actually loaded — bench does not log by default, and "which model is this" is the first thing
  you need when numbers come out slow.
- No CI, no `rustfmt.toml`, no clippy config, no `.cargo/config.toml`, no `rust-toolchain`.

## Prerequisites (see README § Building without Visual Studio)

The `x86_64-pc-windows-gnu` toolchain builds cleanly without MSVC, but needs: MinGW-w64 on PATH
(`ring`/`onig_sys` compile C), and a local ONNX Runtime 1.18.1 with a `dlltool`-generated import
library. `onnxruntime.dll` + `onnxruntime_providers_shared.dll` must sit next to the built exe or it
will not start. The same applies to the MinGW runtime DLLs (`libgomp-1.dll`, `libstdc++-6.dll`,
`libgcc_s_seh-1.dll`, `libwinpthread-1.dll`, `libdl.dll`) next to `whisper-server.exe` and
`llama-server.exe` — with MinGW on PATH they load, launched from the app with a clean PATH they die
instantly and the pipeline hangs in `loading_models`.

## Environment quirks (will bite)

- Shell is **PowerShell 5.1**: chain with `;` / `if ($?)`, never `&&`. `Start-Process` with
  `-RedirectStandardOutput` can hang the caller.
- **`static/index.html` is compiled into the binary** (`include_str!` in `web/mod.rs`) — changing it
  requires a rebuild. `static/app.js` and `static/style.css` are served from disk by tower-http, so
  they take effect on reload. A stale cached `app.js` looks exactly like a broken change.
- Real-time is **not achievable on ≤2-core CPUs**. Don't "fix" latency by adding concurrency — it is
  throughput-bound; concurrency just splits cores and starves each model.
- `main.rs` sizes `RAYON_NUM_THREADS`/`OMP_NUM_THREADS` to the core count (override with those env
  vars). It used to be pinned to 2, which cost ~45s of LLM prefill per translation on a 12-thread
  box instead of ~15s.
- Live WASAPI loopback capture is unreliable on this machine and frequently goes dead (0 frames, no
  error). Verify changes with `bench lat` rather than relying on a live mic/speaker test.
- Models land in `%APPDATA%\kalosm\cache` (kalosm's own cache) and `%LOCALAPPDATA%\realtime-translator\models`
  (`HF_HOME`, set in `main.rs`). Config persists at `%APPDATA%\realtime-translator\config.toml`.

## Architecture (what's not obvious from filenames)

- **Two ML backends run as child processes, not linked in.** `whisper-server.exe` on `127.0.0.1:9183`
  and `llama-server.exe` on `127.0.0.1:8183`. This avoids bindgen/libclang (unavailable on MinGW) and
  keeps a crashing model out of the capture process. Both are managed in
  `pipeline/{whisper_server,llama_server}.rs` with a hand-rolled HTTP/1.1 client — no reqwest.
  Candle/Phi-3 remains a fallback path only; if you see `falling back to candle` in the log, the
  intended backend is not running and the numbers will be 20-40x worse.
- **Two async tasks, not one:** the **worker** (`pipeline/mod.rs::run_pipeline_worker`) does
  capture→VAD→STT and only *enqueues* transcripts; a separate **translator task**
  (`run_translator_task`) drains the queue. This decoupling is intentional — never move translation
  back inline into `process_chunk`.
- Capture sends straight into a `tokio` unbounded channel (`audio/mod.rs`); there is no bridge thread.
- Lifecycle flags live in `state.rs`: `alive`, `ready`, `capture_error` (all `AtomicBool`).
  `Pipeline::stop()` stops capture then arms a 5s watchdog abort.
- Backend selection is per-engine and logged at startup (`stt backend:`, `translation backend:`).

## Hard-won invariants (do not regress)

- **Tokens are streamed to the browser live, so every correctness filter must run *before* a token
  is forwarded.** A check applied to the finished string is useless — the wrong text is already on
  screen. `translate.rs` gates the first `GATE_CHARS` characters and then filters each token; keep
  that ordering.
- **Never send an empty `language` field to whisper-server.** Omitting/blanking it makes whisper
  default to English and silently turns CJK audio into invented English. Always send `"auto"` unless
  a language is configured.
- **Cantonese maps to `zh`, not `yue`.** The `yue` code triggers a long repetition loop.
- **Model discovery must reject `for-tests*` GGUFs** — they load cleanly and return empty text.
- **Prompt carries no few-shot examples, and the glossary lists only terms present in the text.**
  Examples made the small model echo the source verbatim; listing an unused glossary term made it
  inject that term into sentences that never mentioned it.
- **Long inputs must be cut at sentence/clause boundaries**, not word counts — mid-clause cuts get
  echoed or given invented endings.
- **Translation must use the chat API** (`chat.add_message(...)`), not raw `complete()`, and the
  instruction must live in the **system** prompt. On the candle path, putting it in a single user
  message makes Phi-3.5 emit a bare `" "` and stop (translations silently produce 0 chars).
- **Do NOT switch the translator to `tiny_llama_1_1b_chat`** — its chat template returns empty output.
- **VAD must `reset()` whenever the backlog is trimmed** (`MAX_QUEUED_FRAMES` in `pipeline/mod.rs`) —
  otherwise chunks get stitched across dropped audio and Whisper returns empty, making the app look
  frozen.
- **Device loss**: cpal error callbacks set `capture_error`; the worker emits an `Error` event and
  tears down. Keep that path wired through `start_capture`.
- STT model names map in `stt.rs::parse_source` (`tiny_q`,`base`,`small`,`medium`,`large_v2`,`large_q`).
  Loopback = enumerate **output** devices.

## Security / trust

- All model inference is local. Both server processes bind `127.0.0.1` only; the web UI binds
  `0.0.0.0:3000`. Do not widen the server binds.
- First launch downloads model weights from Hugging Face into the caches above. Everything else runs
  offline.
- There are no credentials, tokens or secrets in this repository, and none belong in config.toml.

## Protected / generated

- Gitignored: `/target`, `/models`, `*.log`, `/build*.{txt,err,log}`, `/server_*.log`.
- `Cargo.lock` — regenerate with cargo, never hand-edit.
- `BENCHMARKS.md` is an append-style measurement log with one section per round. Add a new section
  with the measured numbers; do not rewrite or summarise earlier rounds.

## Completion criteria

1. `cargo check` clean, then `cargo build --release` (kill any running `realtime-translator`,
   `llama-server`, `whisper-server` first — the linker cannot overwrite a running exe).
2. `cargo test --lib` passes.
3. `bench lat` on `C:\temp\tools\speech16.wav` (or the `lang_*.wav` clips for non-English) produces
   non-empty transcripts and translations, no `STREAM contained ... foreign-script chars` warnings,
   and backlog that does not grow without bound.
4. If you changed anything in the web layer, confirm the served page reflects it — remember the
   `index.html` rebuild rule.

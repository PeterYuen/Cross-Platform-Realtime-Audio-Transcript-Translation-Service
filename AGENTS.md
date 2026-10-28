# AGENTS.md

Rust single-binary app: WASAPI loopback capture → VAD → Whisper STT → Phi-3 translation → browser UI.
Windows-first. See `README.md` for usage/API; this file is only the non-obvious, agent-relevant context.

## Commands
- `cargo build --release` — **always release.** Debug ML (candle/ort/whisper) is ~10x slower and unusable.
- `cargo check` — fast typecheck (deps cached).
- Run: `target\release\realtime-translator.exe`, then open `http://localhost:3000`.
- No tests, no CI, no lint/rustfmt config. Verify by running the server + polling `/api/debug/audio`.

## Environment quirks (will bite)
- Shell is **PowerShell 5.1**: chain with `;` / `if ($?)`, never `&&`. `Start-Process` with `-RedirectStandardOutput` can hang the caller.
- Real-time is **not achievable on ≤2-core CPUs** (Whisper-tiny ~5x slower than RT; Phi-3 translation minutes/utterance). Don't "fix" latency by adding concurrency — it's throughput-bound; concurrency just splits cores and starves each model.
- Models download once to `%LOCALAPPDATA%\realtime-translator\models` (`HF_HOME` set in `main.rs`). Config persists at `%APPDATA%\realtime-translator\config.toml` (path from `dirs_next::config_dir`).
- `main.rs` sets `RAYON_NUM_THREADS=2`/`OMP_NUM_THREADS=2` unless already set — don't raise these without checking core count.

## Architecture (what's not obvious from filenames)
- Two async tasks, not one: the **worker** (`pipeline/mod.rs::run_pipeline_worker`) does capture→VAD→STT and only *enqueues* transcripts; a separate **translator task** (`run_translator_task`) drains the queue. This decoupling is intentional — never move translation back inline into `process_chunk`.
- Capture sends straight into a `tokio` unbounded channel (`audio/mod.rs`); there is no bridge thread.
- Lifecycle flags live in `state.rs`: `alive`, `ready`, `capture_error` (all `AtomicBool`). `Pipeline::stop()` stops capture then arms a 5s watchdog abort.

## Hard-won invariants (do not regress)
- **Translation must use the chat API** (`chat.add_message(...)`), not raw `complete()` — raw generation never stops.
- **Do NOT switch the translator to `tiny_llama_1_1b_chat`** — its chat template returns empty output. Phi-3-mini is the smallest working translator here.
- **VAD must `reset()` whenever the backlog is trimmed** (`MAX_QUEUED_FRAMES=300` in `pipeline/mod.rs`) — otherwise chunks get stitched across dropped audio and Whisper returns empty, making the app look frozen.
- **Device loss**: cpal error callbacks set `capture_error`; the worker emits an `Error` event and tears down. Keep that path wired through `start_capture`.
- STT model names map in `stt.rs::parse_source` (`tiny_q`,`base`,`small`,`medium`,`large_v2`,`large_q`). Loopback = enumerate **output** devices.

## Verifying a change end-to-end
1. Kill any `realtime-translator` process, `cargo build --release`.
2. Launch, `POST /api/config` (loopback device + `whisper_model=tiny_q`), `POST /api/start`.
3. Play audio; poll `/api/debug/audio` every ~10s. Pass = `transcribe_nonempty` and `translations` both climb over time (not just once).

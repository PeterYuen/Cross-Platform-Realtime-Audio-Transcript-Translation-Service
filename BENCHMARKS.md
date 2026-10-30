# Benchmarks

Measured on the dev box this POC runs on: **Intel i7-1355U (10 cores / 12 threads, no dGPU,
Intel Iris Xe iGPU), 32 GB RAM, Windows 11**, release build, `x86_64-pc-windows-gnu`.

Reproduce with `cargo run --release --bin bench -- <mode>` where mode is
`stt | lat | win | halluc | sess | pos | raw` (`lat` is the end-to-end harness used by most of the
rounds below; `llm`/`qwen05` were candle-era modes since folded into `lat`).
(`src/bin/bench.rs`; STT uses a 16 s TTS clip resampled to 16 kHz.)

## Stage throughput (models loaded alone, no contention)

| Stage | Engine | Result | Real-time? |
|---|---|---|---|
| Whisper `tiny_q` (Q8 GGUF) | candle | **RTF 1.26–1.54×** (16 s clip in 20–24 s) | borderline |
| Phi-3.5-mini Q4_K_M, chat API | candle | **0.16 tok/s** (~6 s/token) | no |
| Phi-3.5-mini Q4_K_M, raw completion | candle | **0.96 tok/s**, 21 s time-to-first-token | no |
| Qwen2.5-0.5B (chat API) | candle | ~0.6 tok/s (≈1.6 s/token) | no |
| Thread count 2 → 8 (`RAYON_NUM_THREADS`) | candle | RTF 1.45 → 1.26 (STT), LLM unchanged | — |

Key observations:

- **Whisper is not the bottleneck.** candle's Whisper is already within ~1.4× of real time.
- **The LLM is the bottleneck by ~1–2 orders of magnitude.** 15 tokens of translation cost 25–90 s.
- The LLM slowness is *not* model size: dropping to a 0.5 B model only gained ~4×. candle spends
  ~21 s prefilling a 40-token prompt, which points at candle's GGUF prefill/decode path, not the
  weights. Adding threads does not help, so the kernels are not parallelised.
- In the live pipeline the numbers get much worse than the isolated ones (STT measured 10–15× RT
  while transcribing) because STT and the LLM contend for the same two rayon threads.

## End-to-end, before any fix (log-derived)

| Event | Latency |
|---|---|
| 0.57 s chunk → transcript | 7.1 s |
| 2.31 s chunk → transcript | 34.9 s |
| 5.00 s chunk → transcript | 45.6 s |
| transcript → translation (39 chars) | 44 s |

Backlog grows without bound, so latency degrades the longer it runs.

## Tuning experiments (round 2)

### Whisper: chunk size dominates cost

Same 16 s clip, transcribed in windows of N seconds:

| Window | Total wall | RTF | Notes |
|---|---|---|---|
| 16 s (one call) | 21 s | **1.26–1.54×** | |
| 5 s (4 calls) | 41 s | **2.57×** | current default was 5 s |
| 2 s (9 calls) | 130 s | **8.13×** | also drops words |
| 1 s (17 calls) | 151 s | **9.44×** | |

Whisper pays a large fixed cost per call, so *shorter* chunks are dramatically slower and less
accurate. Counter-intuitive but decisive: raise `max_chunk_seconds`, do not lower it.

### Translator: where the 21 s goes

Raw completion, Phi-3.5-mini Q4, position-by-position:

```
token#1  +21.26s   <- prefill
token#2  +0.47s
token#10 +0.45s   token#20 +0.45s   token#30 +0.41s   token#40 +0.49s
```

- Decode is **flat with position** → candle *does* reuse its KV cache. Steady state ≈ 2 tok/s.
- The **21 s prefill repeats on every generation** (rep1 of the same test: 20.2 s), so it is a
  per-generation cost, not warm-up. It is paid once per translated caption.

Reusing one chat session instead of building a new `Chat` per translation:

| Turn | Time to first token | Total (30 tokens) |
|---|---|---|
| 1 (fresh session) | 17.2 s | 43.3 s |
| 2 (session reused) | **6.8 s** | 28.9 s |
| 3 | 7.1 s | 20.0 s |

## Changes made from these measurements

| Change | Where | Measured/expected effect |
|---|---|---|
| Reuse one chat session (recycled every 8 turns) | `translate.rs` | prefill 17 s → 7 s per caption |
| Coalesce queued transcripts into one generation | `pipeline/mod.rs` | fewer × more expensive generations; bounds the queue |
| Cap output at ~3× source length (was flat 512 chars) | `translate.rs` | 539-char ramblings → ~40 tokens, ~6× less decode |
| Strip "(Note: …)"/pinyin commentary from output | `translate.rs` | stops showing non-translation text |
| `max_chunk_seconds` default 5 → 10 | `config.rs` | Whisper RTF 2.6× → ~1.5× |

## Round 3: the level bar, and the thread cap

**Why the input meter froze.** `AudioLevel` was emitted from the worker's own `select!` ticker
branch. While the worker was inside a 20–45 s transcription the loop was parked, so no events went
out at all; and when frames arrived fast the `recv()` branch starved the ticker. A dedicated 10 Hz
ticker task now emits levels independently of model work — measured **121 `audio_level` messages in
12 s** (~10 Hz).

The scale was also wrong: `rms * 300` saturates at ~100 % for normal speech (rms ≈ 0.13), so during
real audio the bar sat pinned right — visually identical to a hang. Now mapped to dBFS (-60…0 dB)
with fast attack / slow release.

**The thread cap was costing 3×.** `main.rs` pinned `RAYON_NUM_THREADS=2`, a rule written for a
2-core box. Prefill per translation on this machine:

| RAYON_NUM_THREADS | Prefill | Decode |
|---|---|---|
| 1 | 45.1 s | ~0.7 s/token |
| 2 (what shipped) | ~45 s | — |
| 8 | 21 s | 0.45 s/token |
| 12 (all cores) | **14.9 s** | **0.30 s/token** |

Now sized to `available_parallelism()` instead of hard-coded 2.

**Watchdog.** WASAPI handed us a stream that fired zero callbacks while reporting no error, and
`/api/status` said `stage: ready`. A frame-counter watchdog now raises an explicit error after 15 s
of silence (verified firing on this box).

## Round 4: llama.cpp spike (the real-time answer)

Installed CMake 4.4.4 per-user (`winget install Kitware.CMake --scope user`, no admin) and built
llama.cpp for `x86_64-w64-windows-gnu`. Two gotchas worth recording:
`cpp-httplib` fails on MinGW (`CreateFile2` undeclared) unless the build defines
`-D_WIN32_WINNT=0x0A0000`; and the HTTP server target must be off (`-DLLAMA_BUILD_SERVER=OFF`).

Measured with `llama-bench` on this box (pp = prefill tok/s, tg = decode tok/s):

| Model | threads | pp128 | tg64 | ~caption wall time* |
|---|---|---|---|---|
| Phi-3.5-mini Q4_K_M (3.8 B) | 6 | 16.9 | 5.8 | ~7 s |
| Phi-3.5-mini Q4_K_M | 12 | 30.2 | 4.8 | ~7 s |
| Qwen2.5-0.5B Q4_K_M | 12 | 141 | 7.4 | ~4 s |
| Qwen2.5-0.5B Q4_K_M | **4** | 70 | **13.9** | **~2.5 s** |
| candle (what ships today) | 2–12 | ~3 | 2.0–3.3 | 40–60 s |

\* prefill of a ~40-token prompt + ~30 output tokens.

Decode is memory-bound, so **fewer threads are faster** for generation (Qwen: 13.9 tok/s at 4
threads vs 7.4 at 12). Prefill wants more. Worth splitting them instead of sharing one knob.

A real translation through `llama-completion -t 4 --temp 0`:

```
Text: The pipeline captures system audio and transcribes it with Whisper.
Translation: 管道捕捉系统音频并转录它，使用 Whisper。
```

Understandable, slightly stiff — good enough for rolling captions, not for professional subtitles.

**Conclusion:** real-time on this hardware is reachable, but only by dropping candle for
llama.cpp. With candle the floor is ~40 s per caption no matter how the app is tuned.

## Round 5: config UI behaviour

- **Apply is now hidden unless something actually changed.** It only appears when Audio source,
  Target language or Speech recognition model differ from the saved config; reverting a selection
  hides it again. Verified with a small DOM harness (`C:\temp\tools\uitest.js`, 7 assertions).
- **Found and fixed a silent config-destroying bug:** `buildConfig()` in `static/app.js` sent
  hardcoded `vad_threshold: 0.003, vad_hangover_frames: 15, max_chunk_seconds: 5.0` on every
  save/start, even though the UI never shows those fields. Every Apply/Start was therefore resetting
  the VAD tuning — this is what kept undoing the `max_chunk_seconds = 10` change (it looked like
  another user was fighting the setting). `buildConfig()` now layers the three visible controls on
  top of the server's config so unseen fields survive.

## Round 6: log audit of the live instance

40 minutes of `server11.log` (1054 lines, 16 ERROR):

| Signal | Reading |
|---|---|
| Capture | working again — 507 frame logs, 227 VAD chunks, 78 transcriptions, 26 translations |
| Watchdog | **16 false alarms.** It fired during silence, which is expected: WASAPI loopback emits no callbacks while nothing renders. Fixed to only complain when a stream has produced *zero* frames since Start. |
| Translation size | looked like the 320-char cap was being blown (765 chars). It wasn't — the log printed `String::len()`, i.e. **bytes**, and CJK is 3 bytes/char. Fixed to print characters. |
| Config churn | `max_chunk_seconds` had reverted to 5.0. `POST /api/config` replaced the whole config, so any client that sent a full payload (including a stale browser tab running the old `app.js`) reset fields it never showed. Now merged server-side: a partial body updates only the keys present, and a malformed body is rejected with 400. |
| Throughput | 52 "skipped stale chunk" events — STT keeps roughly a third of chunks, which is the candle-throughput limit, not a bug. |

## Round 7: Whisper producing text that was never said

Reported symptom: `tiny_q` emits captions unrelated to the audio. Reproduced by feeding Whisper
deliberately bad inputs and reading its own confidence numbers (`bench halluc`, `bench win`):

| Input | Output | no_speech |
|---|---|---|
| 5 s digital silence | *(no segment)* | — |
| 5 s low-level noise | *(empty text)* | **0.869** |
| 5 s of real speech | `Pipeline captures system audio, runs voice activity detection.` | 0.005 |
| 2 s windows sliced mid-word | `"a"`, `"??"`, `"hand with never blocks capture."`, `"a lot of things is not real time."` | 0.01–0.50 |

So it is not one bug, it is three, and only the first is a model limitation:

1. **The VAD cut audio mid-word.** When the buffer hit `max_chunk_samples` it emitted wherever it
   happened to be. Whisper-tiny completes a half-word by inventing words — that is where the
   unrelated text comes from. Now the split point is the quietest frame boundary within ~0.6 s of
   the limit, and the tail carries into the next chunk.
2. **No anti-alias filter on the 48→16 kHz downsample.** Linear interpolation decimates straight
   down, folding everything above 8 kHz into the band Whisper's mel filterbank reads. Now a boxcar
   of `ceil(from/to)` samples per output sample. Cheap, and clearly visible in the results below.
3. **Nothing was filtered.** We printed whatever came out, including Whisper's silence answer
   (`no_speech` 0.87) and stray `"a"` / `"??"` fragments. Segments above `no_speech > 0.6` are now
   suppressed and logged as `suppressed hallucination (...)`; meaningless fragments are dropped;
   and the habitual `"a "` prefix artifact is stripped.

Measured before/after on the same four-sentence TTS script:

| | correct transcripts | empty transcriptions | artifacts |
|---|---|---|---|
| before | 1 of 4 usable | 12 of 20 chunks | `a`, `??`, invented sentences |
| after | **4 of 4** | **0 of 4** | none |

`no_speech` on accepted segments landed in 0.00–0.07 versus 0.87 for non-speech, so the 0.6 gate is
sitting in a wide empty gap — it is not a guess.

## Round 8: what actually happens when speech runs past ~20 seconds

Reported symptom: long continuous speech causes delay and the app appears to stick. Measured with a
34 s continuous-speech clip (`bench stt`, 6 threads, machine otherwise idle):

| chunk length | wall/audio | note |
|---|---|---|
| 5 s | 2.59× | fixed per-call cost dominates |
| **10 s** | **1.02–1.21×** | best throughput, and no rambling |
| 17 s | 2.30× | Whisper started generating filler |
| 34 s (one call) | 1.57× | fewer calls, but unstable |

So 10 s chunks are already the right setting — they sit at roughly real-time parity. Two things
nevertheless make long speech fall apart:

**1. Long chunks make Whisper-tiny write its own material.** A 17 s window produced:

```
win0  "猪CAPE, the continuous speech is the hard case for this pipeline…"
win1  "a lot of people who are watching this video need to be in the comments section below.
       If that transcription takes longer than the audio it covers…"
```

That YouTube-outro boilerplate was never said. It is Whisper's decoder running past the end of the
useful content, and every wasted token is also wall-clock we do not have.

**2. Translation starves transcription.** This is the decisive measurement:

| Whisper tiny_q, 10 s chunks | RTF |
|---|---|
| running alone | **1.02×** |
| while candle Phi-3.5 generates concurrently | **1.70×** |

Whisper can keep up with real time on its own. The moment the translator starts generating, the
core pool saturates and Whisper drops to 1.7× — so every 10 s of speech costs 17 s and the backlog
grows without bound. candle Phi-3.5 measures 0.9 tok/s with 13–36 s per caption, which is what
makes the whole thing look frozen.

**And the queue policy made it worse than it needed to.** `run_stt_task` used to keep only the
*newest* queued chunk and discard everything older. Once a backlog formed, every dropped chunk was
speech that simply never appeared — captions stalled, then jumped. That is now replaced with
in-order processing of the whole backlog, capped at 25 s, with an explicit warning when anything is
dropped:

```
stt: 31.2s queued across 4 chunks, working through in order
stt: 48.6s of audio queued, dropping 12.4s of oldest backlog
```

### What would actually close this
The only change that fixes both halves is replacing candle: llama.cpp Qwen2.5-0.5B measured
13.9 tok/s decode (~2.5 s per caption) against candle Phi-3.5's 0.9 tok/s. Cheaper translation
means Whisper keeps its 1.0× instead of being pushed to 1.7×, and the backlog stops growing.

## Round 9: llama.cpp translation swap

Candle Phi-3.5 was the remaining bottleneck: 0.9 tok/s, 15–35 s per caption, and while it generated
it pushed Whisper from 1.02× to 1.70× real time. Swapping the translator to llama.cpp running
Qwen2.5-0.5B (q4_k_m, 491 MB) measured as follows.

**Translation latency** — 6-sentence script, temperature 0:

| backend | per caption |
|---|---|
| candle Phi-3.5 | 15–35 s |
| llama.cpp Qwen-0.5B | **0.4–1.1 s** |

**Contamination of Whisper** (same 34 s clip, 10 s chunks, 6 threads):

| translator running concurrently | Whisper RTF |
|---|---|
| none | 1.02× |
| candle Phi-3.5 | 1.70× |
| llama.cpp Qwen-0.5B, load heavier than real use | 1.24× |

The load generator issued a translation every 3.5 s for 90 s (26 of them), which is several times
busier than the pipeline actually gets — real usage is one caption per ~10 s. So 1.24× should be
read as a worst case, not the expected figure.

### Quality: good enough, not great
Plain "translate this to Chinese" prompting failed badly:

```
Input:  Whisper transcribes each chunk on the worker task.
Output: whispers transcribes each chunk on the worker task.   ← echoed English back
Input:  The device was unplugged, so the pipeline stopped…
Output: 設備插上電源…                                        ← said the opposite
```

Both problems went away once the system prompt demonstrated two worked examples, which is what the
engine now sends. Remaining defects are typical of a 0.5B model: occasional untranslated loanwords
(`captions`, `Pull Request`), and one garbled sentence in six. It reads as a live preview, not a
finished translation.

### How it is wired
`llama-server.exe` runs as a child process and the app talks to it over localhost HTTP, using a
hand-rolled HTTP/1.1 client rather than adding `reqwest` — hyper's client half isn't enabled here
and one localhost port does not justify the dependency. Reasons for this shape:

- Linking llama.cpp into the binary would need bindgen, and bindgen needs libclang, which this
  MinGW toolchain does not have.
- A separate process means a crash in the model takes the translator, not the capture pipeline.

Discovery order for both binary and model: config value → environment → PATH/model cache. If
llama.cpp cannot start, the engine logs a warning and falls back to candle, so nothing stops
working if the dependency is missing.

### Deployment requirement (this will bite)
`llama-server.exe` is dynamically linked against the MinGW runtime. It only starts if
`libgomp-1.dll`, `libstdc++-6.dll`, `libgcc_s_seh-1.dll`, `libwinpthread-1.dll` and `libdl.dll` are
beside it. They have been copied into `C:\temp\llama\build\bin`, and without them the failure is an
obscure Windows loader error rather than something the app can report.

## Round 10: chasing a 100 ms caption

Goal set: transcript **and** translation visible within 100 ms. To measure it honestly I built a
latency harness (`bench lat`) that drives the real VAD, real SttEngine and real TranslateEngine from
a WAV file fed in real time, and measures against the wall-clock instant the audio would have been
captured — so falling behind shows up as growing lag instead of hiding.

### The wall: Whisper costs ~8 s per call no matter how short the audio is

| audio in the call | wall time |
|---|---|
| 0.5 s | 13.3 s |
| 1 s | 8.2 s |
| 2 s | 8.8 s |
| 5 s | 7.6 s |
| 10 s | 9.2 s |

Transcribing half a second of audio costs the same as transcribing ten. Whisper always pads its
input to a 30-second window and runs the full encoder over it, so there is a floor of roughly 8 s
per call that has nothing to do with how much speech there was. Two consequences:

- **Shortening chunks does not reduce latency.** It only multiplies the fixed cost. This explains
  the earlier 2 s-chunk measurement of 8.1× real time.
- **100 ms is unreachable by about 80× on this architecture**, before any tuning is even attempted.

### Where the remaining time goes
4 utterances of continuous speech, 10 s chunks, 6 threads:

| utterance ends | transcript lag | translation lag |
|---|---|---|
| 7.4 s | 10.4 s | 12.5 s |
| 17.0 s | 24.2 s | 25.5 s |
| 26.8 s | 37.1 s | 37.9 s |

With translation switched off entirely the numbers barely move (mean 30.8 s vs 31.4 s), so the
translator is no longer the problem — candle Whisper is simply slower than real time and the queue
grows. Translation now contributes roughly 1 s on top of transcription, not 20.

### What I changed
Translation is streamed token by token from `llama-server` instead of waiting for the finished
sentence. First translated character, measured across utterances: **160–221 ms**.

So the budget is now split like this:

| stage | achieved | target |
|---|---|---|
| audio → transcript | 10–37 s, diverging | 100 ms |
| transcript → first translated character | **0.16–0.22 s** | 100 ms |

### What it would actually take
1. **whisper.cpp instead of candle** for STT. Same architecture, so the 30 s padding floor stays,
   but the encoder is far better optimised; expect roughly 1–2 s per call. That makes the pipeline
   keep up, so lag stops growing, but it does not get near 100 ms.
2. **A streaming ASR model on a GPU** — a Conformer/Transducer or TDT decoder that emits tokens as
   audio arrives, rather than re-encoding a 30-second window every time. This is the only thing that
   meets 100 ms, and it is a different model architecture, not a faster implementation of Whisper.

### Current state
Server runs with llama.cpp translation streaming. Live capture could not be exercised in this round:
WASAPI loopback on this machine is still delivering zero callbacks, so all figures come from the
harness driving the real components, not from a live capture run.

## Round 11: can we stream audio→text→translation continuously?

The idea: emit text as the audio arrives, and start translating each piece immediately, so output
flows continuously instead of in one lump at the end.

**Translation half: already true.** Tokens are streamed from `llama-server` to the UI as they are
decoded. First translated character measured at **156–257 ms** after the transcript exists.

**Audio half: not available in this stack, and not for lack of trying.** Two separate reasons:

1. *The streaming API doesn't do what it looks like.* rwhisper exposes
   `TranscribeChunkedAudioStreamExt`, which takes a stream of audio and yields segments — it looks
   like continuous transcription. Reading its `poll_next` shows it just calls
   `whisper.transcribe(source)` once per stream item and waits. It is sequential chunked
   transcription wearing a streaming costume; every item still pays the full encoder cost.

2. *Whisper emits one segment per call.* To test whether partial text emerges mid-call, segments are
   now forwarded to the translator the instant they are produced. Measured with 20 s chunks
   (several sentences per chunk):

| chunk ends | first text | full text | translation done |
|---|---|---|---|
| 7.4 s | 12.0 s | 12.0 s | 13.6 s |
| 17.0 s | 26.8 s | 26.8 s | 28.7 s |
| 26.8 s | 39.5 s | 39.5 s | 40.2 s |

`first text` equals `full text` on every row. One segment per call, so there is nothing to stream
out early — the decoder finishes the whole chunk and hands over one blob.

So the pipeline is: **bursty** audio→text, then **continuous** text→translation. Making the first
half continuous needs a different decoder, not a different plumbing.

### What would make it genuinely continuous
**Sliding-window partial captions.** Keep a rolling buffer, transcribe the last N seconds every T
seconds, diff against what was already shown, and mark the newest line as provisional until a later
window confirms it. This is how live-caption products behave, and it is plumbing rather than model
work — except that candle pays ~8 s per call, so the window can only refresh every ~8 s, which is
barely better than the current design.

With whisper.cpp the same trick gets interesting: if a call costs ~1 s, the caption refreshes every
second and lag stops diverging. That is the realistic upgrade path, and it is worth building if the
measurements keep pointing at candle being the wall (they have, three rounds running).

**A streaming ASR model** (Conformer/Transducer/TDT, typically on GPU) is the only thing that emits
words as they are spoken. Everything above is an approximation of that behaviour.

## Round 12: whisper.cpp swap — the latency wall came down

Everything before this round was limited by candle costing ~8 s per Whisper call. whisper.cpp does
the same work about 15× faster, so it went in as an optional STT backend with candle as fallback.

**Engine comparison, same model (tiny), same machine:**

| | candle/rwhisper | whisper.cpp |
|---|---|---|
| 5 s clip | ~8 s | **0.74 s** |
| 10 s clip | ~10 s | **0.67 s** |
| 34 s clip | ~35 s | **2.43 s** |

Integrated as `whisper-server.exe` (child process, model resident, multipart POST of an
in-memory PCM16 WAV), same shape as the translator.

### Result: end-to-end latency
34 s clip, 5 s chunks, real-time feed, both servers running:

| utterance ends | transcript | full translated text |
|---|---|---|
| 5.1 s | 0.55 s | 1.81 s |
| 7.4 s | 0.51 s | 0.93 s |
| 12.7 s | 0.52 s | 1.18 s |
| 17.0 s | 0.55 s | 1.35 s |
| 22.2 s | 0.51 s | 1.14 s |
| 26.7 s | 0.47 s | 0.97 s |
| 31.3 s | 0.48 s | 0.97 s |
| 33.9 s | 0.45 s | 0.85 s |

**Mean: 0.51 s to transcript, 1.15 s to full translated text**, down from 31 s. Backlog sits at
0.00 s, so the pipeline now keeps up with continuous speech instead of diverging.

### Two bugs found on the way, both worth knowing about

**1. Model discovery picked a stub.** The whisper.cpp `models/` directory ships
`for-tests-ggml-tiny.bin` alongside the real weights. It loads cleanly and returns empty text, so
every chunk looked like silence. Discovery now rejects anything named `for-tests*`.

**2. The measuring harness was lying.** The feed loop used relative sleeps, so whenever the runtime
was busy the clock drifted and the pipeline looked 15–20 s behind when it was not. Switching to
absolute deadlines (`sleep_until`) dropped `backlog_at_entry` from 16 s to 0.00 s and revealed the
real numbers above. Every latency figure before this point is inflated by the same effect.

### Tuning knob that matters
whisper's language auto-detection costs a whole extra encoder pass per request:

| `stt.language` | transcript | full translation |
|---|---|---|
| `auto` (default) | 0.75 s | 1.67 s |
| `en` | **0.51 s** | **1.15 s** |

If you know the source language, set it — it is a free half-second.

## Round 13: non-English source languages

Asked whether this handles Japanese, Mandarin, Cantonese and Korean. Tested with real TTS audio
(edge-tts voices, converted to 16 kHz mono WAV), run through the actual pipeline.

### It did not work, and the cause was ours

| source | result before fix |
|---|---|
| Japanese | `This online pipeline is made with a system of knowledge and knowledge…` |
| Mandarin | `This is the end of this video. I will continue…` |
| Cantonese | `This is the way to get out of the river.` ×3, 11 s of generation |
| Korean | `This audio type is a function that can be used in a system audio.` |

All four came back as invented English. The reason is a bug in our request, not a model limitation:
whisper-server defaults to `language = "en"` when the `language` form field is absent, and our code
dropped the field whenever the setting was `"auto"` — which was the default. So every non-English
chunk was being transcribed **as if it were English**.

Fix: always send the field, with the literal string `auto` to enable detection. After that:

| source | transcript | full translation |
|---|---|---|
| Japanese | 0.97 s | 1.93 s |
| Mandarin | 0.95 s | 1.90 s |
| Cantonese | 1.42 s | 2.36 s |
| Korean | 0.94 s | 2.00 s |\n
### Two things worth knowing about Cantonese
- whisper's `yue` language code triggers a 10-second repetition loop and returns English garbage.
  Map Cantonese to `zh` instead — it produces correct Traditional Chinese.
- `--detect-language` on the server is not a substitute: it returns only the detected language and
  an empty transcription. Detection has to come from `language=auto` in the request body.

### Translation prompt had the same blind spot
The translator's system prompt said "translate English to Traditional Chinese". On Japanese input it
left kana in the middle of the output; on Korean it mistranslated. Rewording it to name every source
language and adding a Korean example fixed both.

### Accuracy vs speed for non-English
`tiny` is fast but weak here; it confuses homophones throughout CJK (`語音` → `英國`, `擷取` → `結取`).
`base` is noticeably better — correct Korean `감지로`, correct Mandarin `語音管道` — at roughly 4× the
cost:

| model | ~10 s clip | quality |
|---|---|---|
| tiny | ~1.1 s | usable, many homophone errors |
| base | ~4.2 s | clearly better, still keeps up with real time |

Set `stt.model` to switch. For anything beyond a preview, `base` is the better default for CJK.

## Round 13b: what the language test actually showed

I ran the four languages plus English through the real components and read the output, not just
the timings. STT is fine. **Translation is the weak part**, and testing exposed three things.

### 1. Few-shot examples were actively harmful
For long English input, every prompt variant containing worked examples echoed the source back
verbatim:

| prompt | English output |
|---|---|
| instruction only | translated correctly |
| + 1 English example | echoes English |
| + English and Korean examples | echoes English |
| + 2 English examples | echoes English |

So the examples I added in round 9 — which had helped on short sentences — were the reason long
sentences came back untranslated. The prompt is now instruction-only.

### 2. Length, not language, drives the failure
Every echo case was over roughly 100 characters. Translation is now fed sentence-sized pieces
(`split_for_translation`, 80-char budget) instead of whole paragraphs.

### 3. Measured quality, 5 languages

| source | segments | untranslated | mean transcript | mean translation |
|---|---|---|---|---|
| English | 8 | 1 | 1.57 s | 2.89 s |
| Japanese | 1 | 0 | 1.77 s | 3.02 s |
| Mandarin | 1 | 0 | 2.09 s | 3.11 s |
| Cantonese | 3 | 0 | 0.90 s | 1.92 s |
| Korean | 2 | 0 | 0.90 s | 2.26 s |

Typical output quality, as produced:
```
The meter stalls, and the captions fall behind.
  -> 標尺停了，話語落得遙遙相接。        ("stalls" became "stall/counter", meaning lost)

If that transcription takes longer than the audio, the queue grows.
  -> 如果這次轉音時間超過了音频範圍，Q會增大。   ("queue" became the letter "Q")

Real-time performance is not achievable on a two core CPU.
  -> 實時性能在多核心CPU上無法實現。          (says multi-core, not two-core)

이 오디오 파이프라인은 시스템 오디오를 캡처해서... (Korean)
  -> 這串音波儀程是將系統音波采集...            (invented vocabulary: 音波儀程)
```

Loanwords leak through (`chunk`, `case`, `situation`, `pipeline`), simplified and traditional
characters mix, and occasional words are invented. It is a live preview, not a translation.

### Conclusion
Multilingual **transcription** works and is fast. Multilingual **translation** is limited by
Qwen2.5-0.5B, not by anything in our plumbing — and no amount of prompt tuning fixed it. Getting
trustworthy output needs Qwen2.5-1.5B (measured ~5 s per caption, still far better than candle).

## Round 14: capture quality, missed captions, and a local-execution audit

### Are the models local? Yes — verified, not assumed

| process | listening | external connections |
|---|---|---|
| `realtime-translator.exe` | `0.0.0.0:3000` (browser UI) + `127.0.0.1` to llama-server | none |
| `whisper-server.exe` | `127.0.0.1:9183` | none |
| `llama-server.exe` | `127.0.0.1:8183` | none |

`netstat -ano` filtered on all three PIDs shows loopback only. Model weights load from disk
(`C:\temp\whisper\models\ggml-tiny.bin`, `%APPDATA%\kalosm\cache\Qwen\...q4_k_m.gguf`).
`llama-server` is launched without `-ngl`, so it runs entirely on CPU; the whisper.cpp build has
no CUDA/OpenCL/SYCL backend compiled in.

### Bug 1: quiet sources produced no captions at all

The VAD used a fixed absolute threshold of 0.003 (about -50 dBFS). Anything quieter than that never
triggered segmentation, so no chunk, no transcript, no caption — the app looked frozen rather than
quiet. That is almost certainly the "sometimes it misses the translation" report.

The fix tracks a running noise floor (min-tracking: falls instantly, rises slowly) and thresholds
just above it, bounded to keep a silent room from going trigger-happy and a loud room from going
deaf. Measured on a clip scaled so that **0 %** of its frames exceed the old threshold:

| VAD | segments produced |
|---|---|
| fixed 0.003 | **0** |
| adaptive floor | transcribing normally |

### Bug 2: one bad generation deleted four captions

`run_translator_task` joined up to four queued transcripts into a single generation. That was a
candle-era optimisation (≈20 s of fixed overhead per call); with llama.cpp answering in ~0.5 s it
buys nothing and costs everything — if the merged call returned empty, `warn!` logged it and all
four captions were gone. Each item is now translated separately, and an empty result gets one retry
before being counted in the new `empty_translations` counter.

### Bug 3: the decimator was aliasing into Whisper's own band

Downsampling 48 kHz to 16 kHz used a 3-sample moving average, which barely attenuates the first
aliasing lobe. Tones above 8 kHz folded straight back into the 0-8 kHz band Whisper analyses:

| input tone | aliases to | boxcar | 16-tap FIR | suppression |
|---|---|---|---|---|
| 9 kHz | 7 kHz | 0.177 | 0.111 | 4 dB |
| 12 kHz | 4 kHz | 0.100 | 0.026 | **12 dB** |
| 20 kHz | 4 kHz | 0.073 | 0.014 | **15 dB** |

Passband is also flatter (7 kHz: 0.255 vs 0.222 against a 0.300 source), so the audio loses less
of its top end. Cost is ~256k MACs/s, unmeasurable next to the models.

### Bug 4: splitting long text made things worse before it made them better

Cutting at 80 characters by word count still echoed, and cutting mid-clause produced fragments the
model either repeats or invents an ending for:

| cut strategy | pieces | echoed |
|---|---|---|
| 77 chars, word-packed | 2 | 1 |
| 60 chars, on punctuation | 2 | **0** |
| 45 chars, word-packed | 3 | 1 |
| 30 chars, word-packed | 4 | 2 |

So `split_for_translation` cuts at sentence ends first, then clause punctuation, and only falls
back to word packing when there is no boundary available.

### Result

| language | segments | untranslated | transcript | translation |
|---|---|---|---|---|
| English | 8 | 0 | 0.97 s | 2.29 s |
| Japanese | 1 | 0 | 1.06 s | 2.16 s |
| Mandarin | 1 | 0 | 0.89 s | 1.33 s |
| Cantonese | 3 | 0 | 0.88 s | 1.74 s |
| Korean | 2 | 0 | 0.96 s | 2.08 s |

Untranslated captions went from 1-in-8 (English) to zero across every language tested.

## Round 15: translation quality review (2/10 reported) — root causes

A real meeting was run through the pipeline and scored 2/10. Reproducing it locally on a generated
English business meeting translated to Korean reproduced every complaint, which made them easy to
chase down. Five causes, four of them ours.

### Cause 1: the non-Chinese prompt contained a Spanish sentence

`system_prompt_for` had two branches. Chinese got the clean instruction; **everything else** got a
prompt whose only worked example was:

```
Input: The worker captures system audio.
Output: El worker captura el audio del sistema.
```

The live config had `target_language = "Korean"`, so every single request carried that line. A 0.5B
model given a Spanish example will happily produce Spanish-adjacent output, which is exactly the
"Indonesian / Portuguese / German" the reviewer saw. One line of dead prompt was responsible for
most of the language mixing.

Now: no examples in any branch, plus an explicit *never mix two languages* clause.

### Cause 2: "다 다 다" was greedy decoding with no penalty

`temperature: 0` and llama.cpp's default `repeat_penalty = 1.0` is the textbook recipe for a
repetition collapse. Added `repeat_penalty 1.12` / `frequency_penalty 0.25`.

### Cause 3: nothing checked what language came back

Wrong-language output used to flow straight into the transcript. Added `language_ok`, which measures
what fraction of alphabetic characters belong to the target writing system, and retries once with a
stricter prompt when it falls below 95%. The retry keeps whichever answer has the better script
ratio — length is deliberately not a tiebreaker, because a longer answer in the wrong language is
still wrong.

### Cause 4: `[Blank Audio]` was being delivered as a caption

Whisper emits bracketed markers for gaps. They survived every filter and got translated like speech.
`strip_markers` now drops them, so they no longer consume a caption slot.

### Cause 5: the model was too small, not badly prompted

Same English source, same prompt, translated to Korean:

| model | result |
|---|---|
| Qwen2.5-0.5B | answers several sentences **in Chinese**, outright |
| Qwen2.5-1.5B | correct Korean, occasional stray character |

No prompt fixes a 0.5B model for this task. Model discovery now prefers the largest Qwen instruct
GGUF it can find, so the upgrade happens automatically once the file is present.

### Terminology: a glossary, filtered to terms actually present

Small models invent literal equivalents for business jargon. Reviewer examples and the fix:

| term | before | after |
|---|---|---|
| middleware | 중개 서비스 / sometimes untranslated | 미들웨어 |
| data lake | 데이터 라KE (garbled) | 데이터 레이크 |
| cost center | 비용 중심 (literal) | 비용 센터 |
| budgeted positions | 다음 계란에 대한 예산 ("next egg's budget") | 편성 정원 |
| cutoff date | 마지막 날 | 마감일 |

Glossary entries are listed in the prompt **only when the term appears in the text being
translated**. Listing `middleware` unconditionally made the model insert it into a sentence that
never mentioned middleware.

### Measured, English meeting to Korean

| stage | fully Korean | mixed | empty |
|---|---|---|---|
| 0.5B, original prompt | 7/15 | 6 | 2 |
| 1.5B, all fixes | **10/15** | 3 | 2 |

Sample of the current output:
```
Our middleware connects the data lake to the reporting layer, and that part is stable.
  -> 우리의 미들웨어는 데이터 레이크와 보고 단계를 연결합니다. 그 부분은 안정적입니다.

The cutoff date is the fifth of every month.
  -> 마감일은 매월 5일입니다. 그 이후에 발생하는 모든 것이 다음 회차로 이동합니다.

Data security concerns exist, because salary must not be exposed.
  -> 데이터 보안에 대한 우려가 있습니다. 이유는 급여 때문입니다. 정보는 노출되지 않아야 합니다.
```

### What is still wrong, honestly

- **Transcription is the new ceiling.** "headcount" comes out of Whisper-tiny as "headcom", so no
  glossary can fire. Fixing this means `stt.model = ggml-base.bin` (~4x slower, still faster than
  real time).
- **One or two stray CJK characters per paragraph** survive into the Korean output. The retry helps
  but does not eliminate them at 1.5B.
- **Latency is now ~6.6 s per caption** on 4 threads, versus ~1 s for the 0.5B model. Correct
  output costs about 2x real time on this CPU. 8 threads was measured *worse* (16.3 s) because
  Whisper and llama then oversubscribe the 12 cores.
- **Speaker identification does not exist in this pipeline at all.** The 3/10 score is generous;
  there is no diarisation component, so attribution is only as good as utterance ordering.

## Still open

- **Engine swap is still required for real time.** candle costs ~21 s prefill + 2 tok/s decode per
  caption; even perfectly tuned that is ~15–25 s per caption. llama.cpp (CPU or Vulkan on the Iris
  Xe) is the next candidate; blocked on `cmake` not being installed.
- **Capture can silently go dead.** All three WASAPI loopback endpoints reported 0 frames while
  `Audiosrv`/`AudioEndpointBuilder` were running and `/api/status` said `stage: ready`. cpal's error
  callback never fired, so `capture_error` stayed false and the app had no idea it was listening to
  nothing. Needs a heartbeat/watchdog on the frame counter, not just the device-loss flag.


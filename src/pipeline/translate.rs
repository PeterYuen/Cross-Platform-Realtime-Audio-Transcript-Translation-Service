use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use kalosm::language::*;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::config::{GlossaryTerm, TranslationConfig};
use crate::pipeline::llama_server::LlamaServer;

/// How many translated turns we reuse one chat session for before rebuilding it. The session's
/// KV cache keeps the system prompt prefilled, which is what makes turn 2+ much cheaper than
/// turn 1 — but the context window must not be allowed to fill up.
const SESSION_RECYCLE_TURNS: usize = 8;

/// Backend actually in use. Kept explicit because the two differ by a factor of ~30 in latency.
pub struct TranslateEngine {
    backend: Backend,
}

enum Backend {
    /// llama.cpp child process: 0.4-1.1 s per caption.
    Server {
        server: LlamaServer,
        language: Mutex<String>,
        glossary: Vec<GlossaryTerm>,
    },
    /// candle Phi-3: 15-35 s per caption, and it starves Whisper while running.
    Candle {
        model: Llama,
        chat: Mutex<Option<Chat<Llama>>>,
        language: Mutex<String>,
        turns: AtomicUsize,
    },
}

impl TranslateEngine {
    /// Prefer llama.cpp; fall back to candle only if it genuinely cannot run, and say so loudly.
    pub async fn load(cfg: &TranslationConfig) -> Result<Self> {
        if cfg.engine != "candle" {
            match LlamaServer::start(
                Some(&cfg.llama_server),
                Some(&cfg.model),
                cfg.port,
                cfg.threads,
            )
            .await
            {
                Ok(server) => {
                    info!("translation backend: llama.cpp ({} threads)", cfg.threads);
                    return Ok(Self {
                        backend: Backend::Server {
                            server,
                            language: Mutex::new(String::new()),
                            glossary: cfg.glossary.clone(),
                        },
                    });
                }
                Err(e) => {
                    if cfg.engine == "llama" {
                        return Err(anyhow!("llama.cpp translation requested but unavailable: {}", e));
                    }
                    warn!("translation backend: llama.cpp unavailable ({}), falling back to candle", e);
                }
            }
        }

        info!("translation backend: candle (loading Phi-3 mini, this takes a while)...");
        let model = Llama::phi_3()
            .await
            .map_err(|e| anyhow!("Failed to load LLM: {}", e))?;
        info!("Translation model loaded successfully");
        Ok(Self {
            backend: Backend::Candle {
                model,
                chat: Mutex::new(None),
                language: Mutex::new(String::new()),
                turns: AtomicUsize::new(0),
            },
        })
    }

    fn candle_new_chat(model: &Llama, target_language: &str) -> Chat<Llama> {
        // The instruction MUST go in the system prompt: Phi-3.5 returns a bare " " and stops
        // immediately when the instruction is packed into a single user message.
        model.chat().with_system_prompt(format!(
            "You are a professional translator. Translate the user's text into {}. \
             Output ONLY the translated text. Never add explanations, notes, comments or emojis.",
            target_language
        ))
    }

    /// Translate text to target language, streaming tokens through the channel.
    /// Returns the full translated string.
    pub async fn translate_stream(
        &self,
        text: &str,
        target_language: &str,
        token_tx: mpsc::UnboundedSender<String>,
    ) -> Result<String> {
        if text.trim().is_empty() {
            return Ok(String::new());
        }
        match &self.backend {
            Backend::Server { server, language, glossary } => {
                let started = std::time::Instant::now();
                // Scoped so the guard is gone before the await: a parking_lot guard is not Send
                // and holding one across an await point makes the whole task non-Send.
                {
                    let mut lang = language.lock();
                    if lang.as_str() != target_language {
                        *lang = target_language.to_string();
                    }
                }

                let src_chars = text.chars().count();
                // Roughly one token per CJK character, with headroom for longer target scripts.
                let max_tokens = ((src_chars * 2) + 24).clamp(24, 256);
                // Stream tokens straight to the UI: the first characters should appear within a
                // few hundred ms of prefill rather than after the whole sentence has been decoded.
                // Long inputs make this model echo the source instead of translating it (measured:
                // every failure case was over ~100 characters). Sentence-sized pieces translate
                // correctly, so feed it in slices and stitch the output back together.
                let pieces = split_for_translation(text);
                let mut raw = String::new();
                for piece in &pieces {
                    // Tokens are forwarded to the browser as they arrive, which is what makes the
                    // caption appear fast — and it also meant every language and commentary filter
                    // ran *after* the wrong text was already on screen. So hold the first few
                    // characters back and only open the tap once they look right.
                    let mut buf = String::new();
                    let mut released = false;
                    server
                        .chat_stream(
                            &system_prompt_for(target_language, glossary, piece),
                            piece,
                            max_tokens,
                            |tok| {
                                buf.push_str(tok);
                                if released {
                                    // Keep filtering: stray source-language characters arrive
                                    // mid-sentence, not just at the start.
                                    if clean_token(target_language, tok) {
                                        let _ = token_tx.send(tok.to_string());
                                    }
                                    return;
                                }
                                if buf.chars().count() >= GATE_CHARS && passes_gate(target_language, &buf) {
                                    released = true;
                                    let _ = token_tx.send(buf.clone());
                                }
                            },
                        )
                        .await?;
                    if !raw.is_empty() {
                        raw.push(' ');
                    }
                    raw.push_str(&buf);
                }
                let mut result = drop_stray_foreign(target_language, &strip_commentary(&raw));
                // A wrong-language answer is worse than no answer: it silently poisons the whole
                // transcript. One retry with stricter wording recovers most of these.
                if !result.trim().is_empty()
                    && !language_ok(target_language, &result)
                    && foreign_ratio(target_language, &result) > 0.3
                {
                    warn!(
                        "translation came back in the wrong script (wanted {}), retrying",
                        target_language
                    );
                    let mut retry = String::new();
                    for piece in pieces.iter() {
                        let mut streamed = String::new();
                        server
                            .chat_stream(
                                &strict_prompt_for(target_language),
                                piece,
                                max_tokens,
                                |tok| streamed.push_str(tok),
                            )
                            .await?;
                        if !retry.is_empty() {
                            retry.push(' ');
                        }
                        retry.push_str(&streamed);
                    }
                    let retried = strip_commentary(&retry);
                    // Keep whichever answer is more faithful to the target script. Length is not
                    // a tiebreaker — a longer answer in the wrong language is still wrong.
                    if language_ok(target_language, &retried)
                        || script_ratio(target_language, &retried)
                            > script_ratio(target_language, &result)
                    {
                        result = retried;
                    }
                }
                info!(
                    "Translation complete: {} chars in {:.1}s",
                    result.chars().count(),
                    started.elapsed().as_secs_f32()
                );
                Ok(result)
            }
            Backend::Candle {
                model,
                chat: chat_mutex,
                language,
                turns,
            } => {
                let needs_new = {
                    let mut cur = language.lock();
                    if cur.as_str() != target_language {
                        *cur = target_language.to_string();
                        true
                    } else {
                        false
                    }
                };

                let mut chat = chat_mutex.lock().take();
                let recycle = turns.load(Ordering::Relaxed) >= SESSION_RECYCLE_TURNS;
                if chat.is_none() || needs_new || recycle {
                    chat = Some(Self::candle_new_chat(model, target_language));
                    turns.store(0, Ordering::Relaxed);
                }
                let mut chat = chat.expect("chat just set");

                // Use the chat API so the model applies its template and stops at end-of-turn.
                // Raw `complete()` has no stop condition and can generate forever.
                let mut stream = chat.add_message(text).with_sampler(
                    GenerationParameters::new().with_stop_on(Some("\n\n".to_string())),
                );

                let mut full_response = String::new();
                // Generation is the whole cost of a translation (~0.5 s/token), and Phi-3.5 happily
                // expands a 10-word sentence into 500+ characters of notes and pinyin.
                let src_chars = text.chars().count();
                let max_chars = ((src_chars * 3) + 24).clamp(24, 320);

                while let Some(token) = stream.next().await {
                    full_response.push_str(&token);
                    let _ = token_tx.send(token);
                    if full_response.chars().count() >= max_chars {
                        break;
                    }
                }

                turns.fetch_add(1, Ordering::Relaxed);
                *chat_mutex.lock() = Some(chat);

                let result = strip_commentary(&full_response);
                info!("Translation complete: {} chars", result.chars().count());
                Ok(result)
            }
        }
    }
}

/// llama.cpp takes the whole instruction as one system message; keep the shape that was measured
/// to produce correct translations on the small model.
/// Build the translator instruction for a given target language.
///
/// Carries NO worked examples. Two separate measurements drove this:
/// - for Chinese, any example made the 0.5B model echo the source instead of translating;
/// - the previous non-Chinese branch embedded a *Spanish* example sentence, so translating to
///   Korean or German produced output sprinkled with Spanish/Portuguese-looking fragments. That
///   single line is the main reason reviewers saw six languages in one transcript.
///
/// Terms that apply to this target language, formatted for the prompt.
fn glossary_lines(target_language: &str, glossary: &[GlossaryTerm], text: &str) -> String {
    let lowered = text.to_lowercase();
    // Only list terms that actually appear. A glossary the model never needs is a glossary it
    // will happily inject on its own — listing "middleware" made it appear in a sentence that
    // never mentioned middleware.
    let hits: Vec<&GlossaryTerm> = glossary
        .iter()
        .filter(|g| g.language.eq_ignore_ascii_case(target_language))
        .filter(|g| lowered.contains(&g.source.to_lowercase()))
        .collect();
    if hits.is_empty() {
        return String::new();
    }
    let mut out = String::from("
Use these terms exactly as given:
");
    for g in hits {
        out.push_str(&format!("{} -> {}
", g.source, g.target));
    }
    out
}

/// How many characters to hold back before deciding whether to stream them to the browser.
/// Small enough that the first characters still land in a couple of hundred milliseconds.
const GATE_CHARS: usize = 6;

/// True when a character belongs to some *other* language's writing system. Used to spot output
/// that drifted into the source language, or into Japanese/Chinese when Korean was requested.
fn is_foreign_script(c: char, target_language: &str) -> bool {
    // Only letters count, and Latin is treated as neutral: Korean keeps "FTE" and "ID", Chinese
    // keeps product names, and flagging those as foreign made good captions look broken.
    if !c.is_alphabetic() || c < '\u{2b0}' {
        return false;
    }
    !is_in_script(c, script_of(target_language))
}

/// Drop short runs of foreign-script characters. The small model sprinkles isolated 1-4 character
/// fragments of the source language into an otherwise correct translation ("보내您可以"); they are
/// noise, not untranslated content, and a whole untranslated sentence would be far longer and is
/// already handled by the retry below.
fn drop_stray_foreign(target_language: &str, text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if is_foreign_script(chars[i], target_language) {
            let start = i;
            while i < chars.len() && is_foreign_script(chars[i], target_language) {
                i += 1;
            }
            if i - start > 4 {
                out.extend(&chars[start..i]);
            }
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    // Dropping characters leaves double spaces behind.
    let mut cleaned = String::new();
    let mut last_space = false;
    for c in out.chars() {
        let is_ws = c == ' ';
        if is_ws && last_space {
            continue;
        }
        cleaned.push(c);
        last_space = is_ws;
    }
    cleaned.trim().to_string()
}

/// A single streamed token is safe to show when it carries no characters from another script.
fn clean_token(target_language: &str, token: &str) -> bool {
    !token.chars().any(|c| is_foreign_script(c, target_language))
}

/// Whether enough of the answer is in the target script to start showing it. Half rather than
/// ninety-five: Korean legitimately keeps "FTE" or "ID" in Latin letters, and blocking those made
/// short captions never stream at all.
fn passes_gate(target_language: &str, text: &str) -> bool {
    if text.chars().any(|c| is_foreign_script(c, target_language)) {
        return false;
    }
    if META_MARKERS.iter().any(|m| text.to_lowercase().contains(m)) {
        return false;
    }
    let letters = text.chars().filter(|c| c.is_alphabetic()).count();
    if letters == 0 {
        return true;
    }
    script_ratio(target_language, text) >= 0.5
}

fn system_prompt_for(target_language: &str, glossary: &[GlossaryTerm], text: &str) -> String {
    format!(
        "You are a professional translator. Translate the text into {lang}.          Write every word of your answer in {lang} only. Never leave any part of it in the          source language, and never mix two languages in one answer.          Output nothing except the translation.{terms}",
        lang = target_language,
        terms = glossary_lines(target_language, glossary, text)
    )
}

/// Stronger wording used when the first attempt came back in the wrong language.
fn strict_prompt_for(target_language: &str) -> String {
    format!(
        "Translate the following text into {lang}. The answer must be 100% {lang} characters          and words. Any word written in another language is a failure. Text to translate:",
        lang = target_language
    )
}

/// Which writing system the target language uses, so output can be checked against it.
fn script_of(target_language: &str) -> &'static str {
    let t = target_language.to_lowercase();
    if t.contains("korean") {
        "hangul"
    } else if t.contains("japanese") {
        "kana"
    } else if t.contains("chinese") || t.contains("zhong") {
        "han"
    } else if t.contains("arabic") {
        "arabic"
    } else if t.contains("russian") || t.contains("ukrain") {
        "cyrillic"
    } else if t.contains("thai") {
        "thai"
    } else {
        "latin"
    }
}

fn is_in_script(c: char, script: &str) -> bool {
    let n = c as u32;
    match script {
        "hangul" => (0xAC00..=0xD7A3).contains(&n) || (0x1100..=0x11FF).contains(&n) || (0x3130..=0x318F).contains(&n),
        "kana" => (0x3040..=0x30FF).contains(&n) || (0x4E00..=0x9FFF).contains(&n),
        "han" => (0x4E00..=0x9FFF).contains(&n),
        "arabic" => (0x0600..=0x06FF).contains(&n),
        "cyrillic" => (0x0400..=0x04FF).contains(&n),
        "thai" => (0x0E00..=0x0E7F).contains(&n),
        _ => (0x41..=0x5A).contains(&n)
            || (0x61..=0x7A).contains(&n)
            || (0xC0..=0x24F).contains(&n),
    }
}

/// Fraction of alphabetic characters belonging to the target writing system.
fn script_ratio(target_language: &str, text: &str) -> f64 {
    let script = script_of(target_language);
    let mut letters = 0usize;
    let mut in_script = 0usize;
    for c in text.chars() {
        if c.is_whitespace() || c.is_ascii_digit() || c.is_ascii_punctuation() {
            continue;
        }
        if c.is_alphabetic() || is_in_script(c, script) {
            letters += 1;
            if is_in_script(c, script) {
                in_script += 1;
            }
        }
    }
    if letters == 0 {
        0.0
    } else {
        in_script as f64 / letters as f64
    }
}

/// True when enough of the output actually uses the target writing system. This is what catches
/// "우리는만 데이터를... Across that, we are allowed" style code-switching.
fn language_ok(target_language: &str, text: &str) -> bool {
    // 0.7, not 0.95. Korean keeps "ID", "FTE" and "chunk" in Latin letters, and demanding 95%
    // Hangul labelled perfectly good captions as wrong-language - which made every other caption
    // get translated twice and doubled end-to-end latency.
    script_ratio(target_language, text) >= 0.7
}

/// Fraction of letters that belong to some script *other* than the target's. A caption that is
/// mostly another language scores high here, which is what distinguishes a real failure from an
/// ordinary loanword.
fn foreign_ratio(target_language: &str, text: &str) -> f64 {
    let letters = text.chars().filter(|c| c.is_alphabetic()).count();
    if letters == 0 {
        return 0.0;
    }
    text.chars()
        .filter(|c| c.is_alphabetic() && is_foreign_script(*c, target_language))
        .count() as f64
        / letters as f64
}

/// Phi-3.5 likes to append commentary ("(Note: ...)", pinyin glosses, self-congratulation).
/// None of that is part of the translation, and showing it in the caption bar is worse than
/// showing nothing, so cut everything from the first such marker onward.
/// Break long transcripts into pieces the small model can actually handle.
///
/// Above roughly 100 characters it stops translating and repeats the input back. Cutting the text
/// works, but only at real boundaries: splitting mid-clause produces fragments it echoes or
/// invents an ending for. So cut at sentence ends first, then at clause punctuation, and only
/// fall back to packing words when nothing is available.
fn split_for_translation(text: &str) -> Vec<String> {
    const LIMIT: usize = 60;
    let text = text.trim();
    if text.chars().count() <= LIMIT {
        return vec![text.to_string()];
    }

    // First pass: hard sentence boundaries.
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?' | '\u{3002}' | '\u{ff01}' | '\u{ff1f}') {
            sentences.push(std::mem::take(&mut current));
        }
    }
    if !current.trim().is_empty() {
        sentences.push(current);
    }

    // Second pass: sentences too long on their own get cut at clause punctuation.
    let mut units: Vec<String> = Vec::new();
    for sentence in sentences {
        if sentence.chars().count() <= LIMIT {
            units.push(sentence);
            continue;
        }
        let mut buf = String::new();
        for ch in sentence.chars() {
            buf.push(ch);
            if matches!(ch, ',' | ';' | ':' | '\u{ff0c}' | '\u{3001}' | '\u{ff1b}')
                && buf.chars().count() >= LIMIT / 2
            {
                units.push(std::mem::take(&mut buf));
            }
        }
        if !buf.trim().is_empty() {
            units.push(buf);
        }
    }

    let mut pieces: Vec<String> = Vec::new();
    let mut acc = String::new();
    for unit in units {
        if acc.chars().count() + unit.chars().count() > LIMIT && !acc.is_empty() {
            pieces.push(std::mem::take(&mut acc));
        }
        acc.push_str(&unit);
    }
    if !acc.trim().is_empty() {
        pieces.push(acc);
    }

    // Last resort: any piece still over the limit gets packed by words.
    let mut result: Vec<String> = Vec::new();
    for piece in pieces {
        if piece.chars().count() <= LIMIT {
            result.push(piece.trim().to_string());
            continue;
        }
        let mut line = String::new();
        for word in piece.split_whitespace() {
            if line.chars().count() + word.chars().count() > LIMIT && !line.is_empty() {
                result.push(std::mem::take(&mut line));
            }
            if line.is_empty() {
                line = word.to_string();
            } else {
                line.push(' ');
                line.push_str(word);
            }
        }
        if !line.trim().is_empty() {
            result.push(line);
        }
    }
    result.into_iter().filter(|p| !p.trim().is_empty()).collect()
}

/// Phrases that mean "the model is talking about the job, not doing it". Kept deliberately
/// specific: a common word like "provided" would delete real translated content.
const META_MARKERS: &[&str] = &[
    "원본 문장",
    "모든 단어를",
    "번역하지",
    "다른 언어",
    "keep all words",
    "original sentence",
    "do not translate",
    "as an ai",
    "in summary",
    // Safety / process prose. Small models produce this when they think they are reproducing
    // something verbatim, and it buries the actual translation.
    "copyright",
    "copyright holder",
    "for permission",
    "intended as",
    "reproduce it",
    "literal translation",
    "literal english translation",
    "translation based on",
    "based on the original",
    "here's my translation",
    "here is my translation",
    "my translation into",
    "translation into english",
    "translation into korean",
    "translation into chinese",
    "translation into japanese",
    "the original text",
    "provided text",
    "as requested",
    "please note that",
];

/// Lead-ins that end in a colon, e.g. "Here's my translation into English:". Everything before
/// them is commentary; the translation, if any, starts after the colon.
fn split_after_lead_in(text: &str) -> String {
    let lower = text.to_lowercase();
    let marker = lower.rfind("translation into ");
    if let Some(i) = marker {
        if let Some(colon) = lower[i..].find(':') {
            let cut = i + colon + 1;
            let tail = text[cut..].trim();
            if !tail.is_empty() {
                return tail.to_string();
            }
        }
    }
    String::new()
}

/// Drop sentences that are the model talking about the task rather than doing it.
fn drop_meta_sentences(text: &str) -> String {
    let mut kept = String::new();
    for sentence in text.split_inclusive(['.', '!', '?', '\u{3002}', '\u{ff1f}']) {
        let lower = sentence.to_lowercase();
        if META_MARKERS.iter().any(|m| lower.contains(m)) {
            continue;
        }
        kept.push_str(sentence);
    }
    let kept = kept.trim().to_string();
    if kept.is_empty() {
        return kept;
    }
    // A trailing "…:" with nothing after it means the disclaimer ate the whole budget.
    if kept.ends_with(':') {
        return String::new();
    }
    kept
}

fn strip_commentary(s: &str) -> String {
    let markers = [
        "(Note", "Note:", "note that", "(Pinyin", "Pinyin", "(The translation", "(This translation",
        "(Output", "In Chinese", "in Chinese", "(Traditional",
    ];
    let mut cut = s.len();
    for m in markers {
        if let Some(i) = s.find(m) {
            cut = cut.min(i);
        }
    }
    let cleaned = drop_meta_sentences(s[..cut].trim());
    if cleaned.is_empty() {
        return split_after_lead_in(&s[..cut]);
    }
    cleaned
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_copyright_prose_and_keeps_translation() {
        let dirty = "This is a literal English translation based on the original Korean text provided. \
                      If this was intended as copyrighted material, please contact its copyright holder \
                      for permission to reproduce it in full or part. We can send you two reports.";
        assert_eq!(strip_commentary(dirty), "We can send you two reports.");
    }

    #[test]
    fn recovers_translation_after_a_colon_lead_in() {
        let dirty = "Here's my translation into English: The pipeline captures system audio.";
        assert_eq!(strip_commentary(dirty), "The pipeline captures system audio.");
    }

    #[test]
    fn treats_disclaimer_that_ate_the_budget_as_empty() {
        let dirty = "Here's my translation into English:";
        assert!(strip_commentary(dirty).is_empty());
    }

    #[test]
    fn leaves_normal_text_alone() {
        let clean = "The middleware connects the data lake to the reporting layer.";
        assert_eq!(strip_commentary(clean), clean);
    }
}

#[cfg(test)]
mod stray_tests {
    use super::*;

    #[test]
    fn removes_isolated_source_fragments_from_korean_output() {
        let dirty = "우리는 두 개의 보고서를 보내您可以 가져다 사용하도록 합니다.";
        let clean = strip_commentary(&drop_stray_foreign("Korean", dirty));
        assert!(!clean.contains('您'));
        assert!(clean.contains("보내"));
        assert!(clean.contains("가져다"));
    }

    #[test]
    fn keeps_legit_latin_terms_in_korean_output() {
        let ok = "총 시간 동일성을 위해 우리는 FTE을 따로 보고합니다.";
        assert_eq!(drop_stray_foreign("Korean", ok), ok);
    }
}

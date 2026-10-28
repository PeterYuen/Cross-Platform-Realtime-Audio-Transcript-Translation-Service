use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use kalosm::language::*;
use tokio::sync::mpsc;
use tracing::info;

/// Translation engine using a small LLM via kalosm
pub struct TranslateEngine {
    model: Llama,
}

impl TranslateEngine {
    /// Load Phi-3-mini (small, good multilingual translator). Runs on the decoupled translator
    /// task so its latency does not block capture/transcription.
    pub async fn load() -> Result<Self> {
        info!("Loading LLM model (Phi-3 mini)...");
        let model = Llama::phi_3()
            .await
            .map_err(|e| anyhow!("Failed to load LLM: {}", e))?;
        info!("Translation model loaded successfully");
        Ok(Self { model })
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

        // Use the chat API so the model applies its template and stops at end-of-turn.
        // Raw `complete()` has no stop condition and can generate forever.
        let mut chat = self.model.chat();
        let request = format!(
            "Translate the following text to {}. Output ONLY the translation, nothing else.\n\n{}",
            target_language, text
        );
        let mut stream = chat.add_message(request);

        let mut full_response = String::new();
        // Hard cap generation length so a verbose model cannot stall the translator task.
        const MAX_CHARS: usize = 512;

        while let Some(token) = stream.next().await {
            full_response.push_str(&token);
            let _ = token_tx.send(token);
            if full_response.chars().count() >= MAX_CHARS {
                break;
            }
        }

        let result = full_response.trim().to_string();
        info!("Translation complete: {} chars", result.len());
        Ok(result)
    }
}

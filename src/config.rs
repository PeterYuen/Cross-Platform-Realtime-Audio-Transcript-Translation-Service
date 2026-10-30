use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub server_port: u16,
    pub audio_device_name: Option<String>,
    pub audio_device_is_loopback: bool,
    pub target_language: String,
    pub whisper_model: String,
    pub whisper_model_path: Option<PathBuf>,
    pub llm_model_path: Option<PathBuf>,
    pub model_cache_dir: PathBuf,
    pub vad_threshold: f32,
    pub vad_hangover_frames: usize,
    pub max_chunk_seconds: f32,
    pub translation: TranslationConfig,
    pub stt: SttConfig,
}

/// Transcription backend settings. `auto` prefers whisper.cpp (sub-second per chunk) and falls
/// back to candle if whisper-server or a ggml model cannot be found.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SttConfig {
    /// "auto" | "whisper" | "candle"
    pub engine: String,
    /// Path to whisper-server.exe. Empty means search PATH and the usual build output.
    pub server: String,
    /// Path to a ggml model. Empty means search the whisper.cpp model directory.
    pub model: String,
    /// Source language passed to whisper ("auto" = detect). Passing an explicit language skips
    /// whisper's detection pass, which costs a whole extra encoder run on every short chunk.
    pub language: String,
    pub port: u16,
    pub threads: usize,
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            engine: "auto".to_string(),
            server: String::new(),
            model: String::new(),
            language: "auto".to_string(),
            // Distinct from the translator's port so both servers can run side by side.
            port: 9183,
            // 6 measured best: more threads add synchronisation overhead without helping tiny.
            threads: 6,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GlossaryTerm {
    /// Target language this mapping applies to.
    pub language: String,
    /// Term as it appears in the source audio.
    pub source: String,
    /// The fixed translation to use, regardless of what the model would invent.
    pub target: String,
}

/// Translation backend settings. `auto` prefers llama.cpp (sub-second captions) and falls back to
/// candle if llama-server or a GGUF model cannot be found.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TranslationConfig {
    /// "auto" | "llama" | "candle"
    pub engine: String,
    /// Path to llama-server.exe. Empty means search PATH and the usual build output locations.
    pub llama_server: String,
    /// Path to the .gguf model. Empty means search the model cache for a Qwen instruct model.
    pub model: String,
    pub port: u16,
    pub threads: usize,
    /// Fixed terminology. Small models invent literal equivalents for business jargon
    /// ("budgeted positions" -> "next egg's budget"); a glossary pins them down.
    #[serde(default)]
    pub glossary: Vec<GlossaryTerm>,
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            engine: "auto".to_string(),
            llama_server: String::new(),
            model: String::new(),
            port: 8183,
            // 4 measured better than 6 under contention; the model is small enough that extra
            // threads only add sync overhead and steal cores from Whisper.
            threads: 4,
            glossary: default_glossary(),
        }
    }
}

/// Terminology that showed up wrong in review. Add more entries in config.toml to extend.
fn default_glossary() -> Vec<GlossaryTerm> {
    [
        ("Korean", "middleware", "미들웨어"),
        ("Korean", "data lake", "데이터 레이크"),
        ("Korean", "cost center", "비용 센터"),
        ("Korean", "reference ID", "참조 ID"),
        ("Korean", "budgeted position", "편성 정원"),
        ("Korean", "cutoff date", "마감일"),
        ("Korean", "headcount", "인원수"),
        ("Korean", "full time equivalent", "FTE"),
    ]
    .iter()
    .map(|(language, source, target)| GlossaryTerm {
        language: (*language).to_string(),
        source: (*source).to_string(),
        target: (*target).to_string(),
    })
    .collect()
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server_port: 3000,
            audio_device_name: None,
            audio_device_is_loopback: true,
            target_language: "English".to_string(),
            whisper_model: "base".to_string(),
            whisper_model_path: None,
            llm_model_path: None,
            model_cache_dir: dirs_cache(),
            vad_threshold: 0.003,
            vad_hangover_frames: 15,
            // Whisper pays a large fixed cost per call, so short chunks are much slower AND less
            // accurate (measured RTF 1.3x for one 16s call vs 2.6x at 5s chunks, 8.1x at 2s).
            max_chunk_seconds: 10.0,
            translation: TranslationConfig::default(),
            stt: SttConfig::default(),
        }
    }
}

fn dirs_cache() -> PathBuf {
    let mut path = dirs_next::cache_dir().unwrap_or_else(|| std::env::current_dir().unwrap());
    path.push("realtime-translator");
    path.push("models");
    path
}

impl Config {
    pub fn load() -> Self {
        let config_path = Self::config_path();
        if let Some(path) = config_path {
            if let Ok(content) = std::fs::read_to_string(&path) {
                if let Ok(config) = toml::from_str(&content) {
                    return config;
                }
            }
        }
        Self::default()
    }

    pub fn save(&self) {
        if let Some(path) = Self::config_path() {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Ok(content) = toml::to_string_pretty(self) {
                let _ = std::fs::write(&path, content);
            }
        }
    }

    fn config_path() -> Option<PathBuf> {
        let mut path = dirs_next::config_dir()?;
        path.push("realtime-translator");
        path.push("config.toml");
        Some(path)
    }
}

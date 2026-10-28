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
            max_chunk_seconds: 5.0,
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

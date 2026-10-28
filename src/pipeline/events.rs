use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PipelineEvent {
    Transcript {
        text: String,
    },
    TranslationToken {
        token: String,
    },
    TranslationComplete {
        text: String,
    },
    Status {
        running: bool,
    },
    AudioLevel {
        rms: f32,
    },
    Error {
        message: String,
    },
}

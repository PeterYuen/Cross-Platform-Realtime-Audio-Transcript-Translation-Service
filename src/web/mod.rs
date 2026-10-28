pub mod ws;

use axum::{
    extract::State,
    http::StatusCode,
    response::Html,
    routing::{get, post},
    Json, Router,
};
use std::sync::Arc;
use tower_http::services::ServeDir;

use crate::audio;
use crate::state::AppState;

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(index_handler))
        .route("/api/devices", get(list_devices))
        .route("/api/status", get(get_status))
        .route("/api/start", post(start_pipeline))
        .route("/api/stop", post(stop_pipeline))
        .route("/api/config", get(get_config).post(set_config))
        .route("/api/debug/audio", get(debug_audio))
        .route("/ws", get(ws::ws_handler))
        .nest_service("/static", ServeDir::new("static"))
        .with_state(state)
}

async fn index_handler() -> Html<&'static str> {
    Html(include_str!("../../static/index.html"))
}

async fn list_devices() -> Json<Vec<audio::AudioDeviceInfo>> {
    Json(audio::list_capture_devices())
}

async fn get_status(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let running = state.is_running();
    let ready = state.is_ready();
    let stage = if !running {
        "stopped"
    } else if ready {
        "ready"
    } else {
        "loading_models"
    };
    Json(serde_json::json!({
        "running": running,
        "ready": ready,
        "stage": stage,
        "device_error": state.device_error(),
        "target_language": state.config.lock().target_language,
    }))
}

async fn debug_audio(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(state.audio_stats.snapshot())
}

async fn start_pipeline(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    match state.start().await {
        Ok(_) => Json(serde_json::json!({ "ok": true })),
        Err(e) => {
            tracing::error!("Failed to start pipeline: {}", e);
            Json(serde_json::json!({ "ok": false, "error": e.to_string() }))
        }
    }
}

async fn stop_pipeline(State(state): State<Arc<AppState>>) -> StatusCode {
    state.stop();
    StatusCode::OK
}

async fn get_config(State(state): State<Arc<AppState>>) -> Json<crate::config::Config> {
    Json(state.config.lock().clone())
}

async fn set_config(
    State(state): State<Arc<AppState>>,
    Json(new_config): Json<crate::config::Config>,
) -> StatusCode {
    *state.config.lock() = new_config.clone();
    new_config.save();
    StatusCode::OK
}

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
    Json(new_config): Json<serde_json::Value>,
) -> StatusCode {
    // Merge, don't replace. The browser only sends the fields it shows, and a stale tab (or a
    // partial client) used to reset everything else to its own defaults — that is how
    // max_chunk_seconds kept snapping back to 5.0 behind our backs.
    let mut merged = serde_json::to_value(&*state.config.lock())
        .unwrap_or_else(|_| serde_json::json!({}));
    match (merged.as_object_mut(), new_config.as_object()) {
        (Some(base), Some(incoming)) => {
            for (k, v) in incoming {
                base.insert(k.clone(), v.clone());
            }
        }
        // Not an object: fall back to treating it as a whole config.
        _ => match serde_json::from_value::<crate::config::Config>(new_config) {
            Ok(cfg) => {
                cfg.save();
                *state.config.lock() = cfg;
                return StatusCode::OK;
            }
            Err(_) => return StatusCode::BAD_REQUEST,
        },
    }
    match serde_json::from_value::<crate::config::Config>(merged) {
        Ok(cfg) => {
            cfg.save();
            *state.config.lock() = cfg;
            StatusCode::OK
        }
        Err(e) => {
            tracing::warn!("Rejected invalid config payload: {}", e);
            StatusCode::BAD_REQUEST
        }
    }
}

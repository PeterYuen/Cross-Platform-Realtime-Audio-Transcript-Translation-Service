use axum::{
    extract::ws::{WebSocket, WebSocketUpgrade},
    extract::State,
    response::IntoResponse,
};
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::info;

use crate::state::AppState;
use crate::pipeline::events::PipelineEvent;

pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
    let mut rx = state.event_tx.subscribe();

    // Send current state on connect
    let init_msg = serde_json::to_string(&PipelineEvent::Status { running: state.is_running() }).unwrap();
    if socket.send(axum::extract::ws::Message::Text(init_msg.into())).await.is_err() {
        return;
    }

    loop {
        tokio::select! {
            event = rx.recv() => {
                match event {
                    Ok(event) => {
                        let msg = serde_json::to_string(&event).unwrap();
                        if socket.send(axum::extract::ws::Message::Text(msg.into())).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(_) => break,
                }
            }
            msg = socket.recv() => {
                match msg {
                    Some(Ok(axum::extract::ws::Message::Close(_))) => break,
                    Some(Err(_)) => break,
                    Some(Ok(_)) => {} // ignore client messages for now
                    None => break,
                }
            }
        }
    }

    info!("WebSocket client disconnected");
}

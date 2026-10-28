use tracing_subscriber::{fmt, EnvFilter};

use realtime_translator::config::Config;
use realtime_translator::state::AppState;
use realtime_translator::web::build_router;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("info".parse().unwrap())
                // rwhisper emits a benign "receiver is gone" when we abort a stuck worker.
                .add_directive("rwhisper=warn".parse().unwrap()),
        )
        .init();

    tracing::info!("Realtime Audio Translator starting...");

    // Load config
    let config = Config::load();
    let port = config.server_port;

    // Pin model downloads to a local cache dir so inference runs fully offline after the
    // first fetch. Must be set before any model is loaded.
    std::fs::create_dir_all(&config.model_cache_dir).ok();
    std::env::set_var("HF_HOME", &config.model_cache_dir);

    // Cap candle's rayon thread pool so Phi-3 translation cannot saturate a small CPU and
    // starve the latency-critical Whisper transcription that runs alongside it. Respect an
    // explicit override if the user set one.
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        std::env::set_var("RAYON_NUM_THREADS", "2");
    }
    if std::env::var_os("OMP_NUM_THREADS").is_none() {
        std::env::set_var("OMP_NUM_THREADS", "2");
    }

    // Create state
    let state = AppState::new(config);

    // Build router
    let app = build_router(state);

    // Start server
    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("Server listening on http://localhost:{}", port);
    tracing::info!("Open your browser to start using the translator");

    axum::serve(listener, app).await?;
    Ok(())
}

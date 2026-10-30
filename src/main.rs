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

    // Candle runs its matmuls through rayon. The old hard cap of 2 threads was written for a
    // 2-core box; on this 12-thread machine it cost ~45 s of prefill per translation instead of
    // ~15 s (measured: 1 thread 45 s, 8 threads 21 s, 12 threads 15 s). Scale to the cores we
    // actually have, and still honour an explicit override.
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        let n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);
        std::env::set_var("RAYON_NUM_THREADS", n.to_string());
    }
    if std::env::var_os("OMP_NUM_THREADS").is_none() {
        let n = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "2".to_string());
        std::env::set_var("OMP_NUM_THREADS", n);
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

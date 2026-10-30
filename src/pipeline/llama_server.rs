//! llama.cpp translation backend.
//!
//! The candle/Phi-3.5 path works but is unusable for live captions: 0.9 tok/s means 15-35 s per
//! caption, and while it generates it saturates the core pool, which drags Whisper down from
//! 1.0x to 1.7x real time. llama.cpp running Qwen2.5-0.5B does the same job in 0.4-1.1 s, so the
//! translator stops stealing cycles from transcription.
//!
//! Rather than linking llama.cpp into this binary (bindgen + libclang, neither of which is
//! available on this MinGW toolchain), we run `llama-server.exe` as a child process and talk to it
//! over plain localhost HTTP. That keeps the build simple and keeps a crashing model in its own
//! process instead of taking the capture pipeline with it.

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tracing::{debug, info, warn};

/// Directories searched for `llama-server.exe`, in priority order.
pub(crate) fn server_search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    // LLAMA_CPP_HOME = a llama.cpp checkout or build tree.
    if let Ok(home) = std::env::var("LLAMA_CPP_HOME") {
        let base = PathBuf::from(home);
        dirs.push(base.join("build").join("bin"));
        dirs.push(base);
    }
    if let Some(dir) = crate::pipeline::whisper_server::exe_dir() {
        dirs.push(dir);
    }
    dirs.extend(std::env::split_paths(&std::env::var("PATH").unwrap_or_default()));
    dirs.retain(|d| !d.as_os_str().is_empty());
    dirs
}

/// Everything needed to launch llama-server again. Held so the translator can pull itself out of
/// a dead backend instead of failing every caption for the rest of the session.
struct SpawnSpec {
    bin: PathBuf,
    model: PathBuf,
    threads: usize,
    port: u16,
}

pub struct LlamaServer {
    /// `None` when we adopted a server we did not start; we must not kill what is not ours.
    /// Wrapped in a Mutex because `Child` is Send but not Sync, and the engine is shared by
    /// reference across await points.
    child: parking_lot::Mutex<Option<Child>>,
    spec: SpawnSpec,
    addr: SocketAddr,
}

/// True when something answers `/health` on this port twice, a beat apart.
async fn self_alive(addr: SocketAddr) -> bool {
    for probe in 0..2 {
        match LlamaServer::rpc(addr, "GET", "/health", None).await {
            Ok(v) if v.get("status").and_then(|s| s.as_str()) == Some("ok") => {}
            _ => return false,
        }
        if probe == 0 {
            tokio::time::sleep(std::time::Duration::from_millis(400)).await;
        }
    }
    true
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.lock().take() {
            let _ = child.kill();
        }
    }
}

impl LlamaServer {
    /// Start the server, preferring an already-running one on the port (useful while developing).
    pub async fn start(
        server_bin: Option<&str>,
        model: Option<&str>,
        port: u16,
        threads: usize,
    ) -> Result<Self> {
        let addr = SocketAddr::from(([127, 0, 0, 1], port));
        let bin = find_server(server_bin)?;
        let model_path = find_model(model)?;
        let spec = SpawnSpec {
            bin: bin.clone(),
            model: model_path.clone(),
            threads,
            port,
        };

        // Adopting a server we did not start is handy while developing, but it is also how a
        // restart silently dies: a stop() kills its child asynchronously, and if start() arrives
        // before the port is released it happily "adopts" the corpse, which then exits and leaves
        // every later caption failing. Confirm it answers twice before trusting it.
        if self_alive(addr).await {
            info!("translation: reusing llama-server already listening on {}", addr);
            return Ok(Self {
                child: parking_lot::Mutex::new(None),
                spec,
                addr,
            });
        }
        info!(
            "translation: starting llama-server ({} threads, model {})",
            threads,
            model_path.display()
        );

        let me = Self {
            child: parking_lot::Mutex::new(None),
            spec,
            addr,
        };
        me.spawn_child()?;
        me.wait_until_ready().await?;
        Ok(me)
    }

    fn spawn_child(&self) -> Result<()> {
        let mut cmd = Command::new(&self.spec.bin);
        cmd.arg("-m")
            .arg(&self.spec.model)
            .arg("-t")
            .arg(self.spec.threads.to_string())
            .arg("-c")
            .arg("2048")
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(self.spec.port.to_string())
            .arg("--no-webui")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());

        let child = cmd.spawn().map_err(|e| {
            anyhow!(
                "failed to launch {}: {} (is the MinGW runtime next to it?)",
                self.spec.bin.display(),
                e
            )
        })?;
        *self.child.lock() = Some(child);
        Ok(())
    }

    /// Bring the backend back if it died. Without this, one bad restart leaves the app reporting
    /// "ready" while every translation fails forever, and the only fix is to restart the whole app.
    pub async fn revive(&self) -> Result<()> {
        if self_alive(self.addr).await {
            return Ok(());
        }
        warn!("translation: llama-server is not responding, starting a new one");
        if let Some(mut old) = self.child.lock().take() {
            let _ = old.kill();
        }
        self.spawn_child()?;
        self.wait_until_ready().await
    }

    async fn wait_until_ready(&self) -> Result<()> {
        for _ in 0..120 {
            if let Ok(v) = Self::rpc(self.addr, "GET", "/health", None).await {
                if v.get("status").and_then(|s| s.as_str()) == Some("ok") {
                    info!("translation: llama-server ready on {}", self.addr);
                    return Ok(());
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
        Err(anyhow!("llama-server did not become ready within 60 s"))
    }

    /// One chat turn, streaming tokens out as they are produced.
    ///
    /// Worth doing because it is the only part of the pipeline that can approach a 100 ms budget:
    /// the first translated character lands after prefill (~50-100 ms for this model) instead of
    /// after the whole sentence has been generated.
/// Connect to the server, reviving it once if the port is dead. A backend that died after a
/// restart used to stay dead for the rest of the session; this makes the next caption fix it.
async fn connect(&self) -> Result<TcpStream> {
    match TcpStream::connect(self.addr).await {
        Ok(s) => Ok(s),
        Err(e) => {
            warn!("translation: llama-server unreachable ({}); reviving", e);
            self.revive().await?;
            TcpStream::connect(self.addr)
                .await
                .map_err(|e2| anyhow!("cannot reach llama-server on {}: {}", addr_str(self.addr), e2))
        }
    }
}

    pub async fn chat_stream(
        &self,
        system: &str,
        user: &str,
        max_tokens: usize,
        mut on_token: impl FnMut(&str),
    ) -> Result<String> {
        let body = json!({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            // Greedy decoding with no penalty is exactly how you get "다 다 다 다" loops, and that
            // is what the reviewer saw near the end of a long meeting. A mild repeat penalty
            // breaks the cycle without changing wording.
            "repeat_penalty": 1.12,
            "frequency_penalty": 0.25,
            "max_tokens": max_tokens,
            "stream": true,
        });
        let payload = body.to_string();
        let mut stream = self.connect().await?;
        let request = format!(
            "POST /chat/completions HTTP/1.1\r\nHost: {}\r\nUser-Agent: realtime-translator\r\nContent-Type: application/json\r\nAccept: text/event-stream\r\nContent-Length: {}\r\n\r\n{}",
            addr_str(self.addr),
            payload.len(),
            payload
        );
        stream
            .write_all(request.as_bytes())
            .await
            .context("writing streaming request")?;

        let mut reader = SseReader::new(stream);
        reader.read_head().await?;
        let mut full = String::new();
        while let Some(piece) = reader.next_body_chunk().await? {
            let text = String::from_utf8_lossy(&piece).to_string();
            for line in text.lines() {
                let Some(data) = line.strip_prefix("data: ") else {
                    continue;
                };
                if data.trim() == "[DONE]" {
                    return Ok(full);
                }
                let Ok(v) = serde_json::from_str::<Value>(data) else {
                    continue;
                };
                if let Some(delta) = v.pointer("/choices/0/delta/content").and_then(|c| c.as_str()) {
                    if !delta.is_empty() {
                        full.push_str(delta);
                        on_token(delta);
                    }
                }
            }
        }
        Ok(full)
    }

    /// One chat turn. Returns the assistant text.
    pub async fn chat(&self, system: &str, user: &str, max_tokens: usize) -> Result<String> {
        let body = json!({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "repeat_penalty": 1.12,
            "frequency_penalty": 0.25,
            "max_tokens": max_tokens,
            "stream": false,
        });
        let v = Self::rpc(self.addr, "POST", "/chat/completions", Some(body)).await?;
        let text = v
            .pointer("/choices/0/message/content")
            .and_then(|c| c.as_str())
            .ok_or_else(|| anyhow!("unexpected completion response: {}", v))?
            .to_string();
        let n = v
            .pointer("/usage/completion_tokens")
            .and_then(|c| c.as_u64())
            .unwrap_or(0);
        debug!("llama-server generated {} tokens", n);
        Ok(text)
    }

    /// Minimal HTTP/1.1 over a plain socket. axum already pulls hyper in, but hyper's client half
    /// is not enabled here, and adding reqwest just to reach one localhost port is not worth it.
    async fn rpc(
        addr: SocketAddr,
        method: &str,
        path: &str,
        body: Option<Value>,
    ) -> Result<Value> {
        let mut stream = TcpStream::connect(addr)
            .await
            .map_err(|e| anyhow!("cannot reach llama-server on {}: {}", addr, e))?;

        let payload = body.map(|b| b.to_string()).unwrap_or_default();
        let request = format!(
            "{} {} HTTP/1.1\r\nHost: {}\r\nUser-Agent: realtime-translator\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            method, path, addr, payload.len(), payload
        );
        stream
            .write_all(request.as_bytes())
            .await
            .context("writing request to llama-server")?;

        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).await?;

        let split = raw
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .ok_or_else(|| anyhow!("malformed response from llama-server"))?;
        let head = String::from_utf8_lossy(&raw[..split]).to_lowercase();
        let mut body = raw[split + 4..].to_vec();

        if head.contains("transfer-encoding: chunked") {
            body = dechunk(&body)?;
        }
        if !head.contains(" 200 ") && !head.contains("200\r\n") && !head.contains(" 200\r\n") {
            let status = head.lines().next().unwrap_or("").to_string();
            let snippet = String::from_utf8_lossy(&body).chars().take(200).collect::<String>();
            return Err(anyhow!("llama-server said {}: {}", status, snippet));
        }
        let text = String::from_utf8_lossy(&body).to_string();
        let value: Value = serde_json::from_str(&text)
            .map_err(|e| anyhow!("bad json from llama-server ({}): {}", e, &text[..text.len().min(200)]))?;
        Ok(value)
    }
}

/// Undo chunked transfer encoding.
fn dechunk(data: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let end = match data[i..].iter().position(|b| *b == b'\r') {
            Some(e) => i + e,
            None => break,
        };
        let line = String::from_utf8_lossy(&data[i..end]);
        let size = usize::from_str_radix(line.trim().split(';').next().unwrap_or("").trim(), 16)
            .map_err(|_| anyhow!("bad chunk size {:?}", line.trim()))?;
        if size == 0 {
            break;
        }
        let start = end + 2;
        let stop = start + size;
        if stop > data.len() {
            break;
        }
        out.extend_from_slice(&data[start..stop]);
        i = stop + 2;
    }
    Ok(out)
}

fn addr_str(addr: SocketAddr) -> String {
    addr.to_string()
}

/// Incremental reader for an HTTP/1.1 chunked response body, which is what server-sent events
/// arrive over.
struct SseReader {
    stream: TcpStream,
    head_done: bool,
    buf: Vec<u8>,
}

impl SseReader {
    fn new(stream: TcpStream) -> Self {
        Self {
            stream,
            head_done: false,
            buf: Vec::new(),
        }
    }

    /// Read up to the end of the response headers.
    async fn read_head(&mut self) -> Result<()> {
        let mut byte = [0u8; 1];
        loop {
            let n = self.stream.read(&mut byte).await?;
            if n == 0 {
                break;
            }
            self.buf.push(byte[0]);
            if self.buf.len() >= 4 && self.buf[self.buf.len() - 4..] == *b"\r\n\r\n" {
                break;
            }
            if self.buf.len() > 65536 {
                break;
            }
        }
        self.head_done = true;
        Ok(())
    }

    /// Next chunk of the decoded body, or None when the stream has ended.
    async fn next_body_chunk(&mut self) -> Result<Option<Vec<u8>>> {
        if !self.head_done {
            self.read_head().await?;
        }
        // Size line: hex length up to CRLF.
        let mut line = Vec::new();
        let mut byte = [0u8; 1];
        loop {
            let n = self.stream.read(&mut byte).await?;
            if n == 0 {
                return Ok(None);
            }
            if byte[0] == b'\n' {
                break;
            }
            if byte[0] != b'\r' {
                line.push(byte[0]);
            }
        }
        let size = usize::from_str_radix(
            String::from_utf8_lossy(&line).split(';').next().unwrap_or("").trim(),
            16,
        )
        .map_err(|_| anyhow!("bad chunk size {:?}", String::from_utf8_lossy(&line)))?;
        if size == 0 {
            return Ok(None);
        }
        let mut data = vec![0u8; size];
        self.stream.read_exact(&mut data).await?;
        // Trailing CRLF after the chunk payload.
        let mut trail = [0u8; 2];
        let _ = self.stream.read_exact(&mut trail).await;
        Ok(Some(data))
    }
}

fn find_server(explicit: Option<&str>) -> Result<PathBuf> {
    if let Some(p) = explicit.map(str::trim).filter(|p| !p.is_empty()) {
        let path = PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
        return Err(anyhow!("configured llama_server {} does not exist", p));
    }
    if let Ok(p) = std::env::var("RT_LLAMA_SERVER") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
    }
    // PATH lookup
    if let Ok(found) = which("llama-server.exe") {
        return Ok(found);
    }
    for dir in server_search_dirs() {
        let candidate = dir.join("llama-server.exe");
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(anyhow!(
        "llama-server.exe not found. Build llama.cpp with cmake, then point the app at it by any \
         of: [translation].llama_server in config.toml, the RT_LLAMA_SERVER environment variable, \
         LLAMA_CPP_HOME, or by placing it next to realtime-translator.exe."
    ))
}

/// Locate a Qwen GGUF in the kalosm model cache unless one was configured.
fn find_model(explicit: Option<&str>) -> Result<PathBuf> {
    if let Some(p) = explicit.map(str::trim).filter(|p| !p.is_empty()) {
        let path = PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
        return Err(anyhow!("configured model {} does not exist", p));
    }
    let mut roots = Vec::new();
    if let Some(appdata) = std::env::var_os("APPDATA") {
        let mut p = PathBuf::from(appdata);
        p.push("kalosm");
        p.push("cache");
        roots.push(p);
    }
    roots.push(dirs_next::cache_dir().unwrap_or_default().join("realtime-translator/models"));
    // LLAMA_CPP_HOME/models, and a `models` folder next to the executable or next to the checkout,
    // are all convenient places to drop a better model without editing config.
    if let Ok(home) = std::env::var("LLAMA_CPP_HOME") {
        roots.push(PathBuf::from(home).join("models"));
    }
    if let Ok(p) = std::env::var("RT_LLAMA_MODEL") {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Ok(path);
        }
        roots.push(path);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            roots.push(dir.join("models"));
            if let Some(parent) = dir.parent() {
                roots.push(parent.join("models"));
            }
        }
    }

    let mut candidates: Vec<PathBuf> = Vec::new();
    for root in roots {
        collect_gguf(&root, &mut candidates);
    }
    // Prefer Qwen instruct models, then larger ones. Size is the tiebreaker on purpose: with the
    // 0.5B model, translating English to Korean produced output in *Chinese* outright, while the
    // 1.5B model handled it. Bigger is the only reliable signal that quality will hold up.
    candidates.sort_by(|a, b| {
        let rank = |p: &PathBuf| {
            let name = p.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
            match (name.contains("qwen"), name.contains("instruct")) {
                (true, true) => 0u8,
                (true, false) => 1,
                _ => 2,
            }
        };
        let size = |p: &PathBuf| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        rank(a).cmp(&rank(b)).then(size(b).cmp(&size(a)))
    });
    candidates.into_iter().next().ok_or_else(|| {
        anyhow!("no GGUF translation model found in the model cache; set [translation].model")
    })
}

fn collect_gguf(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_gguf(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
            out.push(path);
        }
    }
}

fn which(exe: &str) -> Result<PathBuf> {
    let dirs = std::env::var("PATH").unwrap_or_default();
    for dir in dirs.split(';') {
        let candidate = Path::new(dir).join(exe);
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(anyhow!("{} not on PATH", exe))
}

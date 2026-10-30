//! whisper.cpp backend for transcription.
//!
//! candle/rwhisper costs roughly 8 s per call no matter how short the audio is, because the whole
//! 30-second padded window is re-encoded every time. whisper.cpp does the same work about 15x
//! faster (measured: 0.67 s for a 10 s chunk, 2.4 s for 34 s), which is what turns a pipeline that
//! falls behind into one that keeps up.
//!
//! Same shape as the translator: `whisper-server.exe` runs as a child process and we post audio to
//! it. The model stays resident between requests, so there is no per-call load cost.

use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tracing::{info, warn};

pub struct WhisperServer {
    child: parking_lot::Mutex<Option<Child>>,
    addr: SocketAddr,
    /// Empty means "let whisper detect it", which costs an extra encoder pass per request.
    language: String,
    /// Everything needed to launch the server again, held so a dead backend can be revived
    /// instead of failing every caption for the rest of the session (same design as
    /// LlamaServer::revive; without it one bad restart leaves the app "ready" but deaf).
    spec: SpawnSpec,
}

struct SpawnSpec {
    bin: PathBuf,
    model: PathBuf,
    threads: usize,
}

impl Drop for WhisperServer {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.lock().take() {
            let _ = child.kill();
        }
    }
}

impl WhisperServer {
    pub async fn start(
        server_bin: Option<&str>,
        model: Option<&str>,
        port: u16,
        threads: usize,
        language: &str,
    ) -> Result<Self> {
        let addr = SocketAddr::from(([127, 0, 0, 1], port));
        // Adopting a server we did not start is handy, but it is also how a restart silently
        // dies: stop() kills its child asynchronously, and if start() arrives before the port
        // is released it happily adopts the corpse, which then exits and leaves every later
        // caption failing. Confirm it answers twice, a beat apart, before trusting it.
        if Self::alive(addr).await {
            info!("stt: reusing whisper-server already listening on {}", addr);
            return Ok(Self {
                child: parking_lot::Mutex::new(None),
                addr,
                language: language.trim().to_string(),
                spec: SpawnSpec {
                    bin: PathBuf::new(),
                    model: PathBuf::new(),
                    threads,
                },
            });
        }

        let bin = find_binary(server_bin)?;
        let model_path = find_model(model)?;
        info!(
            "stt: starting whisper-server ({} threads, model {})",
            threads,
            model_path.display()
        );

        let me = Self {
            child: parking_lot::Mutex::new(None),
            addr,
            language: language.trim().to_string(),
            spec: SpawnSpec {
                bin: bin.clone(),
                model: model_path.clone(),
                threads,
            },
        };
        me.spawn_child()?;
        me.wait_until_ready().await?;
        Ok(me)
    }

    fn spawn_child(&self) -> Result<()> {
        let child = Command::new(&self.spec.bin)
            .arg("-m")
            .arg(&self.spec.model)
            .arg("-t")
            .arg(self.spec.threads.to_string())
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(self.addr.port().to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| {
                anyhow!(
                    "failed to launch {}: {} (are the MinGW runtime DLLs next to it?)",
                    self.spec.bin.display(),
                    e
                )
            })?;
        *self.child.lock() = Some(child);
        Ok(())
    }

    /// Bring the backend back if it died. Without this, one dead server leaves the app
    /// reporting "ready" while every transcription fails forever.
    async fn revive(&self) -> Result<()> {
        if Self::alive(self.addr).await {
            return Ok(());
        }
        warn!("stt: whisper-server is not responding, starting a new one");
        if let Some(mut old) = self.child.lock().take() {
            let _ = old.kill();
        }
        if self.spec.bin.as_os_str().is_empty() {
            return Err(anyhow!(
                "whisper-server died and was adopted (not started by us); \
                 restart the app to recover"
            ));
        }
        self.spawn_child()?;
        self.wait_until_ready().await
    }

    async fn wait_until_ready(&self) -> Result<()> {
        for _ in 0..120 {
            if Self::ping(self.addr).await.is_ok() {
                info!("stt: whisper-server ready on {}", self.addr);
                return Ok(());
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
        Err(anyhow!("whisper-server did not become ready within 60 s"))
    }

    /// Two healthy probes a beat apart — one immediate yes can be a dying process answering
    /// its last request before exiting.
    async fn alive(addr: SocketAddr) -> bool {
        for probe in 0..2 {
            if Self::ping(addr).await.is_err() {
                return false;
            }
            if probe == 0 {
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            }
        }
        true
    }

    async fn ping(addr: SocketAddr) -> Result<()> {
        let mut stream = TcpStream::connect(addr).await?;
        stream
            .write_all(b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            .await?;
        let mut buf = Vec::new();
        stream.read_to_end(&mut buf).await?;
        if String::from_utf8_lossy(&buf).contains("OK") {
            Ok(())
        } else {
            Err(anyhow!("not healthy"))
        }
    }

    /// Transcribe mono 16 kHz f32 audio. Returns the full text.
    pub async fn transcribe(&self, samples: &[f32]) -> Result<String> {
        let wav = pcm16_wav(samples);
        if std::env::var("RT_WAV_DUMP").is_ok() {
            static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let i = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = std::path::Path::new(r"C:	emp	ools").join(format!("chunk{:02}.wav", i));
            let _ = std::fs::write(&path, &wav);
        }
        let boundary = "----rtboundary7f3a9c";
        let body = build_multipart(boundary, "audio.wav", &wav, Some(&self.language));

        let request = format!(
            "POST /inference HTTP/1.1\r\nHost: {}\r\nUser-Agent: realtime-translator\r\nContent-Type: multipart/form-data; boundary={}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            self.addr,
            boundary,
            body.len()
        );

        // Connect through the revive path: a server that died after a restart used to stay
        // dead for the rest of the session; this makes the next caption fix it.
        let mut stream = match TcpStream::connect(self.addr).await {
            Ok(s) => s,
            Err(e) => {
                warn!("stt: whisper-server unreachable ({}); reviving", e);
                self.revive().await?;
                TcpStream::connect(self.addr)
                    .await
                    .map_err(|e| anyhow!("cannot reach whisper-server on {}: {}", self.addr, e))?
            }
        };
        stream
            .write_all(request.as_bytes())
            .await
            .context("posting audio to whisper-server")?;
        stream.write_all(&body).await?;
        // Note: do NOT half-close the socket here. httplib treats an early shutdown as the client
        // going away and closes without ever writing a response.
        let _ = &body;

        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).await?;

        let split = raw
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .ok_or_else(|| {
                anyhow!(
                    "malformed response from whisper-server ({} bytes read): {:?}",
                    raw.len(),
                    String::from_utf8_lossy(&raw[..raw.len().min(180)]).to_string()
                )
            })?;
        let head = String::from_utf8_lossy(&raw[..split]).to_lowercase();
        let mut payload = raw[split + 4..].to_vec();
        if head.contains("transfer-encoding: chunked") {
            payload = dechunk(&payload)?;
        }
        if !head.contains("200") {
            return Err(anyhow!(
                "whisper-server error: {}",
                String::from_utf8_lossy(&payload).chars().take(200).collect::<String>()
            ));
        }
        let text = String::from_utf8_lossy(&payload).to_string();
        let value: Value = serde_json::from_str(&text)
            .map_err(|e| anyhow!("bad json from whisper-server ({}): {}", e, &text[..text.len().min(160)]))?;
        let got = value
            .get("text")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string();
        if got.trim().is_empty() {
            tracing::warn!(
                "whisper-server empty (samples={}, resp={:?})",
                samples.len(),
                &text[..text.len().min(160)]
            );
        }
        Ok(got)
    }
}

/// Encode f32 mono audio as an in-memory 16-bit PCM WAV. whisper-server takes a file, so this is
/// what goes over the wire.
fn pcm16_wav(samples: &[f32]) -> Vec<u8> {
    const SR: u32 = 16000;
    let data_len = samples.len() * 2;
    let mut out = Vec::with_capacity(44 + data_len);
    let tag = |s: &'static str| s.as_bytes().to_vec();
    out.extend_from_slice(&tag("RIFF"));
    out.extend_from_slice(&(36 + data_len as u32).to_le_bytes());
    out.extend_from_slice(&tag("WAVE"));
    out.extend_from_slice(&tag("fmt "));
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&SR.to_le_bytes());
    out.extend_from_slice(&(SR * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(&tag("data"));
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round().clamp(-32768.0, 32767.0) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn build_multipart(boundary: &str, filename: &str, wav: &[u8], language: Option<&str>) -> Vec<u8> {
    let mut body = Vec::with_capacity(wav.len() + 512);
    body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body.extend_from_slice(
        format!(
            "Content-Disposition: form-data; name=\"file\"; filename=\"{}\"\r\nContent-Type: application/octet-stream\r\n\r\n",
            filename
        )
        .as_bytes(),
    );
    body.extend_from_slice(wav);
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body.extend_from_slice(
        b"Content-Disposition: form-data; name=\"response_format\"\r\n\r\njson\r\n",
    );
    // Always send the language. With the field absent whisper.cpp falls back to its own default
    // of "en", which silently turns Japanese/Mandarin/Cantonese/Korean audio into invented English.
    if let Some(lang) = language {
        if !lang.is_empty() {
            body.extend_from_slice(
                format!("--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\n{lang}\r\n")
                    .as_bytes(),
            );
        }
    }
    body.extend_from_slice(format!("--{}--\r\n", boundary).as_bytes());
    body
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

fn find_binary(explicit: Option<&str>) -> Result<PathBuf> {
    if let Some(p) = explicit.map(str::trim).filter(|p| !p.is_empty()) {
        let path = PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
        return Err(anyhow!("configured stt server {} does not exist", p));
    }
    if let Ok(p) = std::env::var("RT_WHISPER_SERVER") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
    }
    let exe = "whisper-server.exe";
    for dir in server_search_dirs() {
        let candidate = dir.join(exe);
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(anyhow!(
        "whisper-server.exe not found. Build whisper.cpp, then point the app at it by any of: \
         [stt].server in config.toml, the RT_WHISPER_SERVER environment variable, \
         WHISPER_CPP_HOME, or by placing it next to realtime-translator.exe."
    ))
}

/// Directories searched for `whisper-server.exe`, in priority order.
///
/// Nothing here is machine-specific on purpose: whoever deploys this should be able to point the
/// app at a whisper.cpp build without editing source.
fn server_search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    // WHISPER_CPP_HOME = a whisper.cpp checkout or build tree.
    if let Ok(home) = std::env::var("WHISPER_CPP_HOME") {
        let base = PathBuf::from(home);
        dirs.push(base.join("build").join("bin"));
        dirs.push(base);
    }
    // A self-contained release: the server lives next to this executable.
    if let Some(dir) = exe_dir() {
        dirs.push(dir);
    }
    dirs.extend(std::env::split_paths(&std::env::var("PATH").unwrap_or_default()));
    dirs.retain(|d| !d.as_os_str().is_empty());
    dirs
}

/// The directory containing this executable, when it can be determined.
pub(crate) fn exe_dir() -> Option<PathBuf> {
    std::env::current_exe().ok()?.parent().map(|p| p.to_path_buf())
}

/// Collect `.bin` whisper models from one directory, ignoring unrelated files.
fn collect_bin(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) == Some("bin") {
                out.push(p);
            }
        }
    }
}

fn find_model(explicit: Option<&str>) -> Result<PathBuf> {
    if let Some(p) = explicit.map(str::trim).filter(|p| !p.is_empty()) {
        let path = PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
        return Err(anyhow!("configured stt model {} does not exist", p));
    }
    let mut found = Vec::new();
    if let Ok(p) = std::env::var("RT_WHISPER_MODEL") {
        let path = PathBuf::from(p);
        if path.is_file() {
            found.push(path);
        } else if path.is_dir() {
            collect_bin(&path, &mut found);
        }
    }
    if let Ok(home) = std::env::var("WHISPER_CPP_HOME") {
        collect_bin(&PathBuf::from(home).join("models"), &mut found);
    }
    if let Some(dir) = exe_dir() {
        collect_bin(&dir.join("models"), &mut found);
        collect_bin(&dir, &mut found);
    }
    for dir in std::env::split_paths(&std::env::var("PATH").unwrap_or_default()) {
        collect_bin(&dir.join("models"), &mut found);
    }
    found.sort();
    // Prefer a real production model. The whisper.cpp repo ships `for-tests-ggml-*.bin` stubs that
    // load fine and return empty text, so they must never win discovery over the real weights.
    found.sort_by_key(|p| {
        let name = p
            .file_name()
            .map(|n| n.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        let is_stub = name.starts_with("for-tests") || name.contains("for-tests");
        let tier = if name.contains("tiny") {
            0
        } else if name.contains("base") {
            1
        } else {
            2
        };
        (is_stub, tier)
    });
    found.into_iter().next().ok_or_else(|| {
        anyhow!(
            "no whisper.cpp ggml model found. Set [stt].model in config.toml, point \
             RT_WHISPER_MODEL or WHISPER_CPP_HOME at your model directory, or place the .bin \
             next to realtime-translator.exe."
        )
    })
}

const state = {
    ws: null,
    running: false,
};

document.addEventListener('DOMContentLoaded', async () => {
    await loadDevices();
    await hydrateConfig();
    connectWebSocket();
    setupButtons();
});

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

    ws.onopen = () => {
        console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleEvent(data);
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected, reconnecting in 3s...');
        setTimeout(connectWebSocket, 3000);
    };

    state.ws = ws;
}

function handleEvent(event) {
    switch (event.type) {
        case 'status':
            setRunning(event.running);
            break;
        case 'transcript':
            appendTranscript(event.text);
            break;
        case 'translation_token':
            appendTranslationToken(event.token);
            break;
        case 'translation_complete':
            setTranslationComplete(event.text);
            break;
        case 'audio_level':
            updateLevel(event.rms);
            break;
        case 'error':
            showError(event.message);
            break;
    }
}

async function loadDevices() {
    try {
        const res = await fetch('/api/devices');
        const devices = await res.json();
        const select = document.getElementById('audio-device');
        for (const device of devices) {
            const option = document.createElement('option');
            option.value = device.name;
            option.dataset.loopback = device.is_loopback ? 'true' : 'false';
            option.textContent = device.is_loopback
                ? `🔊 ${device.name} — system audio (loopback)`
                : `🎤 ${device.name} (${device.host})`;
            select.appendChild(option);
        }
    } catch (e) {
        console.error('Failed to load devices:', e);
    }
}

async function hydrateConfig() {
    try {
        const res = await fetch('/api/config');
        const cfg = await res.json();
        const devSel = document.getElementById('audio-device');
        if (cfg.audio_device_name) {
            for (const opt of devSel.options) {
                if (opt.value === cfg.audio_device_name) { devSel.value = opt.value; break; }
            }
        }
        if (cfg.target_language) document.getElementById('target-language').value = cfg.target_language;
        if (cfg.whisper_model) document.getElementById('whisper-model').value = cfg.whisper_model;
    } catch (e) {
        console.error('Failed to load config:', e);
    }
}

function buildConfig() {
    const deviceSelect = document.getElementById('audio-device');
    const langSelect = document.getElementById('target-language');
    const modelSelect = document.getElementById('whisper-model');
    const selected = deviceSelect.options[deviceSelect.selectedIndex];
    const isLoopback = selected && selected.dataset.loopback === 'true';
    return {
        audio_device_name: deviceSelect.value || null,
        audio_device_is_loopback: isLoopback,
        target_language: langSelect.value,
        whisper_model: modelSelect ? modelSelect.value : 'base',
        server_port: 3000,
        vad_threshold: 0.003,
        vad_hangover_frames: 15,
        max_chunk_seconds: 5.0,
    };
}

function setupButtons() {
    document.getElementById('btn-start').addEventListener('click', startPipeline);
    document.getElementById('btn-stop').addEventListener('click', stopPipeline);
    document.getElementById('btn-apply').addEventListener('click', applyConfig);
}

async function startPipeline() {
    hideError();
    setHint('');

    // Persist the shown settings, then start (uses them immediately).
    await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildConfig()),
    });

    const res = await fetch('/api/start', { method: 'POST' });
    const body = await res.json().catch(() => ({ ok: res.ok }));
    if (body.ok) {
        clearOutput();
        setRunning(true);
        setStatusText('Loading models…', 'loading');
        startStatusPolling();
    } else {
        showError(body.error || 'Failed to start pipeline');
    }
}

async function applyConfig() {
    hideError();
    try {
        const res = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(buildConfig()),
        });
        if (!res.ok) throw new Error('HTTP ' + res.status);
        setHint(state.running ? 'Settings saved — restart to apply' : 'Settings saved');
    } catch (e) {
        showError('Failed to save settings');
    }
}

let statusTimer = null;
function startStatusPolling() {
    stopStatusPolling();
    statusTimer = setInterval(pollStatus, 1000);
}
function stopStatusPolling() {
    if (statusTimer) { clearInterval(statusTimer); statusTimer = null; }
}
async function pollStatus() {
    try {
        const res = await fetch('/api/status');
        const s = await res.json();
        if (s.device_error) {
            stopStatusPolling();
            setRunning(false);
            showError('Capture device lost — check your output device and press Start.');
            return;
        }
        if (s.stage === 'ready') {
            stopStatusPolling();
            setStatusText('Running', 'running');
        } else if (s.stage === 'stopped') {
            stopStatusPolling();
            setRunning(false);
        } else {
            setStatusText('Loading models…', 'loading');
        }
    } catch (e) {
        // server unreachable; stop polling
        stopStatusPolling();
    }
}

async function stopPipeline() {
    await fetch('/api/stop', { method: 'POST' });
    stopStatusPolling();
    setRunning(false);
}

function setRunning(running) {
    state.running = running;
    if (!running) stopStatusPolling();
    document.getElementById('btn-start').disabled = running;
    document.getElementById('btn-stop').disabled = !running;
    document.getElementById('status-dot').className = running ? 'dot dot-green' : 'dot dot-red';
    if (running) setStatusText('Running', 'running');
    else setStatusText('Stopped', 'stopped');
}

function setStatusText(text, kind) {
    const dot = document.getElementById('status-dot');
    const label = document.getElementById('status-text');
    if (label) label.textContent = text;
    if (dot) dot.className = kind === 'loading' ? 'dot dot-amber' : (kind === 'running' ? 'dot dot-green' : 'dot dot-red');
}

function appendTranscript(text) {
    const el = document.getElementById('transcript');
    const entry = document.createElement('div');
    entry.className = 'entry';
    entry.textContent = text;
    el.appendChild(entry);
    el.scrollTop = el.scrollHeight;
}

function appendTranslationToken(token) {
    const el = document.getElementById('translation');
    let current = el.querySelector('.current');
    if (!current) {
        current = document.createElement('span');
        current.className = 'current';
        el.appendChild(current);
    }
    current.textContent += token;
    el.scrollTop = el.scrollHeight;
}

function setTranslationComplete(text) {
    const el = document.getElementById('translation');
    const current = el.querySelector('.current');
    if (current) {
        current.className = 'entry';
        current.classList.remove('current');
    }
    el.scrollTop = el.scrollHeight;
}

function clearOutput() {
    document.getElementById('transcript').innerHTML = '';
    document.getElementById('translation').innerHTML = '';
}

function showError(message) {
    const banner = document.getElementById('error-banner');
    banner.textContent = `⚠ ${message}`;
    banner.hidden = false;
    // A fatal error means the worker died; reflect that in the UI.
    setRunning(false);
}

function hideError() {
    const banner = document.getElementById('error-banner');
    banner.hidden = true;
    banner.textContent = '';
}

function setHint(text) {
    const el = document.getElementById('config-hint');
    if (el) el.textContent = text;
}

function updateLevel(rms) {
    const bar = document.getElementById('level-bar');
    const label = document.getElementById('level-value');
    if (!bar || !label) return;
    const pct = Math.min(100, rms * 100 * 3);
    bar.style.width = pct.toFixed(1) + '%';
    label.textContent = rms.toFixed(3);
}

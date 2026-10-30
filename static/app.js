const state = {
    ws: null,
    running: false,
    applying: false,
    appliedPending: false,
    // Last config known to the server. We must build new payloads on top of it: the UI only
    // exposes three settings, and sending hardcoded values for the rest silently reset them on
    // every save (it kept clobbering max_chunk_seconds back to 5.0).
    serverConfig: null,
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
        state.serverConfig = cfg;
        const devSel = document.getElementById('audio-device');
        if (cfg.audio_device_name) {
            for (const opt of devSel.options) {
                if (opt.value === cfg.audio_device_name) { devSel.value = opt.value; break; }
            }
        }
        if (cfg.target_language) document.getElementById('target-language').value = cfg.target_language;
        if (cfg.whisper_model) document.getElementById('whisper-model').value = cfg.whisper_model;
        updateApplyVisibility();
    } catch (e) {
        console.error('Failed to load config:', e);
    }
}

// The three controls that actually need a pipeline restart when they change.
function currentSelection() {
    const devSel = document.getElementById('audio-device');
    const langSel = document.getElementById('target-language');
    const modelSel = document.getElementById('whisper-model');
    return {
        device: devSel.value || null,
        lang: langSel ? langSel.value : null,
        model: modelSel ? modelSel.value : null,
    };
}

function savedSelection() {
    const c = state.serverConfig || {};
    return {
        device: c.audio_device_name || null,
        lang: c.target_language || null,
        model: c.whisper_model || null,
    };
}

function isDirty() {
    return changedParts().length > 0;
}

/// Which of the three controls differ from what the server has. "lang" is listed separately
/// because the translator picks it up on the next caption — it needs no restart.
function changedParts() {
    const a = currentSelection();
    const b = savedSelection();
    const out = [];
    if (a.device !== b.device) out.push('device');
    if (a.model !== b.model) out.push('model');
    if (a.lang !== b.lang) out.push('lang');
    return out;
}

// Apply is always available. Hiding it until something looked different was a bad trade: it made
// the only way to re-run the pipeline impossible without first editing a setting, and "restart to
// pick up what I already changed" is exactly what the button is for.
function updateApplyVisibility() {
    const btn = document.getElementById('btn-apply');
    if (btn) btn.style.display = '';
    const hint = document.getElementById('config-hint');
    if (hint && !isDirty() && /saved|updated/i.test(hint.textContent)) hint.textContent = '';
}

function buildConfig() {
    const deviceSelect = document.getElementById('audio-device');
    const langSelect = document.getElementById('target-language');
    const modelSelect = document.getElementById('whisper-model');
    const selected = deviceSelect.options[deviceSelect.selectedIndex];
    const isLoopback = selected && selected.dataset.loopback === 'true';
    // Start from the server's config and overlay only what the UI actually controls, so fields
    // like vad_threshold / max_chunk_seconds / model paths survive a save.
    return Object.assign({}, state.serverConfig || {}, {
        audio_device_name: deviceSelect.value || null,
        audio_device_is_loopback: isLoopback,
        target_language: langSelect.value,
        whisper_model: modelSelect ? modelSelect.value : 'base',
    });
}

function setupButtons() {
    document.getElementById('btn-start').addEventListener('click', startPipeline);
    document.getElementById('btn-stop').addEventListener('click', stopPipeline);
    document.getElementById('btn-apply').addEventListener('click', applyConfig);
    for (const id of ['audio-device', 'target-language', 'whisper-model']) {
        const el = document.getElementById(id);
        if (el) el.addEventListener('change', updateApplyVisibility);
    }
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
        state.serverConfig = buildConfig();
        updateApplyVisibility();
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
    if (state.applying) return;
    state.applying = true;
    const started = Date.now();
    const tick = setInterval(() => {
        setHint(`Applying settings… ${Math.max(1, Math.round((Date.now() - started) / 1000))}s`);
    }, 1000);
    try {
        const res = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(buildConfig()),
        });
        if (!res.ok) throw new Error('HTTP ' + res.status);
        state.serverConfig = buildConfig();

        if (!state.running) {
            clearInterval(tick);
            setHint('Saved — press Start to apply', 5000);
            return;
        }

        // Apply means restart: stop the running pipeline and start it again so the saved settings
        // are actually read. Doing it here means the user never has to hunt for Stop/Start.
        setRunning(false, { busy: true });
        setStatusText('Restarting…', 'loading');
        await fetch('/api/stop', { method: 'POST' });
        if (!await waitForStage('stopped', 8000)) {
            showError('Saved, but the old pipeline would not stop. Press Stop then Start.');
            setHint('');
            setRunning(false);
            return;
        }
        const start = await fetch('/api/start', { method: 'POST' })
            .then(r => r.json())
            .catch(() => ({ ok: false }));
        if (!start.ok) {
            showError(start.error || 'Settings saved but the pipeline failed to restart');
            setHint('');
            setRunning(false);
            return;
        }
        clearOutput();
        state.appliedPending = true;
        setRunning(true);
        setStatusText('Loading models…', 'loading');
        startStatusPolling();
    } catch (e) {
        showError('Failed to save settings');
        setRunning(state.running);
    } finally {
        clearInterval(tick);
        state.applying = false;
    }
}

/// Wait for the server to report a given stage, so we never start on top of a half-torn-down pipeline.
async function waitForStage(want, timeoutMs) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        try {
            const s = await (await fetch('/api/status')).json();
            if (s.stage === want) return true;
        } catch (e) {
            return false;
        }
        await new Promise(r => setTimeout(r, 200));
    }
    return false;
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
            // Retire the "applying" note. If this poll followed an Apply, confirm it landed.
            if (state.appliedPending) {
                state.appliedPending = false;
                setHint('Applied', 3000);
            } else {
                setHint('');
            }
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

function setRunning(running, opts) {
    state.running = running;
    if (!running) stopStatusPolling();
    const busy = !!(opts && opts.busy);
    document.getElementById('btn-start').disabled = running || busy;
    document.getElementById('btn-stop').disabled = !running || busy;
    document.getElementById('btn-apply').disabled = busy;
    document.getElementById('status-dot').className = running ? 'dot dot-green' : 'dot dot-red';
    if (running) setStatusText('Running', 'running');
    else if (!busy) setStatusText('Stopped', 'stopped');
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

let hintTimer = null;
function setHint(text, autoClearMs) {
    const el = document.getElementById('config-hint');
    if (!el) return;
    el.textContent = text;
    if (hintTimer) {
        clearTimeout(hintTimer);
        hintTimer = null;
    }
    if (text && autoClearMs) {
        hintTimer = setTimeout(() => {
            el.textContent = '';
            hintTimer = null;
        }, autoClearMs);
    }
}

// Level meter state: fast attack, slow release. Without smoothing the bar snaps between frames
// and reads as frozen whenever the pipeline is busy.
let levelSmooth = 0;

function updateLevel(rms) {
    const bar = document.getElementById('level-bar');
    const label = document.getElementById('level-value');
    if (!bar || !label) return;
    // Map dBFS onto the meter. The old `rms * 300` scale pinned to 100% at normal speech levels,
    // so during real audio the bar sat glued to the right edge — indistinguishable from a hang.
    const db = rms > 0 ? 20 * Math.log10(rms) : -80;
    const pct = Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
    levelSmooth = pct > levelSmooth
        ? Math.max(pct, levelSmooth * 0.6 + pct * 0.4)
        : levelSmooth * 0.85 + pct * 0.15;
    bar.style.width = levelSmooth.toFixed(1) + '%';
    label.textContent = (db < -60 ? '-inf' : db.toFixed(1)) + ' dB';
}

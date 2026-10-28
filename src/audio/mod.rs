pub mod vad;

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;
use tracing::{error, info};

use crate::state::AudioStats;

/// Audio sample format used internally: f32 mono at 16kHz
pub const TARGET_SAMPLE_RATE: u32 = 16000;

#[derive(Debug, Clone, serde::Serialize)]
pub struct AudioDeviceInfo {
    pub name: String,
    pub host: String,
    /// True when this device is an output device usable for system-audio loopback capture.
    pub is_loopback: bool,
}

/// List all capturable audio sources:
/// - real input devices (microphones)
/// - output devices, which WASAPI/CoreAudio expose as system-audio loopback sources
pub fn list_capture_devices() -> Vec<AudioDeviceInfo> {
    let mut devices = Vec::new();
    let host = cpal::default_host();
    let host_id = format!("{:?}", host.id());

    if let Ok(device_list) = host.input_devices() {
        for device in device_list {
            devices.push(AudioDeviceInfo {
                name: get_device_name(&device),
                host: host_id.clone(),
                is_loopback: false,
            });
        }
    }

    // Output devices double as loopback capture sources on WASAPI (and CoreAudio on macOS 13+).
    if let Ok(device_list) = host.output_devices() {
        for device in device_list {
            devices.push(AudioDeviceInfo {
                name: get_device_name(&device),
                host: host_id.clone(),
                is_loopback: true,
            });
        }
    }

    devices
}

fn get_device_name(device: &cpal::Device) -> String {
    device
        .name()
        .unwrap_or_else(|_| "Unknown Device".to_string())
}

/// Start capturing audio from the specified device (or default).
/// When `is_loopback` is true the device is treated as an output device and opened in
/// loopback mode (system-audio capture); otherwise it is a normal input device (microphone).
/// Sends f32 mono 16kHz samples through the channel via a dedicated thread.
pub fn start_capture(
    device_name: Option<&str>,
    is_loopback: bool,
    tx: tokio::sync::mpsc::UnboundedSender<Vec<f32>>,
    stats: Arc<AudioStats>,
    capture_error: Arc<AtomicBool>,
) -> Result<AudioStreamHandle> {
    let host = cpal::default_host();

    let device = if let Some(name) = device_name {
        let found = if is_loopback {
            host.output_devices()?
                .find(|d| d.name().map(|n| n.contains(name)).unwrap_or(false))
        } else {
            host.input_devices()?
                .find(|d| d.name().map(|n| n.contains(name)).unwrap_or(false))
        };
        found.ok_or_else(|| anyhow!("Device '{}' not found", name))?
    } else if is_loopback {
        host.default_output_device()
            .ok_or_else(|| anyhow!("No default output device found"))?
    } else {
        host.default_input_device()
            .ok_or_else(|| anyhow!("No default input device found"))?
    };

    // For loopback we must use the device's render (output) mix format, since that is the
    // format WASAPI hands back when capturing a render endpoint.
    let config = if is_loopback {
        device.default_output_config()?
    } else {
        device.default_input_config()?
    };

    info!(
        "Starting capture [{}]: rate={} channels={} format={:?}",
        if is_loopback { "loopback" } else { "mic" },
        config.sample_rate().0,
        config.channels(),
        config.sample_format()
    );

    // Rendezvous channel so build/play errors surface to the caller instead of only the log.
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();

    // Spawn a dedicated thread for cpal (cpal::Stream is not Send)
    let (drop_tx, drop_rx) = mpsc::channel::<()>();
    let handle = std::thread::spawn(move || {
        let sample_rate = config.sample_rate().0;
        let channels = config.channels() as usize;

        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => {
                build_stream_f32(&device, &config, tx.clone(), stats.clone(), sample_rate, channels, capture_error.clone())
            }
            cpal::SampleFormat::I16 => {
                build_stream_i16(&device, &config, tx.clone(), stats.clone(), sample_rate, channels, capture_error.clone())
            }
            cpal::SampleFormat::I32 => {
                build_stream_i32(&device, &config, tx.clone(), stats.clone(), sample_rate, channels, capture_error.clone())
            }
            ref fmt => {
                let msg = format!("Unsupported sample format: {:?}", fmt);
                error!("{}", msg);
                let _ = ready_tx.send(Err(msg));
                return;
            }
        };

        match stream {
            Ok(s) => {
                if let Err(e) = s.play() {
                    let msg = format!("Error playing stream: {}", e);
                    error!("{}", msg);
                    let _ = ready_tx.send(Err(msg));
                    return;
                }
                let _ = ready_tx.send(Ok(()));
                let _ = drop_rx.recv();
                drop(s);
            }
            Err(e) => {
                let msg = format!("Error building stream: {}", e);
                error!("{}", msg);
                let _ = ready_tx.send(Err(msg));
            }
        }
    });

    // Wait briefly for the stream to initialize so failures are reported to the UI.
    match ready_rx.recv_timeout(Duration::from_secs(5)) {
        Ok(Ok(())) => {}
        Ok(Err(msg)) => {
            let _ = drop_tx.send(());
            let _ = handle.join();
            return Err(anyhow!(msg));
        }
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            let _ = drop_tx.send(());
            let _ = handle.join();
            return Err(anyhow!("Audio stream did not initialize within 5s"));
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            let _ = drop_tx.send(());
            let _ = handle.join();
            return Err(anyhow!("Capture thread exited before signaling readiness"));
        }
    }

    Ok(AudioStreamHandle {
        drop_tx: Some(drop_tx),
        thread: Some(handle),
    })
}

fn rms_of(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

fn build_stream_f32(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    tx: tokio::sync::mpsc::UnboundedSender<Vec<f32>>,
    stats: Arc<AudioStats>,
    sample_rate: u32,
    channels: usize,
    capture_error: Arc<AtomicBool>,
) -> Result<cpal::Stream, cpal::BuildStreamError> {
    let sr = sample_rate;
    let flag = capture_error.clone();
    device.build_input_stream(
        &config.config(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mono = downmix_mono(data, channels);
            let level = rms_of(&mono);
            let resampled = if sr != TARGET_SAMPLE_RATE {
                resample_linear(&mono, sr, TARGET_SAMPLE_RATE)
            } else {
                mono
            };
            stats.record(resampled.len(), level);
            let _ = tx.send(resampled);
        },
        {
            let flag = flag.clone();
            move |err| {
                error!("Audio stream error: {}", err);
                flag.store(true, Ordering::Relaxed);
            }
        },
        None,
    )
}

fn build_stream_i16(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    tx: tokio::sync::mpsc::UnboundedSender<Vec<f32>>,
    stats: Arc<AudioStats>,
    sample_rate: u32,
    channels: usize,
    capture_error: Arc<AtomicBool>,
) -> Result<cpal::Stream, cpal::BuildStreamError> {
    let sr = sample_rate;
    let flag = capture_error.clone();
    device.build_input_stream(
        &config.config(),
        move |data: &[i16], _: &cpal::InputCallbackInfo| {
            let mono: Vec<f32> = data
                .chunks(channels)
                .map(|chunk| {
                    chunk
                        .iter()
                        .map(|&s| s as f32 / 32768.0)
                        .sum::<f32>()
                        / channels as f32
                })
                .collect();
            let level = rms_of(&mono);
            let resampled = if sr != TARGET_SAMPLE_RATE {
                resample_linear(&mono, sr, TARGET_SAMPLE_RATE)
            } else {
                mono
            };
            stats.record(resampled.len(), level);
            let _ = tx.send(resampled);
        },
        {
            let flag = flag.clone();
            move |err| {
                error!("Audio stream error: {}", err);
                flag.store(true, Ordering::Relaxed);
            }
        },
        None,
    )
}

fn build_stream_i32(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    tx: tokio::sync::mpsc::UnboundedSender<Vec<f32>>,
    stats: Arc<AudioStats>,
    sample_rate: u32,
    channels: usize,
    capture_error: Arc<AtomicBool>,
) -> Result<cpal::Stream, cpal::BuildStreamError> {
    let sr = sample_rate;
    let flag = capture_error.clone();
    device.build_input_stream(
        &config.config(),
        move |data: &[i32], _: &cpal::InputCallbackInfo| {
            let mono: Vec<f32> = data
                .chunks(channels)
                .map(|chunk| {
                    chunk
                        .iter()
                        .map(|&s| s as f32 / 2147483648.0)
                        .sum::<f32>()
                        / channels as f32
                })
                .collect();
            let level = rms_of(&mono);
            let resampled = if sr != TARGET_SAMPLE_RATE {
                resample_linear(&mono, sr, TARGET_SAMPLE_RATE)
            } else {
                mono
            };
            stats.record(resampled.len(), level);
            let _ = tx.send(resampled);
        },
        {
            let flag = flag.clone();
            move |err| {
                error!("Audio stream error: {}", err);
                flag.store(true, Ordering::Relaxed);
            }
        },
        None,
    )
}

fn downmix_mono(data: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        return data.to_vec();
    }
    data.chunks(channels)
        .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
        .collect()
}

fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let output_len = (samples.len() as f64 * ratio) as usize;
    let mut output = Vec::with_capacity(output_len);
    for i in 0..output_len {
        let src_pos = i as f64 / ratio;
        let idx = src_pos as usize;
        let frac = src_pos - idx as f64;
        if idx + 1 < samples.len() {
            output.push(samples[idx] * (1.0 - frac as f32) + samples[idx + 1] * frac as f32);
        } else if idx < samples.len() {
            output.push(samples[idx]);
        } else {
            output.push(0.0);
        }
    }
    output
}

pub struct AudioStreamHandle {
    drop_tx: Option<mpsc::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl AudioStreamHandle {
    pub fn stop(&mut self) {
        if let Some(tx) = self.drop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for AudioStreamHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

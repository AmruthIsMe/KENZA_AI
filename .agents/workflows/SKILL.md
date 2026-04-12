---
description: KENZA Robot - Full Project Context & Architecture Reference
---

# KENZA AI Robot — Project Skills & Context

## 1. Project Overview

**KENZA** is a sentient robotic companion built on a **Raspberry Pi 5**, created by **Amruth**. It features:
- Conversational AI (wake-word activated, cloud+offline LLM fallback)
- Telepresence (2-way audio/video calls between a laptop app and the robot)
- Vision (camera-based object/person detection)
- Gesture recognition
- Autonomous navigation & patrol modes
- Emotional expression through animated eye displays
- Motor control via an ESP32 bridge or direct GPIO

**Hardware:**
- Raspberry Pi 5 (main brain)
- Pi Camera Module 3 (via libcamera / rpicam-vid)
- USB Mini Microphone (ALSA/PulseAudio)
- Wireless Bluetooth Earphone (speaker output via PulseAudio)
- ESP32 or direct GPIO for motor control (L298N driver)
- GPIO LEDs (listening = green GPIO24, thinking = red GPIO25)

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Raspberry Pi 5                    │
│                                                     │
│  kenza.service (systemd)                            │
│    └─ kenza_launcher.py  (port 8764)                │
│         └─ kenza_server.py  (WS port 8765)          │
│              ├─ mediamtx (WHEP port 8889)            │
│              ├─ kenza_conversation.py (AI engine)    │
│              └─ robot_display.html (Chromium kiosk)  │
│                                                     │
│  pi_motor_server.py  (HTTP port 8080)               │
│                                                     │
└─────────────────────────────────────────────────────┘
          ▲ WebSocket (8765)  ▲ WHEP (8889)
          │                   │
┌─────────┴───────────────────┴──────────────┐
│           Laptop / Phone                    │
│   kenza_app_v2.html  (main control UI)      │
└─────────────────────────────────────────────┘
```

---

## 3. File Map

### Python Modules (on Pi)

| File | Purpose | Port |
|---|---|---|
| `kenza_launcher.py` | Lightweight HTTP daemon; starts/stops `kenza_server.py` on demand | **8764** |
| `kenza_server.py` | Main WebSocket hub — commands, signaling, settings sync, MediaMTX launch | **8765** |
| `kenza_stream.py` | Alternate WHIP streamer (PyAudio mic + PiCamera via aiortc). **Not used in production** — MediaMTX handles streaming directly. | — |
| `kenza_conversation.py` | Conversational AI engine — STT → LLM → TTS pipeline, wake word, emotion detection | — |
| `kenza_autonomy.py` | Autonomous navigation, patrol, obstacle avoidance | — |
| `kenza_gesture.py` | MediaPipe-based gesture recognition | — |
| `kenza_vision.py` | Camera vision — person/object detection | — |
| `pi_motor_server.py` | GPIO motor control HTTP server (F/B/L/R/S endpoints) | **8080** |
| `kenza_ai.py` | LLM integration utilities | — |

### Web Files (in `web/`)

| File | Purpose | Runs On |
|---|---|---|
| `kenza_app_v2.html` | **Main laptop/phone control app** — telepresence, joystick, settings, AI chat | Laptop browser |
| `robot_display.html` | **Robot face display** — animated eyes, call UI, runs in Chromium kiosk on Pi | Pi Chromium |
| `joystick_controller.html` | Standalone joystick control page | Laptop browser |
| `motor_test.html` | Motor testing UI | Laptop browser |
| `stream_viewer.html` | WHEP stream viewer | Laptop browser |
| `eyes_display.html` | Standalone eye animation display | Pi Chromium |

### Config Files

| File | Purpose |
|---|---|
| `config/settings.yaml` | AI providers, voice presets, wake word, GPIO pins, audio thresholds |
| `mediamtx.yml` | MediaMTX streaming server config — WHEP/RTSP/HLS settings |
| `.env` | API keys (GEMINI_API_KEY, GROQ_API_KEY) |
| `kenza.service` | systemd unit — auto-starts `kenza_launcher.py` on boot |
| `kenza.sh` | Shell wrapper to run `kenza_conversation.py` |

---

## 4. Telepresence Call Flow

### Video: Pi → Laptop (via MediaMTX WHEP)
1. `kenza_server.py` starts MediaMTX binary with `mediamtx.yml`
2. MediaMTX runs `rpicam-vid | ffmpeg → RTSP` pipeline (video-only)
3. Laptop `kenza_app_v2.html` connects via WHEP: `http://<pi-ip>:8889/kenza/whep`
4. `TP.pc` (receive-only video transceiver) receives the H264 stream

### Audio & Video: Laptop → Pi (via direct WebRTC)
1. Laptop calls `tpStartLocalStream()` → `getUserMedia({video, audio})`
2. Creates `TP.localPc`, adds tracks, sends SDP offer via WebSocket
3. `kenza_server.py` relays `webrtc_offer` message to robot
4. `robot_display.html` receives offer in `handleWebRTCOffer()`
5. Robot creates `remotePc`, sets remote description, creates answer
6. Robot displays laptop feed in `#remote-video`

### Audio: Pi → Laptop (via direct WebRTC)
1. `robot_display.html` calls `getUserMedia({audio: true})` in `handleWebRTCOffer()`
2. Adds mic track to `remotePc` before creating the SDP answer
3. Laptop's `TP.localPc.ontrack` receives audio, creates `<audio>` element
4. Audio plays through the laptop speakers

> **IMPORTANT:** MediaMTX streams **video only**. The Pi microphone audio goes through the **direct peer-to-peer WebRTC** connection, not MediaMTX.

### Video Off (Pi → Laptop)
- Robot presses camera toggle → sends `cam_toggle` WebSocket message
- Laptop receives it and disables video tracks on `ov-remote-video`

---

## 5. Port Map

| Port | Service | Protocol |
|---|---|---|
| **8764** | `kenza_launcher.py` | HTTP |
| **8765** | `kenza_server.py` | WebSocket |
| **8889** | MediaMTX WHEP | HTTP/WebRTC |
| **8554** | MediaMTX RTSP | RTSP |
| **8080** | `pi_motor_server.py` | HTTP |

---

## 6. WebSocket Message Types

### Relay Messages (forwarded between app ↔ robot)
`call_offer`, `call_answer`, `call_reject`, `call_end`, `call_accepted`, `call_busy`, `call_ping`, `ice_candidate`, `webrtc_offer`, `webrtc_answer`, `update_settings`, `voice_select`, `eye_animation`

### Command Messages (processed by server)
`register`, `get_state`, `switch_mode`, `robot_action`, `play_sound`, `toggle_mic`, `joystick`, `motor`, `esp32_config`, `set_volume`, `ai_message`, `slam_control`, `follow_mode`, `sentry_mode`, `privacy_mode`

### Custom Messages
`cam_toggle` — Robot → Laptop: toggles Pi camera visibility on the laptop

---

## 7. Motor GPIO Pinout

| GPIO | Pin | Motor |
|---|---|---|
| GPIO17 (IN1) | Pin 11 | Left Motor Forward |
| GPIO27 (IN2) | Pin 13 | Left Motor Backward |
| GPIO22 (IN3) | Pin 15 | Right Motor Forward |
| GPIO23 (IN4) | Pin 16 | Right Motor Backward |

Motor endpoints: `http://<pi-ip>:8080/{F,B,L,R,S}`

---

## 8. AI / LLM Fallback Chain

```
Groq (cloud, fast) → Gemini (cloud, multimodal) → Ollama (local) → LlamaGGUF (local file)
```

- **Groq model**: `llama-3.3-70b-versatile`
- **Gemini model**: `gemini-2.0-flash`
- **Ollama model**: `gemma3:270m` (at `http://localhost:11434`)
- **GGUF path**: `models/llama-3.2-3b-instruct.Q4_K_M.gguf`

---

## 9. Common Operations

### Push to GitHub
```bash
cd d:\Main_Project\kenza\KENZA
git add .
git commit -m "commit message"
git push origin main
```
Remote: `https://github.com/AmruthIsMe/KENZA_AI.git`

### Restart Services on Pi
```bash
sudo systemctl restart kenza
# Or manually:
python3 kenza_launcher.py
```

### Start Individual Components
```bash
python3 kenza_server.py           # Main WS server + MediaMTX
python3 pi_motor_server.py        # Motor control
python3 kenza_conversation.py     # AI conversation (standalone)
./kenza.sh                         # Conversation with suppressed JACK errors
```

---

## 10. Known Gotchas

1. **MediaMTX is video-only** — it streams via `rpicam-vid | ffmpeg`. Adding `-f alsa` to the ffmpeg pipeline crashes because PulseAudio holds the ALSA device. Audio must go via the direct WebRTC connection.
2. **Camera exclusivity** — only one process can hold the Pi Camera. MediaMTX (via rpicam-vid) has priority. `kenza_stream.py` and gesture tracking cannot use the camera simultaneously.
3. **`kenza_stream.py` is NOT used in production** — `kenza_server.py` starts MediaMTX directly. The Python WHIP streamer was an earlier approach.
4. **USB Microphone via Chromium** — the Pi's robot_display.html uses `getUserMedia()` in Chromium to capture the USB mic. This works reliably with PulseAudio.
5. **Line endings** — Git warns about LF→CRLF. This is expected (Windows dev ↔ Linux Pi).
6. **`TP.localPc`** vs **`TP.pc`** — `localPc` is the direct WebRTC connection (app↔robot, bidirectional audio/video). `pc` is the WHEP connection (MediaMTX→laptop, receive-only video).

---

## 11. Voice Presets

| Name | Voice ID | Style |
|---|---|---|
| kenza | `en-US-AriaNeural` | Friendly (default) |
| glitch | `en-US-ChristopherNeural` | Robotic/Calm |
| kawaii | `en-US-AnaNeural` | Anime/Child |
| titan | `en-US-EricNeural` | Deep/Authoritative |
| jarvis | `en-GB-RyanNeural` | British Butler |

Wake word: **"kenza"**

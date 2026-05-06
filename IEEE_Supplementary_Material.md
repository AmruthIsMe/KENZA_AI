# Supplementary Material for KENZA IEEE Conference Paper

This document serves as supplementary material for the KENZA IEEE conference paper. It provides the project's README file for quick setup and usage instructions, followed by an overview of the project's hardware and software architecture.

---

## Part 1: Project README

# Kenza AI - Conversational AI Module

Voice-enabled AI assistant with smart online/offline routing.

### Features

- **Wake Word**: Say "Kenza" to activate
- **Smart Routing**: Automatically uses Gemini (online) or Llama (offline)  
- **Human-Like Voice**: Edge-TTS with natural voices
- **Conversation Memory**: Gemini maintains context across turns
- **LED Status**: GPIO LEDs show listening/thinking states

### Quick Start

**1. Install Dependencies**

*On Raspberry Pi:*
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install -r requirements.txt
```

*On Windows/Mac (for testing):*
```bash
pip install -r requirements.txt
```

**2. Configure API Key**

Edit `config/settings.yaml`:
```yaml
api_keys:
  gemini: "your-api-key-here"
```

Or set environment variable:
```bash
export GEMINI_API_KEY="your-api-key"
```

**3. Download Llama Model (for offline mode)**

```bash
# Create models directory
mkdir -p models

# Download a small model (recommended for Pi 5)
# From: https://huggingface.co/TheBloke
# Example: llama-3.2-3b-instruct.Q4_K_M.gguf
```

**4. Run Kenza**

```bash
# Voice mode with wake word
python kenza_ai.py

# Voice mode without wake word (always listening)
python kenza_ai.py --no-wake

# Text-only mode (for testing without mic)
python kenza_ai.py --text

# Test all components
python kenza_ai.py --test
```

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     kenza_ai.py                         │
├─────────────────────────────────────────────────────────┤
│  ConversationManager                                    │
│  ├── SmartRouter                                        │
│  │   ├── GeminiProvider (Online, with memory)          │
│  │   └── LlamaProvider (Offline, fast responses)       │
│  ├── SpeechToText (Google Speech Recognition)          │
│  ├── TextToSpeech (Edge-TTS, human-like voices)        │
│  └── LEDController (GPIO status indicators)            │
└─────────────────────────────────────────────────────────┘
```

### Smart Routing Logic

Llama classifies each request:
- **A (Online)**: Complex/real-time queries → Gemini
- **B (Offline)**: Simple/fast queries → Llama

### GPIO Pinout (Raspberry Pi)

| LED    | BCM Pin | Purpose           |
|--------|---------|-------------------|
| Green  | 24      | Listening/Ready   |
| Red    | 25      | Thinking/Speaking |

### Voice Options

Edge-TTS voices (configure in settings.yaml):
- `en-US-AriaNeural` - Female, natural (default)
- `en-US-GuyNeural` - Male
- `en-GB-SoniaNeural` - British female
- `en-AU-NatashaNeural` - Australian female

### Files

```
kenza/
├── kenza_ai.py           # Main conversational AI
├── config/
│   └── settings.yaml     # Configuration
├── models/               # Llama model files
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Part 2: Project Overview

### 1. High-Level System Architecture

KENZA is a sentient robotic companion system that integrates conversational artificial intelligence, computer vision, hand gesture recognition, and real-time telepresence capabilities. 

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           KENZA Robot System                            │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐ │
│  │   AI Backbone   │    │   Vision Core   │    │  Streaming Engine   │ │
│  │  (kenza_ai.py)  │    │(kenza_vision.py)│    │ (kenza_stream.py)   │ │
│  └────────┬────────┘    └────────┬────────┘    └──────────┬──────────┘ │
│           │                      │                        │            │
│  ┌────────┴────────────────────┴────────────────────────┴──────────┐  │
│  │                   Main Controller (RPI_kenza_main.py)            │  │
│  │   ├── GPIOMotorController    ├── EyeController                   │  │
│  │   ├── AudioController        ├── CommandHandler                  │  │
│  │   └── WebSocket Server       └── Robot State Machine             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2. Hardware Specifications

- **Computing Platform**: Raspberry Pi 5 (4GB/8GB RAM)
- **Motor Control**: L298N Dual H-Bridge driving 4 DC Gear Motors
- **Vision**: Raspberry Pi Camera Module 3 / USB Webcam
- **Audio**: USB Microphone & Speaker with Opus Codec and AEC (Acoustic Echo Cancellation)
- **Display**: HDMI/OLED Display for Eye Animations

### 3. Software Modules

- **AI Core (`kenza_ai.py`)**: Handles smart routing between cloud (Groq/Gemini) and local (Ollama/Llama) LLMs based on query complexity.
- **Vision System (`kenza_vision.py`)**: Uses MediaPipe for real-time face detection, recognition, and tracking.
- **Gesture System (`kenza_gesture.py`)**: MediaPipe Hands-based gesture recognition for UI control (e.g., pinch to select, open palm to navigate).
- **Streaming Engine (`kenza_stream.py`)**: WebRTC-powered bidirectional audio/video streaming with NLMS adaptive filtering for AEC.
- **WebSocket Server (`kenza_server.py`)**: Manages real-time command handling and state synchronization between the robot and web interface.

### 4. Operating Modes

- **Autonomous**: Self-directed behavior including face following, voice commands, and AI chat.
- **Remote**: Telepresence control via joystick, video streaming, and HUD.
- **Sentry**: Security patrol featuring motion detection, alerts, and recording.
- **Privacy**: Disabled sensors; local processing only.

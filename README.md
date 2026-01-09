# Kenza AI - Conversational AI Module

Voice-enabled AI assistant with smart online/offline routing.

## Features

- **Wake Word**: Say "Kenza" to activate
- **Smart Routing**: Automatically uses Gemini (online) or Llama (offline)  
- **Human-Like Voice**: Edge-TTS with natural voices
- **Conversation Memory**: Gemini maintains context across turns
- **LED Status**: GPIO LEDs show listening/thinking states

## Quick Start

### 1. Install Dependencies

**On Raspberry Pi:**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install -r requirements.txt
```

**On Windows/Mac (for testing):**
```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Edit `config/settings.yaml`:
```yaml
api_keys:
  gemini: "your-api-key-here"
```

Or set environment variable:
```bash
export GEMINI_API_KEY="your-api-key"
```

### 3. Download Llama Model (for offline mode)

```bash
# Create models directory
mkdir -p models

# Download a small model (recommended for Pi 5)
# From: https://huggingface.co/TheBloke
# Example: llama-3.2-3b-instruct.Q4_K_M.gguf
```

### 4. Run Kenza

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

## Architecture

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

## Smart Routing Logic

Llama classifies each request:
- **A (Online)**: Complex/real-time queries → Gemini
- **B (Offline)**: Simple/fast queries → Llama

## GPIO Pinout (Raspberry Pi)

| LED    | BCM Pin | Purpose           |
|--------|---------|-------------------|
| Green  | 24      | Listening/Ready   |
| Red    | 25      | Thinking/Speaking |

## Voice Options

Edge-TTS voices (configure in settings.yaml):
- `en-US-AriaNeural` - Female, natural (default)
- `en-US-GuyNeural` - Male
- `en-GB-SoniaNeural` - British female
- `en-AU-NatashaNeural` - Australian female

## Files

```
kenza/
├── kenza_ai.py           # Main conversational AI
├── config/
│   └── settings.yaml     # Configuration
├── models/               # Llama model files
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

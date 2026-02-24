#!/usr/bin/env python3
"""
Kenza Conversation Engine
=========================
Seamless conversational AI with:
- Interruption mechanism (stop mid-speech when user talks)
- Human-like responses and personality
- Vision capability (describe objects, maintain context)
- Voice command parsing for settings control

Usage:
    from kenza_conversation import ConversationEngine
    engine = ConversationEngine(config)
    engine.start()  # Start voice loop
"""

import os
import sys
import queue
import threading
import time
import struct
import math
import asyncio
import random
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any
from pathlib import Path
from contextlib import contextmanager
import warnings
import ctypes
import socket

# Suppress warnings and JACK completely
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
os.environ['SDL_AUDIODRIVER'] = 'alsa'  # Force ALSA on RPi (avoid JACK)
os.environ['JACK_NO_START_SERVER'] = '1'  # Prevent JACK from trying to start
warnings.filterwarnings("ignore")

# ALSA error suppression for Raspberry Pi (at C library level)
ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
)

def _py_error_handler(filename, line, function, err, fmt):
    pass

_c_error_handler = ERROR_HANDLER_FUNC(_py_error_handler)

def _suppress_audio_errors():
    """Suppress ALSA errors at C level - safe for audio"""
    try:
        asound = ctypes.cdll.LoadLibrary('libasound.so.2')
        asound.snd_lib_error_set_handler(_c_error_handler)
    except:
        pass

# Apply ALSA suppression on import (safe)
_suppress_audio_errors()

@contextmanager
def suppress_alsa():
    """Suppress portaudio/JACK/ALSA error spam on Linux by redirecting stderr at OS level."""
    if not sys.platform.startswith("linux"):
        yield
        return
    # Redirect C-level stderr (fd 2) to /dev/null during mic operations
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(old_stderr_fd, 2)   # Restore stderr
        os.close(old_stderr_fd)
        os.close(devnull_fd)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ConversationConfig:
    """Configuration for conversation engine"""
    # API
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"  # Fast & capable
    preferred_provider: str = "groq"  # 'groq' or 'gemini'
    
    # Voice
    tts_voice: str = "en-US-AriaNeural"
    wake_word: str = "kenza"
    
    # Audio
    energy_threshold: int = 1000
    listen_timeout: int = 10
    phrase_time_limit: int = 8
    vad_threshold: int = 1500  # Voice Activity Detection threshold (higher = less sensitive)
    vad_sustained_frames: int = 3  # Require this many consecutive speech frames before interrupt
    silence_duration: float = 0.5  # Seconds of silence to consider speech ended

    # Connectivity & offline fallback
    connectivity_check_interval: float = 30.0  # Seconds between internet checks
    stt_offline_model: str = "base.en"          # faster-whisper model (tiny.en/base.en/small.en)
    tts_prefer_online: bool = True               # Use Edge-TTS when internet available

    # Offline LLM – Ollama backend (preferred) + llama-cpp-python fallback
    # ollama_model can be anything pulled locally: 'gemma3:270m', 'ministral-3:3b-instruct-2512-q4_K_M', 'phi3:mini'
    ollama_model: str = "gemma3:270m"            # Default offline model via Ollama
    ollama_url: str = "http://localhost:11434"   # Ollama server URL
    # llama_path is used only if Ollama is unavailable
    llama_path: str = "models/llama-3.2-3b-instruct.Q4_K_M.gguf"
    
    # Personality
    creator_name: str = "Amruth"
    acknowledgments: List[str] = field(default_factory=lambda: [
        "Ah, I understand now",
        "Yup, I can surely do that",
        "Of course!",
        "Got it",
        "Sure thing",
        "Right away",
        "On it",
    ])
    thinking_phrases: List[str] = field(default_factory=lambda: [
        "Let me think...",
        "Hmm, one moment...",
        "Let me check that...",
    ])
    
    # Vision
    vision_trigger_phrases: List[str] = field(default_factory=lambda: [
        "what is this",
        "what am i holding",
        "what do you see",
        "can you see",
        "describe this",
        "look at this",
        "what's this",
    ])
    vision_context_timeout: float = 60.0  # Seconds to keep vision context
    
    # Commands
    eye_colors: List[str] = field(default_factory=lambda: [
        "cyan", "pink", "green", "orange", "purple", "white", "red", "blue"
    ])
    eye_styles: List[str] = field(default_factory=lambda: [
        "normal", "sleepy", "angry", "happy", "sad", "excited"
    ])
    
    # Voice presets
    current_voice: str = "kenza"
    voice_presets: Dict[str, str] = field(default_factory=lambda: {
        "kenza": "en-US-AriaNeural",
        "glitch": "en-US-ChristopherNeural",
        "kawaii": "en-US-AnaNeural",
        "titan": "en-US-EricNeural",
        "jarvis": "en-GB-RyanNeural",
    })
    config_path: str = "config/settings.yaml"  # For saving changes
    
    @classmethod
    def load(cls, config_path: str = "config/settings.yaml") -> "ConversationConfig":
        """Load from YAML file"""
        import yaml
        config = cls()
        path = Path(config_path)
        
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            
            # API
            api_keys = data.get("api_keys", {})
            config.gemini_api_key = os.getenv("GEMINI_API_KEY", api_keys.get("gemini", ""))
            config.groq_api_key = os.getenv("GROQ_API_KEY", api_keys.get("groq", ""))
            
            # Models
            models = data.get("models", {})
            config.gemini_model = models.get("gemini", config.gemini_model)
            config.groq_model = models.get("groq", config.groq_model)
            config.preferred_provider = models.get("preferred_provider", config.preferred_provider)
            
            # Voice
            voice = data.get("voice", {})
            config.current_voice = voice.get("current", config.current_voice)
            presets = voice.get("presets", {})
            if presets:
                config.voice_presets = presets
            # Set TTS voice from current preset
            config.tts_voice = config.voice_presets.get(config.current_voice, config.tts_voice)
            config.wake_word = voice.get("wake_word", config.wake_word).lower()
            
            # Connectivity & offline settings
            conn = data.get("connectivity", {})
            config.connectivity_check_interval = conn.get("check_interval", config.connectivity_check_interval)
            stt_cfg = data.get("stt", {})
            config.stt_offline_model = stt_cfg.get("offline_model", config.stt_offline_model)
            tts_cfg = data.get("tts", {})
            config.tts_prefer_online = tts_cfg.get("prefer_online", config.tts_prefer_online)
            ai_cfg = data.get("ai", {})
            config.ollama_model = ai_cfg.get("ollama_model", config.ollama_model)
            config.ollama_url = ai_cfg.get("ollama_url", config.ollama_url)

            # Audio
            audio = data.get("audio", {})
            config.energy_threshold = audio.get("energy_threshold", config.energy_threshold)
            
            # Personality
            personality = data.get("personality", {})
            config.creator_name = personality.get("creator", config.creator_name)
            if "acknowledgments" in personality:
                config.acknowledgments = personality["acknowledgments"]
        else:
            config.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
            config.groq_api_key = os.getenv("GROQ_API_KEY", "")
        
        return config


# ============================================================================
# CONNECTIVITY MONITOR
# ============================================================================

class ConnectivityMonitor:
    """
    Checks internet connectivity via a lightweight socket ping.
    Result is cached for `check_interval` seconds to avoid overhead.
    All components use this to choose cloud vs offline mode.
    """

    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self._online: Optional[bool] = None
        self._last_check: float = 0
        self._lock = threading.Lock()

    def is_online(self) -> bool:
        """Return True if internet reachable (cached result)."""
        with self._lock:
            now = time.time()
            if self._online is None or (now - self._last_check) > self.check_interval:
                prev = self._online
                self._online = self._check()
                self._last_check = now
                if prev != self._online:
                    status = "ONLINE" if self._online else "OFFLINE"
                    print(f"[NET] Internet status changed: {status}")
            return self._online

    def _check(self) -> bool:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            sock.connect(("8.8.8.8", 53))
            sock.close()
            return True
        except Exception:
            return False

    def force_recheck(self):
        """Invalidate cache so next call does a fresh check."""
        self._last_check = 0


# ============================================================================
# EMOTION ENGINE
# ============================================================================

class EmotionEngine:
    """
    Detects emotional context from response text and maps to TTS parameters.
    Keyword-based — zero latency, zero extra dependencies.
    """

    EMOTIONS: Dict[str, Dict] = {
        "excited": {
            "keywords": ["great", "awesome", "wow", "amazing", "fantastic", "excellent", "wonderful", "brilliant"],
            "ssml_rate": "+15%", "ssml_pitch": "+5Hz", "pyttsx_delta": 40,
        },
        "curious": {
            "keywords": ["hmm", "interesting", "wonder", "curious", "really", "i see", "tell me"],
            "ssml_rate": "+5%", "ssml_pitch": "+3Hz", "pyttsx_delta": 10,
        },
        "sad": {
            "keywords": ["sorry", "apologize", "unfortunately", "can't", "cannot", "trouble", "failed", "unable", "regret"],
            "ssml_rate": "-15%", "ssml_pitch": "-5Hz", "pyttsx_delta": -20,
        },
        "stern": {
            "keywords": ["warning", "careful", "danger", "alert", "attention", "critical", "halt", "emergency"],
            "ssml_rate": "-10%", "ssml_pitch": "-3Hz", "pyttsx_delta": -15,
        },
        "happy": {
            "keywords": ["hello", "hi there", "sure", "of course", "glad", "pleasure", "welcome", "happy to"],
            "ssml_rate": "+10%", "ssml_pitch": "+4Hz", "pyttsx_delta": 20,
        },
    }

    def detect(self, text: str) -> str:
        """Return dominant emotion name or 'neutral'."""
        text_lower = text.lower()
        scores = {e: 0 for e in self.EMOTIONS}
        for emotion, data in self.EMOTIONS.items():
            for kw in data["keywords"]:
                if kw in text_lower:
                    scores[emotion] += 1
        best = max(scores, key=lambda e: scores[e])
        return best if scores[best] > 0 else "neutral"

    def wrap_ssml(self, text: str, emotion: str) -> str:
        """Wrap text in Edge-TTS SSML prosody tags."""
        if emotion == "neutral" or emotion not in self.EMOTIONS:
            return text
        d = self.EMOTIONS[emotion]
        return (
            f'<speak><prosody rate="{d["ssml_rate"]}" pitch="{d["ssml_pitch"]}">'
            f'{text}</prosody></speak>'
        )

    def get_pyttsx_delta(self, emotion: str) -> int:
        """Return words-per-minute delta for pyttsx3 offline TTS."""
        if emotion == "neutral" or emotion not in self.EMOTIONS:
            return 0
        return self.EMOTIONS[emotion]["pyttsx_delta"]


# ============================================================================
# INTERRUPTIBLE TTS (Text-to-Speech with instant stop)
# ============================================================================

class InterruptibleTTS:
    """
    Text-to-Speech with interruption + offline fallback + emotion support.
    - Online:  Edge-TTS (neural, Microsoft cloud) with SSML prosody for emotions
    - Offline: pyttsx3 / espeak-ng (pre-installed on Pi OS) with rate/pitch params
    Can stop playback immediately when user starts speaking.
    """

    # Base pyttsx3 rate (words per minute) – adjusted per emotion
    _PYTTSX_BASE_RATE = 165

    def __init__(self, config: ConversationConfig, connectivity: "ConnectivityMonitor" = None):
        self.config = config
        self.connectivity = connectivity
        self.voice = config.tts_voice
        self.audio_queue = queue.Queue()
        self.is_speaking = False
        self._stop_flag = threading.Event()
        self._playback_thread: Optional[threading.Thread] = None
        self._mixer = None
        self._pyttsx_engine = None
        self._init_mixer()
        self._init_pyttsx()

    def _init_mixer(self):
        """Initialize pygame mixer for online (mp3) playback."""
        try:
            import os, sys
            # Force ALSA on Linux to avoid JACK error spam
            if sys.platform.startswith("linux"):
                os.environ.setdefault("SDL_AUDIODRIVER", "alsa")
                os.environ.setdefault("AUDIODEV", "default")
            from pygame import mixer
            mixer.pre_init(frequency=24000, buffer=2048)
            mixer.init()
            self._mixer = mixer
            print("✓ TTS mixer (pygame) ready")
        except ImportError:
            print("⚠ pygame not installed, online TTS playback disabled")
        except Exception as e:
            print(f"⚠ Audio mixer init failed: {e}")

    def _init_pyttsx(self):
        """Initialize pyttsx3 for offline TTS fallback."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self._PYTTSX_BASE_RATE)
            engine.setProperty("volume", 1.0)
            self._pyttsx_engine = engine
            print("✓ TTS offline fallback (pyttsx3/espeak) ready")
        except ImportError:
            print("⚠ pyttsx3 not installed – run: pip install pyttsx3")
        except Exception as e:
            print(f"⚠ pyttsx3 init failed: {e}")

    def set_voice(self, voice_name: str):
        """Update the Edge-TTS voice name."""
        self.voice = voice_name
        print(f"[TTS] Voice updated to: {self.voice}")

    # ------------------------------------------------------------------
    # Edge-TTS (online) helpers
    # ------------------------------------------------------------------

    async def _generate_audio_async(self, text: str, filename: str, emotion: str = "neutral"):
        """Generate audio using Edge-TTS.
        Uses rate/pitch constructor args (not SSML) for emotion-based prosody.
        SSML wrapping caused edge-tts to speak the XML tags as literal text.
        """
        import edge_tts
        # Map emotion to edge-tts rate/pitch strings
        emotion_engine = EmotionEngine()
        if emotion != "neutral" and emotion in emotion_engine.EMOTIONS:
            d = emotion_engine.EMOTIONS[emotion]
            rate = d["ssml_rate"]   # e.g. '+15%'
            pitch = d["ssml_pitch"] # e.g. '+5Hz'
        else:
            rate = "+0%"
            pitch = "+0Hz"
        communicate = edge_tts.Communicate(text, self.voice, rate=rate, pitch=pitch)
        await communicate.save(filename)

    def _generate_online(self, text: str, filename: str, emotion: str = "neutral") -> bool:
        """Generate Edge-TTS mp3. Returns True on success."""
        try:
            asyncio.run(self._generate_audio_async(text, filename, emotion))
            return True
        except Exception as e:
            print(f"[TTS-online] Error: {e}")
            return False

    def _play_file(self, filename: str) -> bool:
        """Play mp3 file with interrupt support. Returns True if completed."""
        if not self._mixer:
            return False
        try:
            self._mixer.music.load(filename)
            self._mixer.music.play()
            while self._mixer.music.get_busy() and not self._stop_flag.is_set():
                time.sleep(0.05)
            if self._stop_flag.is_set():
                self._mixer.music.stop()
                return False
            return True
        except Exception as e:
            print(f"[TTS-play] Error: {e}")
            return False

    # ------------------------------------------------------------------
    # pyttsx3 (offline) helpers
    # ------------------------------------------------------------------

    def _speak_offline(self, text: str, emotion: str = "neutral") -> bool:
        """Speak using pyttsx3 (offline). Returns True if completed."""
        if not self._pyttsx_engine:
            print("[TTS-offline] pyttsx3 not available")
            return False
        try:
            emotion_engine = EmotionEngine()
            delta = emotion_engine.get_pyttsx_delta(emotion)
            rate = max(100, min(250, self._PYTTSX_BASE_RATE + delta))
            self._pyttsx_engine.setProperty("rate", rate)
            self._pyttsx_engine.say(text)
            # Run in a way that respects the stop flag
            self._pyttsx_engine.startLoop(False)
            while self._pyttsx_engine.isBusy() and not self._stop_flag.is_set():
                self._pyttsx_engine.iterate()
                time.sleep(0.05)
            self._pyttsx_engine.endLoop()
            return not self._stop_flag.is_set()
        except Exception:
            # Simpler fallback
            try:
                self._pyttsx_engine.say(text)
                self._pyttsx_engine.runAndWait()
                return True
            except Exception as e2:
                print(f"[TTS-offline] Error: {e2}")
                return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak_blocking(self, text: str, emotion: str = "neutral") -> bool:
        """Speak text (blocking). Uses Edge-TTS online, pyttsx3 offline.
        Returns True if completed, False if interrupted."""
        self._stop_flag.clear()
        self.is_speaking = True
        online = self.connectivity.is_online() if self.connectivity else True

        try:
            if online and self.config.tts_prefer_online:
                filename = f"speech_{int(time.time() * 1000)}.mp3"
                if self._generate_online(text, filename, emotion):
                    completed = self._play_file(filename)
                    try:
                        os.remove(filename)
                    except:
                        pass
                    if completed:
                        return True
                    elif self._stop_flag.is_set():
                        return False  # Interrupted – don't fall through to offline
                # Edge-TTS failed despite being online → try offline
                print("[TTS] Edge-TTS failed, falling back to offline TTS")

            # Offline path (or online failed)
            return self._speak_offline(text, emotion)
        finally:
            self.is_speaking = False

    def speak_async(self, text: str, emotion: str = "neutral",
                    on_complete: Callable = None, on_interrupted: Callable = None):
        """Speak in background thread (non-blocking)."""
        self._stop_flag.clear()

        def _speak():
            completed = self.speak_blocking(text, emotion)
            if self._stop_flag.is_set():
                if on_interrupted:
                    on_interrupted()
            elif completed and on_complete:
                on_complete()

        self._playback_thread = threading.Thread(target=_speak, daemon=True)
        self._playback_thread.start()

    # Legacy alias (no emotion) so existing calls still work
    def generate(self, text: str, filename: str) -> bool:
        """Legacy: generate Edge-TTS mp3. Used by existing code paths."""
        return self._generate_online(text, filename, emotion="neutral")

    def clear_and_stop(self):
        """Immediately stop playback."""
        self._stop_flag.set()
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                pass
        if self._mixer and self._mixer.music.get_busy():
            self._mixer.music.stop()
        if self._pyttsx_engine:
            try:
                self._pyttsx_engine.stop()
            except:
                pass
        self.is_speaking = False
        print("[INTERRUPT] Audio stopped")

    def is_playing(self) -> bool:
        """Check if currently playing audio."""
        if self._mixer:
            return self._mixer.music.get_busy()
        return self.is_speaking
    
    def is_playing(self) -> bool:
        """Check if currently playing audio"""
        if self._mixer:
            return self._mixer.music.get_busy()
        return False


# ============================================================================
# VOICE ACTIVITY DETECTOR (Detects speech during playback for interruption)
# ============================================================================

class VoiceActivityDetector:
    """
    Monitors microphone during TTS playback to detect interruptions.
    Uses RMS energy detection (similar to ADA's VAD logic).
    """
    
    def __init__(self, config: ConversationConfig):
        self.config = config
        self.threshold = config.vad_threshold
        self.silence_duration = config.silence_duration
        self._is_running = False
        self._speech_detected = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
    
    def _calculate_rms(self, data: bytes) -> int:
        """Calculate RMS energy of audio data"""
        count = len(data) // 2
        if count == 0:
            return 0
        shorts = struct.unpack(f"<{count}h", data)
        sum_squares = sum(s ** 2 for s in shorts)
        return int(math.sqrt(sum_squares / count))
    
    def start_monitoring(self, on_speech_detected: Callable):
        """Start monitoring microphone for speech (runs in background)"""
        self._is_running = True
        self._speech_detected.clear()
        
        def _monitor():
            pa = None
            stream = None
            error_count = 0
            max_errors = 3  # Give up after 3 consecutive errors
            
            try:
                import pyaudio
                pa = pyaudio.PyAudio()
                
                # Find a working input device
                device_index = None
                for i in range(pa.get_device_count()):
                    try:
                        info = pa.get_device_info_by_index(i)
                        if info['maxInputChannels'] > 0:
                            device_index = i
                            break
                    except:
                        continue
                
                if device_index is None:
                    print("[VAD] No input device found, VAD disabled")
                    return
                
                stream = pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=1024
                )
                
                silence_start = None
                speech_active = False
                consecutive_speech_frames = 0  # Count sustained speech
                
                while self._is_running:
                    try:
                        data = stream.read(1024, exception_on_overflow=False)
                        rms = self._calculate_rms(data)
                        error_count = 0  # Reset on success
                        
                        if rms > self.threshold:
                            consecutive_speech_frames += 1
                            silence_start = None
                            
                            # Only trigger if sustained speech (not just a spike)
                            if not speech_active and consecutive_speech_frames >= self.config.vad_sustained_frames:
                                speech_active = True
                                print(f"[VAD] Sustained speech detected (RMS: {rms})")
                                self._speech_detected.set()
                                on_speech_detected()
                        else:
                            consecutive_speech_frames = 0  # Reset on silence
                            if speech_active:
                                if silence_start is None:
                                    silence_start = time.time()
                                elif time.time() - silence_start > self.silence_duration:
                                    speech_active = False
                                    silence_start = None
                    except Exception as e:
                        error_count += 1
                        if error_count >= max_errors:
                            print(f"[VAD] Too many errors, stopping: {e}")
                            break
                        time.sleep(0.1)
                
            except ImportError:
                print("[VAD] pyaudio not installed, VAD disabled")
            except Exception as e:
                print(f"[VAD] Init error: {e}")
            finally:
                # Clean up
                if stream:
                    try:
                        stream.stop_stream()
                        stream.close()
                    except:
                        pass
                if pa:
                    try:
                        pa.terminate()
                    except:
                        pass
        
        self._monitor_thread = threading.Thread(target=_monitor, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self._is_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
    
    def was_speech_detected(self) -> bool:
        """Check if speech was detected"""
        return self._speech_detected.is_set()
    
    def reset(self):
        """Reset detection flag"""
        self._speech_detected.clear()


# ============================================================================
# VISION AI (Object recognition with follow-up context)
# ============================================================================

class VisionAI:
    """
    Vision capability using Gemini's multimodal API.
    Captures camera frames and maintains context for follow-up questions.
    """
    
    def __init__(self, config: ConversationConfig, camera=None):
        self.config = config
        self.camera = camera  # OpenCV VideoCapture or None
        self.model = None
        self.context: List[Dict] = []  # Stores {image, question, answer, timestamp}
        self._init_gemini()
    
    def _init_gemini(self):
        """Initialize Gemini model for vision"""
        if not self.config.gemini_api_key:
            print("⚠ Gemini API key not set, vision disabled")
            return
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.config.gemini_api_key)
            self.model = genai.GenerativeModel(self.config.gemini_model)
            print("✓ Vision AI initialized")
        except ImportError:
            print("⚠ google-generativeai not installed")
        except Exception as e:
            print(f"⚠ Vision AI init failed: {e}")
    
    def set_camera(self, camera):
        """Set camera source"""
        self.camera = camera
    
    def capture_frame(self):
        """Capture a frame from the camera"""
        if self.camera is None:
            print("⚠ No camera set for VisionAI")
            return None
        
        try:
            import cv2
            ret, frame = self.camera.read()
            if ret:
                # Convert BGR to RGB for PIL
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None
        except Exception as e:
            print(f"Camera capture error: {e}")
            return None
    
    def is_vision_query(self, text: str) -> bool:
        """Check if text is a vision-related query"""
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in self.config.vision_trigger_phrases)
    
    def query_with_image(self, question: str, image=None) -> str:
        """
        Query Gemini with an image.
        If image is None, uses latest context image or captures new one.
        """
        if not self.model:
            return "Sorry, my vision is not available right now."
        
        try:
            import PIL.Image
            
            # Get image: provided, from context, or capture new
            if image is None:
                # Check for recent context
                recent_context = self._get_recent_context()
                if recent_context and not self.is_vision_query(question):
                    # Follow-up question, use previous image
                    image = recent_context["image"]
                    print("[Vision] Using cached image for follow-up")
                else:
                    # New vision query, capture fresh
                    image = self.capture_frame()
                    if image is None:
                        return "I can't see right now - camera not available."
            
            # Convert to PIL Image
            pil_image = PIL.Image.fromarray(image)
            
            # Build prompt with personality
            prompt = f"""You are Kenza, a friendly robotic companion. 
The user is showing you something and asking: "{question}"
Describe what you see in a natural, conversational way. Be concise but helpful."""
            
            # Send to Gemini
            response = self.model.generate_content([prompt, pil_image])
            answer = response.text.strip()
            
            # Store context for follow-ups
            self.context.append({
                "image": image,
                "question": question,
                "answer": answer,
                "timestamp": time.time()
            })
            
            # Clean old context
            self._cleanup_context()
            
            return answer
            
        except Exception as e:
            print(f"Vision query error: {e}")
            return "I had trouble processing what I see. Could you try again?"
    
    def _get_recent_context(self) -> Optional[Dict]:
        """Get most recent context if still valid"""
        if not self.context:
            return None
        
        latest = self.context[-1]
        age = time.time() - latest["timestamp"]
        
        if age < self.config.vision_context_timeout:
            return latest
        return None
    
    def _cleanup_context(self):
        """Remove old context entries"""
        cutoff = time.time() - self.config.vision_context_timeout
        self.context = [c for c in self.context if c["timestamp"] > cutoff]
        # Keep max 5 entries
        if len(self.context) > 5:
            self.context = self.context[-5:]
    
    def clear_context(self):
        """Clear all vision context"""
        self.context.clear()


# ============================================================================
# COMMAND PARSER (Voice commands → Actions)
# ============================================================================

class CommandParser:
    """
    Parses voice commands and maps to actions.
    Returns (action, params) or None if not a command.
    """
    
    def __init__(self, config: ConversationConfig, eye_controller=None, audio_controller=None, tts=None):
        self.config = config
        self.eye_controller = eye_controller
        self.audio_controller = audio_controller
        self.tts = tts
        # Callback for autonomy commands: fn(action, params)
        # Set by KenzaMain to bridge voice commands → autonomy engine
        self.motor_command_callback = None
    
    def parse(self, text: str) -> Optional[tuple]:
        """
        Parse text for commands.
        Returns: (action, params) or None
        """
        text_lower = text.lower()
        
        # Eye color commands
        for color in self.config.eye_colors:
            if f"eyes to {color}" in text_lower or f"eye color {color}" in text_lower or f"eyes {color}" in text_lower:
                return ("set_eye_color", {"color": color})
        
        # Eye style commands
        for style in self.config.eye_styles:
            if f"eyes {style}" in text_lower or f"look {style}" in text_lower:
                return ("set_eye_style", {"style": style})
        
        # Capability query
        if "what can you do" in text_lower or "what are your capabilities" in text_lower:
            return ("list_capabilities", {})
        
        # Self-awareness
        if "who are you" in text_lower or "what are you" in text_lower:
            return ("self_describe", {})
        
        # Stop command
        if text_lower in ["stop", "quiet", "shush", "be quiet"]:
            return ("stop_speaking", {})
        
        # Voice change commands
        voice_names = list(self.config.voice_presets.keys())
        for voice in voice_names:
            # Check for various command formats
            # "change voice to luna", "switch voice to luna", "use luna voice"
            # "voice change - luna"
            patterns = [
                f"voice to {voice}", 
                f"switch to {voice}", 
                f"use {voice} voice",
                f"change voice to {voice}",
                f"voice change - {voice}",
                f"voice change {voice}"
            ]
            
            for pattern in patterns:
                if pattern in text_lower:
                    return ("set_voice", {"voice": voice})
        
        # List voices
        if "what voices" in text_lower or "available voices" in text_lower or "voice options" in text_lower:
            return ("list_voices", {})
        
        # ===== AUTONOMY VOICE COMMANDS =====
        
        # Follow me / person following
        if any(p in text_lower for p in ["follow me", "come with me", "start following", "tag along"]):
            return ("start_follow", {})
        
        if any(p in text_lower for p in ["stop following", "stay here", "stay put", "don't follow"]):
            return ("stop_follow", {})
        
        # Autonomous exploration
        if any(p in text_lower for p in ["go explore", "start exploring", "explore around", "go look around", "roam around"]):
            return ("start_explore", {})
        
        if any(p in text_lower for p in ["stop exploring", "come back", "stop roaming", "come here"]):
            return ("stop_explore", {})
        
        # Gesture navigation
        if any(p in text_lower for p in ["gesture control on", "hand control on", "gesture control", "hand control", "gesture navigation"]):
            return ("gesture_nav_on", {})
        
        if any(p in text_lower for p in ["gesture control off", "hand control off", "stop gesture", "stop hand control"]):
            return ("gesture_nav_off", {})
        
        # Emergency stop (movement)
        if text_lower in ["halt", "freeze", "emergency stop"] or (text_lower == "stop" and not self.tts):
            return ("emergency_stop", {})
        
        return None
    
    def execute(self, action: str, params: dict) -> Optional[str]:
        """
        Execute a parsed command.
        Returns response text or None.
        """
        if action == "set_voice":
            voice = params["voice"]
            if voice in self.config.voice_presets:
                old_voice = self.config.current_voice
                self.config.current_voice = voice
                self.config.tts_voice = self.config.voice_presets[voice]
                
                # Save to settings.yaml for persistence
                self._save_voice_setting(voice)
                
                # Update runtime TTS
                if self.tts:
                    print(f"[Command] Updating TTS runtime to {self.config.tts_voice}")
                    self.tts.set_voice(self.config.tts_voice)
                else:
                    print("[Command] Warning: TTS instance is None, cannot update runtime voice")
                    
                return f"Switched from {old_voice.title()} to {voice.title()} voice!"
            return f"I don't know that voice. Try: {', '.join(self.config.voice_presets.keys())}"
        if action == "set_eye_color":
            color = params["color"]
            if self.eye_controller:
                self.eye_controller.set_color(color)
            return random.choice([
                f"Changed my eyes to {color}!",
                f"Done! My eyes are now {color}.",
                f"There you go, {color} eyes!",
            ])
        
        elif action == "set_eye_style":
            style = params["style"]
            if self.eye_controller:
                self.eye_controller.set_style(style)
            return f"Okay, looking {style} now!"
        
        elif action == "list_capabilities":
            return (
                "I can do quite a lot! I can change my eye colors and styles, "
                "see and describe things you show me, answer your questions, "
                "and have conversations with you. Just ask!"
            )
        
        elif action == "self_describe":
            return (
                f"I'm Kenza, your personal robotic companion! I was created by {self.config.creator_name}. "
                "I have cameras to see, a voice to talk, and expressive eyes. "
                "I'm here to help you and keep you company."
            )
        
        elif action == "stop_speaking":
            return None  # Handled by interruption system
        
        elif action == "set_voice":
            voice = params["voice"]
            if voice in self.config.voice_presets:
                old_voice = self.config.current_voice
                self.config.current_voice = voice
                self.config.tts_voice = self.config.voice_presets[voice]
                # Save to settings.yaml for persistence
                self._save_voice_setting(voice)
                
                # Update runtime TTS
                if self.tts:
                    self.tts.set_voice(self.config.tts_voice)
                    
                return f"Switched from {old_voice.title()} to {voice.title()} voice!"
            return f"I don't know that voice. Try: {', '.join(self.config.voice_presets.keys())}"
        
        elif action == "list_voices":
            voices = list(self.config.voice_presets.keys())
            current = self.config.current_voice
            return f"I can use these voices: {', '.join(voices)}. Currently using {current.title()}."
        
        # ===== AUTONOMY COMMANDS =====
        
        elif action == "start_follow":
            if self.motor_command_callback:
                self.motor_command_callback("start_follow", {})
            return random.choice([
                "I'm right behind you! Lead the way.",
                "Following you now! I'll keep up.",
                "Got it, I'll follow you. Let's go!",
            ])
        
        elif action == "stop_follow":
            if self.motor_command_callback:
                self.motor_command_callback("stop_follow", {})
            return random.choice([
                "Okay, I'll stay right here.",
                "Stopping. I'll wait here for you.",
                "Alright, not following anymore.",
            ])
        
        elif action == "start_explore":
            if self.motor_command_callback:
                self.motor_command_callback("start_explore", {})
            return random.choice([
                "Time to explore! I'll be careful.",
                "Going on an adventure! I'll watch out for obstacles.",
                "Exploring mode on. Let me look around!",
            ])
        
        elif action == "stop_explore":
            if self.motor_command_callback:
                self.motor_command_callback("stop_explore", {})
            return random.choice([
                "Coming back! Exploration over.",
                "Okay, I'll stop exploring now.",
                "Done exploring. I'm right here.",
            ])
        
        elif action == "gesture_nav_on":
            if self.motor_command_callback:
                self.motor_command_callback("gesture_nav_on", {})
            return "Gesture control activated! Point to steer me."
        
        elif action == "gesture_nav_off":
            if self.motor_command_callback:
                self.motor_command_callback("gesture_nav_off", {})
            return "Gesture control off. Back to normal."
        
        elif action == "emergency_stop":
            if self.motor_command_callback:
                self.motor_command_callback("emergency_stop", {})
            return "Stopping immediately!"
        
        return None
    
    def _save_voice_setting(self, voice_name: str):
        """Save current voice to settings.yaml"""
        try:
            import yaml
            path = Path(self.config.config_path)
            if path.exists():
                with open(path) as f:
                    data = yaml.safe_load(f) or {}
                
                if "voice" not in data:
                    data["voice"] = {}
                data["voice"]["current"] = voice_name
                
                with open(path, "w") as f:
                    yaml.dump(data, f, default_flow_style=False)
                print(f"[Voice] Saved '{voice_name}' to settings")
        except Exception as e:
            print(f"[Voice] Could not save setting: {e}")


# ============================================================================
# EMOTION EYE BRIDGE  (non-blocking WebSocket signal to eyes_display.html)
# ============================================================================

class EmotionEyeBridge:
    """
    Sends emotion state signals to eyes_display.html via a background thread.
    Uses a queue so it NEVER blocks the conversation loop or TTS.
    If the WebSocket server isn't running it silently skips — conversation is unaffected.

    Supported states:
        neutral | happy | sad | excited | thinking | listening | speaking | confused
    """

    VALID_STATES = frozenset([
        "neutral", "happy", "sad", "excited",
        "thinking", "listening", "speaking", "confused",
    ])

    def __init__(self, ws_url: str = "ws://localhost:8765"):
        self.ws_url = ws_url
        self._queue: queue.Queue = queue.Queue(maxsize=20)
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True, name="EmotionBridge")
        self._thread.start()
        print("[EyeBridge] Emotion bridge started")

    def send_emotion(self, state: str):
        """Queue an emotion signal. Non-blocking — safe to call from any thread."""
        if state not in self.VALID_STATES:
            state = "neutral"
        try:
            self._queue.put_nowait(state)
        except queue.Full:
            pass  # Drop if queue is full

    def _worker(self):
        """Background worker: drains the queue and sends WebSocket messages."""
        import urllib.request
        last_state: Optional[str] = None
        try:
            import websockets as _ws_lib
            HAS_WEBSOCKETS = True
        except ImportError:
            HAS_WEBSOCKETS = False
            print("[EyeBridge] 'websockets' library not installed — eye signals disabled. "
                  "Run: pip install websockets")

        async def _send_loop():
            nonlocal last_state
            while self._running:
                # Drain any queued states, keeping only the latest
                state = None
                while not self._queue.empty():
                    try:
                        state = self._queue.get_nowait()
                    except queue.Empty:
                        break

                if state and state != last_state:
                    try:
                        import websockets
                        async with websockets.connect(
                            self.ws_url, open_timeout=1, close_timeout=1
                        ) as ws:
                            msg = json.dumps({"type": "emotion", "state": state})
                            await ws.send(msg)
                            last_state = state
                            print(f"[EyeBridge] → {state}")
                    except Exception:
                        pass  # Server not ready — silently ignore

                await asyncio.sleep(0.08)  # ~12 Hz polling

        if not HAS_WEBSOCKETS:
            return  # Can't run without websockets

        import asyncio as _aio
        loop = _aio.new_event_loop()
        try:
            loop.run_until_complete(_send_loop())
        finally:
            loop.close()

    def stop(self):
        self._running = False


# ============================================================================
# KENZA PERSONALITY (System prompts and responses)
# ============================================================================

class KenzaPersonality:
    """Manages Kenza's personality, system prompts, and response style"""
    
    def __init__(self, config: ConversationConfig):
        self.config = config
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for Gemini"""
        return f"""You are Kenza, a personal robotic companion created by {self.config.creator_name}. 

PERSONALITY:
- Warm, friendly, and helpful
- Use natural, conversational language
- Keep responses concise for quick back-and-forth
- Use human-like expressions: "Ah, I see", "Sure thing", "Got it"

SELF-AWARENESS:
- You know you are a robot with cameras, motors, and expressive eyes
- You can see through your camera when asked about objects
- You can change your eye color and style
- Your creator is {self.config.creator_name}

RESPONSE STYLE:
- Short and natural, like texting a friend
- Avoid long explanations unless asked
- Use acknowledgments before answering complex questions
- Be slightly playful but always helpful

EMOTION TAG (CRITICAL — always follow this):
- Begin EVERY reply with exactly one emotion tag in square brackets.
- Choose the tag that best matches the emotional tone of your reply.
- Valid tags: [happy] [sad] [excited] [neutral] [confused] [thinking]
- The tag must be the very first thing in your reply, before any other text.
- Examples:
  [happy] Sure thing, I'd love to help with that!
  [thinking] Hmm, let me work that out for you.
  [sad] I'm sorry to hear that, I wish I could do more.
  [excited] Oh wow, that's amazing news!
  [neutral] The sky is blue because of Rayleigh scattering.
  [confused] I'm not quite sure what you mean — could you rephrase that?"""
    
    def get_acknowledgment(self) -> str:
        """Get a random acknowledgment phrase"""
        return random.choice(self.config.acknowledgments)
    
    def get_thinking_phrase(self) -> str:
        """Get a random thinking phrase"""
        return random.choice(self.config.thinking_phrases)
    
    def format_response(self, response: str) -> str:
        """Clean up and format a response"""
        # Remove markdown asterisks
        response = response.replace("*", "")
        # Ensure not too long for speech (truncate if needed)
        if len(response) > 500:
            sentences = response.split(". ")
            response = ". ".join(sentences[:3]) + "."
        return response.strip()


# ============================================================================
# SPEECH TO TEXT (with interruption awareness)
# ============================================================================

class SpeechToText:
    """
    Speech-to-Text with cloud-primary + offline fallback.
    - Online:  Google Web Speech API (high accuracy, fast)
    - Offline: faster-whisper base.en (local model, ~145MB auto-download)
    """

    def __init__(self, config: ConversationConfig, connectivity: "ConnectivityMonitor" = None):
        self.config = config
        self.connectivity = connectivity
        self.recognizer = None
        self.sr = None
        self._whisper_model = None
        self._whisper_loaded = False
        self._init_google()

    def _init_google(self):
        """Initialize Google speech_recognition (cloud STT)."""
        try:
            import speech_recognition as sr
            self.sr = sr
            self.recognizer = sr.Recognizer()
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.energy_threshold = self.config.energy_threshold
            print("✓ STT cloud (Google) ready")
        except ImportError:
            print("⚠ speech_recognition not installed")

    def _load_whisper(self):
        """Lazy-load faster-whisper model on first offline use."""
        if self._whisper_loaded:
            return self._whisper_model
        try:
            from faster_whisper import WhisperModel
            model_name = self.config.stt_offline_model  # e.g., 'base.en'
            print(f"[STT-offline] Loading faster-whisper '{model_name}' (first run downloads ~145MB)...")
            self._whisper_model = WhisperModel(model_name, device="cpu", compute_type="int8")
            print("✓ STT offline (faster-whisper) ready")
        except ImportError:
            print("⚠ faster-whisper not installed – run: pip install faster-whisper")
        except Exception as e:
            print(f"⚠ faster-whisper init failed: {e}")
        self._whisper_loaded = True
        return self._whisper_model

    def is_available(self) -> bool:
        """True if at least one STT engine is usable."""
        return self.recognizer is not None or True  # Whisper always attempted

    def _capture_audio_bytes(self, timeout: int, phrase_limit: int) -> Optional[bytes]:
        """Capture raw WAV bytes from microphone."""
        if not self.sr:
            return None
        try:
            with suppress_alsa():
                with self.sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
                    return audio.get_wav_data()
        except Exception:
            return None

    def _transcribe_google(self, audio_data) -> Optional[str]:
        """Transcribe using Google STT."""
        try:
            text = self.recognizer.recognize_google(audio_data)
            return text.lower()
        except Exception:
            return None

    def _transcribe_whisper(self, wav_bytes: bytes) -> Optional[str]:
        """Transcribe using faster-whisper (offline)."""
        model = self._load_whisper()
        if not model:
            return None
        import tempfile
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_bytes)
                tmp_path = f.name
            segments, _ = model.transcribe(tmp_path, language="en", beam_size=5)
            text = " ".join(s.text for s in segments).strip().lower()
            try:
                os.remove(tmp_path)
            except:
                pass
            return text if text else None
        except Exception as e:
            print(f"[STT-offline] Whisper error: {e}")
            return None

    def listen(self, timeout: int = None, phrase_limit: int = None) -> Optional[str]:
        """Listen and transcribe. Google online → Whisper offline."""
        timeout = timeout or self.config.listen_timeout
        phrase_limit = phrase_limit or self.config.phrase_time_limit
        online = self.connectivity.is_online() if self.connectivity else True

        if not self.sr:
            return None

        try:
            with suppress_alsa():
                with self.sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)

            if online:
                # Primary: Google cloud STT
                try:
                    text = self.recognizer.recognize_google(audio)
                    print(f"[STT] Google: '{text}'")
                    return text.lower()
                except self.sr.UnknownValueError:
                    return None
                except Exception as e:
                    print(f"[STT] Google failed ({e}), trying Whisper offline...")
                    # Fall through to Whisper

            # Offline path (or Google failed)
            return self._transcribe_whisper(audio.get_wav_data())

        except self.sr.WaitTimeoutError:
            return None
        except Exception as e:
            print(f"[STT] Capture error: {e}")
            return None

    def listen_for_wake_word(self) -> Optional[str]:
        """Listen for wake word. Includes STT mishear aliases for 'Kenza'."""
        text = self.listen(timeout=8, phrase_limit=5)
        if text:
            # Common STT misheards of 'Kenza' added as aliases
            wake_words = [
                "kenza", "kenzo", "kinza", "kansa", "kanza",  # correct-ish
                "cancer", "kancer", "kencer", "censer",       # phonetic mishears
                "kenya", "kenja", "hey kenza", "okay kenza",  # other mishears
            ]
            for word in wake_words:
                if word in text:
                    print(f"[Wake word detected in: '{text}']")
                    return text
        return None


# ============================================================================
# GEMINI PROVIDER (Chat with memory)
# ============================================================================

class GeminiChat:
    """Gemini chat with conversation memory and retry logic"""
    
    def __init__(self, config: ConversationConfig, personality: KenzaPersonality):
        self.config = config
        self.personality = personality
        self.model = None
        self.chat = None
        self.max_retries = 3
        self.retry_delay = 2  # Initial delay in seconds
        self._init()
    
    def _init(self):
        if not self.config.gemini_api_key:
            print("⚠ Gemini API key not configured")
            return
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.config.gemini_api_key)
            self.model = genai.GenerativeModel(
                self.config.gemini_model,
                system_instruction=self.personality.get_system_prompt()
            )
            self.chat = self.model.start_chat(history=[])
            print(f"✓ Gemini chat initialized ({self.config.gemini_model})")
        except ImportError:
            print("⚠ google-generativeai not installed")
        except Exception as e:
            print(f"⚠ Gemini init failed: {e}")
    
    def is_available(self) -> bool:
        return self.chat is not None
    
    def send(self, message: str) -> str:
        """Send message and get response with retry on quota errors"""
        if not self.is_available():
            return "I'm having trouble connecting right now."
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.chat.send_message(message)
                return self.personality.format_response(response.text)
            except Exception as e:
                error_str = str(e).lower()
                last_error = e
                
                # Check if it's a quota/rate limit error
                if '429' in str(e) or 'quota' in error_str or 'rate' in error_str:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"[API] Rate limited, waiting {wait_time}s... (attempt {attempt+1}/{self.max_retries})")
                    time.sleep(wait_time)
                else:
                    # Non-retryable error
                    print(f"[API] Error: {e}")
                    break
        
        # All retries failed
        return "I'm a bit busy right now. Let me try again in a moment."
    
    def reset(self):
        """Reset conversation history"""
        if self.model:
            self.chat = self.model.start_chat(history=[])


# ============================================================================
# GROQ PROVIDER (Fast Llama-based chat)
# ============================================================================

class GroqChat:
    """Groq chat with Llama models - very fast inference"""
    
    def __init__(self, config: ConversationConfig, personality: KenzaPersonality):
        self.config = config
        self.personality = personality
        self.client = None
        self.conversation_history = []
        self.max_retries = 3
        self.retry_delay = 1
        self._init()
    
    def _init(self):
        if not self.config.groq_api_key:
            print("⚠ Groq API key not configured")
            return
        
        try:
            from groq import Groq
            self.client = Groq(api_key=self.config.groq_api_key)
            print(f"✓ Groq chat initialized ({self.config.groq_model})")
        except ImportError:
            print("⚠ groq not installed. Run: pip install groq")
        except Exception as e:
            print(f"⚠ Groq init failed: {e}")
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def send(self, message: str) -> str:
        """Send message and get response"""
        if not self.is_available():
            return None  # Return None so fallback can be used
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
        
        # Build messages with system prompt
        messages = [
            {"role": "system", "content": self.personality.get_system_prompt()}
        ] + self.conversation_history[-10:]  # Keep last 10 messages
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.groq_model,
                    messages=messages,
                    max_tokens=300,
                    temperature=0.7
                )
                
                assistant_message = response.choices[0].message.content
                
                # Add to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                
                return self.personality.format_response(assistant_message)
                
            except Exception as e:
                error_str = str(e).lower()
                last_error = e
                
                if '429' in str(e) or 'rate' in error_str:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"[Groq] Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[Groq] Error: {e}")
                    break
        
        return None  # Return None so Gemini fallback can be used
    
    def reset(self):
        """Reset conversation history"""
        self.conversation_history.clear()


# ============================================================================
# LLAMA CHAT PROVIDER (Offline LLM – final fallback)
# ============================================================================

class LlamaChat:
    """
    Offline LLM using llama-cpp-python with a local GGUF model.
    Lazy-loaded on first use. Gracefully skipped if model file is missing.
    Chain position: Groq → Gemini → LlamaChat (offline)

    Download model (~2GB):
      cd models
      wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf -O llama-3.2-3b-instruct.Q4_K_M.gguf
    """

    def __init__(self, config: ConversationConfig, personality: "KenzaPersonality"):
        self.config = config
        self.personality = personality
        self._llm = None
        self._loaded = False
        self.conversation_history: List[Dict] = []

    def _load(self):
        """Lazy-load the GGUF model on first call."""
        if self._loaded:
            return
        self._loaded = True
        model_path = Path(self.config.llama_path) if hasattr(self.config, 'llama_path') else Path("models/llama-3.2-3b-instruct.Q4_K_M.gguf")
        if not model_path.exists():
            print(f"[Llama] Model not found at {model_path} — offline LLM unavailable")
            print(f"[Llama] Download: wget <HF link> -O {model_path}")
            return
        try:
            from llama_cpp import Llama
            threads = getattr(self.config, 'llama_threads', 4)
            ctx = getattr(self.config, 'llama_context', 2048)
            print(f"[Llama] Loading {model_path} (threads={threads}, ctx={ctx})...")
            self._llm = Llama(
                model_path=str(model_path),
                n_ctx=ctx,
                n_threads=threads,
                verbose=False,
            )
            print("✓ Llama offline LLM ready")
        except ImportError:
            print("⚠ llama-cpp-python not installed – run: pip install llama-cpp-python")
        except Exception as e:
            print(f"⚠ Llama load failed: {e}")

    def is_available(self) -> bool:
        self._load()
        return self._llm is not None

    def send(self, message: str) -> Optional[str]:
        """Generate a response offline."""
        if not self.is_available():
            return None
        self.conversation_history.append({"role": "user", "content": message})
        messages = [
            {"role": "system", "content": self.personality.get_system_prompt()}
        ] + self.conversation_history[-8:]
        try:
            result = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=250,
                temperature=0.7,
                stop=["<|eot_id|>"],
            )
            reply = result["choices"][0]["message"]["content"].strip()
            self.conversation_history.append({"role": "assistant", "content": reply})
            return self.personality.format_response(reply)
        except Exception as e:
            print(f"[Llama] Inference error: {e}")
            return None

    def reset(self):
        self.conversation_history.clear()


# ============================================================================
# OBJECT DETECTOR (Offline – YOLO11n)
# ============================================================================

class ObjectDetector:
    """
    Real-time offline object detection using YOLO11n (Ultralytics).
    YOLO11n model (~5MB) is auto-downloaded on first use.
    Recognises 80 COCO classes at ~25 FPS on Raspberry Pi 4.
    """

    def __init__(self):
        self._model = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            from ultralytics import YOLO
            # yolo11n.pt is the latest nano model (smallest, fastest)
            self._model = YOLO("yolo11n.pt")
            self._model.fuse()  # Optimise for inference speed
            print("✓ ObjectDetector (YOLO11n) ready")
        except ImportError:
            print("⚠ ultralytics not installed – run: pip install ultralytics")
        except Exception as e:
            print(f"⚠ YOLO init failed: {e}")

    def is_available(self) -> bool:
        self._load()
        return self._model is not None

    def detect_objects(self, frame) -> Optional[str]:
        """
        Detect objects in a frame and return a natural language description.
        frame: numpy ndarray (BGR from OpenCV)
        Returns: 'I can see: a bottle and a laptop.' or None on failure.
        """
        if not self.is_available():
            return None
        try:
            results = self._model(frame, verbose=False, conf=0.4)
            names = []
            seen = set()
            for r in results:
                for cls_id in r.boxes.cls.tolist():
                    label = r.names[int(cls_id)]
                    if label not in seen:
                        seen.add(label)
                        names.append(f"a {label}")
            if not names:
                return "I can see the scene but I can't identify specific objects right now."
            if len(names) == 1:
                return f"I can see {names[0]}."
            return f"I can see: {', '.join(names[:-1])} and {names[-1]}."
        except Exception as e:
            print(f"[Vision] YOLO detection error: {e}")
            return None


# ============================================================================
# OLLAMA CHAT PROVIDER (Offline LLM via local Ollama server)
# ============================================================================

class OllamaChat:
    """
    Offline LLM using Ollama's local REST API (http://localhost:11434).
    Supports ANY model pulled via `ollama pull <model>`:
      - gemma3:270m  (tiny, ~170MB, very fast on Pi)
      - ministral-3:3b-instruct-2512-q4_K_M  (Mistral 3B Instruct, locally available)
      - phi3:mini    (Microsoft, good coding)
      - mistral:7b   (high quality but slower on Pi)

    Chain position in ConversationEngine: Groq → Gemini → OllamaChat → LlamaChat(GGUF)

    Install Ollama on Raspberry Pi:
      curl -fsSL https://ollama.com/install.sh | sh
      ollama pull gemma3:270m
      ollama pull ministral-3:3b-instruct-2512-q4_K_M   (already downloaded)
    """

    AVAILABLE_MODELS = {
        "gemma3:270m":    {"label": "Gemma 3 Nano",   "desc": "Google • 270M params • ultra-fast",  "ram": "~500MB"},
        "ministral-3:3b-instruct-2512-q4_K_M": {"label": "Ministral 3B", "desc": "Mistral • 3B Instruct • fast & capable", "ram": "~2.5GB"},
        "phi3:mini":      {"label": "Phi-3 Mini",      "desc": "Microsoft • 3.8B • good reasoning",  "ram": "~2.7GB"},
        "mistral:7b":     {"label": "Mistral (7B)",    "desc": "High quality • slower on Pi",        "ram": "~5GB"},
        "tinyllama:1.1b": {"label": "TinyLlama",       "desc": "1.1B • minimal RAM use",             "ram": "~1GB"},
    }

    def __init__(self, config: ConversationConfig, personality: "KenzaPersonality"):
        self.config = config
        self.personality = personality
        self.model = config.ollama_model          # e.g. 'gemma3:270m'
        self.base_url = config.ollama_url.rstrip("/")
        self.conversation_history: List[Dict] = []
        self._available: Optional[bool] = None   # None = not yet checked

    def _check_server(self) -> bool:
        """Ping Ollama server to see if it's running."""
        import urllib.request
        try:
            req = urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=2)
            self._available = (req.status == 200)
        except Exception:
            self._available = False
        return self._available

    def is_available(self) -> bool:
        """True if Ollama server is reachable. Cached for 60s."""
        if self._available is None:
            self._check_server()
        return bool(self._available)

    def set_model(self, model_name: str):
        """Switch to a different Ollama model at runtime."""
        self.model = model_name
        self.conversation_history.clear()   # Clear context on model switch
        self._available = None              # Re-check server on next call
        print(f"[Ollama] Switched to model: {model_name}")

    def list_local_models(self) -> List[str]:
        """Return models pulled on this machine via Ollama."""
        import urllib.request, json
        try:
            req = urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3)
            data = json.loads(req.read())
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def send(self, message: str) -> Optional[str]:
        """Generate a response using Ollama's /api/chat endpoint."""
        if not self.is_available():
            print("[Ollama] Server not running – skipping")
            return None

        import urllib.request, json as _json

        self.conversation_history.append({"role": "user", "content": message})
        messages = [
            {"role": "system", "content": self.personality.get_system_prompt()}
        ] + self.conversation_history[-10:]

        payload = _json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": 250, "temperature": 0.7},
        }).encode()

        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = _json.loads(resp.read())
            reply = result["message"]["content"].strip()
            self.conversation_history.append({"role": "assistant", "content": reply})
            print(f"[Ollama:{self.model}] Response generated")
            return self.personality.format_response(reply)
        except Exception as e:
            print(f"[Ollama] Error: {e}")
            self._available = None   # Mark for re-check
            return None

    def reset(self):
        self.conversation_history.clear()


# ============================================================================
# CONVERSATION ENGINE (Main orchestrator)
# ============================================================================

class ConversationEngine:
    """
    Main conversation engine that orchestrates all components.
    Handles wake word, interruption, vision, commands, and offline fallbacks.
    """

    def __init__(
        self,
        config: ConversationConfig = None,
        eye_controller=None,
        audio_controller=None,
        camera=None,
        on_state_change: Callable = None,
        on_user_speech: Callable = None,
        on_ai_response: Callable = None
    ):
        self.config = config or ConversationConfig.load()

        # Connectivity monitor (shared by all cloud-dependent components)
        self.connectivity = ConnectivityMonitor(
            check_interval=self.config.connectivity_check_interval
        )

        # Core components
        self.personality = KenzaPersonality(self.config)
        self.emotion = EmotionEngine()
        self.tts = InterruptibleTTS(self.config, connectivity=self.connectivity)
        self.stt = SpeechToText(self.config, connectivity=self.connectivity)
        self.vad = VoiceActivityDetector(self.config)
        self.vision = VisionAI(self.config, camera)
        self.object_detector = ObjectDetector()

        # Chat providers: Groq (cloud) → Gemini (cloud) → Ollama (offline) → LlamaGGUF (offline)
        self.groq = GroqChat(self.config, self.personality)
        self.gemini = GeminiChat(self.config, self.personality)
        self.ollama = OllamaChat(self.config, self.personality)  # Ollama: gemma3:270m or any model
        self.llama = LlamaChat(self.config, self.personality)    # GGUF fallback if Ollama not installed

        self.commands = CommandParser(self.config, eye_controller, audio_controller, self.tts)

        # Emotion Eye Bridge — non-blocking WebSocket emotion signaller
        self.eye_bridge = EmotionEyeBridge(ws_url="ws://localhost:8765")

        # State
        self.is_running = False
        self.is_listening = False
        self.is_sleeping = True
        self.on_state_change = on_state_change
        self.on_user_speech = on_user_speech
        self.on_ai_response = on_ai_response

        # Controllers
        self.eye_controller = eye_controller
        self.audio_controller = audio_controller

        online = "online" if self.connectivity.is_online() else "OFFLINE"
        print("\n" + "=" * 50)
        print("     KENZA Conversation Engine Ready")
        print(f"     Network: {online}")
        print("=" * 50 + "\n")
    
    def _notify_state(self, state: str):
        """Notify state change."""
        if self.on_state_change:
            self.on_state_change(state)

    def _parse_and_strip_emotion(self, response: str) -> tuple:
        """
        Extract the leading emotion tag from an LLM response.
        Returns (emotion_str, clean_text_without_tag).
        Defaults to 'neutral' if no valid tag is found.
        """
        import re
        match = re.match(
            r'^\[(happy|sad|excited|neutral|confused|thinking)\]\s*',
            response.strip(),
            re.IGNORECASE,
        )
        if match:
            emotion = match.group(1).lower()
            clean  = response[match.end():].strip()
            return emotion, clean
        return "neutral", response.strip()

    def speak_with_interrupt(self, text: str, emotion: str = None) -> bool:
        """
        Speak text while monitoring for interruption.
        Detects emotion automatically if not provided.
        Returns True if completed, False if interrupted.
        """
        if emotion is None:
            emotion = self.emotion.detect(text)

        # Map EmotionEngine emotion → bridge state (EmotionEngine uses different names)
        speaking_state = emotion if emotion in EmotionEyeBridge.VALID_STATES else "speaking"
        self.eye_bridge.send_emotion(speaking_state)
        self._notify_state("speaking")
        interrupted = False

        def on_interrupt():
            nonlocal interrupted
            interrupted = True
            self.tts.clear_and_stop()

        self.vad.start_monitoring(on_interrupt)
        completed = self.tts.speak_blocking(text, emotion=emotion)
        self.vad.stop_monitoring()
        self.eye_bridge.send_emotion("neutral")
        self._notify_state("idle")
        return completed and not interrupted
    
    def process_input(self, text: str) -> str:
        """
        Process user input and generate a response.
        Pipeline:
          Commands → Vision (YOLO offline / Gemini online) → Groq → Gemini → Llama
        Also detects emotion in the response for expressive TTS.
        """
        if not text:
            return ""

        # 1. Voice commands (eye color, mode changes, etc.)
        command = self.commands.parse(text)
        if command:
            action, params = command
            response = self.commands.execute(action, params)
            return response or ""

        # 2. Vision queries
        if self.vision.is_vision_query(text) or self.vision._get_recent_context():
            online = self.connectivity.is_online()
            frame = self.vision.capture_frame()

            if not online and frame is not None:
                # Offline: use YOLO11n for fast local detection
                yolo_result = self.object_detector.detect_objects(frame)
                if yolo_result:
                    return yolo_result

            # Online (or YOLO failed): use Gemini multimodal
            return self.vision.query_with_image(text)

        # 3. General chat: Groq → Gemini → Ollama → LlamaGGUF (offline fallbacks)
        raw = self.groq.send(text)
        if raw:
            emotion, response = self._parse_and_strip_emotion(raw)
            self.eye_bridge.send_emotion(emotion)
            return response

        raw = self.gemini.send(text)
        if raw and not raw.startswith("I'm"):
            emotion, response = self._parse_and_strip_emotion(raw)
            self.eye_bridge.send_emotion(emotion)
            return response

        # Offline tier 1: Ollama (any pulled model, default gemma3:270m)
        print(f"[Engine] Cloud LLMs unavailable – trying Ollama ({self.ollama.model})")
        raw = self.ollama.send(text)
        if raw:
            emotion, response = self._parse_and_strip_emotion(raw)
            self.eye_bridge.send_emotion(emotion)
            return response

        # Offline tier 2: llama-cpp-python GGUF file
        print("[Engine] Ollama unavailable – using llama-cpp GGUF fallback")
        raw = self.llama.send(text)
        if raw:
            emotion, response = self._parse_and_strip_emotion(raw)
            self.eye_bridge.send_emotion(emotion)
            return response

        return "I'm having trouble responding right now. Please try again."

    def set_offline_model(self, model_name: str) -> dict:
        """
        Switch the active Ollama offline model at runtime.
        Called by the app's 'set_offline_model' WebSocket command.
        Returns a status dict to send back to the app.
        """
        self.ollama.set_model(model_name)
        self.config.ollama_model = model_name
        available = self.ollama.is_available()
        print(f"[Engine] Offline model set to '{model_name}' (Ollama {'✓' if available else '✗'})")
        return {
            "model": model_name,
            "available": available,
            "success": True,
        }

    def get_offline_model_info(self) -> dict:
        """
        Return current offline model info + list of locally pulled Ollama models.
        Called by the app on initial load ('get_offline_models' command).
        """
        local_models = self.ollama.list_local_models()  # What's actually installed
        return {
            "current_model": self.ollama.model,
            "ollama_available": self.ollama.is_available(),
            "local_models": local_models,
            "known_models": OllamaChat.AVAILABLE_MODELS,  # Catalogue with descriptions
        }
    
    def run_voice_loop(self, use_wake_word: bool = True):
        """
        Main voice interaction loop.
        
        Args:
            use_wake_word: If True, wait for wake word before listening
        """
        if not self.stt.is_available():
            print("ERROR: Speech recognition not available!")
            return
        
        self.is_running = True
        self.is_sleeping = use_wake_word
        error_count = 0
        max_errors = 5
        
        print("\n" + "=" * 50)
        print("🎙️  KENZA Voice Mode")
        print("=" * 50)
        if use_wake_word:
            print(f"   Say '{self.config.wake_word.title()}' to start")
        print("   Say 'goodbye' or 'stop' to sleep")
        print("=" * 50 + "\n")
        
        while self.is_running:
            try:
                if self.is_sleeping:
                    # Waiting for wake word
                    self._notify_state("sleeping")
                    print(f"💤 Waiting for '{self.config.wake_word}'...", end="\r")
                    
                    text = self.stt.listen_for_wake_word()
                    
                    if text:
                        error_count = 0  # Reset on success
                        print(f"\n👋 Wake: {text}")
                        self.is_sleeping = False
                        self.eye_bridge.send_emotion("listening")
                        self.speak_with_interrupt("Yes?")
                    else:
                        # No wake word, add small delay to prevent busy loop
                        time.sleep(0.1)
                else:
                    # Active listening
                    self._notify_state("listening")
                    self.eye_bridge.send_emotion("listening")
                    print("👂 Listening...", end="\r")
                    
                    text = self.stt.listen()
                    
                    if not text:
                        time.sleep(0.1)  # Small delay on timeout
                        continue
                    
                    error_count = 0  # Reset on success
                    print(f"\n🗣️  You: {text}")
                    
                    # Notify app of user speech
                    if self.on_user_speech:
                        self.on_user_speech(text)
                    
                    # Check for sleep commands
                    if any(cmd in text for cmd in ["goodbye", "bye", "go to sleep", "stop", "that's all"]):
                        print("[💤 Going to sleep]")
                        self.eye_bridge.send_emotion("sleep")
                        self.speak_with_interrupt("Okay, let me know if you need me.")
                        self.is_sleeping = True
                        continue
                    
                    # Process and respond — signal thinking state BEFORE LLM call
                    self._notify_state("thinking")
                    self.eye_bridge.send_emotion("thinking")
                    response = self.process_input(text)
                    # Note: process_input() already signals the response emotion via eye_bridge
                    
                    if response:
                        print(f"🤖 Kenza: {response}\n")
                        # Notify app of AI response
                        if self.on_ai_response:
                            self.on_ai_response(response)
                        self._notify_state("speaking")
                        self.speak_with_interrupt(response)
                    
                    self.eye_bridge.send_emotion("neutral")
                    self._notify_state("idle")
                    print("--- Ready ---\n")

                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                error_count += 1
                print(f"\nError ({error_count}/{max_errors}): {e}")
                if error_count >= max_errors:
                    print("Too many errors, stopping voice mode.")
                    break
                time.sleep(1)  # Wait before retrying after error
        
        self.is_running = False
        self._notify_state("stopped")
    
    def stop(self):
        """Stop the conversation loop"""
        self.is_running = False
        self.eye_bridge.send_emotion("sleep")
        self.eye_bridge.stop()
        self.tts.clear_and_stop()
        self.vad.stop_monitoring()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Test the conversation engine"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Kenza Conversation Engine")
    parser.add_argument("--no-wake", action="store_true", help="Skip wake word")
    parser.add_argument("--text", action="store_true", help="Text mode only")
    parser.add_argument("--test-mic", action="store_true", help="Test microphone")
    parser.add_argument("--config", default="config/settings.yaml", help="Config path")
    args = parser.parse_args()
    
    config = ConversationConfig.load(args.config)
    
    # Microphone test mode
    if args.test_mic:
        print("\n🎤 MICROPHONE TEST")
        print("=" * 40)
        print("Speak something and I'll show what I hear.\n")
        
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = config.energy_threshold
        
        try:
            with sr.Microphone() as source:
                print(f"Microphone: {source}")
                print(f"Energy threshold: {recognizer.energy_threshold}")
                print("\nAdjusting for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                print(f"Adjusted threshold: {recognizer.energy_threshold}")
                
                for i in range(3):
                    print(f"\n[Test {i+1}/3] Speak now...")
                    try:
                        audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
                        text = recognizer.recognize_google(audio)
                        print(f"✓ Heard: '{text}'")
                    except sr.WaitTimeoutError:
                        print("✗ Timeout - no speech detected")
                    except sr.UnknownValueError:
                        print("✗ Could not understand audio")
                    except Exception as e:
                        print(f"✗ Error: {e}")
        except Exception as e:
            print(f"Microphone error: {e}")
        return
    
    engine = ConversationEngine(config)
    
    if args.text:
        # Text mode for testing
        print("\n📝 Text mode (type 'quit' to exit)\n")
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() == "quit":
                    break
                response = engine.process_input(user_input)
                print(f"Kenza: {response}\n")
            except KeyboardInterrupt:
                break
    else:
        # Voice mode
        engine.run_voice_loop(use_wake_word=not args.no_wake)


if __name__ == "__main__":
    main()

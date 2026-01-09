#!/usr/bin/env python3
"""
Kenza AI - Conversational AI Module
Based on Google Gemini + Offline Llama LLM with Edge-TTS

Features:
- Wake word detection ("Kenza")
- Smart routing between Online (Gemini) and Offline (Llama)
- Human-like voice synthesis (Edge-TTS)
- Conversation memory (Gemini chat session)
- GPIO LED status indicators
- ALSA log suppression for clean output

Usage:
    python kenza_ai.py              # Full voice mode with wake word
    python kenza_ai.py --no-wake    # Skip wake word, always listening
    python kenza_ai.py --text       # Text-only mode (no mic/speaker)
    python kenza_ai.py --test       # Test components
"""

import os
import sys
import queue
import threading
import time
import warnings
import asyncio
import ctypes
import argparse
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path

import yaml

# ============================================================================
# HARDCORE LOG SUPPRESSION (ALSA/C-Level for Raspberry Pi)
# ============================================================================
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# C-level error handler to hide ALSA warnings
ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
)

def py_error_handler(filename, line, function, err, fmt):
    pass

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

@contextmanager
def suppress_alsa_errors():
    """Context manager to suppress ALSA error messages on Linux/Pi"""
    try:
        asound = ctypes.cdll.LoadLibrary('libasound.so')
        asound.snd_lib_error_set_handler(c_error_handler)
        yield
        asound.snd_lib_error_set_handler(None)
    except OSError:
        # Not on Linux or libasound not available
        yield


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration loaded from settings.yaml or defaults"""
    
    # API Keys
    gemini_api_key: str = ""
    
    # Models
    gemini_model: str = "gemini-2.0-flash"
    llama_model_path: str = "models/llama-3.2-3b-instruct.Q4_K_M.gguf"
    llama_context_size: int = 2048
    llama_threads: int = 4
    
    # Voice
    tts_voice: str = "en-US-AriaNeural"  # Edge-TTS voice
    wake_word: str = "kenza"
    
    # Audio
    energy_threshold: int = 1000
    listen_timeout: int = 10
    phrase_time_limit: int = 8
    
    # GPIO (Raspberry Pi)
    led_listening_pin: int = 24  # Green LED
    led_thinking_pin: int = 25   # Red LED
    gpio_enabled: bool = True
    
    # Sleep commands
    sleep_commands: list = None
    
    def __init__(self):
        self.sleep_commands = ["that's all", "go to sleep", "stop", "goodbye", "bye"]
    
    @classmethod
    def load(cls, config_path: str = "config/settings.yaml") -> "Config":
        """Load configuration from YAML file"""
        config = cls()
        config_file = Path(config_path)
        
        if config_file.exists():
            with open(config_file) as f:
                data = yaml.safe_load(f) or {}
            
            # API Keys (prefer environment variables)
            api_keys = data.get("api_keys", {})
            config.gemini_api_key = os.getenv("GEMINI_API_KEY", api_keys.get("gemini", ""))
            
            # Models
            models = data.get("models", {})
            config.gemini_model = models.get("gemini", config.gemini_model)
            config.llama_model_path = models.get("llama_path", config.llama_model_path)
            config.llama_context_size = models.get("llama_context", config.llama_context_size)
            config.llama_threads = models.get("llama_threads", config.llama_threads)
            
            # Voice
            voice = data.get("voice", {})
            config.tts_voice = voice.get("edge_tts_voice", config.tts_voice)
            config.wake_word = voice.get("wake_word", config.wake_word).lower()
            
            # Audio
            audio = data.get("audio", {})
            config.energy_threshold = audio.get("energy_threshold", config.energy_threshold)
            config.listen_timeout = audio.get("listen_timeout", config.listen_timeout)
            config.phrase_time_limit = audio.get("phrase_time_limit", config.phrase_time_limit)
            
            # GPIO
            gpio = data.get("gpio", {})
            config.led_listening_pin = gpio.get("led_listening", config.led_listening_pin)
            config.led_thinking_pin = gpio.get("led_thinking", config.led_thinking_pin)
            config.gpio_enabled = gpio.get("enabled", config.gpio_enabled)
            
            # Sleep commands
            config.sleep_commands = data.get("sleep_commands", config.sleep_commands)
        else:
            # Try environment variable for API key
            config.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        
        return config


# ============================================================================
# GPIO LED CONTROLS (with fallback for non-Pi systems)
# ============================================================================

class LEDController:
    """Controls GPIO LEDs for status indication (with fallback)"""
    
    def __init__(self, config: Config):
        self.config = config
        self.led_listening = None
        self.led_thinking = None
        self._initialize()
    
    def _initialize(self):
        """Initialize GPIO LEDs"""
        if not self.config.gpio_enabled:
            return
        
        try:
            from gpiozero import LED
            self.led_listening = LED(self.config.led_listening_pin)
            self.led_thinking = LED(self.config.led_thinking_pin)
            print("✓ GPIO LEDs initialized")
        except ImportError:
            print("⚠ gpiozero not available (not on Pi?), LEDs disabled")
        except Exception as e:
            print(f"⚠ GPIO init failed: {e}, LEDs disabled")
    
    def listening_on(self):
        """Green LED on - listening for input"""
        if self.led_listening:
            self.led_listening.on()
    
    def listening_off(self):
        """Green LED off"""
        if self.led_listening:
            self.led_listening.off()
    
    def thinking_on(self):
        """Red LED on - thinking/speaking"""
        if self.led_thinking:
            self.led_thinking.on()
    
    def thinking_off(self):
        """Red LED off"""
        if self.led_thinking:
            self.led_thinking.off()
    
    def all_off(self):
        """Turn off all LEDs"""
        self.listening_off()
        self.thinking_off()


# ============================================================================
# TEXT-TO-SPEECH (Edge-TTS - Human-like voices)
# ============================================================================

class TextToSpeech:
    """Edge-TTS based speech synthesis with human-like voices"""
    
    def __init__(self, config: Config):
        self.config = config
        self.voice = config.tts_voice
        self._mixer_initialized = False
        self._initialize_mixer()
    
    def _initialize_mixer(self):
        """Initialize pygame mixer for audio playback"""
        try:
            from pygame import mixer
            mixer.pre_init(frequency=24000, buffer=2048)
            mixer.init()
            self.mixer = mixer
            self._mixer_initialized = True
            print(f"✓ Audio initialized (voice: {self.voice})")
        except ImportError:
            print("⚠ pygame not installed, audio playback disabled")
        except Exception as e:
            print(f"⚠ Audio init failed: {e}")
    
    async def _generate_audio(self, text: str, filename: str):
        """Generate audio file using Edge-TTS"""
        import edge_tts
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(filename)
    
    def generate(self, text: str, filename: str) -> bool:
        """Generate TTS audio file synchronously"""
        try:
            import edge_tts
            asyncio.run(self._generate_audio(text, filename))
            return True
        except ImportError:
            print("⚠ edge-tts not installed")
            return False
        except Exception as e:
            print(f"TTS Error: {e}")
            return False
    
    def play(self, filename: str, blocking: bool = True):
        """Play an audio file"""
        if not self._mixer_initialized:
            return
        
        try:
            self.mixer.music.load(filename)
            self.mixer.music.play()
            if blocking:
                while self.mixer.music.get_busy():
                    time.sleep(0.1)
        except Exception as e:
            print(f"Playback error: {e}")
    
    def stop(self):
        """Stop audio playback"""
        if self._mixer_initialized:
            self.mixer.music.stop()
    
    def is_playing(self) -> bool:
        """Check if audio is currently playing"""
        if self._mixer_initialized:
            return self.mixer.music.get_busy()
        return False
    
    def speak(self, text: str, led_controller: LEDController = None):
        """Generate and play speech"""
        filename = f"speech_{int(time.time() * 1000)}.mp3"
        
        if self.generate(text, filename):
            if led_controller:
                led_controller.thinking_on()
            
            self.play(filename)
            
            if led_controller:
                led_controller.thinking_off()
            
            # Cleanup
            try:
                os.remove(filename)
            except:
                pass
            return True
        return False


# ============================================================================
# SPEECH-TO-TEXT (Google Speech Recognition)
# ============================================================================

class SpeechToText:
    """Google Speech Recognition for STT"""
    
    def __init__(self, config: Config):
        self.config = config
        self.recognizer = None
        self.microphone = None
        self._initialize()
    
    def _initialize(self):
        """Initialize speech recognizer"""
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.energy_threshold = self.config.energy_threshold
            self.sr = sr
            print("✓ Speech recognition initialized")
        except ImportError:
            print("⚠ speech_recognition not installed")
    
    def is_available(self) -> bool:
        """Check if STT is available"""
        return self.recognizer is not None
    
    def listen(self, timeout: int = None, phrase_limit: int = None) -> str:
        """Listen for speech and return recognized text"""
        if not self.is_available():
            return None
        
        timeout = timeout or self.config.listen_timeout
        phrase_limit = phrase_limit or self.config.phrase_time_limit
        
        try:
            with suppress_alsa_errors():
                with self.sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(
                        source, 
                        timeout=timeout, 
                        phrase_time_limit=phrase_limit
                    )
                    text = self.recognizer.recognize_google(audio)
                    return text.lower()
        except self.sr.WaitTimeoutError:
            return None
        except self.sr.UnknownValueError:
            return None
        except Exception as e:
            print(f"STT Error: {e}")
            return None
    
    def listen_for_wake_word(self, wake_word: str) -> str:
        """Listen specifically for wake word, return full phrase if found"""
        try:
            with suppress_alsa_errors():
                with self.sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(
                        source, 
                        timeout=2, 
                        phrase_time_limit=3
                    )
                    text = self.recognizer.recognize_google(audio).lower()
                    if wake_word in text:
                        return text
                    return None
        except:
            return None


# ============================================================================
# AI PROVIDERS (Gemini + Llama)
# ============================================================================

class GeminiProvider:
    """Google Gemini API with conversation memory"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.chat_session = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Gemini client with chat session for memory"""
        if not self.config.gemini_api_key or len(self.config.gemini_api_key) < 5:
            print("⚠ Gemini API key not configured")
            return
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.config.gemini_api_key)
            self.model = genai.GenerativeModel(self.config.gemini_model)
            # Start chat session for conversation memory
            self.chat_session = self.model.start_chat(history=[])
            print(f"✓ Gemini initialized ({self.config.gemini_model})")
        except ImportError:
            print("⚠ google-generativeai not installed")
        except Exception as e:
            print(f"⚠ Gemini init failed: {e}")
    
    def is_available(self) -> bool:
        """Check if Gemini is ready"""
        return self.chat_session is not None
    
    def send(self, message: str) -> str:
        """Send message and get response (with memory)"""
        if not self.is_available():
            raise RuntimeError("Gemini not initialized")
        
        response = self.chat_session.send_message(message)
        return response.text.strip()


class LlamaProvider:
    """Offline Llama LLM using llama-cpp-python"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Llama model"""
        model_path = Path(self.config.llama_model_path)
        
        if not model_path.exists():
            print(f"⚠ Llama model not found at {model_path}")
            print("  Download from: https://huggingface.co/TheBloke")
            return
        
        try:
            from llama_cpp import Llama
            print("⏳ Loading Offline Llama Model...")
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=self.config.llama_context_size,
                n_threads=self.config.llama_threads,
                verbose=False
            )
            print("✓ Offline Llama Model Loaded")
        except ImportError:
            print("⚠ llama-cpp-python not installed")
        except Exception as e:
            print(f"⚠ Llama init failed: {e}")
    
    def is_available(self) -> bool:
        """Check if Llama is ready"""
        return self.llm is not None
    
    def send(self, message: str, max_tokens: int = 150) -> str:
        """Send message and get response (single-turn, no memory)"""
        if not self.is_available():
            raise RuntimeError("Llama not initialized")
        
        output = self.llm(message, max_tokens=max_tokens, temperature=0.7)
        return output["choices"][0]["text"].strip()
    
    def classify(self, message: str) -> str:
        """Quick classification for routing (A=Online, B=Offline)"""
        if not self.is_available():
            return "A"  # Default to online if Llama not available
        
        prompt = f"Classify: '{message}'. Answer A for (Real-time/Complex/Internet) or B for (Simple/Offline). Answer only A or B."
        output = self.llm(prompt, max_tokens=5, temperature=0)
        return output["choices"][0]["text"].strip().upper()


# ============================================================================
# SMART AI ROUTER
# ============================================================================

class SmartRouter:
    """Routes requests between Online (Gemini) and Offline (Llama)"""
    
    def __init__(self, gemini: GeminiProvider, llama: LlamaProvider):
        self.gemini = gemini
        self.llama = llama
    
    def route(self, request: str) -> str:
        """Decide between online/offline and get response"""
        try:
            # Use Llama to classify the request
            if self.llama.is_available():
                decision = self.llama.classify(request)
            else:
                decision = "A"  # Default to online
            
            if decision.startswith("A") and self.gemini.is_available():
                # Online mode - use Gemini (with memory)
                print(f"\n[🟢 ONLINE] {request[:50]}...")
                response = self.gemini.send(request)
            elif self.llama.is_available():
                # Offline mode - use Llama (faster, no memory)
                print(f"\n[🔵 OFFLINE] {request[:50]}...")
                response = self.llama.send(request)
            else:
                return "I'm having trouble with both online and offline modes. Please check my configuration."
            
            # Clean response (remove markdown asterisks)
            return response.replace("*", "")
            
        except Exception as e:
            print(f"\n[⚠️ ERROR] {e}")
            
            # Fallback chain
            if self.llama.is_available():
                try:
                    return self.llama.send(request).replace("*", "")
                except:
                    pass
            
            return "I'm having trouble responding right now. Please try again."


# ============================================================================
# CONVERSATION MANAGER
# ============================================================================

class ConversationManager:
    """Main orchestrator for the conversation system"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize components
        print("\n" + "=" * 50)
        print("     KENZA AI - Initializing Components")
        print("=" * 50 + "\n")
        
        self.leds = LEDController(config)
        self.tts = TextToSpeech(config)
        self.stt = SpeechToText(config)
        self.gemini = GeminiProvider(config)
        self.llama = LlamaProvider(config)
        self.router = SmartRouter(self.gemini, self.llama)
        
        print("\n" + "=" * 50 + "\n")
    
    def chat(self, message: str) -> str:
        """Process a text message and return response"""
        return self.router.route(message)
    
    def voice_chat(self, text: str) -> str:
        """Process voice input and speak response"""
        self.leds.thinking_on()
        response = self.router.route(text)
        self.leds.thinking_off()
        
        # Speak the response
        self.tts.speak(response, self.leds)
        
        return response
    
    def is_sleep_command(self, text: str) -> bool:
        """Check if text contains a sleep command"""
        return any(cmd in text for cmd in self.config.sleep_commands)


# ============================================================================
# THREADED AUDIO PIPELINE (for async TTS)
# ============================================================================

def chat_thread(request: str, router: SmartRouter, text_queue: queue.Queue, 
                llm_done: threading.Event, stop_event: threading.Event):
    """Thread for AI processing"""
    try:
        reply = router.route(request)
        text_queue.put(reply)
    except Exception as e:
        print(f"Chat error: {e}")
    finally:
        llm_done.set()


def tts_thread(text_queue: queue.Queue, tts: TextToSpeech, audio_queue: queue.Queue,
               llm_done: threading.Event, tts_done: threading.Event, stop_event: threading.Event):
    """Thread for TTS generation"""
    counter = 0
    while not stop_event.is_set():
        try:
            text = text_queue.get(timeout=0.5)
            if text:
                filename = f"speech_{counter}.mp3"
                if tts.generate(text, filename):
                    audio_queue.put(filename)
                    counter += 1
            text_queue.task_done()
        except queue.Empty:
            if llm_done.is_set() and text_queue.empty():
                break
    tts_done.set()


def playback_thread(audio_queue: queue.Queue, tts: TextToSpeech, leds: LEDController,
                    tts_done: threading.Event, stop_event: threading.Event):
    """Thread for audio playback"""
    while not stop_event.is_set():
        try:
            mp3_file = audio_queue.get(timeout=0.5)
            leds.thinking_on()
            tts.play(mp3_file)
            leds.thinking_off()
            audio_queue.task_done()
            try:
                os.remove(mp3_file)
            except:
                pass
        except queue.Empty:
            if tts_done.is_set() and audio_queue.empty():
                break


def process_with_threads(request: str, manager: ConversationManager):
    """Process request using threaded pipeline for better responsiveness"""
    text_q = queue.Queue()
    audio_q = queue.Queue()
    llm_done = threading.Event()
    tts_done = threading.Event()
    stop_event = threading.Event()
    
    t1 = threading.Thread(target=chat_thread, 
                          args=(request, manager.router, text_q, llm_done, stop_event))
    t2 = threading.Thread(target=tts_thread,
                          args=(text_q, manager.tts, audio_q, llm_done, tts_done, stop_event))
    t3 = threading.Thread(target=playback_thread,
                          args=(audio_q, manager.tts, manager.leds, tts_done, stop_event))
    
    t1.start()
    t2.start()
    t3.start()
    
    t1.join()
    t2.join()
    t3.join()


# ============================================================================
# MAIN LOOPS
# ============================================================================

def run_voice_mode(manager: ConversationManager, use_wake_word: bool = True):
    """Voice-based conversation with optional wake word"""
    print("\n" + "=" * 50)
    print("🎙️  KENZA AI - Voice Mode")
    print("=" * 50)
    if use_wake_word:
        print(f"   Say '{manager.config.wake_word.title()}' to start")
    print("   Say 'goodbye' or 'stop' to sleep")
    print("=" * 50 + "\n")
    
    if not manager.stt.is_available():
        print("ERROR: Speech recognition not available!")
        return
    
    sleeping = use_wake_word  # Start asleep if using wake word
    
    while True:
        try:
            if sleeping:
                # Waiting for wake word
                manager.leds.listening_on()
                print(f"💤 Waiting for '{manager.config.wake_word}'...", end="\r")
                
                text = manager.stt.listen_for_wake_word(manager.config.wake_word)
                manager.leds.listening_off()
                
                if text:
                    print(f"\nUser: {text}")
                    sleeping = False
                    # Play wake response
                    manager.tts.speak("Yes?", manager.leds)
            else:
                # Active listening mode
                manager.leds.listening_on()
                print("👂 Listening...", end="\r")
                
                text = manager.stt.listen()
                manager.leds.listening_off()
                
                if not text:
                    continue
                
                print(f"\nUser: {text}")
                
                # Check for sleep command
                if manager.is_sleep_command(text):
                    print("[💤 Going back to sleep]")
                    manager.tts.speak("Okay, let me know if you need me.", manager.leds)
                    sleeping = True
                    continue
                
                # Process with threaded pipeline
                process_with_threads(text, manager)
                print("--- Ready ---\n")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            manager.leds.all_off()
            break
        except Exception as e:
            manager.leds.all_off()
            print(f"\nError: {e}")


def run_text_mode(manager: ConversationManager):
    """Text-based conversation mode"""
    print("\n" + "=" * 50)
    print("   KENZA AI - Text Mode")
    print("=" * 50)
    print("   Type 'quit' to exit, 'clear' to reset")
    print("=" * 50 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit"]:
                print("\nKenza: Goodbye! 👋\n")
                break
            
            response = manager.chat(user_input)
            print(f"Kenza: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nKenza: Goodbye! 👋\n")
            break


def run_test_mode(manager: ConversationManager):
    """Test all components"""
    print("\n" + "=" * 50)
    print("   KENZA AI - Component Test")
    print("=" * 50 + "\n")
    
    # Test providers
    print("AI Providers:")
    print(f"  Gemini: {'✓ Ready' if manager.gemini.is_available() else '✗ Not configured'}")
    print(f"  Llama:  {'✓ Ready' if manager.llama.is_available() else '✗ Model not found'}")
    
    # Test audio
    print("\nAudio:")
    print(f"  STT:    {'✓ Ready' if manager.stt.is_available() else '✗ Not available'}")
    print(f"  TTS:    {'✓ Ready' if manager.tts._mixer_initialized else '✗ Not available'}")
    
    # Test conversation
    print("\nTesting conversation...")
    try:
        response = manager.chat("Say 'test successful' in exactly two words.")
        print(f"Response: {response}")
        print("✓ Conversation test passed!")
    except Exception as e:
        print(f"✗ Test failed: {e}")
    
    # Test TTS
    print("\nTesting TTS...")
    try:
        manager.tts.speak("Kenza AI test successful.", manager.leds)
        print("✓ TTS test passed!")
    except Exception as e:
        print(f"✗ TTS test failed: {e}")
    
    print("\n" + "=" * 50 + "\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Kenza AI - Conversational AI Module")
    parser.add_argument("--text", action="store_true", help="Text-only mode (no voice)")
    parser.add_argument("--no-wake", action="store_true", help="Skip wake word, always listening")
    parser.add_argument("--test", action="store_true", help="Run component tests")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Load configuration
    config = Config.load(args.config)
    
    # Initialize conversation manager
    manager = ConversationManager(config)
    
    try:
        if args.test:
            run_test_mode(manager)
        elif args.text:
            run_text_mode(manager)
        else:
            run_voice_mode(manager, use_wake_word=not args.no_wake)
    finally:
        manager.leds.all_off()


if __name__ == "__main__":
    main()

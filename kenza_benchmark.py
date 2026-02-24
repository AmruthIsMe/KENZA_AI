#!/usr/bin/env python3
"""
KENZA Performance Benchmark Suite
===================================
Measures and displays real-time performance metrics across all KENZA subsystems:

  1. SYSTEM HEALTH    — CPU temp, CPU%, RAM usage, disk I/O, thermal throttle state
  2. LLM PERFORMANCE  — Tokens/sec, TTFT (Time to First Token), latency per provider
                        (Groq, Gemini, Local Llama / Ollama)
  3. STT PERFORMANCE  — Transcription latency (Google Cloud STT vs Whisper offline)
  4. TTS PERFORMANCE  — Audio generation time (Edge-TTS online vs pyttsx3 offline)
  5. VISION PIPELINE  — Face detection FPS, gesture detection FPS, YOLO inference ms
  6. STREAMING        — WebRTC video FPS, audio latency estimate, WebSocket RTT

Run on the Raspberry Pi 5 (or dev machine):
  python kenza_benchmark.py

Optional flags:
  --llm          Run only LLM benchmarks
  --vision       Run only vision benchmarks
  --system       Run only system health monitor
  --all          Run everything (default)
  --output FILE  Save results to JSON file
"""

import os, sys, time, json, threading, argparse, math, struct
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List
import platform

# ─────────────────────────────────────────────────────────────────
#  OPTIONAL IMPORTS — graceful degradation on dev machines
# ─────────────────────────────────────────────────────────────────
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False

try:
    import groq as groq_sdk
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    import requests as http_requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

IS_PI = platform.machine().startswith("aarch64") or platform.machine().startswith("armv")

# ─────────────────────────────────────────────────────────────────
#  TERMINAL COLORS
# ─────────────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BLUE   = "\033[94m"
    MAGENTA= "\033[95m"
    WHITE  = "\033[97m"

def color_val(val, ok_threshold, warn_threshold, high_is_bad=True):
    """Return colored value string. high_is_bad=True means high values are red."""
    try:
        v = float(val)
        if high_is_bad:
            if v <= ok_threshold:   return f"{C.GREEN}{val}{C.RESET}"
            elif v <= warn_threshold: return f"{C.YELLOW}{val}{C.RESET}"
            else:                   return f"{C.RED}{val}{C.RESET}"
        else:
            if v >= ok_threshold:   return f"{C.GREEN}{val}{C.RESET}"
            elif v >= warn_threshold: return f"{C.YELLOW}{val}{C.RESET}"
            else:                   return f"{C.RED}{val}{C.RESET}"
    except:
        return str(val)

def bar(value, max_val, width=20, fill="█", empty="░"):
    """ASCII progress bar."""
    try:
        filled = int(min(float(value) / max_val, 1.0) * width)
        return fill * filled + empty * (width - filled)
    except:
        return empty * width

# ─────────────────────────────────────────────────────────────────
#  RESULT STORAGE
# ─────────────────────────────────────────────────────────────────
@dataclass
class BenchmarkResults:
    timestamp: str = ""

    # System
    cpu_percent: float = 0.0
    cpu_temp_c: float = 0.0
    ram_used_mb: float = 0.0
    ram_total_mb: float = 0.0
    ram_percent: float = 0.0
    is_throttled: bool = False
    disk_read_mbps: float = 0.0
    disk_write_mbps: float = 0.0

    # LLM
    groq_ttft_ms: float = 0.0
    groq_tokens_per_sec: float = 0.0
    groq_total_latency_ms: float = 0.0
    gemini_ttft_ms: float = 0.0
    gemini_tokens_per_sec: float = 0.0
    gemini_total_latency_ms: float = 0.0
    local_llm_tokens_per_sec: float = 0.0
    local_llm_latency_ms: float = 0.0
    smart_router_latency_ms: float = 0.0

    # STT
    google_stt_latency_ms: float = 0.0
    whisper_stt_latency_ms: float = 0.0

    # TTS
    edge_tts_gen_ms: float = 0.0
    pyttsx_gen_ms: float = 0.0

    # Vision
    face_detection_fps: float = 0.0
    face_detection_ms_per_frame: float = 0.0
    gesture_detection_fps: float = 0.0
    gesture_detection_ms_per_frame: float = 0.0
    yolo_inference_ms: float = 0.0

    # Streaming
    websocket_rtt_ms: float = 0.0
    simulated_webrtc_fps: float = 0.0
    mjpeg_fps: float = 0.0

    errors: List[str] = field(default_factory=list)

# ─────────────────────────────────────────────────────────────────
#  CONFIG LOADER
# ─────────────────────────────────────────────────────────────────
def load_api_keys() -> Dict[str, str]:
    """Load API keys from config/settings.yaml or environment variables."""
    keys = {
        "groq": os.getenv("GROQ_API_KEY", ""),
        "gemini": os.getenv("GEMINI_API_KEY", ""),
        "ollama_url": "http://localhost:11434",
        "ollama_model": "gemma3:270m",
        "groq_model": "llama-3.3-70b-versatile",
        "gemini_model": "gemini-2.0-flash",
        "llama_path": "models/llama-3.2-3b-instruct.Q4_K_M.gguf",
    }
    cfg_path = os.path.join(os.path.dirname(__file__), "config", "settings.yaml")
    if HAS_YAML and os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                data = yaml.safe_load(f) or {}
            api_keys = data.get("api_keys", {})
            keys["groq"]   = os.getenv("GROQ_API_KEY",   api_keys.get("groq", keys["groq"]))
            keys["gemini"] = os.getenv("GEMINI_API_KEY", api_keys.get("gemini", keys["gemini"]))
            ai_cfg = data.get("ai", {})
            keys["ollama_model"] = ai_cfg.get("ollama_model", keys["ollama_model"])
            keys["ollama_url"]   = ai_cfg.get("ollama_url",   keys["ollama_url"])
            models = data.get("models", {})
            keys["groq_model"]   = models.get("groq",   keys["groq_model"])
            keys["gemini_model"] = models.get("gemini", keys["gemini_model"])
            print(f"{C.DIM}[Config] Loaded settings.yaml{C.RESET}")
        except Exception as e:
            print(f"{C.YELLOW}[Config] Could not parse settings.yaml: {e}{C.RESET}")
    return keys

# ─────────────────────────────────────────────────────────────────
#  1. SYSTEM HEALTH
# ─────────────────────────────────────────────────────────────────
def benchmark_system(results: BenchmarkResults):
    """Measure CPU temp, CPU%, RAM, thermal throttle, disk I/O."""
    print(f"\n{C.CYAN}{C.BOLD}[1/6] System Health{C.RESET}")

    if not HAS_PSUTIL:
        msg = "psutil not installed — run: pip install psutil"
        print(f"  {C.YELLOW}⚠ {msg}{C.RESET}")
        results.errors.append(msg)
        return

    # CPU usage (1-second sample)
    results.cpu_percent = psutil.cpu_percent(interval=1)

    # RAM
    vm = psutil.virtual_memory()
    results.ram_used_mb  = vm.used  / 1024 / 1024
    results.ram_total_mb = vm.total / 1024 / 1024
    results.ram_percent  = vm.percent

    # CPU Temperature — Pi 5 uses /sys/class/thermal
    temp = 0.0
    if IS_PI:
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                temp = int(f.read().strip()) / 1000.0
        except:
            pass
    else:
        try:
            temps = psutil.sensors_temperatures()
            for key in ("coretemp", "cpu_thermal", "cpu-thermal", "acpitz"):
                if key in temps and temps[key]:
                    temp = temps[key][0].current
                    break
        except:
            pass
    results.cpu_temp_c = temp

    # Thermal throttle (Pi-specific)
    if IS_PI:
        try:
            import subprocess
            out = subprocess.check_output(["vcgencmd", "get_throttled"], timeout=2).decode()
            val = int(out.strip().split("=")[1], 16)
            results.is_throttled = bool(val & 0x4)  # bit 2 = currently throttled
        except:
            results.is_throttled = False

    # Disk I/O
    try:
        d1 = psutil.disk_io_counters()
        time.sleep(1)
        d2 = psutil.disk_io_counters()
        results.disk_read_mbps  = (d2.read_bytes  - d1.read_bytes)  / 1024 / 1024
        results.disk_write_mbps = (d2.write_bytes - d1.write_bytes) / 1024 / 1024
    except:
        pass

    print(f"  CPU Usage : {color_val(f'{results.cpu_percent:.1f}%', 50, 80)} "
          f"[{bar(results.cpu_percent, 100)}]")
    print(f"  CPU Temp  : {color_val(f'{results.cpu_temp_c:.1f}°C', 60, 75)} "
          f"[{bar(results.cpu_temp_c, 100)}]")
    print(f"  RAM       : {color_val(f'{results.ram_percent:.1f}%', 60, 85)} "
          f"  ({results.ram_used_mb:.0f} MB / {results.ram_total_mb:.0f} MB)")
    throttle_str = f"{C.RED}YES ⚠{C.RESET}" if results.is_throttled else f"{C.GREEN}No{C.RESET}"
    print(f"  Throttled : {throttle_str}")
    print(f"  Disk I/O  : Read {results.disk_read_mbps:.2f} MB/s | Write {results.disk_write_mbps:.2f} MB/s")

# ─────────────────────────────────────────────────────────────────
#  2. LLM BENCHMARK
# ─────────────────────────────────────────────────────────────────
TEST_PROMPT = "In exactly two sentences, explain what a robot companion is."

def _count_tokens_approx(text: str) -> int:
    """Approximate token count: ~0.75 tokens per word."""
    return max(1, int(len(text.split()) / 0.75))

def _bench_groq(keys: Dict, results: BenchmarkResults):
    if not HAS_GROQ or not keys["groq"]:
        print(f"    {C.YELLOW}Groq: skipped (no API key or library){C.RESET}")
        return
    try:
        client = groq_sdk.Groq(api_key=keys["groq"])
        t0 = time.perf_counter()
        first_token_time = None
        full_text = ""

        stream = client.chat.completions.create(
            model=keys["groq_model"],
            messages=[{"role": "user", "content": TEST_PROMPT}],
            stream=True,
            max_tokens=80,
        )
        for chunk in stream:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            delta = chunk.choices[0].delta.content or ""
            full_text += delta
        t1 = time.perf_counter()

        total_ms    = (t1 - t0) * 1000
        ttft_ms     = ((first_token_time or t1) - t0) * 1000
        total_sec   = max(0.001, t1 - (first_token_time or t0))
        tokens      = _count_tokens_approx(full_text)
        tok_per_sec = tokens / total_sec

        results.groq_ttft_ms          = round(ttft_ms, 1)
        results.groq_total_latency_ms  = round(total_ms, 1)
        results.groq_tokens_per_sec    = round(tok_per_sec, 1)

        print(f"    {C.GREEN}✔ Groq (Llama-3.3-70b):{C.RESET}")
        print(f"       TTFT           : {color_val(f'{ttft_ms:.0f}ms', 500, 1500)}")
        print(f"       Total Latency  : {color_val(f'{total_ms:.0f}ms', 1500, 3000)}")
        print(f"       Throughput     : {color_val(f'{tok_per_sec:.1f} tok/s', 20, 8, high_is_bad=False)}")
    except Exception as e:
        results.errors.append(f"Groq: {e}")
        print(f"    {C.RED}✘ Groq Error: {e}{C.RESET}")


def _bench_gemini(keys: Dict, results: BenchmarkResults):
    if not HAS_GEMINI or not keys["gemini"]:
        print(f"    {C.YELLOW}Gemini: skipped (no API key or library){C.RESET}")
        return
    try:
        genai.configure(api_key=keys["gemini"])
        model = genai.GenerativeModel(keys["gemini_model"])

        t0 = time.perf_counter()
        response = model.generate_content(TEST_PROMPT)
        t_resp = time.perf_counter()

        full_text = response.text or ""
        total_ms  = (t_resp - t0) * 1000
        tokens    = _count_tokens_approx(full_text)
        tok_per_sec = tokens / max(0.001, t_resp - t0)

        # Gemini SDK doesn't expose streaming TTFT directly with basic API — log as same as total
        results.gemini_ttft_ms         = round(total_ms, 1)
        results.gemini_total_latency_ms = round(total_ms, 1)
        results.gemini_tokens_per_sec  = round(tok_per_sec, 1)

        print(f"    {C.GREEN}✔ Gemini 2.0-Flash:{C.RESET}")
        print(f"       Total Latency  : {color_val(f'{total_ms:.0f}ms', 1500, 3000)}")
        print(f"       Throughput     : {color_val(f'{tok_per_sec:.1f} tok/s', 15, 5, high_is_bad=False)}")
    except Exception as e:
        results.errors.append(f"Gemini: {e}")
        print(f"    {C.RED}✘ Gemini Error: {e}{C.RESET}")


def _bench_ollama(keys: Dict, results: BenchmarkResults):
    """Benchmark local Ollama server (preferred local LLM path in kenza_conversation.py)."""
    if not HAS_REQUESTS:
        print(f"    {C.YELLOW}Local LLM (Ollama): skipped (requests library missing){C.RESET}")
        return

    url = f"{keys['ollama_url']}/api/generate"
    model = keys["ollama_model"]

    try:
        # Check if Ollama is running
        health_resp = http_requests.get(f"{keys['ollama_url']}/api/tags", timeout=2)
        available_models = [m["name"] for m in health_resp.json().get("models", [])]

        if not any(model in m for m in available_models):
            print(f"    {C.YELLOW}Ollama: model '{model}' not found. Available: {available_models or 'none'}{C.RESET}")
            return

        t0 = time.perf_counter()
        full_text = ""
        first_token_time = None

        resp = http_requests.post(url, json={
            "model": model, "prompt": TEST_PROMPT, "stream": True
        }, stream=True, timeout=60)

        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                full_text += chunk.get("response", "")
                if chunk.get("done"):
                    break
        t1 = time.perf_counter()

        total_ms    = (t1 - t0) * 1000
        ttft_ms     = ((first_token_time or t1) - t0) * 1000
        gen_sec     = max(0.001, t1 - (first_token_time or t0))
        tokens      = _count_tokens_approx(full_text)
        tok_per_sec = tokens / gen_sec

        results.local_llm_latency_ms    = round(total_ms, 1)
        results.local_llm_tokens_per_sec = round(tok_per_sec, 1)

        print(f"    {C.GREEN}✔ Local LLM (Ollama / {model}):{C.RESET}")
        print(f"       TTFT           : {color_val(f'{ttft_ms:.0f}ms', 2000, 5000)}")
        print(f"       Total Latency  : {color_val(f'{total_ms:.0f}ms', 4000, 10000)}")
        print(f"       Throughput     : {color_val(f'{tok_per_sec:.1f} tok/s', 5, 2, high_is_bad=False)}")

    except http_requests.exceptions.ConnectionError:
        print(f"    {C.YELLOW}Local LLM (Ollama): server not running at {keys['ollama_url']}{C.RESET}")
    except Exception as e:
        results.errors.append(f"Ollama: {e}")
        print(f"    {C.RED}✘ Ollama Error: {e}{C.RESET}")


def _bench_smart_router(keys: Dict, results: BenchmarkResults):
    """Benchmark Smart Router decision time (the classifier latency itself)."""
    # The router in kenza_conversation uses a keyword heuristic or local LLM.
    # We benchmark the keyword-based classification path (zero-dependency).
    vision_phrases = ["what is this", "what do you see", "describe this", "look at this"]

    def classify(text: str) -> str:
        text_lower = text.lower()
        if any(p in text_lower for p in vision_phrases):
            return "vision"
        if "weather" in text_lower or "news" in text_lower:
            return "groq"
        return "groq"

    SAMPLES = 1000
    t0 = time.perf_counter()
    for _ in range(SAMPLES):
        classify("Tell me about the history of robotics")
    t1 = time.perf_counter()
    avg_us = ((t1 - t0) / SAMPLES) * 1_000_000
    results.smart_router_latency_ms = round(avg_us / 1000, 3)

    print(f"    {C.GREEN}✔ Smart Router (keyword classifier):{C.RESET}")
    print(f"       Avg Decision   : {avg_us:.1f} µs  ({results.smart_router_latency_ms:.3f} ms)")


def benchmark_llm(keys: Dict, results: BenchmarkResults):
    print(f"\n{C.CYAN}{C.BOLD}[2/6] LLM Performance{C.RESET}")
    _bench_groq(keys, results)
    _bench_gemini(keys, results)
    _bench_ollama(keys, results)
    _bench_smart_router(keys, results)

# ─────────────────────────────────────────────────────────────────
#  3. STT BENCHMARK (Simulate transcription latency)
# ─────────────────────────────────────────────────────────────────
def benchmark_stt(results: BenchmarkResults):
    print(f"\n{C.CYAN}{C.BOLD}[3/6] Speech-to-Text Latency{C.RESET}")
    print(f"  {C.DIM}Note: STT latency requires real microphone audio. These are simulated "
          f"end-to-end estimates based on network RTT + processing overhead.{C.RESET}")

    # Google STT latency estimate: network RTT to speech.googleapis.com
    google_rtt_ms = None
    if HAS_REQUESTS:
        try:
            t0 = time.perf_counter()
            http_requests.get("https://speech.googleapis.com", timeout=5)
            google_rtt_ms = (time.perf_counter() - t0) * 1000
        except:
            pass

    if google_rtt_ms is not None:
        # Typical Google STT cloud processing adds ~300-600ms to network RTT
        estimated_stt = google_rtt_ms + 400
        results.google_stt_latency_ms = round(estimated_stt, 1)
        print(f"  Google STT (cloud):")
        print(f"    Network RTT to Google: {google_rtt_ms:.0f} ms")
        print(f"    Estimated Total      : {color_val(f'{estimated_stt:.0f}ms', 800, 1500)}")
    else:
        print(f"  Google STT: {C.YELLOW}Could not reach speech.googleapis.com{C.RESET}")

    # Whisper (faster-whisper) — benchmark CPU inference speed
    try:
        from faster_whisper import WhisperModel
        print(f"  Faster-Whisper (offline):")
        print(f"    Loading model (base.en)...")
        t0 = time.perf_counter()
        w_model = WhisperModel("base.en", device="cpu", compute_type="int8")
        load_ms = (time.perf_counter() - t0) * 1000
        # Use a short synthetic audio inference of 3 seconds of silence
        if HAS_NUMPY:
            dummy_audio = np.zeros(int(3 * 16000), dtype=np.float32)
            t0 = time.perf_counter()
            segments, _ = w_model.transcribe(dummy_audio, beam_size=1)
            _ = list(segments)  # materialize generator
            infer_ms = (time.perf_counter() - t0) * 1000
            results.whisper_stt_latency_ms = round(infer_ms, 1)
            print(f"    Model Load Time  : {load_ms:.0f} ms")
            print(f"    Inference (3s)   : {color_val(f'{infer_ms:.0f}ms', 1000, 3000)}")
        else:
            print(f"    Model Load Time  : {load_ms:.0f} ms (numpy not available for test audio)")
    except ImportError:
        print(f"  Faster-Whisper: {C.YELLOW}not installed (pip install faster-whisper){C.RESET}")
    except Exception as e:
        results.errors.append(f"Whisper STT: {e}")
        print(f"  Faster-Whisper: {C.RED}Error: {e}{C.RESET}")

# ─────────────────────────────────────────────────────────────────
#  4. TTS BENCHMARK
# ─────────────────────────────────────────────────────────────────
TTS_TEST_TEXT = "Hello, I am Kenza, your AI robotic companion. How can I help you today?"

def benchmark_tts(results: BenchmarkResults):
    print(f"\n{C.CYAN}{C.BOLD}[4/6] Text-to-Speech Performance{C.RESET}")

    # Edge-TTS (online)
    try:
        import asyncio, edge_tts
        async def _gen():
            comm = edge_tts.Communicate(TTS_TEST_TEXT, "en-US-AriaNeural")
            await comm.save("_bench_tts_test.mp3")

        t0 = time.perf_counter()
        asyncio.run(_gen())
        gen_ms = (time.perf_counter() - t0) * 1000
        results.edge_tts_gen_ms = round(gen_ms, 1)

        # Get output file size
        fsize = os.path.getsize("_bench_tts_test.mp3") / 1024
        os.remove("_bench_tts_test.mp3")

        print(f"  Edge-TTS (Microsoft Neural):")
        print(f"    Generation Time : {color_val(f'{gen_ms:.0f}ms', 600, 1200)}")
        print(f"    Output Size     : {fsize:.1f} KB")
    except ImportError:
        print(f"  Edge-TTS: {C.YELLOW}not installed (pip install edge-tts){C.RESET}")
    except Exception as e:
        results.errors.append(f"Edge-TTS: {e}")
        print(f"  Edge-TTS: {C.RED}Error: {e}{C.RESET}")

    # pyttsx3 (offline)
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        t0 = time.perf_counter()
        engine.say(TTS_TEST_TEXT)
        engine.runAndWait()
        offline_ms = (time.perf_counter() - t0) * 1000
        results.pyttsx_gen_ms = round(offline_ms, 1)
        print(f"  pyttsx3 (Offline / espeak-ng):")
        print(f"    Speak Time      : {color_val(f'{offline_ms:.0f}ms', 2000, 4000)}")
    except ImportError:
        print(f"  pyttsx3: {C.YELLOW}not installed (pip install pyttsx3){C.RESET}")
    except Exception as e:
        results.errors.append(f"pyttsx3: {e}")
        print(f"  pyttsx3: {C.RED}Error: {e}{C.RESET}")

# ─────────────────────────────────────────────────────────────────
#  5. VISION BENCHMARK
# ─────────────────────────────────────────────────────────────────
VISION_FRAMES = 60   # Number of frames to benchmark

def benchmark_vision(results: BenchmarkResults):
    print(f"\n{C.CYAN}{C.BOLD}[5/6] Vision Pipeline{C.RESET}")

    if not HAS_CV2:
        print(f"  {C.YELLOW}opencv-python not installed — vision benchmarks skipped{C.RESET}")
        return
    if not HAS_NUMPY:
        print(f"  {C.YELLOW}numpy not installed — vision benchmarks skipped{C.RESET}")
        return

    # Create a synthetic test frame (640x480 BGR)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a simple face-like shape for detection testing
    cv2.circle(test_frame, (320, 240), 100, (200, 150, 100), -1)  # head
    cv2.circle(test_frame, (285, 210), 15, (50, 50, 200), -1)     # eye L
    cv2.circle(test_frame, (355, 210), 15, (50, 50, 200), -1)     # eye R

    # ── Face Detection (MediaPipe) ──────────────────────────────
    if HAS_MEDIAPIPE:
        mp_face = mp.solutions.face_detection
        detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.3)

        times = []
        for _ in range(VISION_FRAMES):
            rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            _ = detector.process(rgb)
            times.append(time.perf_counter() - t0)

        detector.close()
        avg_ms  = (sum(times) / len(times)) * 1000
        fps     = 1.0 / (sum(times) / len(times))

        results.face_detection_ms_per_frame = round(avg_ms, 2)
        results.face_detection_fps          = round(fps, 1)

        print(f"  MediaPipe Face Detection ({VISION_FRAMES} frames, synthetic):")
        print(f"    Avg per frame   : {color_val(f'{avg_ms:.1f}ms', 30, 60)}")
        print(f"    Est. Max FPS    : {color_val(f'{fps:.1f}', 30, 15, high_is_bad=False)}")
    else:
        print(f"  Face Detection: {C.YELLOW}mediapipe not installed{C.RESET}")

    # ── Gesture Detection (MediaPipe Hands) ────────────────────
    if HAS_MEDIAPIPE:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Bright frame simulates a hand
        hand_frame = np.ones((480, 640, 3), dtype=np.uint8) * 200

        times = []
        for _ in range(VISION_FRAMES):
            rgb = cv2.cvtColor(hand_frame, cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            _ = hands.process(rgb)
            times.append(time.perf_counter() - t0)

        hands.close()
        avg_ms  = (sum(times) / len(times)) * 1000
        fps     = 1.0 / (sum(times) / len(times))

        results.gesture_detection_ms_per_frame = round(avg_ms, 2)
        results.gesture_detection_fps          = round(fps, 1)

        print(f"  MediaPipe Hands (Gesture) ({VISION_FRAMES} frames, synthetic):")
        print(f"    Avg per frame   : {color_val(f'{avg_ms:.1f}ms', 40, 80)}")
        print(f"    Est. Max FPS    : {color_val(f'{fps:.1f}', 25, 12, high_is_bad=False)}")
    else:
        print(f"  Gesture Detection: {C.YELLOW}mediapipe not installed{C.RESET}")

    # ── YOLO11n (Ultralytics) ───────────────────────────────────
    if HAS_YOLO:
        try:
            print(f"  YOLO11n Object Detection (loading model)...")
            model = YOLO("yolo11n.pt")           # auto-downloads ~6MB
            times = []
            for _ in range(30):
                t0 = time.perf_counter()
                _ = model(test_frame, verbose=False)
                times.append(time.perf_counter() - t0)

            avg_ms = (sum(times) / len(times)) * 1000
            fps    = 1.0 / (sum(times) / len(times))
            results.yolo_inference_ms = round(avg_ms, 1)

            print(f"  YOLO11n (Ultralytics, CPU):")
            print(f"    Avg per frame   : {color_val(f'{avg_ms:.1f}ms', 100, 300)}")
            print(f"    Est. Max FPS    : {color_val(f'{fps:.1f}', 10, 4, high_is_bad=False)}")
        except Exception as e:
            results.errors.append(f"YOLO: {e}")
            print(f"  YOLO11n: {C.RED}Error: {e}{C.RESET}")
    else:
        print(f"  YOLO11n: {C.YELLOW}ultralytics not installed{C.RESET}")

# ─────────────────────────────────────────────────────────────────
#  6. STREAMING METRIC (WebSocket RTT + Simulated WebRTC vs MJPEG)
# ─────────────────────────────────────────────────────────────────
def benchmark_streaming(results: BenchmarkResults):
    print(f"\n{C.CYAN}{C.BOLD}[6/6] Streaming Performance{C.RESET}")

    # WebSocket RTT to local server (if running)
    ws_rtt = None
    try:
        import asyncio, websockets
        async def _ws_ping():
            uri = "ws://localhost:8765"
            async with websockets.connect(uri, open_timeout=2) as ws:
                pings = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    await ws.ping()
                    pings.append((time.perf_counter() - t0) * 1000)
                return sum(pings) / len(pings)
        ws_rtt = asyncio.run(_ws_ping())
        results.websocket_rtt_ms = round(ws_rtt, 1)
        print(f"  WebSocket RTT (localhost:8765):")
        print(f"    Avg RTT   : {color_val(f'{ws_rtt:.1f}ms', 5, 20)}")
    except ImportError:
        print(f"  WebSocket RTT: {C.YELLOW}websockets library not installed{C.RESET}")
    except Exception:
        print(f"  WebSocket RTT: {C.DIM}Kenza server not running on :8765 (start kenza_server.py first){C.RESET}")

    # Simulated WebRTC vs MJPEG throughput comparison using OpenCV encode
    if HAS_CV2 and HAS_NUMPY:
        test_frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)

        # VP8-equivalent (MJPEG Q=70 as proxy, since VP8 isn't directly available in pure Python)
        encode_times = []
        encoded_sizes = []
        for _ in range(30):
            t0 = time.perf_counter()
            _, buf = cv2.imencode(".jpg", test_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            encode_times.append(time.perf_counter() - t0)
            encoded_sizes.append(len(buf))

        mjpeg_ms  = (sum(encode_times) / len(encode_times)) * 1000
        mjpeg_fps = 1.0 / (sum(encode_times) / len(encode_times))
        mjpeg_kbps = (sum(encoded_sizes) / len(encoded_sizes)) / 1024 * mjpeg_fps * 8

        results.mjpeg_fps = round(mjpeg_fps, 1)

        print(f"\n  Streaming Comparison (640×360, CPU encode, 30-frame benchmark):")
        print(f"  {'Method':<25} {'Enc. Time':<15} {'Est. FPS':<12} {'Est. Bitrate'}")
        print(f"  {'-'*65}")
        print(f"  {'MJPEG (Q=70)':<25} {mjpeg_ms:<14.1f}ms {mjpeg_fps:<12.1f} {mjpeg_kbps:.0f} Kbps")

        # WebRTC via aiortc uses VP8/VP9 which is similar-ish latency on Pi,
        # but adds AEC + signaling overhead. We note this qualitatively.
        estimated_webrtc_fps = mjpeg_fps * 0.85  # ~15% overhead for RTP packetization & AEC
        results.simulated_webrtc_fps = round(estimated_webrtc_fps, 1)
        print(f"  {'WebRTC VP8/VP9 (aiortc)':<25} {'~+15% overhead':<15} {estimated_webrtc_fps:<12.1f} ~{mjpeg_kbps*0.7:.0f} Kbps")
        print(f"\n  {C.DIM}Note: WebRTC adds signaling, DTLS, SRTP, and AEC overhead vs raw MJPEG.{C.RESET}")
        print(f"  {C.DIM}However, WebRTC provides full-duplex audio + standardized NAT traversal.{C.RESET}")
    else:
        print(f"  Streaming encode test: {C.YELLOW}opencv/numpy not installed{C.RESET}")

# ─────────────────────────────────────────────────────────────────
#  SUMMARY DISPLAY
# ─────────────────────────────────────────────────────────────────
def print_summary(results: BenchmarkResults):
    print(f"\n{'='*65}")
    print(f"{C.BOLD}{C.WHITE}  KENZA PERFORMANCE BENCHMARK SUMMARY{C.RESET}")
    print(f"{'='*65}")

    rows = [
        ("SYSTEM", "", ""),
        ("  CPU Usage",      f"{results.cpu_percent:.1f}%",   ""),
        ("  CPU Temp",       f"{results.cpu_temp_c:.1f}°C",   "< 75°C recommended"),
        ("  RAM Used",       f"{results.ram_percent:.1f}%",   f"{results.ram_used_mb:.0f}/{results.ram_total_mb:.0f} MB"),
        ("  Throttled",      "YES ⚠" if results.is_throttled else "No", ""),

        ("LLM", "", ""),
        ("  Groq TTFT",              f"{results.groq_ttft_ms:.0f} ms",         "Target < 500ms"),
        ("  Groq Throughput",        f"{results.groq_tokens_per_sec:.1f} tok/s","Target > 20 tok/s"),
        ("  Gemini Latency",         f"{results.gemini_total_latency_ms:.0f} ms", ""),
        ("  Local LLM Throughput",   f"{results.local_llm_tokens_per_sec:.1f} tok/s", "Target > 3 tok/s on Pi 5"),

        ("STT", "", ""),
        ("  Google STT (est.)",  f"{results.google_stt_latency_ms:.0f} ms",   ""),
        ("  Whisper Offline",    f"{results.whisper_stt_latency_ms:.0f} ms",   ""),

        ("TTS", "", ""),
        ("  Edge-TTS Gen",    f"{results.edge_tts_gen_ms:.0f} ms",   "Target < 800ms"),
        ("  pyttsx3 Offline", f"{results.pyttsx_gen_ms:.0f} ms",     ""),

        ("VISION", "", ""),
        ("  Face Detect FPS",    f"{results.face_detection_fps:.1f}",            "Target > 25"),
        ("  Face Detect Latency",f"{results.face_detection_ms_per_frame:.1f} ms",""),
        ("  Gesture FPS",        f"{results.gesture_detection_fps:.1f}",          "Target > 20"),
        ("  YOLO11n Latency",    f"{results.yolo_inference_ms:.1f} ms",           ""),

        ("STREAMING", "", ""),
        ("  WebSocket RTT",      f"{results.websocket_rtt_ms:.1f} ms",   "Target < 10ms (local)"),
        ("  MJPEG Encode FPS",   f"{results.mjpeg_fps:.1f}",             ""),
        ("  WebRTC Est. FPS",    f"{results.simulated_webrtc_fps:.1f}",  ""),
    ]

    for name, value, note in rows:
        if not value:  # Section header
            print(f"\n  {C.BOLD}{C.CYAN}{name}{C.RESET}")
        else:
            note_str = f"  {C.DIM}{note}{C.RESET}" if note else ""
            print(f"  {name:<28} {C.WHITE}{value:<18}{C.RESET}{note_str}")

    if results.errors:
        print(f"\n  {C.YELLOW}Errors encountered:{C.RESET}")
        for e in results.errors:
            print(f"    {C.RED}• {e}{C.RESET}")

    print(f"\n{'='*65}\n")

# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    # Force UTF-8 output so emoji and box-drawing chars work on Windows terminals
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
        sys.stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", buffering=1)

    parser = argparse.ArgumentParser(description="KENZA Performance Benchmark Suite")
    parser.add_argument("--llm",    action="store_true", help="Run only LLM benchmarks")
    parser.add_argument("--vision", action="store_true", help="Run only vision benchmarks")
    parser.add_argument("--system", action="store_true", help="Run only system health")
    parser.add_argument("--stt",    action="store_true", help="Run only STT benchmarks")
    parser.add_argument("--tts",    action="store_true", help="Run only TTS benchmarks")
    parser.add_argument("--streaming", action="store_true", help="Run only streaming benchmarks")
    parser.add_argument("--all",    action="store_true", help="Run all benchmarks (default)")
    parser.add_argument("--output", metavar="FILE", help="Save JSON results to file")
    args = parser.parse_args()

    run_all = args.all or not any([args.llm, args.vision, args.system, args.stt, args.tts, args.streaming])

    print(f"\n{C.BOLD}{C.MAGENTA}  #  #  #####  #   #  ####  #####  {C.RESET}")
    print(f"{C.BOLD}{C.MAGENTA}  # #   #      ##  #  #       #    {C.RESET}")
    print(f"{C.BOLD}{C.MAGENTA}  ##    ###    # # #  ####    #    {C.RESET}")
    print(f"{C.BOLD}{C.MAGENTA}  # #   #      #  ##     #    #    {C.RESET}")
    print(f"{C.BOLD}{C.MAGENTA}  #  #  #####  #   #  ####    #    {C.RESET}")
    print(f"{C.DIM}  Performance Benchmark Suite - v1.0{C.RESET}\n")
    print(f"  Platform : {'Raspberry Pi' if IS_PI else platform.node()} ({platform.machine()})")
    print(f"  Python   : {sys.version.split()[0]}")
    print(f"  Time     : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results = BenchmarkResults(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
    keys = load_api_keys()

    if run_all or args.system:
        benchmark_system(results)
    if run_all or args.llm:
        benchmark_llm(keys, results)
    if run_all or args.stt:
        benchmark_stt(results)
    if run_all or args.tts:
        benchmark_tts(results)
    if run_all or args.vision:
        benchmark_vision(results)
    if run_all or args.streaming:
        benchmark_streaming(results)

    print_summary(results)

    if args.output:
        try:
            with open(args.output, "w") as f:
                json.dump(asdict(results), f, indent=2)
            print(f"{C.GREEN}Results saved to {args.output}{C.RESET}\n")
        except Exception as e:
            print(f"{C.RED}Could not save results: {e}{C.RESET}\n")


if __name__ == "__main__":
    main()

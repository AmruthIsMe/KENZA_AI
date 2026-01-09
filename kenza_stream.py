#!/usr/bin/env python3
"""
Kenza Robot - Two-Way Audio/Video Streaming
============================================
Extends the existing video streaming to support:
- Outbound: Pi Camera + Pi Microphone → Browser
- Inbound: Browser Microphone → Pi Speaker (with Echo Cancellation)

Based on MediaMTX + aiortc + WHIP protocol.

Usage:
    python kenza_stream.py              # Start streaming
    python kenza_stream.py --no-video   # Audio only
    python kenza_stream.py --test-audio # Test audio devices

Requirements:
    pip install aiortc aiohttp av numpy pyaudio
    
    On Pi: sudo apt-get install portaudio19-dev
"""

import os
import asyncio
import aiohttp
import av
import numpy as np
import subprocess
import time
import sys
import re
import requests
import socket
import threading
import queue
import warnings
from collections import deque
import fractions

# Suppress warnings
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
warnings.filterwarnings("ignore")

# Try to import Pi-specific modules
try:
    from picamera2 import Picamera2
    HAS_PICAMERA = True
except ImportError:
    HAS_PICAMERA = False
    print("⚠ PiCamera2 not available (not on Pi?)")

try:
    import miniupnpc
    HAS_UPNP = True
except ImportError:
    HAS_UPNP = False

try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False
    print("⚠ PyAudio not available")

from aiortc import (
    RTCPeerConnection,
    RTCConfiguration,
    RTCIceServer,
    RTCSessionDescription,
    VideoStreamTrack,
    MediaStreamTrack
)
from aiortc.contrib.media import MediaPlayer


# =============================================================================
# CONFIGURATION
# =============================================================================

class StreamConfig:
    # Video
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 360
    FRAME_RATE = 30
    
    # Audio
    AUDIO_SAMPLE_RATE = 48000
    AUDIO_CHANNELS = 1
    AUDIO_CHUNK_SIZE = 1024
    
    # MediaMTX
    MEDIA_MTX_PATH = "./mediamtx"
    CONFIG_FILE = "mediamtx.yml"
    STREAM_NAME = "kenza"
    
    # Echo Cancellation
    AEC_ENABLED = True
    AEC_FILTER_LENGTH = 4096  # Adaptive filter taps
    AEC_MU = 0.01  # Learning rate (NLMS)
    
    @property
    def local_push_url(self):
        return f"http://127.0.0.1:8889/{self.STREAM_NAME}/whip"


CONFIG = StreamConfig()


# =============================================================================
# NETWORK UTILITIES
# =============================================================================

def get_local_ip():
    """Gets the Pi's local IP address"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


def setup_global_network(local_ip):
    """Auto-configures Router (UPnP) and MediaMTX config"""
    print("[NETWORK] Configuring Global Access...")
    
    public_ip = "Unknown"
    
    if HAS_UPNP:
        try:
            upnp = miniupnpc.UPnP()
            upnp.discoverdelay = 200
            upnp.discover()
            upnp.selectigd()
            public_ip = upnp.externalipaddress()
            
            upnp.addportmapping(8889, 'TCP', local_ip, 8889, 'MediaMTX_TCP', '')
            upnp.addportmapping(8189, 'UDP', local_ip, 8189, 'MediaMTX_UDP', '')
            print(f"    > Router Configured! Public IP: {public_ip}")
        except Exception as e:
            print(f"    ! UPnP Failed: {e}")
    
    # Fallback to get public IP
    if public_ip == "Unknown":
        try:
            public_ip = requests.get('https://api.ipify.org', timeout=3).text
        except:
            pass

    # Update MediaMTX Config
    if public_ip and public_ip != "Unknown":
        try:
            if os.path.exists(CONFIG.CONFIG_FILE):
                with open(CONFIG.CONFIG_FILE, 'r') as f:
                    lines = f.readlines()
                with open(CONFIG.CONFIG_FILE, 'w') as f:
                    for line in lines:
                        if "webrtcAdditionalHosts" in line:
                            f.write(f'webrtcAdditionalHosts: [ "{public_ip}" ]\n')
                        else:
                            f.write(line)
                print("    > mediamtx.yml updated with Public IP.")
        except Exception as e:
            print(f"    ! Config update failed: {e}")
            
    return public_ip


# =============================================================================
# AUDIO DEVICE UTILITIES (USB + Bluetooth + Default)
# =============================================================================

def get_bluetooth_audio_devices():
    """
    Find connected Bluetooth audio devices via PulseAudio.
    Returns: (input_device, output_device) tuple or (None, None)
    """
    bt_input = None
    bt_output = None
    
    try:
        # Check for Bluetooth sources (microphones)
        sources = subprocess.check_output(
            "pactl list sources short 2>/dev/null", 
            shell=True
        ).decode()
        
        for line in sources.split('\n'):
            if 'bluez' in line.lower() or 'bluetooth' in line.lower():
                parts = line.split('\t')
                if len(parts) >= 2:
                    bt_input = parts[1]  # Device name
                    break
        
        # Check for Bluetooth sinks (speakers/headphones)
        sinks = subprocess.check_output(
            "pactl list sinks short 2>/dev/null", 
            shell=True
        ).decode()
        
        for line in sinks.split('\n'):
            if 'bluez' in line.lower() or 'bluetooth' in line.lower():
                parts = line.split('\t')
                if len(parts) >= 2:
                    bt_output = parts[1]  # Device name
                    break
                    
    except Exception as e:
        pass
    
    return bt_input, bt_output


def get_audio_input_device():
    """
    Find the best audio input device.
    Priority: USB Mic > Bluetooth > System Default
    """
    print("[AUDIO] Scanning for Input Devices...")
    
    # 1. Try USB Mic first (best quality, lowest latency)
    try:
        result = subprocess.check_output("arecord -l", shell=True).decode()
        match = re.search(r"card (\d+):.*USB", result, re.IGNORECASE)
        if match:
            card_id = match.group(1)
            print(f"    > Found USB Audio Input: plughw:{card_id},0")
            return f"plughw:{card_id},0"
    except:
        pass
    
    # 2. Try Bluetooth (TWS/Headset)
    bt_input, _ = get_bluetooth_audio_devices()
    if bt_input:
        print(f"    > Found Bluetooth Input: {bt_input}")
        # For Bluetooth, we use PulseAudio's "default" which routes to BT
        # Or we can use the specific device via pulse
        return "default"  # PulseAudio handles BT routing
    
    # 3. Fallback to system default
    print("    > Using System Default Input (PulseAudio)")
    return "default"


def get_audio_output_device():
    """
    Find the best audio output device.
    Priority: USB Speaker > Bluetooth > Pi Audio Jack > Default
    """
    print("[AUDIO] Scanning for Output Devices...")
    
    # 1. Try USB Audio output first
    try:
        result = subprocess.check_output("aplay -l", shell=True).decode()
        match = re.search(r"card (\d+):.*USB", result, re.IGNORECASE)
        if match:
            card_id = match.group(1)
            print(f"    > Found USB Audio Output: plughw:{card_id},0")
            return f"plughw:{card_id},0"
    except:
        pass
    
    # 2. Try Bluetooth (TWS/Headphones)
    _, bt_output = get_bluetooth_audio_devices()
    if bt_output:
        print(f"    > Found Bluetooth Output: {bt_output}")
        return "default"  # PulseAudio routes to BT automatically
    
    # 3. Check for Pi headphone jack
    try:
        result = subprocess.check_output("aplay -l", shell=True).decode()
        if "Headphones" in result or "bcm2835" in result:
            print("    > Using Pi 3.5mm Audio Jack")
            return "default"
    except:
        pass
    
    # 4. Fallback
    print("    > Using System Default Output (PulseAudio)")
    return "default"


def get_bluetooth_device_index():
    """
    Get PyAudio device index for Bluetooth audio.
    Returns: (input_index, output_index) or (None, None)
    """
    if not HAS_PYAUDIO:
        return None, None
    
    bt_input_idx = None
    bt_output_idx = None
    
    try:
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            name = info['name'].lower()
            
            if 'bluez' in name or 'bluetooth' in name or 'bt' in name:
                if info['maxInputChannels'] > 0 and bt_input_idx is None:
                    bt_input_idx = i
                if info['maxOutputChannels'] > 0 and bt_output_idx is None:
                    bt_output_idx = i
        p.terminate()
    except:
        pass
    
    return bt_input_idx, bt_output_idx


def set_bluetooth_as_default():
    """
    Set connected Bluetooth device as the default PulseAudio sink/source.
    Call this to ensure audio routes through Bluetooth.
    """
    print("[BLUETOOTH] Setting as default audio device...")
    
    try:
        # Get Bluetooth sink
        sinks = subprocess.check_output(
            "pactl list sinks short", shell=True
        ).decode()
        
        for line in sinks.split('\n'):
            if 'bluez' in line.lower():
                sink_name = line.split('\t')[1]
                subprocess.run(
                    f"pactl set-default-sink {sink_name}", 
                    shell=True, check=True
                )
                print(f"    > Default sink set to: {sink_name}")
                break
        
        # Get Bluetooth source
        sources = subprocess.check_output(
            "pactl list sources short", shell=True
        ).decode()
        
        for line in sources.split('\n'):
            if 'bluez' in line.lower() and 'monitor' not in line.lower():
                source_name = line.split('\t')[1]
                subprocess.run(
                    f"pactl set-default-source {source_name}", 
                    shell=True, check=True
                )
                print(f"    > Default source set to: {source_name}")
                break
                
        return True
        
    except Exception as e:
        print(f"    ! Failed to set Bluetooth as default: {e}")
        return False


def list_audio_devices():
    """List all available audio devices including Bluetooth"""
    print("\n" + "=" * 60)
    print("  AUDIO DEVICES".center(60))
    print("=" * 60)
    
    # ALSA devices
    print("\n📢 ALSA OUTPUT DEVICES (Speakers):")
    try:
        result = subprocess.check_output("aplay -l 2>/dev/null", shell=True).decode()
        if result.strip():
            print(result)
        else:
            print("  No ALSA output devices found")
    except:
        print("  Could not list ALSA output devices")
    
    print("\n🎤 ALSA INPUT DEVICES (Microphones):")
    try:
        result = subprocess.check_output("arecord -l 2>/dev/null", shell=True).decode()
        if result.strip():
            print(result)
        else:
            print("  No ALSA input devices found")
    except:
        print("  Could not list ALSA input devices")
    
    # PulseAudio devices (includes Bluetooth)
    print("\n🔵 PULSEAUDIO SINKS (Output - includes Bluetooth):")
    try:
        result = subprocess.check_output(
            "pactl list sinks short 2>/dev/null", shell=True
        ).decode()
        if result.strip():
            for line in result.strip().split('\n'):
                parts = line.split('\t')
                if len(parts) >= 2:
                    is_bt = '🎧' if 'bluez' in parts[1].lower() else '  '
                    print(f"  {is_bt} {parts[1]}")
        else:
            print("  No PulseAudio sinks found")
    except:
        print("  PulseAudio not available")
    
    print("\n🔵 PULSEAUDIO SOURCES (Input - includes Bluetooth):")
    try:
        result = subprocess.check_output(
            "pactl list sources short 2>/dev/null", shell=True
        ).decode()
        if result.strip():
            for line in result.strip().split('\n'):
                parts = line.split('\t')
                if len(parts) >= 2 and 'monitor' not in parts[1].lower():
                    is_bt = '🎧' if 'bluez' in parts[1].lower() else '  '
                    print(f"  {is_bt} {parts[1]}")
        else:
            print("  No PulseAudio sources found")
    except:
        print("  PulseAudio not available")
    
    # Bluetooth devices
    print("\n🎧 BLUETOOTH AUDIO DEVICES:")
    bt_in, bt_out = get_bluetooth_audio_devices()
    if bt_in or bt_out:
        if bt_in:
            print(f"  Input:  {bt_in}")
        if bt_out:
            print(f"  Output: {bt_out}")
    else:
        print("  No Bluetooth audio devices connected")
        print("  To connect: bluetoothctl → connect <MAC>")
    
    # PyAudio devices
    if HAS_PYAUDIO:
        print("\n🐍 PYAUDIO DEVICES:")
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            in_ch = info['maxInputChannels']
            out_ch = info['maxOutputChannels']
            if in_ch > 0 or out_ch > 0:
                direction = []
                if in_ch > 0:
                    direction.append(f"in:{in_ch}")
                if out_ch > 0:
                    direction.append(f"out:{out_ch}")
                is_bt = '🎧' if 'bluez' in info['name'].lower() else '  '
                print(f"  {is_bt} [{i}] {info['name']} ({', '.join(direction)})")
        p.terminate()
    
    print("\n" + "=" * 60)


# =============================================================================
# ECHO CANCELLATION (NLMS Adaptive Filter)
# =============================================================================

class EchoCanceller:
    """
    Acoustic Echo Cancellation using Normalized Least Mean Squares (NLMS)
    
    Prevents the speaker output from feeding back into the microphone.
    """
    
    def __init__(self, filter_length=4096, mu=0.01):
        self.filter_length = filter_length
        self.mu = mu  # Step size / learning rate
        self.weights = np.zeros(filter_length, dtype=np.float32)
        self.speaker_buffer = deque(maxlen=filter_length)
        
        # Initialize buffer with zeros
        for _ in range(filter_length):
            self.speaker_buffer.append(0.0)
        
        print(f"[AEC] Initialized (filter_length={filter_length}, mu={mu})")
    
    def add_speaker_sample(self, samples: np.ndarray):
        """Add samples that were played to the speaker (reference signal)"""
        for sample in samples.flatten():
            self.speaker_buffer.append(float(sample))
    
    def cancel_echo(self, mic_samples: np.ndarray) -> np.ndarray:
        """
        Remove echo from microphone signal using NLMS
        
        Args:
            mic_samples: Raw microphone input (may contain echo)
            
        Returns:
            Cleaned signal with echo removed
        """
        output = np.zeros_like(mic_samples, dtype=np.float32)
        
        for i, mic_sample in enumerate(mic_samples.flatten()):
            # Get current speaker buffer as array
            x = np.array(list(self.speaker_buffer), dtype=np.float32)
            
            # Estimate echo: y_hat = w^T * x
            echo_estimate = np.dot(self.weights, x)
            
            # Error signal (cleaned output)
            error = mic_sample - echo_estimate
            output[i] = error
            
            # NLMS weight update: w = w + mu * error * x / (x^T * x + epsilon)
            norm = np.dot(x, x) + 1e-10  # Avoid division by zero
            self.weights += (self.mu * error * x) / norm
        
        return output.reshape(mic_samples.shape)
    
    def reset(self):
        """Reset the adaptive filter"""
        self.weights = np.zeros(self.filter_length, dtype=np.float32)
        self.speaker_buffer.clear()
        for _ in range(self.filter_length):
            self.speaker_buffer.append(0.0)


# =============================================================================
# AUDIO PLAYER (For Incoming Audio from Browser)
# =============================================================================

class AudioPlayer:
    """
    Plays incoming WebRTC audio through the Pi's speaker.
    Runs in a separate thread for low-latency playback.
    """
    
    def __init__(self, echo_canceller: EchoCanceller = None):
        self.echo_canceller = echo_canceller
        self.audio_queue = queue.Queue(maxsize=100)
        self.running = False
        self.thread = None
        self.pyaudio = None
        self.stream = None
        
    def start(self):
        """Start the audio playback thread"""
        if not HAS_PYAUDIO:
            print("[AUDIO PLAYER] PyAudio not available, skipping")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.thread.start()
        print("[AUDIO PLAYER] Started")
    
    def stop(self):
        """Stop the audio playback thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.pyaudio:
            self.pyaudio.terminate()
        print("[AUDIO PLAYER] Stopped")
    
    def _playback_loop(self):
        """Main playback loop running in thread"""
        try:
            self.pyaudio = pyaudio.PyAudio()
            
            # Find output device
            output_device = None
            for i in range(self.pyaudio.get_device_count()):
                info = self.pyaudio.get_device_info_by_index(i)
                if info['maxOutputChannels'] > 0:
                    output_device = i
                    break
            
            self.stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=CONFIG.AUDIO_CHANNELS,
                rate=CONFIG.AUDIO_SAMPLE_RATE,
                output=True,
                output_device_index=output_device,
                frames_per_buffer=CONFIG.AUDIO_CHUNK_SIZE
            )
            
            while self.running:
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                    
                    # Feed to echo canceller (so it knows what was played)
                    if self.echo_canceller and len(audio_data) > 0:
                        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                        self.echo_canceller.add_speaker_sample(samples / 32768.0)
                    
                    # Play the audio
                    self.stream.write(audio_data)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[AUDIO PLAYER] Error: {e}")
                    
        except Exception as e:
            print(f"[AUDIO PLAYER] Failed to initialize: {e}")
    
    def play(self, audio_data: bytes):
        """Queue audio data for playback"""
        try:
            self.audio_queue.put_nowait(audio_data)
        except queue.Full:
            pass  # Drop frame if queue is full


# =============================================================================
# INCOMING AUDIO TRACK HANDLER
# =============================================================================

class AudioReceiver:
    """
    Receives audio from the remote WebRTC peer (browser)
    and routes it to the AudioPlayer for playback.
    """
    
    def __init__(self, player: AudioPlayer):
        self.player = player
        self.running = False
        self.track = None
        self.task = None
    
    async def handle_track(self, track: MediaStreamTrack):
        """Handle incoming audio track from WebRTC"""
        print(f"[AUDIO RECEIVER] Received track: {track.kind}")
        
        if track.kind != "audio":
            return
        
        self.track = track
        self.running = True
        
        print("[AUDIO RECEIVER] Started receiving audio from browser")
        
        try:
            while self.running:
                try:
                    # Receive audio frame from WebRTC
                    frame = await asyncio.wait_for(track.recv(), timeout=1.0)
                    
                    # Convert to raw PCM bytes
                    # aiortc gives us av.AudioFrame
                    audio_array = frame.to_ndarray()
                    
                    # Ensure correct format (mono, int16)
                    if len(audio_array.shape) > 1:
                        audio_array = audio_array.mean(axis=0)  # Mix to mono
                    
                    # Convert to int16
                    audio_int16 = (audio_array * 32767).astype(np.int16)
                    audio_bytes = audio_int16.tobytes()
                    
                    # Send to player
                    self.player.play(audio_bytes)
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"[AUDIO RECEIVER] Error: {e}")
                    break
                    
        except asyncio.CancelledError:
            pass
        
        print("[AUDIO RECEIVER] Stopped")
    
    def stop(self):
        """Stop receiving audio"""
        self.running = False


# =============================================================================
# VIDEO TRACK (PiCamera2) - From your existing code
# =============================================================================

class PiCameraVideoStreamTrack(VideoStreamTrack):
    """PiCamera2 video stream for WebRTC"""
    kind = "video"

    def __init__(self):
        super().__init__()
        if not HAS_PICAMERA:
            raise RuntimeError("PiCamera2 not available")
        
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (CONFIG.FRAME_WIDTH, CONFIG.FRAME_HEIGHT), "format": "RGB888"}, 
            controls={"FrameRate": CONFIG.FRAME_RATE}
        )
        self.picam2.configure(config)
        self.picam2.start()
        print(f"[CAMERA] Started @ {CONFIG.FRAME_WIDTH}x{CONFIG.FRAME_HEIGHT} {CONFIG.FRAME_RATE}fps")

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = self.picam2.capture_array()
        # PiCamera2 with RGB888 may output BGR, convert to RGB
        frame_rgb = frame[:, :, ::-1].copy()  # BGR to RGB
        video_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

    def stop(self):
        self.picam2.stop()
        print("[CAMERA] Stopped")


# =============================================================================
# AUDIO TRACK (PyAudio-based for Bluetooth/PulseAudio support)
# =============================================================================

class PyAudioStreamTrack(MediaStreamTrack):
    """
    Custom audio track using PyAudio for microphone capture.
    Works with Bluetooth/PulseAudio unlike FFmpeg's MediaPlayer.
    """
    kind = "audio"
    
    def __init__(self, sample_rate=48000, channels=1):
        super().__init__()
        self.sample_rate = sample_rate
        self.channels = channels
        self.samples_per_frame = 960  # 20ms at 48kHz
        self._timestamp = 0
        self._audio_queue = queue.Queue(maxsize=50)
        self._running = False
        self._thread = None
        self._pyaudio = None
        self._stream = None
        
        self._start_capture()
    
    def _start_capture(self):
        """Start the audio capture thread"""
        if not HAS_PYAUDIO:
            print("[AUDIO TRACK] PyAudio not available")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
    
    def _capture_loop(self):
        """Background thread for capturing audio"""
        try:
            self._pyaudio = pyaudio.PyAudio()
            
            # Find input device (prefer pulse/default for Bluetooth)
            input_device = None
            for i in range(self._pyaudio.get_device_count()):
                info = self._pyaudio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    # Prefer pulse device
                    if 'pulse' in info['name'].lower():
                        input_device = i
                        break
                    elif input_device is None:
                        input_device = i
            
            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=input_device,
                frames_per_buffer=self.samples_per_frame
            )
            
            print(f"[AUDIO TRACK] Capturing from device {input_device}")
            
            while self._running:
                try:
                    data = self._stream.read(self.samples_per_frame, exception_on_overflow=False)
                    self._audio_queue.put(data)
                except Exception as e:
                    if self._running:
                        print(f"[AUDIO TRACK] Capture error: {e}")
                    break
                    
        except Exception as e:
            print(f"[AUDIO TRACK] Failed to initialize: {e}")
    
    async def recv(self):
        """Receive next audio frame for WebRTC"""
        # Get audio data from queue
        try:
            data = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._audio_queue.get(timeout=1.0)
            )
        except queue.Empty:
            # Generate silence if no audio
            data = b'\x00' * (self.samples_per_frame * 2 * self.channels)
        
        # Convert to numpy array
        audio_array = np.frombuffer(data, dtype=np.int16)
        
        # Create audio frame
        audio_frame = av.AudioFrame.from_ndarray(
            audio_array.reshape(1, -1),
            format='s16',
            layout='mono' if self.channels == 1 else 'stereo'
        )
        audio_frame.sample_rate = self.sample_rate
        audio_frame.pts = self._timestamp
        audio_frame.time_base = fractions.Fraction(1, self.sample_rate)
        
        self._timestamp += self.samples_per_frame
        
        return audio_frame
    
    def stop(self):
        """Stop the audio capture"""
        self._running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pyaudio:
            self._pyaudio.terminate()
        if self._thread:
            self._thread.join(timeout=2)
        print("[AUDIO TRACK] Stopped")


# =============================================================================
# MEDIA MTX SERVER
# =============================================================================

def start_mediamtx_server():
    """Starts the MediaMTX binary in background"""
    print("[SYSTEM] Starting MediaMTX Server...")
    
    if not os.path.exists(CONFIG.MEDIA_MTX_PATH):
        print(f"[WARNING] MediaMTX not found at '{CONFIG.MEDIA_MTX_PATH}'")
        print("          Download from: https://github.com/bluenviron/mediamtx/releases")
        return None
    
    try:
        return subprocess.Popen(
            [CONFIG.MEDIA_MTX_PATH], 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f"[ERROR] Could not start MediaMTX: {e}")
        return None


# =============================================================================
# MAIN WEBRTC CLIENT
# =============================================================================

async def run_stream(enable_video=True, enable_audio=True):
    """
    Main streaming function with two-way audio support.
    """
    
    # === SETUP ===
    local_ip = get_local_ip()
    public_ip = setup_global_network(local_ip)
    
    server_process = start_mediamtx_server()
    if server_process:
        await asyncio.sleep(2)  # Wait for server to boot
    
    # === ECHO CANCELLATION ===
    echo_canceller = None
    if CONFIG.AEC_ENABLED:
        echo_canceller = EchoCanceller(
            filter_length=CONFIG.AEC_FILTER_LENGTH,
            mu=CONFIG.AEC_MU
        )
    
    # === AUDIO PLAYER (For incoming audio from browser) ===
    audio_player = AudioPlayer(echo_canceller)
    audio_player.start()
    
    # === AUDIO RECEIVER ===
    audio_receiver = AudioReceiver(audio_player)
    
    # === WEBRTC SETUP ===
    print("[WEBRTC] Initializing Tracks...")
    
    video_track = None
    audio_input = None
    
    # Video Track (PiCamera)
    if enable_video and HAS_PICAMERA:
        try:
            video_track = PiCameraVideoStreamTrack()
        except Exception as e:
            print(f"[WARNING] Camera failed: {e}")
    
    # Audio Input Track (Microphone) - Use PyAudio for Bluetooth support
    audio_track = None
    if enable_audio:
        audio_dev = get_audio_input_device()
        
        # Use custom PyAudio track (works with Bluetooth/PulseAudio)
        try:
            audio_track = PyAudioStreamTrack(
                sample_rate=CONFIG.AUDIO_SAMPLE_RATE,
                channels=CONFIG.AUDIO_CHANNELS
            )
            print("[AUDIO] Microphone initialized via PyAudio")
        except Exception as e:
            print(f"[WARNING] Microphone failed: {e}")
            audio_track = None
    
    # Peer Connection
    config = RTCConfiguration(iceServers=[
        RTCIceServer(urls=["stun:stun.l.google.com:19302"])
    ])
    pc = RTCPeerConnection(configuration=config)
    
    # Add outgoing tracks
    if video_track:
        pc.addTrack(video_track)
    if audio_track:
        pc.addTrack(audio_track)
    
    # Handle incoming tracks (audio from browser)
    @pc.on("track")
    async def on_track(track):
        print(f"[WEBRTC] Incoming track: {track.kind}")
        if track.kind == "audio":
            asyncio.create_task(audio_receiver.handle_track(track))
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"[WEBRTC] Connection state: {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
    
    # === WHIP HANDSHAKE ===
    print(f"[WEBRTC] Connecting to {CONFIG.local_push_url}...")
    
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                CONFIG.local_push_url,
                data=pc.localDescription.sdp,
                headers={"Content-Type": "application/sdp"}
            ) as resp:
                if resp.status not in [200, 201]:
                    print(f"[ERROR] Connection Failed: {resp.status}")
                    print(await resp.text())
                    return
                
                answer = await resp.text()
                await pc.setRemoteDescription(
                    RTCSessionDescription(sdp=answer, type="answer")
                )
    except aiohttp.ClientError as e:
        print(f"[ERROR] Could not connect to MediaMTX: {e}")
        print("        Make sure MediaMTX is running!")
        return
    
    # === LIVE STATUS ===
    print("\n" + "=" * 60)
    print("  KENZA STREAM IS LIVE  ".center(60, "#"))
    print("=" * 60)
    print(f" [LOCAL]  http://{local_ip}:8889/{CONFIG.STREAM_NAME}")
    if public_ip != "Unknown":
        print(f" [GLOBAL] http://{public_ip}:8889/{CONFIG.STREAM_NAME}")
    print("-" * 60)
    print(" Features:")
    print(f"   Video:     {'✓ Enabled' if video_track else '✗ Disabled'}")
    print(f"   Audio Out: {'✓ Mic → Browser' if audio_track else '✗ Disabled'}")
    print(f"   Audio In:  ✓ Browser → Speaker")
    print(f"   AEC:       {'✓ Enabled' if echo_canceller else '✗ Disabled'}")
    print("-" * 60)
    print(" Press Ctrl+C to stop.")
    print("=" * 60 + "\n")
    
    # === KEEP ALIVE ===
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        print("\n[STOPPING] Cleaning up...")
        audio_receiver.stop()
        audio_player.stop()
        await pc.close()
        if video_track:
            video_track.stop()
        if audio_track:
            audio_track.stop()
        if server_process:
            server_process.terminate()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Kenza Two-Way Audio/Video Stream")
    parser.add_argument("--no-video", action="store_true", help="Audio only, no video")
    parser.add_argument("--no-audio", action="store_true", help="Video only, no audio")
    parser.add_argument("--bluetooth", action="store_true", help="Use Bluetooth TWS as audio device")
    parser.add_argument("--test-audio", action="store_true", help="List audio devices")
    parser.add_argument("--test-aec", action="store_true", help="Test echo cancellation")
    args = parser.parse_args()
    
    if args.test_audio:
        list_audio_devices()
        return
    
    # Set Bluetooth as default if requested
    if args.bluetooth:
        print("\n🎧 Bluetooth mode enabled")
        if set_bluetooth_as_default():
            print("    > Bluetooth audio activated!\n")
        else:
            print("    ! Could not set Bluetooth. Make sure TWS is connected.\n")
            print("    Tip: Run 'bluetoothctl' then 'connect <MAC>' to connect your TWS\n")
    
    if args.test_aec:
        print("\n=== TESTING ECHO CANCELLATION ===\n")
        aec = EchoCanceller(filter_length=256, mu=0.05)
        
        # Simulate speaker output
        speaker_signal = np.sin(2 * np.pi * 440 * np.arange(1000) / 16000)
        aec.add_speaker_sample(speaker_signal)
        
        # Simulate mic input with echo
        mic_signal = speaker_signal * 0.5 + np.random.randn(1000) * 0.1
        
        # Cancel echo
        cleaned = aec.cancel_echo(mic_signal)
        
        print(f"  Original mic RMS:  {np.sqrt(np.mean(mic_signal**2)):.4f}")
        print(f"  Cleaned signal RMS: {np.sqrt(np.mean(cleaned**2)):.4f}")
        print("  ✓ AEC test complete")
        return
    
    try:
        asyncio.run(run_stream(
            enable_video=not args.no_video,
            enable_audio=not args.no_audio
        ))
    except KeyboardInterrupt:
        print("\n\n👋 Stopped by user")


if __name__ == "__main__":
    main()

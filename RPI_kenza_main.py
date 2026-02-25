#!/usr/bin/env python3
"""
RPI_kenza_main.py - KENZA Robot Unified Boot Application
=========================================================
Single-command boot script for KENZA robot.
Run this ONE file on boot — it starts everything:

  ┌─────────────────────────────────────────────────┐
  │  SERVICES STARTED AUTOMATICALLY ON BOOT         │
  │                                                 │
  │  ► WebSocket Server      (port 8765) — control  │
  │  ► HTTP Motor Server     (port 8080) — motors   │
  │  ► WebRTC Signaling Relay          — telepres.  │
  │  ► Chromium Kiosk Display          — robot_ui   │
  │  ► Conversation/Voice Engine       — AI chat    │
  │  ► Telemetry Broadcaster           — live data  │
  └─────────────────────────────────────────────────┘

Usage:
    python RPI_kenza_main.py              # Normal boot
    python RPI_kenza_main.py --pairing    # Start in pairing mode
    python RPI_kenza_main.py --debug      # Enable debug logging
    python RPI_kenza_main.py --no-display # Skip Chromium launch
    python RPI_kenza_main.py --no-voice   # Skip voice mode

Requirements:
    pip install websockets opencv-python pyzbar gpiozero
"""

import os
import sys
import json
import asyncio
import subprocess
import re
import time
import threading
import signal
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Set, Optional, Callable
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class KenzaConfig:
    """Central configuration for Kenza"""
    # WebSocket server
    ws_host: str = "0.0.0.0"
    ws_port: int = 8765
    
    # Camera
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    
    # Pairing
    pairing_timeout: int = 120  # seconds to wait for QR scan
    
    # Paths
    config_dir: str = str(Path.home() / ".kenza")
    
    def __post_init__(self):
        os.makedirs(self.config_dir, exist_ok=True)


CONFIG = KenzaConfig()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("kenza")


# ============================================================================
# OPTIONAL IMPORTS (graceful degradation)
# ============================================================================

# WebSocket
try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    log.warning("websockets not installed. Run: pip install websockets")

# OpenCV for camera
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    log.warning("opencv-python not installed. Run: pip install opencv-python")

# pyzbar for QR decoding
try:
    from pyzbar.pyzbar import decode as decode_qr
    HAS_PYZBAR = True
except ImportError:
    HAS_PYZBAR = False
    log.warning("pyzbar not installed. Run: pip install pyzbar")

# Serial for ESP32 communication
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    log.warning("pyserial not installed. Run: pip install pyserial")

# Autonomy engine
try:
    from kenza_autonomy import AutonomyEngine
    HAS_AUTONOMY = True
except ImportError:
    HAS_AUTONOMY = False
    log.warning("kenza_autonomy not available — autonomous features disabled")

# HTTP server for motor fallback (port 8080)
from http.server import BaseHTTPRequestHandler, HTTPServer as _HTTPServer


# ============================================================================
# ROBOT STATE
# ============================================================================

@dataclass
class RobotState:
    """Current state of the robot"""
    # Connection
    is_paired: bool = False
    wifi_ssid: str = ""
    
    # Motors
    motor_direction: str = "S"  # F/B/L/R/S
    motor_speed: int = 0
    
    # Modes
    follow_mode: bool = False
    sentry_mode: bool = False
    privacy_mode: bool = False
    autonomous_mode: bool = False
    obstacle_avoidance: bool = True
    gesture_nav: bool = False
    
    # Eyes
    eye_color: str = "cyan"
    eye_style: str = "normal"
    
    # Audio
    mic_muted: bool = False
    current_voice: str = "aria"
    
    # Telemetry
    battery_percent: int = 95
    wifi_signal: int = -45
    cpu_temp: float = 38.0
    storage_percent: int = 64
    
    def to_dict(self) -> dict:
        return asdict(self)


# Global robot state
ROBOT_STATE = RobotState()


# ============================================================================
# HTTP MOTOR SERVER (port 8080 — mirrors pi_motor_server.py)
# ============================================================================

class _HttpMotorHandler(BaseHTTPRequestHandler):
    """
    Minimal HTTP handler for motor commands.
    Accepts GET /<direction> where direction is F/B/L/R/S.
    Compatible with the web app's HTTP fallback path.
    """
    motor_ctrl = None  # Injected by HttpMotorServer after GPIO is ready

    def do_GET(self):
        direction = self.path.strip('/').upper()
        if direction in ('F', 'B', 'L', 'R', 'S') and _HttpMotorHandler.motor_ctrl:
            _HttpMotorHandler.motor_ctrl.send_motor_command(direction, 0 if direction == 'S' else 100)
            log.debug(f"[HTTP] Motor: {direction}")
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(b'OK')

    def log_message(self, fmt, *args):
        pass  # Suppress noisy HTTP logs


class HttpMotorServer:
    """
    Runs the HTTP motor server on port 8080 in a background daemon thread.
    This allows the web app to send motor commands via HTTP as well as WebSocket.
    """
    def __init__(self, port: int = 8080):
        self.port = port
        self._server: Optional[_HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self, motor_ctrl):
        """Start the HTTP server and inject the motor controller."""
        _HttpMotorHandler.motor_ctrl = motor_ctrl
        self._server = _HTTPServer(('', self.port), _HttpMotorHandler)

        def _serve():
            log.info(f"🌐 HTTP Motor Server started on port {self.port}")
            self._server.serve_forever()

        self._thread = threading.Thread(target=_serve, daemon=True, name='HttpMotorServer')
        self._thread.start()

    def stop(self):
        if self._server:
            self._server.shutdown()
            log.info("🌐 HTTP Motor Server stopped")


# ============================================================================
# HARDWARE CONTROLLERS
# ============================================================================

class GPIOMotorController:
    """
    Controls motors directly via Raspberry Pi GPIO.
    Uses gpiozero for Pi 5 compatibility.
    Based on pi_motor_server.py configuration.
    
    GPIO Pin Mapping:
        IN1 = GPIO17 (Pin 11)
        IN2 = GPIO27 (Pin 13)
        IN3 = GPIO22 (Pin 15)
        IN4 = GPIO23 (Pin 16)
    """
    
    def __init__(self):
        self.initialized = False
        self.IN1 = None
        self.IN2 = None
        self.IN3 = None
        self.IN4 = None
        
    def connect(self) -> bool:
        """Initialize GPIO pins for motor control"""
        try:
            from gpiozero import OutputDevice
            
            self.IN1 = OutputDevice(17)  # GPIO17 → Pin 11
            self.IN2 = OutputDevice(27)  # GPIO27 → Pin 13
            self.IN3 = OutputDevice(22)  # GPIO22 → Pin 15
            self.IN4 = OutputDevice(23)  # GPIO23 → Pin 16
            
            self.stop()  # Ensure motors are stopped initially
            self.initialized = True
            log.info("🔌 GPIO Motor Controller initialized (Pi GPIO)")
            return True
            
        except ImportError:
            log.warning("gpiozero not available - motor control disabled")
            log.warning("Install with: pip install gpiozero")
            return False
        except Exception as e:
            log.error(f"GPIO initialization failed: {e}")
            return False
    
    def send_motor_command(self, direction: str, speed: int):
        """
        Execute motor command.
        Direction: F=Forward, B=Backward, L=Left, R=Right, S=Stop
        Speed: 0-100 (currently binary - on/off)
        """
        ROBOT_STATE.motor_direction = direction
        ROBOT_STATE.motor_speed = speed
        
        if not self.initialized:
            log.debug(f"[SIM] Motor: {direction} @ {speed}%")
            return
        
        # Stop if speed is 0
        if speed == 0:
            self.stop()
            return
            
        # Execute direction command
        if direction == 'F':
            self._forward()
        elif direction == 'B':
            self._backward()
        elif direction == 'L':
            self._left()
        elif direction == 'R':
            self._right()
        elif direction == 'S':
            self.stop()
        else:
            log.warning(f"Unknown direction: {direction}")
    
    def _forward(self):
        """Move forward"""
        self.IN1.on()
        self.IN2.off()
        self.IN3.on()
        self.IN4.off()
        log.debug("Motor: FORWARD")
    
    def _backward(self):
        """Move backward"""
        self.IN1.off()
        self.IN2.on()
        self.IN3.off()
        self.IN4.on()
        log.debug("Motor: BACKWARD")
    
    def _left(self):
        """Turn left"""
        self.IN1.off()
        self.IN2.on()
        self.IN3.on()
        self.IN4.off()
        log.debug("Motor: LEFT")
    
    def _right(self):
        """Turn right"""
        self.IN1.on()
        self.IN2.off()
        self.IN3.off()
        self.IN4.on()
        log.debug("Motor: RIGHT")
    
    def stop(self):
        """Stop all motors"""
        if self.initialized:
            self.IN1.off()
            self.IN2.off()
            self.IN3.off()
            self.IN4.off()
        ROBOT_STATE.motor_direction = "S"
        ROBOT_STATE.motor_speed = 0
        log.debug("Motor: STOP")
    
    def disconnect(self):
        """Clean up GPIO resources"""
        self.stop()
        if self.IN1:
            self.IN1.close()
        if self.IN2:
            self.IN2.close()
        if self.IN3:
            self.IN3.close()
        if self.IN4:
            self.IN4.close()
        self.initialized = False
        log.info("🔌 GPIO Motor Controller cleaned up")


class EyeController:
    """
    Controls the robot's eye display.
    Integrates with the eye animation system.
    """
    
    def __init__(self):
        self.current_color = "cyan"
        self.current_style = "normal"
        self.colors = {
            "cyan": (0, 255, 255),
            "pink": (255, 105, 180),
            "green": (0, 255, 128),
            "orange": (255, 165, 0),
            "purple": (147, 112, 219),
            "white": (255, 255, 255)
        }
    
    def set_color(self, color: str):
        """Set eye color"""
        if color in self.colors:
            self.current_color = color
            ROBOT_STATE.eye_color = color
            log.info(f"👁️ Eye color: {color}")
            # TODO: Send to actual eye display
            return True
        return False
    
    def set_style(self, style: str):
        """Set eye style (normal, sleepy, angry, happy, etc.)"""
        self.current_style = style
        ROBOT_STATE.eye_style = style
        log.info(f"👁️ Eye style: {style}")
        # TODO: Send to actual eye display
        return True


class AudioController:
    """
    Controls audio input/output.
    Manages microphone, speaker, and TTS.
    """
    
    def __init__(self):
        self.mic_muted = False
        self.current_voice = "aria"
        self.voices = ["aria", "maya", "zephyr"]
    
    def set_mic_muted(self, muted: bool):
        """Mute/unmute microphone"""
        self.mic_muted = muted
        ROBOT_STATE.mic_muted = muted
        log.info(f"🎤 Microphone: {'muted' if muted else 'active'}")
        # TODO: Actually mute mic
    
    def set_voice(self, voice: str):
        """Set TTS voice"""
        if voice in self.voices:
            self.current_voice = voice
            ROBOT_STATE.current_voice = voice
            log.info(f"🗣️ Voice set to: {voice}")
            return True
        return False
    
    def speak(self, text: str):
        """Text-to-speech output"""
        log.info(f"🗣️ Speaking: {text[:50]}...")
        # TODO: Integrate with TTS engine


class AIHandler:
    """
    Handles AI chat and voice interactions.
    Integrates with the new ConversationEngine for seamless interaction.
    """
    
    def __init__(self, audio_controller: AudioController, eye_controller: EyeController = None,
                 camera=None, on_emotion_fn=None):
        self.audio = audio_controller
        self.eyes = eye_controller
        self.camera = camera
        self.conversation_engine = None
        self.voice_thread = None
        self._running = False
        self.on_emotion_fn = on_emotion_fn  # callback(emotion: str) for WS broadcast
        self._init_engine()
    
    def _init_engine(self):
        """Initialize the conversation engine"""
        try:
            from kenza_conversation import ConversationEngine, ConversationConfig
            
            config = ConversationConfig.load("config/settings.yaml")
            self.conversation_engine = ConversationEngine(
                config=config,
                eye_controller=self.eyes,
                audio_controller=self.audio,
                camera=self.camera,
                on_state_change=self._on_state_change,
                on_emotion=self._on_emotion,
            )
            log.info("🧠 Conversation Engine initialized")
        except ImportError as e:
            log.warning(f"Could not load ConversationEngine: {e}")
        except Exception as e:
            log.error(f"ConversationEngine init failed: {e}")
    
    def _on_state_change(self, state: str):
        """Handle conversation state changes"""
        log.debug(f"AI State: {state}")

    def _on_emotion(self, emotion: str):
        """Handle emotion from conversation engine — update local eyes and fire WS callback."""
        # Local eye LED/servo update
        if self.eyes:
            self.eyes.set_style(emotion)
        # WebSocket broadcast (set via KenzaMain.run)
        if self.on_emotion_fn:
            try:
                self.on_emotion_fn(emotion)
            except Exception:
                pass
    
    def set_camera(self, camera):
        """Set camera for vision capability"""
        self.camera = camera
        if self.conversation_engine and self.conversation_engine.vision:
            self.conversation_engine.vision.set_camera(camera)
    
    def start_voice_mode(self, use_wake_word: bool = True):
        """Start voice interaction in background thread"""
        if self._running and use_wake_word:
            # If already running and wake_word is requested, it's a no-op
            return False
        
        if self._running and not use_wake_word:
            # If already running but a non-wake-word mode is requested,
            # stop the current one and restart.
            log.info("Restarting voice mode for non-wake-word interaction.")
            self.stop_voice_mode()
            # Give a moment for the thread to clean up
            time.sleep(0.1)

        if not self.conversation_engine:
            return False
        
        self._running = True
        
        def _run_voice():
            try:
                self.conversation_engine.run_voice_loop(use_wake_word=use_wake_word)
            except Exception as e:
                log.error(f"Voice mode error: {e}")
            finally:
                self._running = False
        
        self.voice_thread = threading.Thread(target=_run_voice, daemon=True)
        self.voice_thread.start()
        log.info("🎙️ Voice mode started")
        return True
    
    def stop_voice_mode(self):
        """Stop voice interaction"""
        if self.conversation_engine:
            self.conversation_engine.stop()
        self._running = False
        log.info("🔇 Voice mode stopped")
    
    def interrupt(self):
        """Interrupt current speech"""
        if self.conversation_engine:
            self.conversation_engine.tts.clear_and_stop()
            log.info("🛑 Speech interrupted")
    
    async def process_message(self, message: str) -> str:
        """
        Process a user message and generate response.
        Returns the AI response.
        """
        log.info(f"🧠 AI processing: {message[:50]}...")
        
        if self.conversation_engine:
            try:
                response = self.conversation_engine.process_input(message)
                return response
            except Exception as e:
                log.error(f"AI error: {e}")
                return f"I had trouble processing that: {e}"
        else:
            return "AI engine not available. Please check configuration."
    
    async def process_message_with_speech(self, message: str) -> str:
        """Process message and speak the response"""
        response = await self.process_message(message)
        
        if response and self.conversation_engine:
            # Speak asynchronously
            self.conversation_engine.tts.speak_async(response)
        
        return response



# ============================================================================
# QR SCANNER MODULE
# ============================================================================

class QRScanner:
    """
    Camera-based QR code scanner for WiFi pairing.
    Parses standard WiFi QR format: WIFI:T:WPA;S:ssid;P:password;;
    """
    
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.camera = None
        self.running = False
        self._on_qr_callback: Optional[Callable] = None
        
    def start(self):
        """Start the camera and begin scanning"""
        if not HAS_CV2 or not HAS_PYZBAR:
            log.error("QR scanning requires opencv-python and pyzbar")
            return False
            
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                log.error(f"Failed to open camera {self.camera_index}")
                return False
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.camera_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.camera_height)
            self.running = True
            log.info("📷 QR Scanner started")
            return True
        except Exception as e:
            log.error(f"Camera error: {e}")
            return False
    
    def stop(self):
        """Stop the camera"""
        self.running = False
        if self.camera:
            self.camera.release()
            self.camera = None
        log.info("📷 QR Scanner stopped")
    
    def on_qr_detected(self, callback: Callable):
        """Register callback for QR detection: callback(ssid, password)"""
        self._on_qr_callback = callback
    
    def scan_once(self) -> Optional[tuple]:
        """
        Scan for a single frame and return WiFi credentials if found.
        Returns: (ssid, password) or None
        """
        if not self.camera or not self.running:
            return None
            
        ret, frame = self.camera.read()
        if not ret:
            return None
        
        # Decode QR codes in frame
        codes = decode_qr(frame)
        for code in codes:
            data = code.data.decode('utf-8')
            wifi_creds = self.parse_wifi_qr(data)
            if wifi_creds:
                log.info(f"📱 QR detected: SSID={wifi_creds[0]}")
                return wifi_creds
        
        return None
    
    def scan_loop(self, timeout: int = 120):
        """
        Continuously scan for QR codes until found or timeout.
        Calls registered callback when WiFi QR is detected.
        """
        start_time = time.time()
        log.info(f"🔍 Scanning for WiFi QR code (timeout: {timeout}s)...")
        
        while self.running and (time.time() - start_time) < timeout:
            result = self.scan_once()
            if result:
                if self._on_qr_callback:
                    self._on_qr_callback(result[0], result[1])
                return result
            time.sleep(0.1)  # Small delay to reduce CPU usage
        
        log.warning("⏱ QR scan timeout")
        return None
    
    @staticmethod
    def parse_wifi_qr(data: str) -> Optional[tuple]:
        """
        Parse WiFi QR code format: WIFI:T:WPA;S:ssid;P:password;;
        Returns: (ssid, password) or None
        """
        if not data.startswith('WIFI:'):
            return None
        
        try:
            # Extract SSID
            ssid_match = re.search(r'S:([^;]+);', data)
            # Extract Password
            pass_match = re.search(r'P:([^;]*);', data)
            
            if ssid_match:
                ssid = ssid_match.group(1)
                password = pass_match.group(1) if pass_match else ""
                return (ssid, password)
        except Exception as e:
            log.error(f"QR parse error: {e}")
        
        return None


# ============================================================================
# WIFI MANAGER MODULE
# ============================================================================

class WiFiManager:
    """
    WiFi network connection manager using nmcli (NetworkManager).
    """
    
    @staticmethod
    def is_available() -> bool:
        """Check if nmcli is available"""
        try:
            subprocess.run(['nmcli', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    @staticmethod
    def connect(ssid: str, password: str) -> bool:
        """
        Connect to a WiFi network.
        Returns: True if connection successful
        """
        log.info(f"📶 Connecting to WiFi: {ssid}")
        
        try:
            # First, try to delete existing connection with same name
            subprocess.run(
                ['nmcli', 'connection', 'delete', ssid],
                capture_output=True
            )
            
            # Create new connection
            cmd = [
                'nmcli', 'device', 'wifi', 'connect', ssid,
                'password', password,
                '--wait', '30'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                log.info(f"✅ Connected to {ssid}")
                return True
            else:
                log.error(f"WiFi connection failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            log.error("WiFi connection timed out")
            return False
        except Exception as e:
            log.error(f"WiFi error: {e}")
            return False
    
    @staticmethod
    def get_current_ip() -> Optional[str]:
        """Get current IP address on the connected network"""
        try:
            result = subprocess.run(
                ['hostname', '-I'],
                capture_output=True, text=True, check=True
            )
            ips = result.stdout.strip().split()
            return ips[0] if ips else None
        except Exception:
            return None
    
    @staticmethod
    def get_current_ssid() -> Optional[str]:
        """Get currently connected WiFi SSID"""
        try:
            result = subprocess.run(
                ['iwgetid', '-r'],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip() or None
        except Exception:
            return None
    
    @staticmethod
    def is_connected() -> bool:
        """Check if connected to any network"""
        ip = WiFiManager.get_current_ip()
        return ip is not None and not ip.startswith('127.')


# ============================================================================
# WEBSOCKET SERVER MODULE
# ============================================================================

class KenzaWebSocket:
    """
    WebSocket server for app communication.
    Handles bidirectional messaging AND WebRTC telepresence signaling relay.

    Roles:
      - 'controller' : laptop / phone running kenza_app_v2.html
      - 'robot'      : this RPi running robot_display.html
      - 'display'    : robot_display.html (local browser on RPi)

    Telepresence signaling messages relayed:
      register, call_offer, call_answer, call_reject, call_end, ice_candidate
    """

    # Role constants
    ROLE_CONTROLLER = 'controller'
    ROLE_ROBOT      = 'robot'
    ROLE_DISPLAY    = 'display'

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set = set()
        # role -> set of websockets
        self._by_role: Dict[str, Set] = {
            self.ROLE_CONTROLLER: set(),
            self.ROLE_ROBOT:      set(),
            self.ROLE_DISPLAY:    set(),
        }
        self._client_roles: Dict = {}   # websocket -> role
        self._client_names: Dict = {}   # websocket -> name
        self.server = None
        self._message_handlers: Dict[str, Callable] = {}
        self._loop = None

    def on_message(self, msg_type: str, handler: Callable):
        """Register a handler for a specific message type"""
        self._message_handlers[msg_type] = handler

    async def _relay_to_role(self, role: str, payload: dict, exclude=None):
        """Send a JSON payload to all clients with a given role."""
        targets = self._by_role.get(role, set()) - ({exclude} if exclude else set())
        if targets:
            msg = json.dumps(payload)
            await asyncio.gather(*[c.send(msg) for c in targets], return_exceptions=True)

    async def _relay_to_opposite(self, sender_role: str, payload: dict, sender):
        """Relay a signaling message to the opposite peer."""
        target_role = self.ROLE_ROBOT if sender_role == self.ROLE_CONTROLLER else self.ROLE_CONTROLLER
        await self._relay_to_role(target_role, payload)

    async def handler(self, websocket, path=None):
        """Handle a new WebSocket connection."""
        self.clients.add(websocket)
        client_ip = websocket.remote_address[0]
        log.info(f"🔗 Client connected: {client_ip}")
        role = None

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get('type', 'unknown')

                    # ─────────────────────────────────────────────────────
                    # REGISTER: client announces its role
                    # ─────────────────────────────────────────────────────
                    if msg_type == 'register':
                        role = data.get('role', 'controller')
                        name = data.get('name', role)
                        self._client_roles[websocket] = role
                        self._client_names[websocket] = name
                        self._by_role.setdefault(role, set()).add(websocket)
                        log.info(f"📱 Registered [{role}] '{name}' from {client_ip}")

                        # Notify controllers if a robot/display joined
                        if role in (self.ROLE_ROBOT, self.ROLE_DISPLAY):
                            await self._relay_to_role(self.ROLE_CONTROLLER, {
                                'type': 'peer_online', 'role': role, 'name': name
                            })
                        # Tell the robot who is online among controllers
                        elif role == self.ROLE_CONTROLLER:
                            for rws in self._by_role[self.ROLE_ROBOT]:
                                rname = self._client_names.get(rws, 'KENZA HomeBot')
                                await websocket.send(json.dumps({
                                    'type': 'peer_online', 'role': 'robot', 'name': rname
                                }))
                        continue  # no further processing for register

                    # ─────────────────────────────────────────────────────
                    # TELEPRESENCE SIGNALING — relay between peers
                    # ─────────────────────────────────────────────────────
                    SIGNALING = {'call_offer', 'call_answer', 'call_reject',
                                 'call_end', 'ice_candidate', 'call_busy'}
                    if msg_type in SIGNALING:
                        sender_role = self._client_roles.get(websocket, self.ROLE_CONTROLLER)
                        await self._relay_to_opposite(sender_role, data, sender=websocket)
                        continue

                    # ─────────────────────────────────────────────────────
                    # NORMAL APP COMMANDS — handled by registered handlers
                    # ─────────────────────────────────────────────────────
                    if msg_type in self._message_handlers:
                        response = self._message_handlers[msg_type](data)
                        if response:
                            await websocket.send(json.dumps(response))

                except json.JSONDecodeError:
                    log.warning(f"Invalid JSON: {message[:100]}")

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            # Clean up role tracking
            role = self._client_roles.pop(websocket, None)
            name = self._client_names.pop(websocket, None)
            if role:
                self._by_role.get(role, set()).discard(websocket)
                # Notify controllers if the robot went offline
                if role in (self.ROLE_ROBOT, self.ROLE_DISPLAY):
                    await self._relay_to_role(self.ROLE_CONTROLLER, {
                        'type': 'peer_offline', 'role': role, 'name': name
                    })
            log.info(f"🔌 Disconnected: [{role or '?'}] '{name or '?'}' ({client_ip})")

    async def broadcast(self, data: dict):
        """Send message to all connected clients"""
        if not self.clients:
            return
        message = json.dumps(data)
        await asyncio.gather(
            *[client.send(message) for client in self.clients],
            return_exceptions=True
        )

    def broadcast_sync(self, data: dict):
        """Synchronous broadcast (for calling from non-async context)"""
        if self._loop and self.clients:
            asyncio.run_coroutine_threadsafe(
                self.broadcast(data),
                self._loop
            )

    async def start(self):
        """Start the WebSocket server"""
        if not HAS_WEBSOCKETS:
            log.error("websockets module required")
            return
        self._loop = asyncio.get_event_loop()
        self.server = await websockets.serve(
            self.handler,
            self.host,
            self.port
        )
        ip = WiFiManager.get_current_ip() or self.host
        log.info(f"🌐 WebSocket server started: ws://{ip}:{self.port}")

    async def stop(self):
        """Stop the WebSocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            log.info("🌐 WebSocket server stopped")



# ============================================================================
# PAIRING SERVICE
# ============================================================================

class PairingService:
    """
    Orchestrates the pairing flow:
    1. Start QR scanning
    2. Parse WiFi credentials from QR
    3. Connect to WiFi network
    4. Broadcast pairing success to app
    """
    
    def __init__(self, ws_server: KenzaWebSocket):
        self.ws_server = ws_server
        self.qr_scanner = QRScanner(CONFIG.camera_index)
        self.is_pairing = False
        
    def start_pairing(self, timeout: int = 120) -> bool:
        """
        Start the pairing process.
        Blocks until pairing succeeds or times out.
        Returns: True if pairing successful
        """
        if self.is_pairing:
            log.warning("Pairing already in progress")
            return False
            
        self.is_pairing = True
        log.info("🔐 Starting pairing mode...")
        
        # Start QR scanner
        if not self.qr_scanner.start():
            self.is_pairing = False
            return False
        
        try:
            # Scan for WiFi QR code
            result = self.qr_scanner.scan_loop(timeout)
            
            if not result:
                log.error("No WiFi QR code detected")
                return False
            
            ssid, password = result
            
            # Connect to WiFi
            if not WiFiManager.connect(ssid, password):
                log.error("Failed to connect to WiFi")
                return False
            
            # Wait a bit for network to stabilize
            time.sleep(2)
            
            # Get our IP address
            ip = WiFiManager.get_current_ip()
            if not ip:
                log.error("Failed to get IP address")
                return False
            
            # Broadcast pairing success
            log.info(f"🎉 Pairing successful! IP: {ip}")
            self.ws_server.broadcast_sync({
                'type': 'pairing_success',
                'data': {
                    'ip': ip,
                    'ssid': ssid,
                    'timestamp': int(time.time())
                }
            })
            
            return True
            
        finally:
            self.qr_scanner.stop()
            self.is_pairing = False
    
    def start_pairing_async(self, timeout: int = 120):
        """Start pairing in a background thread"""
        thread = threading.Thread(
            target=self.start_pairing,
            args=(timeout,),
            daemon=True
        )
        thread.start()
        return thread


# ============================================================================
# MAIN KENZA APPLICATION
# ============================================================================

class KenzaMain:
    """
    Main Kenza application that orchestrates ALL boot services:

      - WebSocket   server    (port 8765) — real-time control & signaling
      - HTTP Motor  server    (port 8080) — HTTP fallback for motor cmds
      - Chromium kiosk display           — robot_display.html full-screen
      - Conversation/Voice engine        — wake-word AI interaction
      - Telemetry broadcaster            — live battery, temp, wifi
      - Autonomy engine (optional)       — follow/explore/sentry modes

    Commands handled via WebSocket from kenza_app_v2.html:
      motor, privacy_mode, update_settings, voice_select, toggle_mic,
      ai_message, follow_mode, sentry_mode, autonomous_mode, gesture_nav,
      get_state, ping, start_voice_mode, stop_voice_mode, interrupt
    """

    # Path to robot display HTML (relative to this script)
    DISPLAY_HTML = str(Path(__file__).parent / 'web' / 'robot_display.html')

    def __init__(self, auto_voice: bool = True, auto_display: bool = True):
        # Boot flags
        self.auto_voice   = auto_voice
        self.auto_display = auto_display

        # Core services
        self.ws_server       = KenzaWebSocket(CONFIG.ws_host, CONFIG.ws_port)
        self.http_motor_srv  = HttpMotorServer(port=8080)
        self.pairing_service = PairingService(self.ws_server)
        self.running = False

        # Hardware controllers
        self.motors = GPIOMotorController()
        self.eyes   = EyeController()
        self.audio  = AudioController()
        
        # Wire WS emotion broadcast into AIHandler
        def _ws_emotion_broadcast(emotion: str):
            self.ws_server.broadcast_sync({'type': 'emotion', 'state': emotion})
        self.ai     = AIHandler(self.audio, eye_controller=self.eyes, on_emotion_fn=_ws_emotion_broadcast)

        # Autonomy engine
        self.autonomy = None
        if HAS_AUTONOMY:
            self.autonomy = AutonomyEngine(
                motor_callback=self._autonomy_motor_callback,
                status_callback=self._autonomy_status_callback,
            )
            log.info("🧠 Autonomy engine initialized")

        # Telemetry broadcast interval
        self.telemetry_interval = 2.0  # seconds
        self._telemetry_task = None

        # Register message handlers
        self._setup_handlers()

        # Wire voice commands → autonomy engine bridge
        self._wire_voice_autonomy()
    
    def _wire_voice_autonomy(self):
        """Bridge voice commands from CommandParser to autonomy engine"""
        if not self.autonomy:
            return
        
        # Access the CommandParser through the conversation engine
        engine = getattr(self.ai, 'conversation_engine', None)
        if engine is None:
            engine = getattr(self.ai, 'engine', None)
        if engine is None:
            log.debug("No conversation engine found for voice-autonomy bridge")
            return
        
        commands = getattr(engine, 'commands', None)
        if commands is None:
            log.debug("No CommandParser found on engine")
            return
        
        def _voice_autonomy_callback(action: str, params: dict):
            """Route voice commands to the appropriate autonomy handler"""
            action_map = {
                "start_follow": lambda: self._handle_follow_mode({'data': {'enabled': True}}),
                "stop_follow": lambda: self._handle_follow_mode({'data': {'enabled': False}}),
                "start_explore": lambda: self._handle_autonomous_mode({'data': {'enabled': True}}),
                "stop_explore": lambda: self._handle_autonomous_mode({'data': {'enabled': False}}),
                "gesture_nav_on": lambda: self._handle_gesture_nav({'data': {'enabled': True}}),
                "gesture_nav_off": lambda: self._handle_gesture_nav({'data': {'enabled': False}}),
                "emergency_stop": lambda: (self.autonomy.stop(), self.motors.stop()),
            }
            fn = action_map.get(action)
            if fn:
                fn()
                log.info(f"Voice → Autonomy: {action}")
        
        commands.motor_command_callback = _voice_autonomy_callback
        log.info("Voice → Autonomy bridge connected")
        
    def _setup_handlers(self):
        """Set up WebSocket message handlers for all app commands"""
        
        # Pairing
        self.ws_server.on_message('start_pairing', self._handle_start_pairing)
        
        # Motor control (from joystick)
        self.ws_server.on_message('motor', self._handle_motor)
        
        # Privacy mode toggle
        self.ws_server.on_message('privacy_mode', self._handle_privacy_mode)
        
        # Settings updates (eye color, etc.)
        self.ws_server.on_message('update_settings', self._handle_update_settings)
        
        # Voice selection
        self.ws_server.on_message('voice_select', self._handle_voice_select)
        self.ws_server.on_message('get_voices', self._handle_get_voices)
        
        # Microphone toggle
        self.ws_server.on_message('toggle_mic', self._handle_toggle_mic)
        
        # AI chat message
        self.ws_server.on_message('ai_message', self._handle_ai_message)
        
        # Follow mode toggle
        self.ws_server.on_message('follow_mode', self._handle_follow_mode)
        
        # Sentry mode toggle
        self.ws_server.on_message('sentry_mode', self._handle_sentry_mode)
        
        # Autonomy controls
        self.ws_server.on_message('autonomous_mode', self._handle_autonomous_mode)
        self.ws_server.on_message('obstacle_avoidance', self._handle_obstacle_avoidance)
        self.ws_server.on_message('gesture_nav', self._handle_gesture_nav)
        
        # Get current state
        self.ws_server.on_message('get_state', self._handle_get_state)
        
        # Ping/pong keep-alive
        self.ws_server.on_message('ping', self._handle_ping)

        # Voice mode control
        self.ws_server.on_message('start_voice_mode', self._handle_start_voice_mode)
        self.ws_server.on_message('stop_voice_mode', self._handle_stop_voice_mode)
        self.ws_server.on_message('interrupt', self._handle_interrupt)

        # Offline LLM model selection (Ollama)
        self.ws_server.on_message('set_offline_model', self._handle_set_offline_model)
        self.ws_server.on_message('get_offline_models', self._handle_get_offline_models)
        
    # =========== COMMAND HANDLERS ===========
    
    def _handle_start_pairing(self, data):
        """Handle request to start pairing mode"""
        timeout = data.get('data', {}).get('timeout', 120)
        self.pairing_service.start_pairing_async(timeout)
        return {'type': 'pairing_started', 'data': {'timeout': timeout}}
    
    def _handle_motor(self, data):
        """
        Handle motor control commands from joystick.
        Data: {direction: 'F/B/L/R/S', speed: 0-100}
        """
        cmd = data.get('data', {})
        direction = cmd.get('direction', 'S')
        speed = cmd.get('speed', 0)
        
        log.debug(f"🎮 Motor command: {direction} @ {speed}%")
        
        # Send to GPIO motor controller
        self.motors.send_motor_command(direction, speed)
        
        # No response needed for motor commands (high frequency)
        return None
    
    def _handle_privacy_mode(self, data):
        """
        Handle privacy mode toggle.
        Data: {enabled: true/false}
        """
        enabled = data.get('data', {}).get('enabled', False)
        ROBOT_STATE.privacy_mode = enabled
        
        if enabled:
            log.info("🔴 Privacy mode ENABLED - cameras/mics disabled")
            # TODO: Actually disable cameras/microphones
        else:
            log.info("🟢 Privacy mode DISABLED")
        
        return {'type': 'privacy_mode_ack', 'data': {'enabled': enabled}}
    
    def _handle_update_settings(self, data):
        """
        Handle settings updates (eye color, styles, etc.)
        Data: {eyes: {color: 'cyan'}, ...}
        """
        settings = data.get('data', {})
        
        # Eye settings
        if 'eyes' in settings:
            eye_settings = settings['eyes']
            if 'color' in eye_settings:
                self.eyes.set_color(eye_settings['color'])
            if 'style' in eye_settings:
                self.eyes.set_style(eye_settings['style'])
        
        log.info(f"⚙️ Settings updated: {settings}")
        return {'type': 'settings_ack', 'data': settings}
    
    def _handle_voice_select(self, data):
        """
        Handle voice selection for TTS.
        Data: {voice: 'aria'/'luna'/'nova'/'rex'/'sonia'}
        """
        voice = data.get('data', {}).get('voice', 'aria').lower()
        
        # Check if engine is available and voice is valid
        if hasattr(self.ai, 'engine') and self.ai.engine:
            config = self.ai.engine.config
            if voice in config.voice_presets:
                # Update config
                config.current_voice = voice
                config.tts_voice = config.voice_presets[voice]
                
                # Update TTS engine
                if hasattr(self.ai.engine, 'tts'):
                    self.ai.engine.tts.set_voice(config.tts_voice)
                
                # Save to settings
                self.ai.engine.commands._save_voice_setting(voice)
                
                return {'type': 'voice_select_ack', 'data': {'voice': voice, 'success': True}}
        
        return {'type': 'voice_select_ack', 'data': {'voice': voice, 'success': False}}
        
    def _handle_get_voices(self, data):
        """Return available voice presets"""
        voices = {}
        current = "aria"
        
        if hasattr(self.ai, 'engine') and self.ai.engine:
            config = self.ai.engine.config
            voices = config.voice_presets
            current = config.current_voice
            
        return {
            'type': 'voices_list', 
            'data': {
                'voices': list(voices.keys()), 
                'current': current,
                'presets': voices
            }
        }

    def _handle_set_offline_model(self, data):
        """
        Handle offline LLM model switch.
        Data: {model: 'gemma3:270m'}
        Switches the Ollama model used when internet is unavailable.
        """
        model = data.get('data', {}).get('model', 'gemma3:270m')
        engine = getattr(self.ai, 'conversation_engine', None)

        if engine and hasattr(engine, 'set_offline_model'):
            result = engine.set_offline_model(model)
            log.info(f"🤖 Offline model switched to: {model}")
            return {'type': 'offline_model_ack', 'data': result}

        return {'type': 'offline_model_ack', 'data': {'model': model, 'success': False,
                                                       'error': 'Engine not available'}}

    def _handle_get_offline_models(self, data):
        """
        Return info about available offline models.
        The app calls this on the AI Settings page load.
        """
        engine = getattr(self.ai, 'conversation_engine', None)

        if engine and hasattr(engine, 'get_offline_model_info'):
            info = engine.get_offline_model_info()
            return {'type': 'offline_models_info', 'data': info}

        return {
            'type': 'offline_models_info',
            'data': {
                'current_model': 'gemma3:270m',
                'ollama_available': False,
                'local_models': [],
                'known_models': {},
            }
        }
    
    def _handle_toggle_mic(self, data):
        """
        Handle microphone mute toggle.
        Data: {muted: true/false}
        """
        muted = data.get('data', {}).get('muted', False)
        self.audio.set_mic_muted(muted)
        
        return {'type': 'mic_status', 'data': {'muted': muted}}
    
    def _handle_ai_message(self, data):
        """
        Handle AI chat message.
        Data: {message: 'user text'}
        """
        message = data.get('data', {}).get('message', '')
        
        if not message:
            return None
        
        # Process in background and send response when ready
        asyncio.create_task(self._process_ai_message(message))
        
        # Immediate acknowledgment
        return {'type': 'ai_thinking', 'data': {'status': 'processing'}}
    
    async def _process_ai_message(self, message: str):
        """Background task to process AI message and send response"""
        try:
            response = await self.ai.process_message(message)
            
            # Broadcast AI response to all clients
            await self.ws_server.broadcast({
                'type': 'ai_response',
                'data': {
                    'message': response,
                    'timestamp': int(time.time())
                }
            })
            
            # Optional: Speak the response
            # self.audio.speak(response)
            
        except Exception as e:
            log.error(f"AI processing error: {e}")
            await self.ws_server.broadcast({
                'type': 'ai_error',
                'data': {'error': str(e)}
            })
    
    def _handle_follow_mode(self, data):
        """
        Handle follow mode toggle.
        Data: {enabled: true/false}
        Uses autonomy engine with MediaPipe Pose tracking.
        """
        enabled = data.get('data', {}).get('enabled', False)
        ROBOT_STATE.follow_mode = enabled
        
        if enabled:
            log.info("🚶 Follow mode ENABLED — locking onto person via pose tracking")
            # Stop other autonomy modes
            ROBOT_STATE.sentry_mode = False
            ROBOT_STATE.autonomous_mode = False
            ROBOT_STATE.gesture_nav = False
            if self.autonomy:
                self.autonomy.start_follow()
        else:
            log.info("🚶 Follow mode DISABLED")
            if self.autonomy:
                self.autonomy.stop()
            self.motors.stop()
        
        return {'type': 'follow_mode_ack', 'data': {'enabled': enabled}}
    
    def _handle_sentry_mode(self, data):
        """
        Handle sentry/security mode toggle.
        Data: {enabled: true/false}
        Uses autonomy engine with frame-differencing motion detection.
        """
        enabled = data.get('data', {}).get('enabled', False)
        ROBOT_STATE.sentry_mode = enabled
        
        if enabled:
            log.info("🔒 Sentry mode ENABLED — monitoring for motion")
            ROBOT_STATE.follow_mode = False
            ROBOT_STATE.autonomous_mode = False
            ROBOT_STATE.gesture_nav = False
            if self.autonomy:
                self.autonomy.start_sentry()
        else:
            log.info("🔓 Sentry mode DISABLED")
            if self.autonomy:
                self.autonomy.stop()
        
        return {'type': 'sentry_mode_ack', 'data': {'enabled': enabled}}
    
    def _handle_autonomous_mode(self, data):
        """
        Handle autonomous exploration toggle.
        Data: {enabled: true/false}
        """
        enabled = data.get('data', {}).get('enabled', False)
        ROBOT_STATE.autonomous_mode = enabled
        
        if enabled:
            log.info("🧭 Autonomous exploration ENABLED")
            ROBOT_STATE.follow_mode = False
            ROBOT_STATE.sentry_mode = False
            ROBOT_STATE.gesture_nav = False
            if self.autonomy:
                self.autonomy.start_explore()
        else:
            log.info("🧭 Autonomous exploration DISABLED")
            if self.autonomy:
                self.autonomy.stop()
            self.motors.stop()
        
        return {'type': 'autonomous_mode_ack', 'data': {'enabled': enabled}}
    
    def _handle_obstacle_avoidance(self, data):
        """
        Handle collision avoidance toggle.
        Data: {enabled: true/false}
        """
        enabled = data.get('data', {}).get('enabled', True)
        ROBOT_STATE.obstacle_avoidance = enabled
        
        if self.autonomy:
            self.autonomy.set_collision_avoidance(enabled)
        
        log.info(f"🛡️ Collision avoidance {'ENABLED' if enabled else 'DISABLED'}")
        return {'type': 'obstacle_avoidance_ack', 'data': {'enabled': enabled}}
    
    def _handle_gesture_nav(self, data):
        """
        Handle gesture navigation toggle.
        Data: {enabled: true/false}
        """
        enabled = data.get('data', {}).get('enabled', False)
        ROBOT_STATE.gesture_nav = enabled
        
        if enabled:
            log.info("✋ Gesture navigation ENABLED")
            ROBOT_STATE.follow_mode = False
            ROBOT_STATE.sentry_mode = False
            ROBOT_STATE.autonomous_mode = False
            if self.autonomy:
                self.autonomy.start_gesture_nav()
        else:
            log.info("✋ Gesture navigation DISABLED")
            if self.autonomy:
                self.autonomy.stop()
            self.motors.stop()
        
        return {'type': 'gesture_nav_ack', 'data': {'enabled': enabled}}
    
    def _handle_get_state(self, data):
        """Return current robot state"""
        return {'type': 'robot_state', 'data': ROBOT_STATE.to_dict()}
    
    def _handle_ping(self, data):
        """Handle ping/pong for connection keep-alive"""
        return {'type': 'pong', 'data': {'timestamp': int(time.time())}}
    
    def _handle_start_voice_mode(self, data):
        """
        Start voice interaction mode.
        Data: {use_wake_word: true/false}
        If engine is already running, this is a no-op (engine is always-on).
        """
        use_wake_word = data.get('data', {}).get('use_wake_word', True)
        # If engine is already running, and we're requesting wake_word mode, it's a no-op.
        # If engine is running but we're requesting non-wake_word mode, restart it.
        if self.ai._running and use_wake_word:
            return {
                'type': 'voice_mode_status',
                'data': {'running': True, 'use_wake_word': use_wake_word, 'note': 'already_running'}
            }
        
        success = self.ai.start_voice_mode(use_wake_word=use_wake_word)
        return {
            'type': 'voice_mode_status',
            'data': {'running': success, 'use_wake_word': use_wake_word}
        }
    
    def _handle_stop_voice_mode(self, data):
        """Stop voice interaction mode"""
        self.ai.stop_voice_mode()
        return {'type': 'voice_mode_status', 'data': {'running': False}}
    
    def _handle_interrupt(self, data):
        """Interrupt current speech (manual interrupt from app)"""
        self.ai.interrupt()
        return {'type': 'interrupt_ack', 'data': {'success': True}}
    
    # =========== AUTONOMY CALLBACKS ===========
    
    def _autonomy_motor_callback(self, direction: str, speed: int):
        """Called by AutonomyEngine to send motor commands"""
        self.motors.send_motor_command(direction, speed)
    
    def _autonomy_status_callback(self, status: dict):
        """Called by AutonomyEngine to broadcast status to app"""
        self.ws_server.broadcast_sync(status)
    
    # =========== TELEMETRY ===========
    
    @staticmethod
    def _read_cpu_temp() -> float:
        """Read real CPU temperature from the Pi's thermal zone."""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp') as f:
                return round(int(f.read().strip()) / 1000.0, 1)
        except Exception:
            return ROBOT_STATE.cpu_temp

    @staticmethod
    def _read_wifi_rssi() -> int:
        """Read WiFi RSSI from iwconfig; returns dBm as a negative int."""
        try:
            out = subprocess.check_output(['iwconfig'], stderr=subprocess.DEVNULL).decode()
            match = re.search(r'Signal level=(-\d+)', out)
            return int(match.group(1)) if match else ROBOT_STATE.wifi_signal
        except Exception:
            return ROBOT_STATE.wifi_signal

    async def _broadcast_telemetry(self):
        """Periodically broadcast real telemetry to connected clients"""
        while self.running:
            try:
                # Read real values where possible
                cpu_temp = self._read_cpu_temp()
                wifi     = self._read_wifi_rssi()
                ROBOT_STATE.cpu_temp    = cpu_temp
                ROBOT_STATE.wifi_signal = wifi

                telemetry = {
                    'battery': ROBOT_STATE.battery_percent,
                    'wifi':    wifi,
                    'temp':    cpu_temp,
                    'storage': ROBOT_STATE.storage_percent,
                    'motor_dir':   ROBOT_STATE.motor_direction,
                    'motor_speed': ROBOT_STATE.motor_speed,
                }
                await self.ws_server.broadcast({
                    'type': 'telemetry',
                    'data': telemetry
                })
            except Exception as e:
                log.error(f"Telemetry broadcast error: {e}")
            await asyncio.sleep(self.telemetry_interval)
    
    # =========== MAIN RUN LOOP ===========
    
    @staticmethod
    def _launch_display():
        """
        Launch robot_display.html in Chromium kiosk mode (full-screen, no chrome).
        Runs in a background thread so it does not block startup.
        Tries chromium-browser first (Pi OS), then chromium (some distros).
        """
        html_path = KenzaMain.DISPLAY_HTML
        if not Path(html_path).exists():
            log.warning(f"🔭 Display HTML not found at: {html_path}")
            return

        chromium_bins = ['chromium-browser', 'chromium']
        for binary in chromium_bins:
            try:
                subprocess.Popen([
                    binary,
                    '--kiosk',
                    '--noerrdialogs',
                    '--disable-infobars',
                    '--disable-restore-session-state',
                    '--no-first-run',
                    '--disable-session-crashed-bubble',
                    f'file://{html_path}'
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                log.info(f"🖥️  Launched {binary} kiosk: {html_path}")
                return
            except FileNotFoundError:
                continue
        log.warning("🖥️  Chromium not found — display will not auto-launch. Install with: sudo apt install chromium-browser")

    async def run(self, start_pairing: bool = False):
        """
        Main run loop — starts ALL services on boot.

        Services started:
          1. GPIO motor controller
          2. HTTP motor server (port 8080)
          3. WebSocket server  (port 8765 — includes WebRTC signaling relay)
          4. Telemetry broadcaster (every 2 s)
          5. Chromium kiosk display (robot_display.html)
          6. Voice / conversation engine (wake-word listening)
          7. Optional pairing mode
        """
        self.running = True
        log.info("🤖 KENZA booting — starting all services...")

        # 1. GPIO motors
        self.motors.connect()

        # 2. HTTP motor server (port 8080)
        self.http_motor_srv.start(self.motors)

        # 3. WebSocket server (port 8765)
        await self.ws_server.start()

        # 4. Telemetry broadcaster
        self._telemetry_task = asyncio.create_task(self._broadcast_telemetry())

        # 5. Chromium kiosk display
        if self.auto_display:
            display_thread = threading.Thread(
                target=self._launch_display, daemon=True, name='ChromiumDisplay'
            )
            display_thread.start()
        else:
            log.info("🖥️  Display auto-launch skipped (--no-display)")

        # 6. Voice / conversation engine (always-on with wake word)
        if self.auto_voice:
            # Small delay to let WS server settle before mic init
            await asyncio.sleep(1.5)

            # Wire WS emotion broadcast into AIHandler BEFORE starting voice mode
            def _ws_emotion_broadcast(emotion: str):
                self.ws_server.broadcast_sync({'type': 'emotion', 'state': emotion})

            self.ai.on_emotion_fn = _ws_emotion_broadcast

            # Propagate callback to already-initialised ConversationEngine
            if self.ai.conversation_engine:
                self.ai.conversation_engine.on_emotion = lambda e: self.ai._on_emotion(e)

            self.ai.start_voice_mode(use_wake_word=True)
            log.info("🎤 Voice mode started — say 'Kenza' to begin")
        else:
            log.info("🎤 Voice mode skipped (--no-voice)")

        # 7. Pairing mode (optional)
        if start_pairing:
            self.pairing_service.start_pairing_async()

        log.info("\n" + "=" * 52)
        ip = WiFiManager.get_current_ip() or '0.0.0.0'
        log.info(f"  ✅  KENZA is ONLINE at {ip}")
        log.info(f"  🌐  WebSocket : ws://{ip}:8765")
        log.info(f"  🚗  HTTP Motor: http://{ip}:8080/F|B|L|R|S")
        log.info(f"  🖥️   Display  : robot_display.html (kiosk)")
        log.info(f"  🎤  Voice    : listening for 'Kenza'")
        log.info("=" * 52 + "\n")

        # Keep running
        try:
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

        # Cleanup
        log.info("🛑 Shutting down...")
        if self._telemetry_task:
            self._telemetry_task.cancel()
        if self.autonomy:
            self.autonomy.close()
        self.ai.stop_voice_mode()
        self.motors.disconnect()
        self.http_motor_srv.stop()
        await self.ws_server.stop()
        log.info("🤖 KENZA shutdown complete")
    
    def stop(self):
        """Signal the application to stop"""
        self.running = False


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main boot entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='KENZA Robot — Unified Boot Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Starts all services automatically:
  WebSocket server (8765)  — control & WebRTC signaling
  HTTP motor server (8080)  — /F /B /L /R /S endpoints
  Chromium kiosk display    — robot_display.html full-screen
  Voice engine              — wake-word AI conversation
  Telemetry broadcaster     — real battery/temp/wifi data
        """
    )
    parser.add_argument('--pairing', action='store_true',
                        help='Start in QR pairing mode')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--port', type=int, default=8765,
                        help='WebSocket port (default: 8765)')
    parser.add_argument('--no-display', action='store_true',
                        help='Skip Chromium kiosk launch')
    parser.add_argument('--no-voice', action='store_true',
                        help='Skip voice / wake-word mode')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    CONFIG.ws_port = args.port

    # Create application
    app = KenzaMain(
        auto_voice=not args.no_voice,
        auto_display=not args.no_display,
    )

    # Graceful shutdown on SIGINT / SIGTERM
    def signal_handler(sig, frame):
        log.info("Received shutdown signal")
        app.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run everything
    try:
        asyncio.run(app.run(start_pairing=args.pairing))
    except KeyboardInterrupt:
        log.info("Interrupted by user")


if __name__ == '__main__':
    main()

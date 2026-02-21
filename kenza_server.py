#!/usr/bin/env python3
"""
Kenza Server - Unified WebSocket for App UI
============================================
Combines gesture tracking with settings sync and robot control.

Features:
- Gesture data streaming to UI (from kenza_gesture.py)
- Real-time settings sync (bidirectional)
- Mode switching (autonomous/remote)
- Robot action commands
- Telemetry streaming

Usage:
    python kenza_server.py              # Start full server
    python kenza_server.py --no-gesture # Skip gesture tracking (for testing)
    python kenza_server.py --port 8765  # Custom port

Requirements:
    pip install websockets aiohttp
"""

import os
import sys
import json
import asyncio
import time
import threading
from dataclasses import dataclass, asdict
from typing import Dict, Set, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    print("⚠ websockets not installed. Run: pip install websockets")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    print("⚠ aiohttp not installed. Run: pip install aiohttp")

# Try to import gesture tracker
try:
    from kenza_gesture import GestureTracker, GestureType
    HAS_GESTURE = True
except ImportError:
    HAS_GESTURE = False
    print("⚠ kenza_gesture module not found")

# Try to import camera
try:
    from picamera2 import Picamera2
    HAS_PICAMERA = True
except ImportError:
    HAS_PICAMERA = False

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ServerConfig:
    PORT: int = 8765
    GESTURE_FPS: int = 30
    TELEMETRY_INTERVAL: float = 1.0  # seconds
    
CONFIG = ServerConfig()


# =============================================================================
# ROBOT STATE (Simulated)
# =============================================================================

@dataclass
class RobotState:
    """Current robot state and settings"""
    # Mode
    mode: str = "autonomous"  # "autonomous" or "remote"
    
    # Eyes
    eye_color: str = "cyan"
    eye_style: str = "normal"
    
    # Voice
    voice_volume: int = 80
    voice_pitch: int = 50
    
    # Movement  
    max_speed: int = 70
    obstacle_avoidance: bool = True
    
    # Telemetry
    battery: int = 95
    wifi_rssi: int = -45
    motor_temp: int = 38
    
    # Connection
    paired: bool = False
    
    # ESP32 Motor Controller
    esp32_ip: str = "192.168.4.1"
    esp32_connected: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# COMMAND HANDLERS
# =============================================================================

class CommandHandler:
    """Handles incoming commands from the app"""
    
    def __init__(self, state: RobotState):
        self.state = state
        self.callbacks = {}
    
    async def handle(self, message: str, websocket) -> Optional[Dict]:
        """Process incoming command and return response"""
        try:
            cmd = json.loads(message)
            cmd_type = cmd.get('type', '')
            data = cmd.get('data', {})
            
            if cmd_type == 'update_settings':
                return await self._update_settings(data)
            elif cmd_type == 'switch_mode':
                return await self._switch_mode(data)
            elif cmd_type == 'robot_action':
                return await self._robot_action(data)
            elif cmd_type == 'play_sound':
                return await self._play_sound(data)
            elif cmd_type == 'toggle_mic':
                return await self._toggle_mic(data)
            elif cmd_type == 'joystick':
                return await self._joystick(data)
            elif cmd_type == 'motor':
                return await self._motor(data)
            elif cmd_type == 'esp32_config':
                return await self._esp32_config(data)
            elif cmd_type == 'get_state':
                return {'type': 'state', 'data': self.state.to_dict()}
            # New handlers for expanded app
            elif cmd_type == 'slam_control':
                return await self._slam_control(data)
            elif cmd_type == 'follow_mode':
                return await self._follow_mode(data)
            elif cmd_type == 'sentry_mode':
                return await self._sentry_mode(data)
            elif cmd_type == 'privacy_mode':
                return await self._privacy_mode(data)
            elif cmd_type == 'voice_select':
                return await self._voice_select(data)
            elif cmd_type == 'ai_message':
                return await self._ai_message(data)
            else:
                print(f"[CMD] Unknown command: {cmd_type}")
                return None
                
        except json.JSONDecodeError:
            return None
        except Exception as e:
            print(f"[CMD] Error: {e}")
            return None
    
    async def _update_settings(self, data: Dict) -> Dict:
        """Update robot settings"""
        if 'eyes' in data:
            eyes = data['eyes']
            if 'color' in eyes:
                self.state.eye_color = eyes['color']
                print(f"[SETTINGS] Eye color → {self.state.eye_color}")
            if 'style' in eyes:
                self.state.eye_style = eyes['style']
                
        if 'voice' in data:
            voice = data['voice']
            if 'volume' in voice:
                self.state.voice_volume = voice['volume']
            if 'pitch' in voice:
                self.state.voice_pitch = voice['pitch']
                
        if 'speed' in data:
            self.state.max_speed = data['speed']
            print(f"[SETTINGS] Max speed → {self.state.max_speed}%")
            
        if 'obstacle_avoidance' in data:
            self.state.obstacle_avoidance = data['obstacle_avoidance']
            
        return {'type': 'settings_updated', 'data': self.state.to_dict()}
    
    async def _switch_mode(self, data: Dict) -> Dict:
        """Switch between autonomous and remote mode"""
        mode = data.get('mode', 'autonomous')
        self.state.mode = mode
        print(f"[MODE] Switched to → {mode.upper()}")
        
        # Here you would trigger actual mode switch logic
        # e.g., stop autonomous behaviors, enable joystick control
        
        return {'type': 'mode_changed', 'data': {'mode': mode}}
    
    async def _robot_action(self, data: Dict) -> Dict:
        """Trigger robot action (wave, dance, etc.)"""
        action = data.get('action', '')
        print(f"[ACTION] Triggered → {action}")
        
        # Here you would call actual robot control
        # e.g., self.robot.wave(), self.robot.dance()
        
        return {'type': 'action_started', 'data': {'action': action}}
    
    async def _play_sound(self, data: Dict) -> Dict:
        """Play a sound on the robot"""
        sound = data.get('sound', 'meow.wav')
        print(f"[SOUND] Playing → {sound}")
        
        # Here you would play the actual sound file
        # import pygame; pygame.mixer.music.load(sound); pygame.mixer.music.play()
        
        return {'type': 'sound_played', 'data': {'sound': sound}}
    
    async def _toggle_mic(self, data: Dict) -> Dict:
        """Toggle microphone mute"""
        muted = data.get('muted', False)
        print(f"[MIC] {'Muted' if muted else 'Unmuted'}")
        return {'type': 'mic_toggled', 'data': {'muted': muted}}
    
    async def _joystick(self, data: Dict) -> Dict:
        """Handle joystick input for remote control"""
        x = data.get('x', 0)  # -1 to 1
        y = data.get('y', 0)  # -1 to 1
        
        if self.state.mode == 'remote':
            # Here you would send to motor controller
            # left_speed = (y + x) * self.state.max_speed
            # right_speed = (y - x) * self.state.max_speed
            pass
            
        return None  # No response needed for high-frequency joystick updates
    
    async def _motor(self, data: Dict) -> Dict:
        """Forward motor commands to ESP32"""
        direction = data.get('direction', 'S')
        speed = data.get('speed', 0)
        
        # Map direction to ESP32 endpoint
        endpoint = f"http://{self.state.esp32_ip}/{direction}"
        
        if HAS_AIOHTTP:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(endpoint, timeout=aiohttp.ClientTimeout(total=0.5)) as resp:
                        if resp.status == 200:
                            self.state.esp32_connected = True
                        else:
                            self.state.esp32_connected = False
            except Exception as e:
                self.state.esp32_connected = False
                # Silent fail - high frequency commands
        else:
            # Fallback to sync requests (blocking, not ideal)
            try:
                import requests
                requests.get(endpoint, timeout=0.5)
                self.state.esp32_connected = True
            except:
                self.state.esp32_connected = False
        
        return None  # No response for high-frequency motor commands
    
    async def _esp32_config(self, data: Dict) -> Dict:
        """Configure ESP32 IP address"""
        ip = data.get('ip', '192.168.4.1')
        self.state.esp32_ip = ip
        print(f"[ESP32] IP configured: {ip}")
        
        # Test connection
        try:
            if HAS_AIOHTTP:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://{ip}/S", 
                        timeout=aiohttp.ClientTimeout(total=1.0)
                    ) as resp:
                        self.state.esp32_connected = resp.status == 200
            else:
                import requests
                resp = requests.get(f"http://{ip}/S", timeout=1.0)
                self.state.esp32_connected = resp.status_code == 200
        except:
            self.state.esp32_connected = False
        
        status = 'connected' if self.state.esp32_connected else 'unreachable'
        print(f"[ESP32] Status: {status}")
        
        return {
            'type': 'esp32_status',
            'data': {
                'ip': ip,
                'connected': self.state.esp32_connected
            }
        }
    
    async def _slam_control(self, data: Dict) -> Dict:
        """Control SLAM exploration"""
        action = data.get('action', 'status')  # start, stop, status
        print(f"[SLAM] Action: {action}")
        # TODO: Connect to actual SLAM system
        return {'type': 'slam_status', 'data': {'status': 'mapping' if action == 'start' else 'idle'}}
    
    async def _follow_mode(self, data: Dict) -> Dict:
        """Toggle Follow Me mode"""
        enabled = data.get('enabled', False)
        print(f"[FOLLOW] {'Enabled' if enabled else 'Disabled'}")
        # TODO: Connect to person tracking system
        return {'type': 'follow_status', 'data': {'enabled': enabled}}
    
    async def _sentry_mode(self, data: Dict) -> Dict:
        """Toggle Sentry/Security patrol mode"""
        enabled = data.get('enabled', False)
        start_time = data.get('start', '22:00')
        end_time = data.get('end', '06:00')
        print(f"[SENTRY] {'Enabled' if enabled else 'Disabled'} ({start_time} - {end_time})")
        # TODO: Connect to patrol scheduler
        return {'type': 'sentry_status', 'data': {'enabled': enabled, 'schedule': f'{start_time}-{end_time}'}}
    
    async def _privacy_mode(self, data: Dict) -> Dict:
        """Toggle Privacy Mode (disable camera/mic)"""
        enabled = data.get('enabled', False)
        print(f"[PRIVACY] {'🔴 Active - Camera/Mic disabled' if enabled else '🟢 Inactive'}")
        # TODO: Actually disable camera and mic hardware
        return {'type': 'privacy_status', 'data': {'enabled': enabled}}
    
    async def _voice_select(self, data: Dict) -> Dict:
        """Select voice avatar for TTS"""
        voice = data.get('voice', 'default')
        print(f"[VOICE] Selected: {voice}")
        # TODO: Connect to TTS system with voice profiles
        return {'type': 'voice_changed', 'data': {'voice': voice}}
    
    async def _ai_message(self, data: Dict) -> Dict:
        """Process AI chat message"""
        message = data.get('message', '')
        print(f"[AI] User: {message}")
        # TODO: Connect to LLM backend (Qwen/Mistral/Llama)
        # For now, return a placeholder response
        response = f"I heard you say: '{message}'. AI backend integration pending."
        return {'type': 'ai_response', 'data': {'message': response}}



# =============================================================================
# WEBSOCKET SERVER
# =============================================================================

class KenzaServer:
    """Unified WebSocket server for Kenza app"""
    
    def __init__(self, enable_gesture: bool = True):
        self.state = RobotState()
        self.handler = CommandHandler(self.state)
        self.clients: Set = set()
        self.enable_gesture = enable_gesture and HAS_GESTURE
        self.tracker = None
        self.camera = None
        self.running = False
        
    async def start(self, port: int = CONFIG.PORT):
        """Start the WebSocket server"""
        if not HAS_WEBSOCKETS:
            print("❌ Cannot start server: websockets not installed")
            return
            
        self.running = True
        
        # Initialize gesture tracking
        if self.enable_gesture:
            self._init_camera()
            if HAS_GESTURE:
                self.tracker = GestureTracker()
                print("[GESTURE] Tracker initialized")
        
        # Get local IP
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('10.255.255.255', 1))
            local_ip = s.getsockname()[0]
            s.close()
        except:
            local_ip = '127.0.0.1'
        
        print("\n" + "=" * 50)
        print("  KENZA SERVER RUNNING  ".center(50, "="))
        print("=" * 50)
        print(f"\n  WebSocket: ws://{local_ip}:{port}")
        print(f"  Gesture:   {'✓ Enabled' if self.enable_gesture else '✗ Disabled'}")
        print(f"\n  Open kenza_app.html and connect!\n")
        print("=" * 50)
        print("\nPress Ctrl+C to stop.\n")
        
        async with websockets.serve(self._handle_client, "0.0.0.0", port):
            await asyncio.Future()  # Run forever
    
    def _init_camera(self):
        """Initialize camera for gesture tracking"""
        if HAS_PICAMERA:
            try:
                self.camera = Picamera2()
                config = self.camera.create_video_configuration(main={"size": (640, 480)})
                self.camera.configure(config)
                self.camera.start()
                print("[CAMERA] PiCamera2 initialized")
            except Exception as e:
                print(f"[CAMERA] PiCamera2 failed: {e}")
                self.camera = None
        
        if self.camera is None and HAS_OPENCV:
            try:
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print("[CAMERA] OpenCV camera initialized")
            except Exception as e:
                print(f"[CAMERA] OpenCV failed: {e}")
                self.camera = None
    
    async def _handle_client(self, websocket):
        """Handle a WebSocket client connection"""
        self.clients.add(websocket)
        print(f"[WS] Client connected ({len(self.clients)} total)")
        
        # Send initial state
        await websocket.send(json.dumps({
            'type': 'connected',
            'data': self.state.to_dict()
        }))
        
        try:
            # Create tasks for sending and receiving
            receive_task = asyncio.create_task(self._receive_loop(websocket))
            gesture_task = asyncio.create_task(self._gesture_loop(websocket))
            telemetry_task = asyncio.create_task(self._telemetry_loop(websocket))
            
            await asyncio.gather(receive_task, gesture_task, telemetry_task)
            
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"[WS] Client disconnected ({len(self.clients)} total)")
    
    async def _receive_loop(self, websocket):
        """Receive and process commands from client"""
        try:
            async for message in websocket:
                response = await self.handler.handle(message, websocket)
                if response:
                    await websocket.send(json.dumps(response))
        except websockets.exceptions.ConnectionClosed:
            pass
    
    async def _gesture_loop(self, websocket):
        """Stream gesture data to client"""
        if not self.enable_gesture or not self.tracker:
            return
            
        frame_delay = 1.0 / CONFIG.GESTURE_FPS
        
        try:
            while True:
                frame = self._capture_frame()
                if frame is not None:
                    result = self.tracker.process_frame(frame)
                    gesture_data = result.to_dict()
                    gesture_data['type'] = 'gesture'
                    
                    await websocket.send(json.dumps(gesture_data))
                
                await asyncio.sleep(frame_delay)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f"[GESTURE] Error: {e}")
    
    async def _telemetry_loop(self, websocket):
        """Stream telemetry data to client"""
        try:
            while True:
                # Simulate telemetry updates (replace with real readings)
                self.state.battery = max(0, self.state.battery - 0.01)  # Slow drain
                
                telemetry = {
                    'type': 'telemetry',
                    'data': {
                        'battery': round(self.state.battery),
                        'wifi_rssi': self.state.wifi_rssi,
                        'motor_temp': self.state.motor_temp
                    }
                }
                
                await websocket.send(json.dumps(telemetry))
                await asyncio.sleep(CONFIG.TELEMETRY_INTERVAL)
        except websockets.exceptions.ConnectionClosed:
            pass
    
    def _capture_frame(self):
        """Capture a frame from camera"""
        if self.camera is None:
            return None
            
        if HAS_PICAMERA and isinstance(self.camera, Picamera2):
            frame = self.camera.capture_array()
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = self.camera.read()
            if ret:
                return cv2.flip(frame, 1)  # Mirror
            return None
    
    def stop(self):
        """Stop the server"""
        self.running = False
        if self.tracker:
            self.tracker.close()
        if self.camera:
            if HAS_PICAMERA and isinstance(self.camera, Picamera2):
                self.camera.stop()
            else:
                self.camera.release()


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Kenza Server - Unified WebSocket")
    parser.add_argument("--port", type=int, default=CONFIG.PORT, help="WebSocket port")
    parser.add_argument("--no-gesture", action="store_true", help="Disable gesture tracking")
    args = parser.parse_args()
    
    server = KenzaServer(enable_gesture=not args.no_gesture)
    
    try:
        asyncio.run(server.start(port=args.port))
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
    finally:
        server.stop()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Kenza Robot - Hand Gesture Tracking
====================================
Single-file module for:
- Hand detection using MediaPipe
- Gesture recognition (pinch, fist, palm, point, peace)
- Cursor/pointer position from index finger
- UI control events (click, drag, hover)
- Robot trick triggers

Gestures:
    - Open Palm    → Hover/Release
    - Pinch        → Click/Select
    - Closed Fist  → Drag mode
    - Point        → Direction (Up/Down/Left/Right)
    - Peace Sign   → Special action

Usage:
    python kenza_gesture.py --test
    
    # In your code:
    from kenza_gesture import GestureTracker
    tracker = GestureTracker()
    gesture = tracker.process_frame(frame)

Requirements:
    pip install mediapipe opencv-python
"""

import cv2
import math
import time
import numpy as np
from typing import Optional, Dict, Tuple, NamedTuple, List
from dataclasses import dataclass
from enum import Enum

# Try to import MediaPipe
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("⚠ MediaPipe not installed. Run: pip install mediapipe")


# =============================================================================
# GESTURE TYPES
# =============================================================================

class GestureType(Enum):
    NONE = "None"
    OPEN_PALM = "Open Palm"       # All fingers extended - hover/release
    CLOSED_FIST = "Closed Fist"   # All fingers closed - drag mode
    PINCH = "Pinch"               # Thumb + Index together - click
    POINT_UP = "Point Up"
    POINT_DOWN = "Point Down"
    POINT_LEFT = "Point Left"
    POINT_RIGHT = "Point Right"
    PEACE = "Peace Sign"          # Index + Middle extended
    THUMBS_UP = "Thumbs Up"
    THUMBS_DOWN = "Thumbs Down"


class UIAction(Enum):
    NONE = "none"
    HOVER = "hover"       # Moving cursor
    CLICK = "click"       # Pinch detected
    DRAG_START = "drag_start"   # Fist closed
    DRAGGING = "dragging"       # Fist moving
    DRAG_END = "drag_end"       # Fist opened
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"


# =============================================================================
# DATA TYPES
# =============================================================================

class HandPosition(NamedTuple):
    """Normalized hand position (0-1)"""
    x: float  # 0 = left, 1 = right
    y: float  # 0 = top, 1 = bottom


@dataclass
class GestureResult:
    """Result of gesture detection"""
    gesture: GestureType
    action: UIAction
    position: Optional[HandPosition]  # Index finger tip position
    confidence: float
    landmarks: Optional[List] = None
    
    def to_dict(self) -> Dict:
        return {
            'gesture': self.gesture.value,
            'action': self.action.value,
            'x': self.position.x if self.position else 0.5,
            'y': self.position.y if self.position else 0.5,
            'confidence': self.confidence
        }


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class GestureConfig:
    # Detection settings
    MAX_HANDS: int = 1
    MIN_DETECTION_CONFIDENCE: float = 0.7
    MIN_TRACKING_CONFIDENCE: float = 0.5
    
    # Gesture thresholds
    PINCH_THRESHOLD: float = 0.05  # Distance for pinch detection
    CLICK_HOLD_TIME: float = 0.1   # Seconds to register click
    
    # Display
    DRAW_LANDMARKS: bool = True
    GESTURE_FONT = cv2.FONT_HERSHEY_SIMPLEX


CONFIG = GestureConfig()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_distance(p1, p2) -> float:
    """Calculate distance between two landmarks"""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def is_finger_extended(landmarks, tip_idx: int, pip_idx: int) -> bool:
    """Check if a finger is extended (tip above PIP joint)"""
    return landmarks[tip_idx].y < landmarks[pip_idx].y


def is_thumb_extended(landmarks) -> bool:
    """Check if thumb is extended (accounts for hand orientation)"""
    # Check if right or left hand based on wrist-pinky angle
    is_right_hand = landmarks[5].x > landmarks[17].x
    
    if is_right_hand:
        return landmarks[4].x < landmarks[3].x
    else:
        return landmarks[4].x > landmarks[3].x


# =============================================================================
# GESTURE TRACKER
# =============================================================================

class GestureTracker:
    """
    Tracks hand gestures using MediaPipe.
    Provides gesture type and UI action for controlling interfaces.
    """
    
    def __init__(self):
        if not HAS_MEDIAPIPE:
            raise RuntimeError("MediaPipe not installed")
        
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=CONFIG.MAX_HANDS,
            min_detection_confidence=CONFIG.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=CONFIG.MIN_TRACKING_CONFIDENCE
        )
        
        # State tracking
        self.previous_gesture = GestureType.NONE
        self.drag_active = False
        self.pinch_start_time = 0
        self.last_pinch = False
        
        print("[GESTURE] Hand Gesture Tracker initialized")
    
    def process_frame(self, frame: np.ndarray) -> GestureResult:
        """
        Process a frame and return detected gesture.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            GestureResult with gesture type, UI action, and position
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            # No hand detected
            if self.drag_active:
                self.drag_active = False
                return GestureResult(
                    gesture=GestureType.NONE,
                    action=UIAction.DRAG_END,
                    position=None,
                    confidence=0.0
                )
            return GestureResult(
                gesture=GestureType.NONE,
                action=UIAction.NONE,
                position=None,
                confidence=0.0
            )
        
        # Get first hand landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark
        
        # Get index finger tip position (for cursor)
        position = HandPosition(x=lm[8].x, y=lm[8].y)
        
        # Check finger states
        index_extended = is_finger_extended(lm, 8, 6)
        middle_extended = is_finger_extended(lm, 12, 10)
        ring_extended = is_finger_extended(lm, 16, 14)
        pinky_extended = is_finger_extended(lm, 20, 18)
        thumb_extended = is_thumb_extended(lm)
        
        # Pinch detection (thumb tip to index tip)
        pinch_dist = get_distance(lm[4], lm[8])
        is_pinching = pinch_dist < CONFIG.PINCH_THRESHOLD
        
        # Determine gesture
        gesture = GestureType.NONE
        action = UIAction.HOVER
        
        # Priority: Pinch > Fist > Palm > Pointing
        if is_pinching:
            gesture = GestureType.PINCH
            action = UIAction.CLICK
            
        elif not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            gesture = GestureType.CLOSED_FIST
            if not self.drag_active:
                self.drag_active = True
                action = UIAction.DRAG_START
            else:
                action = UIAction.DRAGGING
                
        elif index_extended and middle_extended and ring_extended and pinky_extended:
            gesture = GestureType.OPEN_PALM
            if self.drag_active:
                self.drag_active = False
                action = UIAction.DRAG_END
            else:
                action = UIAction.HOVER
                
        elif index_extended and not middle_extended and not ring_extended and not pinky_extended:
            # Pointing - determine direction
            tip = lm[8]
            mcp = lm[5]
            dx = tip.x - mcp.x
            dy = tip.y - mcp.y
            
            if abs(dy) > abs(dx):
                gesture = GestureType.POINT_UP if dy < 0 else GestureType.POINT_DOWN
                action = UIAction.SCROLL_UP if dy < 0 else UIAction.SCROLL_DOWN
            else:
                gesture = GestureType.POINT_RIGHT if dx > 0 else GestureType.POINT_LEFT
                
        elif index_extended and middle_extended and not ring_extended and not pinky_extended:
            gesture = GestureType.PEACE
            
        elif thumb_extended and not index_extended and not middle_extended:
            # Check thumb direction for thumbs up/down
            if lm[4].y < lm[3].y:
                gesture = GestureType.THUMBS_UP
            else:
                gesture = GestureType.THUMBS_DOWN
        
        # Reset drag if no fist
        if gesture != GestureType.CLOSED_FIST and self.drag_active:
            self.drag_active = False
            action = UIAction.DRAG_END
        
        self.previous_gesture = gesture
        
        return GestureResult(
            gesture=gesture,
            action=action,
            position=position,
            confidence=1.0,  # MediaPipe doesn't provide per-detection confidence
            landmarks=lm
        )
    
    def draw_on_frame(self, frame: np.ndarray, result: GestureResult) -> np.ndarray:
        """Draw gesture info and landmarks on frame"""
        output = frame.copy()
        
        if result.landmarks:
            # Draw hand landmarks
            h, w = frame.shape[:2]
            
            # Draw connections manually for headless compatibility
            for id, lm in enumerate(result.landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(output, (cx, cy), 5, (0, 255, 0), -1)
            
            # Draw cursor position (index tip)
            if result.position:
                cursor_x = int(result.position.x * w)
                cursor_y = int(result.position.y * h)
                
                # Cursor indicator
                color = (0, 255, 255) if result.action == UIAction.CLICK else (255, 0, 255)
                cv2.circle(output, (cursor_x, cursor_y), 15, color, 3)
                
                if result.action == UIAction.DRAGGING:
                    cv2.circle(output, (cursor_x, cursor_y), 20, (0, 0, 255), 2)
        
        # Draw gesture text
        cv2.putText(output, result.gesture.value, (50, 80), 
                    CONFIG.GESTURE_FONT, 2, (0, 255, 0), 3)
        cv2.putText(output, f"Action: {result.action.value}", (50, 140),
                    CONFIG.GESTURE_FONT, 1, (255, 255, 0), 2)
        
        return output
    
    def close(self):
        self.hands.close()


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def create_gesture_processor():
    """
    Create a gesture processing function for integration.
    
    Usage:
        from kenza_gesture import create_gesture_processor
        process = create_gesture_processor()
        
        # In video loop:
        gesture_data = process(frame)
        print(gesture_data)  # {'gesture': 'Pinch', 'action': 'click', 'x': 0.5, 'y': 0.3}
    """
    tracker = GestureTracker()
    
    def process(frame: np.ndarray) -> Dict:
        result = tracker.process_frame(frame)
        return result.to_dict()
    
    return process


# =============================================================================
# COMMAND LINE TEST
# =============================================================================

def test_with_camera():
    """Test gesture detection with camera (headless mode)"""
    print("\n=== Hand Gesture Test (Headless) ===\n")
    
    try:
        from picamera2 import Picamera2
        picam = Picamera2()
        config = picam.create_video_configuration(main={"size": (640, 480)})
        picam.configure(config)
        picam.start()
        use_picam = True
        print("[CAMERA] Using PiCamera2")
    except:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        use_picam = False
        print("[CAMERA] Using USB/Default camera")
    
    tracker = GestureTracker()
    
    print("\n[GESTURES]")
    print("  Open Palm  → Hover")
    print("  Pinch      → Click")
    print("  Fist       → Drag")
    print("  Point      → Direction")
    print("\nPress Ctrl+C to stop.\n")
    
    frame_count = 0
    start_time = time.time()
    last_print = 0
    last_gesture = GestureType.NONE
    
    try:
        while True:
            if use_picam:
                frame = picam.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)  # Mirror for intuitive control
            
            # Process gesture
            result = tracker.process_frame(frame)
            
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Print when gesture changes or every second
            if result.gesture != last_gesture or time.time() - last_print >= 1.0:
                last_print = time.time()
                last_gesture = result.gesture
                
                if result.gesture != GestureType.NONE:
                    pos_str = f"({result.position.x:.2f}, {result.position.y:.2f})" if result.position else "N/A"
                    print(f"[{result.gesture.value}] action:{result.action.value} pos:{pos_str}")
                else:
                    print("[NO HAND] Waiting for hand...")
                
                print(f"[FPS] {fps:.1f}\n")
            
            time.sleep(0.033)  # ~30 FPS
    
    except KeyboardInterrupt:
        print("\n\n👋 Stopped")
    
    finally:
        tracker.close()
        if use_picam:
            picam.stop()
        else:
            cap.release()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Kenza Gesture - Hand Tracking")
    parser.add_argument("--test", action="store_true", help="Test with camera (console output)")
    parser.add_argument("--serve", action="store_true", help="Start WebSocket server for UI")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port (default: 8765)")
    args = parser.parse_args()
    
    if args.serve:
        run_websocket_server(port=args.port)
    elif args.test:
        test_with_camera()
    else:
        parser.print_help()


async def gesture_websocket_handler(websocket, tracker, use_picam, camera):
    """Handle WebSocket connections and send gesture data"""
    import json
    
    print(f"[WS] Client connected")
    
    try:
        while True:
            # Capture frame
            if use_picam:
                frame = camera.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = camera.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
            
            # Process gesture
            result = tracker.process_frame(frame)
            
            # Send to browser
            data = result.to_dict()
            await websocket.send(json.dumps(data))
            
            await asyncio.sleep(0.033)  # ~30 FPS
    
    except Exception as e:
        print(f"[WS] Client disconnected: {e}")


def run_websocket_server(port=8765):
    """Run WebSocket server for gesture UI"""
    import asyncio
    
    try:
        import websockets
    except ImportError:
        print("❌ websockets not installed. Run: pip install websockets")
        return
    
    print("\n=== Gesture WebSocket Server ===\n")
    
    # Initialize camera
    try:
        from picamera2 import Picamera2
        picam = Picamera2()
        config = picam.create_video_configuration(main={"size": (640, 480)})
        picam.configure(config)
        picam.start()
        use_picam = True
        camera = picam
        print("[CAMERA] Using PiCamera2")
    except:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        use_picam = False
        camera = cap
        print("[CAMERA] Using USB/Default camera")
    
    tracker = GestureTracker()
    
    # Handler for websockets v15+ (only receives websocket, not path)
    async def handler(websocket):
        await gesture_websocket_handler(websocket, tracker, use_picam, camera)
    
    async def main_server():
        import socket
        local_ip = socket.gethostbyname(socket.gethostname())
        
        print(f"\n🖐️ Gesture server running!")
        print(f"   WebSocket: ws://localhost:{port}")
        print(f"   WebSocket: ws://{local_ip}:{port}")
        print(f"\n   Open gesture_ui.html in browser and connect.\n")
        print("Press Ctrl+C to stop.\n")
        
        async with websockets.serve(handler, "0.0.0.0", port):
            await asyncio.Future()  # Run forever
    
    try:
        asyncio.run(main_server())
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
    finally:
        tracker.close()
        if use_picam:
            camera.stop()
        else:
            camera.release()


# Add asyncio import at runtime for server
import asyncio


if __name__ == "__main__":
    main()

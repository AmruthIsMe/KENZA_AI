#!/usr/bin/env python3
"""
Kenza Robot - Face Detection & Recognition
===========================================
Single-file module for:
- Real-time face detection using MediaPipe
- Owner recognition (learns and identifies you)
- Face position tracking (for motor following in future)

Usage:
    # Add your face to known faces
    python kenza_vision.py --add-face "Amruth"
    
    # Test face detection with camera
    python kenza_vision.py --test
    
    # Get face position (for motor control)
    from kenza_vision import FaceTracker
    tracker = FaceTracker()
    position = tracker.get_face_position(frame)

Requirements:
    pip install mediapipe opencv-python numpy
"""

import os
import cv2
import numpy as np
import pickle
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple, NamedTuple
from dataclasses import dataclass
from collections import deque

# Try to import MediaPipe
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("⚠ MediaPipe not installed. Run: pip install mediapipe")

# Try to import face_recognition (optional, for better recognition)
try:
    import face_recognition
    HAS_FACE_RECOGNITION = True
except ImportError:
    HAS_FACE_RECOGNITION = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class VisionConfig:
    # Detection settings
    MIN_DETECTION_CONFIDENCE: float = 0.5
    MIN_TRACKING_CONFIDENCE: float = 0.5
    
    # Recognition settings
    KNOWN_FACES_DIR: str = "known_faces"
    RECOGNITION_THRESHOLD: float = 0.6
    
    # Tracking settings
    SMOOTHING_FRAMES: int = 5  # Average position over N frames
    
    # Display settings
    BOX_COLOR_KNOWN: Tuple[int, int, int] = (0, 255, 0)     # Green for owner
    BOX_COLOR_UNKNOWN: Tuple[int, int, int] = (0, 165, 255)  # Orange for unknown
    BOX_COLOR_NO_FACE: Tuple[int, int, int] = (128, 128, 128)
    
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE: float = 0.6
    FONT_THICKNESS: int = 2


CONFIG = VisionConfig()


# =============================================================================
# DATA TYPES
# =============================================================================

class FacePosition(NamedTuple):
    """Position of face relative to frame center (-1 to 1)"""
    x: float  # -1 = far left, 0 = center, 1 = far right
    y: float  # -1 = top, 0 = center, 1 = bottom
    size: float  # 0-1, relative size of face in frame
    confidence: float


@dataclass
class DetectedFace:
    """Information about a detected face"""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    position: FacePosition
    name: Optional[str] = None  # None if unknown
    is_owner: bool = False
    landmarks: Optional[Dict] = None


# =============================================================================
# FACE DETECTOR (MediaPipe)
# =============================================================================

class FaceDetector:
    """
    Detects faces using MediaPipe Face Detection.
    Fast and accurate, optimized for edge devices.
    """
    
    def __init__(self):
        if not HAS_MEDIAPIPE:
            raise RuntimeError("MediaPipe not installed")
        
        self.mp_face = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        
        self.detector = self.mp_face.FaceDetection(
            model_selection=0,  # 0 = short range (2m), 1 = full range (5m)
            min_detection_confidence=CONFIG.MIN_DETECTION_CONFIDENCE
        )
        
        print("[VISION] MediaPipe Face Detector initialized")
    
    def detect(self, frame: np.ndarray) -> List[DetectedFace]:
        """
        Detect all faces in a frame.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            List of DetectedFace objects
        """
        h, w = frame.shape[:2]
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)
        
        faces = []
        
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Calculate position relative to frame center
                center_x = x + width / 2
                center_y = y + height / 2
                
                # Normalize to -1 to 1 range
                pos_x = (center_x / w) * 2 - 1
                pos_y = (center_y / h) * 2 - 1
                size = (width * height) / (w * h)
                
                position = FacePosition(
                    x=pos_x,
                    y=pos_y,
                    size=size,
                    confidence=detection.score[0]
                )
                
                faces.append(DetectedFace(
                    bbox=(x, y, width, height),
                    position=position,
                    landmarks=None
                ))
        
        return faces
    
    def close(self):
        self.detector.close()


# =============================================================================
# FACE RECOGNIZER (Identifies Owner)
# =============================================================================

class FaceRecognizer:
    """
    Recognizes known faces (owner, family).
    Uses face_recognition library if available, else simple encoding comparison.
    """
    
    def __init__(self, known_faces_dir: str = None):
        self.known_faces_dir = Path(known_faces_dir or CONFIG.KNOWN_FACES_DIR)
        self.known_faces_dir.mkdir(parents=True, exist_ok=True)
        
        self.known_encodings: Dict[str, List[np.ndarray]] = {}
        self.owner_name: Optional[str] = None
        
        self._load_known_faces()
        
        print(f"[VISION] Face Recognizer initialized ({len(self.known_encodings)} known faces)")
    
    def _load_known_faces(self):
        """Load known face encodings from disk"""
        encodings_file = self.known_faces_dir / "encodings.pkl"
        
        if encodings_file.exists():
            try:
                with open(encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_encodings = data.get('encodings', {})
                    self.owner_name = data.get('owner', None)
                print(f"    > Loaded {len(self.known_encodings)} known faces")
                if self.owner_name:
                    print(f"    > Owner: {self.owner_name}")
            except Exception as e:
                print(f"    ! Failed to load encodings: {e}")
    
    def _save_known_faces(self):
        """Save known face encodings to disk"""
        encodings_file = self.known_faces_dir / "encodings.pkl"
        
        try:
            with open(encodings_file, 'wb') as f:
                pickle.dump({
                    'encodings': self.known_encodings,
                    'owner': self.owner_name
                }, f)
        except Exception as e:
            print(f"[VISION] Failed to save encodings: {e}")
    
    def add_face(self, frame: np.ndarray, name: str, is_owner: bool = False) -> bool:
        """
        Add a face to the known faces database.
        
        Args:
            frame: Image containing a face
            name: Name of the person
            is_owner: Whether this is the robot's owner
            
        Returns:
            True if face was added successfully
        """
        if not HAS_FACE_RECOGNITION:
            print("[VISION] face_recognition library not installed")
            print("         Run: pip install face_recognition")
            return False
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find face and get encoding
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if not face_locations:
            print("[VISION] No face found in image")
            return False
        
        # Get encoding for the first face
        encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        if not encodings:
            print("[VISION] Could not encode face")
            return False
        
        # Add to database
        if name not in self.known_encodings:
            self.known_encodings[name] = []
        
        self.known_encodings[name].append(encodings[0])
        
        if is_owner:
            self.owner_name = name
        
        # Save image for reference
        face_img_path = self.known_faces_dir / f"{name}_{len(self.known_encodings[name])}.jpg"
        cv2.imwrite(str(face_img_path), frame)
        
        self._save_known_faces()
        
        print(f"[VISION] Added face for '{name}' (owner={is_owner})")
        return True
    
    def recognize(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Tuple[Optional[str], bool]:
        """
        Recognize a face in the given bounding box.
        
        Args:
            frame: Full image
            face_bbox: (x, y, width, height) of the face
            
        Returns:
            (name, is_owner) or (None, False) if unknown
        """
        if not HAS_FACE_RECOGNITION or not self.known_encodings:
            return None, False
        
        x, y, w, h = face_bbox
        
        # Extract face region with some padding
        pad = int(min(w, h) * 0.2)
        y1 = max(0, y - pad)
        y2 = min(frame.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(frame.shape[1], x + w + pad)
        
        face_img = frame[y1:y2, x1:x2]
        
        if face_img.size == 0:
            return None, False
        
        # Convert to RGB
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Get encoding
        face_locations = face_recognition.face_locations(rgb_face)
        if not face_locations:
            return None, False
        
        encodings = face_recognition.face_encodings(rgb_face, face_locations)
        if not encodings:
            return None, False
        
        face_encoding = encodings[0]
        
        # Compare with known faces
        best_match = None
        best_distance = CONFIG.RECOGNITION_THRESHOLD
        
        for name, known_encs in self.known_encodings.items():
            distances = face_recognition.face_distance(known_encs, face_encoding)
            min_distance = min(distances) if len(distances) > 0 else 1.0
            
            if min_distance < best_distance:
                best_distance = min_distance
                best_match = name
        
        if best_match:
            is_owner = (best_match == self.owner_name)
            return best_match, is_owner
        
        return None, False


# =============================================================================
# FACE TRACKER (Position Tracking for Motor Control)
# =============================================================================

class FaceTracker:
    """
    Tracks face position for motor control.
    Provides smoothed position data for following behavior.
    """
    
    def __init__(self):
        self.detector = FaceDetector() if HAS_MEDIAPIPE else None
        self.recognizer = FaceRecognizer()
        
        # Position history for smoothing
        self.position_history = deque(maxlen=CONFIG.SMOOTHING_FRAMES)
        
        # Last known position
        self.last_position: Optional[FacePosition] = None
        self.last_face: Optional[DetectedFace] = None
        self.frames_since_detection = 0
        
        print("[VISION] Face Tracker initialized")
    
    def process_frame(self, frame: np.ndarray, recognize: bool = True) -> List[DetectedFace]:
        """
        Process a frame and return detected faces with recognition.
        
        Args:
            frame: BGR image
            recognize: Whether to run face recognition
            
        Returns:
            List of detected faces with names if recognized
        """
        if not self.detector:
            return []
        
        # Detect faces
        faces = self.detector.detect(frame)
        
        # Run recognition on each face
        if recognize and faces:
            for face in faces:
                name, is_owner = self.recognizer.recognize(frame, face.bbox)
                face.name = name
                face.is_owner = is_owner
        
        # Update tracking
        if faces:
            # Track the largest face (assumed to be primary)
            primary_face = max(faces, key=lambda f: f.position.size)
            self.position_history.append(primary_face.position)
            self.last_position = self._get_smoothed_position()
            self.last_face = primary_face
            self.frames_since_detection = 0
        else:
            self.frames_since_detection += 1
        
        return faces
    
    def _get_smoothed_position(self) -> Optional[FacePosition]:
        """Get smoothed position from history"""
        if not self.position_history:
            return None
        
        positions = list(self.position_history)
        
        avg_x = sum(p.x for p in positions) / len(positions)
        avg_y = sum(p.y for p in positions) / len(positions)
        avg_size = sum(p.size for p in positions) / len(positions)
        avg_conf = sum(p.confidence for p in positions) / len(positions)
        
        return FacePosition(x=avg_x, y=avg_y, size=avg_size, confidence=avg_conf)
    
    def get_motor_command(self) -> Dict[str, float]:
        """
        Get motor command for following the face.
        
        Returns:
            {
                'pan': -1 to 1 (left to right),
                'tilt': -1 to 1 (up to down),
                'approach': -1 to 1 (back up to move closer),
                'confidence': 0 to 1
            }
        """
        if self.last_position is None or self.frames_since_detection > 30:
            return {'pan': 0, 'tilt': 0, 'approach': 0, 'confidence': 0}
        
        pos = self.last_position
        
        # Pan: if face is to the right (x > 0), pan right (positive)
        pan = pos.x
        
        # Tilt: if face is below center (y > 0), tilt down (positive)
        tilt = pos.y
        
        # Approach: if face is small, move closer
        # Target size is ~0.1 of frame
        target_size = 0.1
        approach = (target_size - pos.size) * 5
        approach = max(-1, min(1, approach))
        
        # Reduce confidence over time if no detection
        decay = max(0, 1 - self.frames_since_detection / 30)
        confidence = pos.confidence * decay
        
        return {
            'pan': pan,
            'tilt': tilt,
            'approach': approach,
            'confidence': confidence
        }
    
    def draw_faces(self, frame: np.ndarray, faces: List[DetectedFace]) -> np.ndarray:
        """
        Draw bounding boxes and names on frame.
        """
        output = frame.copy()
        
        for face in faces:
            x, y, w, h = face.bbox
            
            # Choose color based on recognition
            if face.is_owner:
                color = CONFIG.BOX_COLOR_KNOWN
                label = f"👑 {face.name}"
            elif face.name:
                color = CONFIG.BOX_COLOR_KNOWN
                label = face.name
            else:
                color = CONFIG.BOX_COLOR_UNKNOWN
                label = "Unknown"
            
            # Draw box
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            
            # Draw label background
            label_size = cv2.getTextSize(label, CONFIG.FONT, CONFIG.FONT_SCALE, CONFIG.FONT_THICKNESS)[0]
            cv2.rectangle(output, (x, y - label_size[1] - 10), (x + label_size[0] + 10, y), color, -1)
            
            # Draw label text
            cv2.putText(output, label, (x + 5, y - 5), CONFIG.FONT, CONFIG.FONT_SCALE, (0, 0, 0), CONFIG.FONT_THICKNESS)
            
            # Draw position indicator (for debugging motor following)
            if face.is_owner or (len(faces) == 1):
                pos = face.position
                indicator = f"x:{pos.x:.2f} y:{pos.y:.2f}"
                cv2.putText(output, indicator, (x, y + h + 20), CONFIG.FONT, 0.5, color, 1)
        
        return output
    
    def add_owner_face(self, frame: np.ndarray, name: str) -> bool:
        """Add the owner's face from a frame"""
        return self.recognizer.add_face(frame, name, is_owner=True)
    
    def close(self):
        if self.detector:
            self.detector.close()


# =============================================================================
# INTEGRATION WITH KENZA_STREAM (Video Processing)
# =============================================================================

def create_face_processor():
    """
    Create a face processing function for integration with kenza_stream.py
    
    Usage in kenza_stream.py:
        from kenza_vision import create_face_processor
        process_frame = create_face_processor()
        
        # In video capture loop:
        processed_frame = process_frame(frame)
    """
    tracker = FaceTracker()
    
    def process(frame: np.ndarray) -> np.ndarray:
        faces = tracker.process_frame(frame)
        return tracker.draw_faces(frame, faces)
    
    return process


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def test_with_camera():
    """Test face detection with camera (headless mode - console output)"""
    import sys
    import select
    import threading
    
    print("\n=== Face Detection Test (Headless) ===\n")
    
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
        use_picam = False
        print("[CAMERA] Using USB/Default camera")
    
    tracker = FaceTracker()
    
    print("\n[HEADLESS MODE] Commands:")
    print("  Type 'a' + Enter → Add your face as owner")
    print("  Ctrl+C → Stop\n")
    
    frame_count = 0
    start_time = time.time()
    last_print = 0
    current_frame = None
    capture_face_flag = False
    
    # Thread to listen for keyboard input
    def input_listener():
        nonlocal capture_face_flag
        while True:
            try:
                cmd = input().strip().lower()
                if cmd == 'a':
                    capture_face_flag = True
            except:
                break
    
    input_thread = threading.Thread(target=input_listener, daemon=True)
    input_thread.start()
    
    try:
        while True:
            # Capture frame
            if use_picam:
                frame = picam.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = cap.read()
                if not ret:
                    break
            
            current_frame = frame.copy()
            
            # Check if user wants to capture face
            if capture_face_flag:
                capture_face_flag = False
                print("\n📸 Capturing face...")
                name = input("Enter your name: ").strip()
                if name and current_frame is not None:
                    if tracker.add_owner_face(current_frame, name):
                        print(f"✅ Added {name} as owner!\n")
                    else:
                        print("❌ Failed to add face. Make sure face is visible.\n")
                else:
                    print("❌ Cancelled\n")
            
            # Process frame
            faces = tracker.process_frame(frame, recognize=HAS_FACE_RECOGNITION)
            cmd = tracker.get_motor_command()
            
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Print status every second
            if time.time() - last_print >= 1.0:
                last_print = time.time()
                
                if faces:
                    for i, face in enumerate(faces):
                        name = face.name or "Unknown"
                        owner = " 👑" if face.is_owner else ""
                        print(f"[FACE {i+1}] {name}{owner} | pos: ({face.position.x:.2f}, {face.position.y:.2f}) | conf: {face.position.confidence:.2f}")
                    print(f"[MOTOR] pan:{cmd['pan']:.2f} tilt:{cmd['tilt']:.2f} approach:{cmd['approach']:.2f}")
                else:
                    print("[NO FACE] Looking...")
                print(f"[FPS] {fps:.1f}\n")
            
            time.sleep(0.033)  # ~30 FPS cap
    
    except KeyboardInterrupt:
        print("\n\n👋 Stopped")
    
    finally:
        tracker.close()
        if use_picam:
            picam.stop()
        else:
            cap.release()


def add_face_from_image(image_path: str, name: str, is_owner: bool = False):
    """Add a face from an image file"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return False
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not read image: {image_path}")
        return False
    
    recognizer = FaceRecognizer()
    return recognizer.add_face(frame, name, is_owner)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Kenza Vision - Face Detection & Recognition")
    parser.add_argument("--test", action="store_true", help="Test with camera")
    parser.add_argument("--add-face", metavar="NAME", help="Add your face with given name")
    parser.add_argument("--image", metavar="PATH", help="Image file to add face from")
    parser.add_argument("--owner", action="store_true", help="Mark added face as owner")
    args = parser.parse_args()
    
    if args.add_face:
        if args.image:
            success = add_face_from_image(args.image, args.add_face, args.owner)
        else:
            print("Use --image to specify an image, or use --test mode to capture from camera")
            return
        
        if success:
            print(f"✅ Added face for '{args.add_face}'")
        else:
            print("❌ Failed to add face")
    
    elif args.test:
        test_with_camera()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

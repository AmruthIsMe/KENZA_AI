#!/usr/bin/env python3
"""
Kenza Robot - Autonomous Movement Engine
==========================================
Camera-based autonomy pipeline:
- Object detection (MobileNet-SSD or contour fallback)
- Collision avoidance (zone-based proximity analysis)
- Reactive path planning (state machine)
- Person following with pose tracking + identity memory
- Gesture-based navigation

Runs as a daemon thread, sends motor commands via callback,
broadcasts status via WebSocket.

Usage:
    from kenza_autonomy import AutonomyEngine

    engine = AutonomyEngine(motor_callback=my_motor_fn, status_callback=my_status_fn)
    engine.start_explore()      # autonomous roaming
    engine.start_follow()       # follow nearest person
    engine.start_gesture_nav()  # hand-gesture steering
    engine.stop()
"""

import time
import threading
import logging
import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Tuple
from collections import deque

import numpy as np

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

log = logging.getLogger("kenza.autonomy")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AutonomyConfig:
    """Configuration for the autonomy engine"""
    # Detection
    detection_fps: int = 10
    frame_width: int = 640
    frame_height: int = 480

    # Collision avoidance
    obstacle_area_threshold: float = 0.08     # bbox area ratio to trigger avoidance
    obstacle_close_threshold: float = 0.15    # very close — emergency stop
    zone_left_boundary: float = 0.33          # frame percentage
    zone_right_boundary: float = 0.66

    # Path planning
    turn_duration: float = 0.6               # seconds to turn before re-evaluating
    explore_speed: int = 60                   # motor speed for exploration (0-100)

    # Person following
    follow_speed: int = 50
    follow_distance_min: float = 0.10         # min bbox area (too close → back up)
    follow_distance_max: float = 0.03         # max bbox area (too far → move closer)
    follow_center_deadzone: float = 0.15      # normalized, no turn needed
    identity_match_threshold: float = 0.55    # histogram correlation threshold
    identity_memory_timeout: float = 3.0      # seconds before forgetting target
    pose_confidence: float = 0.5

    # Gesture navigation
    gesture_speed: int = 55

    # MobileNet-SSD model paths
    model_prototxt: str = "models/MobileNetSSD_deploy.prototxt"
    model_weights: str = "models/MobileNetSSD_deploy.caffemodel"


CONFIG = AutonomyConfig()


# =============================================================================
# ENUMS
# =============================================================================

class AutonomyState(Enum):
    IDLE = "idle"
    EXPLORING = "exploring"
    AVOIDING = "avoiding"
    TURNING_LEFT = "turning_left"
    TURNING_RIGHT = "turning_right"
    FOLLOWING = "following"
    FOLLOWING_SEARCHING = "searching"
    SENTRY = "sentry"
    GESTURE_NAV = "gesture_nav"


class AutonomyMode(Enum):
    NONE = "none"
    EXPLORE = "explore"
    FOLLOW = "follow"
    SENTRY = "sentry"
    GESTURE = "gesture"


# =============================================================================
# OBJECT DETECTOR
# =============================================================================

# MobileNet-SSD COCO class labels
MOBILENET_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Classes that are physical obstacles
OBSTACLE_CLASSES = {
    "bicycle", "bus", "car", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "pottedplant", "sheep", "sofa", "train", "bottle"
}


@dataclass
class Detection:
    """A single detected object"""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]   # (x, y, w, h)
    center_x: float                    # normalized 0-1
    center_y: float                    # normalized 0-1
    area_ratio: float                  # bbox area / frame area


class ObjectDetector:
    """
    Detects objects using MobileNet-SSD (DNN) or contour-based fallback.
    """

    def __init__(self):
        self.net = None
        self.use_dnn = False
        self._load_model()

    def _load_model(self):
        """Try to load MobileNet-SSD, fall back to contour detection"""
        if not HAS_CV2:
            log.warning("OpenCV not available — detection disabled")
            return

        proto = os.path.join(os.path.dirname(__file__), CONFIG.model_prototxt)
        weights = os.path.join(os.path.dirname(__file__), CONFIG.model_weights)

        if os.path.exists(proto) and os.path.exists(weights):
            try:
                self.net = cv2.dnn.readNetFromCaffe(proto, weights)
                self.use_dnn = True
                log.info("🧠 MobileNet-SSD loaded for object detection")
            except Exception as e:
                log.warning(f"Failed to load DNN model: {e}, using contour fallback")
        else:
            log.info("📐 Using contour-based obstacle detection (no DNN model found)")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a frame"""
        if not HAS_CV2:
            return []

        if self.use_dnn and self.net is not None:
            return self._detect_dnn(frame)
        else:
            return self._detect_contours(frame)

    def _detect_dnn(self, frame: np.ndarray) -> List[Detection]:
        """Detect using MobileNet-SSD"""
        h, w = frame.shape[:2]
        frame_area = h * w
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
        )
        self.net.setInput(blob)
        detections_raw = self.net.forward()

        results = []
        for i in range(detections_raw.shape[2]):
            confidence = float(detections_raw[0, 0, i, 2])
            if confidence < 0.4:
                continue

            class_id = int(detections_raw[0, 0, i, 1])
            if class_id < 0 or class_id >= len(MOBILENET_CLASSES):
                continue

            label = MOBILENET_CLASSES[class_id]
            box = detections_raw[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue

            results.append(Detection(
                label=label,
                confidence=confidence,
                bbox=(x1, y1, bw, bh),
                center_x=(x1 + bw / 2) / w,
                center_y=(y1 + bh / 2) / h,
                area_ratio=(bw * bh) / frame_area,
            ))

        return results

    def _detect_contours(self, frame: np.ndarray) -> List[Detection]:
        """Fallback: detect large blobs via contour analysis"""
        h, w = frame.shape[:2]
        frame_area = h * w

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        # Use adaptive thresholding to find dark/distinct regions
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 10
        )

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for contour in contours:
            area = cv2.contourArea(contour)
            area_ratio = area / frame_area

            # Only consider significant sized objects
            if area_ratio < 0.02:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            results.append(Detection(
                label="obstacle",
                confidence=min(area_ratio * 5, 0.99),
                bbox=(x, y, bw, bh),
                center_x=(x + bw / 2) / w,
                center_y=(y + bh / 2) / h,
                area_ratio=area_ratio,
            ))

        return results


# =============================================================================
# COLLISION AVOIDANCE
# =============================================================================

@dataclass
class CollisionResult:
    """Result of collision analysis"""
    blocked_left: bool = False
    blocked_center: bool = False
    blocked_right: bool = False
    emergency_stop: bool = False
    obstacle_count: int = 0
    clearest_zone: str = "center"   # "left", "center", "right"
    closest_distance: float = 1.0   # 0 = touching, 1 = far away


class CollisionAvoidance:
    """
    Analyzes detected objects to determine which zones are blocked.
    """

    def __init__(self):
        self.enabled = True

    def analyze(self, detections: List[Detection], frame_w: int, frame_h: int) -> CollisionResult:
        """
        Analyze detections and return collision result.
        
        Args:
            detections: list of Detection dicts or Detection objects
            frame_w: frame width
            frame_h: frame height
        """
        result = CollisionResult()

        if not self.enabled:
            return result

        obstacles = []
        for d in detections:
            # Support both Detection objects and dicts
            if isinstance(d, dict):
                label = d.get('label', 'obstacle')
                area_ratio = d.get('area_ratio', 0)
                center_x = d.get('center_x', 0.5)
            else:
                label = d.label
                area_ratio = d.area_ratio
                center_x = d.center_x

            # Skip persons (we're detecting them separately for following)
            if label == "person":
                continue

            if area_ratio >= CONFIG.obstacle_area_threshold:
                obstacles.append({
                    'label': label,
                    'area_ratio': area_ratio,
                    'center_x': center_x,
                })

        result.obstacle_count = len(obstacles)

        if not obstacles:
            return result

        # Analyze zones
        left_max = 0.0
        center_max = 0.0
        right_max = 0.0

        for obs in obstacles:
            cx = obs['center_x']
            ar = obs['area_ratio']

            if cx < CONFIG.zone_left_boundary:
                left_max = max(left_max, ar)
            elif cx > CONFIG.zone_right_boundary:
                right_max = max(right_max, ar)
            else:
                center_max = max(center_max, ar)

        result.blocked_left = left_max >= CONFIG.obstacle_area_threshold
        result.blocked_center = center_max >= CONFIG.obstacle_area_threshold
        result.blocked_right = right_max >= CONFIG.obstacle_area_threshold

        # Emergency stop if anything is very close
        max_area = max(left_max, center_max, right_max)
        result.closest_distance = max(0.0, 1.0 - (max_area / CONFIG.obstacle_close_threshold))
        result.emergency_stop = max_area >= CONFIG.obstacle_close_threshold

        # Find clearest zone
        zone_scores = {
            "left": left_max,
            "center": center_max,
            "right": right_max,
        }
        result.clearest_zone = min(zone_scores, key=zone_scores.get)

        return result


# =============================================================================
# PATH PLANNER
# =============================================================================

class PathPlanner:
    """
    Reactive path planner using a simple state machine.
    EXPLORING → (obstacle) → AVOIDING → TURNING → EXPLORING
    """

    def __init__(self):
        self.state = AutonomyState.IDLE
        self._turn_start = 0.0
        self._turn_direction = "right"

    def plan(self, collision: CollisionResult) -> Tuple[str, int]:
        """
        Given collision result, return (direction, speed).
        direction: 'F', 'B', 'L', 'R', 'S'
        speed: 0-100
        """
        if collision.emergency_stop:
            self.state = AutonomyState.AVOIDING
            return ('S', 0)

        if self.state == AutonomyState.EXPLORING or self.state == AutonomyState.IDLE:
            if collision.blocked_center:
                # Start turning toward clearest zone
                self.state = AutonomyState.TURNING_LEFT if collision.clearest_zone == "left" else AutonomyState.TURNING_RIGHT
                self._turn_start = time.time()
                self._turn_direction = collision.clearest_zone
            else:
                self.state = AutonomyState.EXPLORING
                return ('F', CONFIG.explore_speed)

        if self.state in (AutonomyState.TURNING_LEFT, AutonomyState.TURNING_RIGHT):
            # Turn for a set duration, then re-evaluate
            if time.time() - self._turn_start > CONFIG.turn_duration:
                self.state = AutonomyState.EXPLORING
                return ('F', CONFIG.explore_speed)

            direction = 'L' if self.state == AutonomyState.TURNING_LEFT else 'R'
            return (direction, CONFIG.explore_speed)

        if self.state == AutonomyState.AVOIDING:
            # Back up briefly then turn
            if not collision.blocked_center and not collision.emergency_stop:
                self.state = AutonomyState.EXPLORING
                return ('F', CONFIG.explore_speed)
            # Turn toward clearest zone
            if collision.clearest_zone == "left":
                return ('L', CONFIG.explore_speed // 2)
            else:
                return ('R', CONFIG.explore_speed // 2)

        return ('S', 0)

    def reset(self):
        self.state = AutonomyState.IDLE
        self._turn_start = 0.0


# =============================================================================
# PERSON IDENTITY (Appearance Matching)
# =============================================================================

class PersonIdentity:
    """
    Stores a color-histogram signature of a target person for re-identification.
    Uses the torso region (between shoulders and hips from pose landmarks)
    to build a robust appearance model.
    """

    def __init__(self):
        self.histogram = None          # HSV color histogram of torso
        self.last_position = None      # (center_x, center_y) normalized
        self.last_bbox_area = 0.0      # for distance estimation
        self.last_seen = 0.0           # timestamp
        self.locked = False            # whether we have a locked target

    def build_from_frame(self, frame: np.ndarray, pose_landmarks, frame_w: int, frame_h: int) -> bool:
        """
        Build identity from pose landmarks — use torso region.
        MediaPipe Pose landmarks: 11=left_shoulder, 12=right_shoulder, 23=left_hip, 24=right_hip
        """
        if pose_landmarks is None:
            return False

        try:
            lm = pose_landmarks.landmark

            # Get torso bounding box from shoulder and hip landmarks
            left_shoulder = lm[11]
            right_shoulder = lm[12]
            left_hip = lm[23]
            right_hip = lm[24]

            # Check visibility
            if any(l.visibility < CONFIG.pose_confidence for l in [left_shoulder, right_shoulder, left_hip, right_hip]):
                return False

            # Pixel coordinates
            x_coords = [left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x]
            y_coords = [left_shoulder.y, right_shoulder.y, left_hip.y, right_hip.y]

            x1 = int(min(x_coords) * frame_w)
            y1 = int(min(y_coords) * frame_h)
            x2 = int(max(x_coords) * frame_w)
            y2 = int(max(y_coords) * frame_h)

            # Clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)

            if x2 - x1 < 20 or y2 - y1 < 20:
                return False

            # Extract torso region and compute HSV histogram
            torso = frame[y1:y2, x1:x2]
            hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            self.histogram = hist

            # Store position (center of full body, not just torso)
            all_x = [l.x for l in lm if l.visibility > CONFIG.pose_confidence]
            all_y = [l.y for l in lm if l.visibility > CONFIG.pose_confidence]

            self.last_position = (
                sum(all_x) / len(all_x) if all_x else 0.5,
                sum(all_y) / len(all_y) if all_y else 0.5,
            )

            # Bbox area for distance estimation
            body_w = max(x_coords) - min(x_coords)
            body_h = max(y_coords) - min(y_coords)
            self.last_bbox_area = body_w * body_h

            self.last_seen = time.time()
            self.locked = True
            return True

        except (IndexError, AttributeError) as e:
            log.debug(f"Failed to build identity: {e}")
            return False

    def match(self, frame: np.ndarray, pose_landmarks, frame_w: int, frame_h: int) -> float:
        """
        Compare a detected person against stored identity.
        Returns correlation score (0.0 to 1.0). Higher = better match.
        """
        if self.histogram is None or pose_landmarks is None:
            return 0.0

        try:
            lm = pose_landmarks.landmark
            left_shoulder = lm[11]
            right_shoulder = lm[12]
            left_hip = lm[23]
            right_hip = lm[24]

            if any(l.visibility < CONFIG.pose_confidence for l in [left_shoulder, right_shoulder, left_hip, right_hip]):
                return 0.0

            x_coords = [left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x]
            y_coords = [left_shoulder.y, right_shoulder.y, left_hip.y, right_hip.y]

            x1 = max(0, int(min(x_coords) * frame_w))
            y1 = max(0, int(min(y_coords) * frame_h))
            x2 = min(frame_w, int(max(x_coords) * frame_w))
            y2 = min(frame_h, int(max(y_coords) * frame_h))

            if x2 - x1 < 20 or y2 - y1 < 20:
                return 0.0

            torso = frame[y1:y2, x1:x2]
            hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

            # Compare using correlation
            score = cv2.compareHist(self.histogram, hist, cv2.HISTCMP_CORREL)
            return max(0.0, score)

        except Exception:
            return 0.0

    def update_position(self, pose_landmarks, frame_w: int, frame_h: int):
        """Update last known position from pose landmarks"""
        if pose_landmarks is None:
            return

        try:
            lm = pose_landmarks.landmark
            visible = [(l.x, l.y) for l in lm if l.visibility > CONFIG.pose_confidence]
            if visible:
                self.last_position = (
                    sum(p[0] for p in visible) / len(visible),
                    sum(p[1] for p in visible) / len(visible),
                )

                x_coords = [lm[11].x, lm[12].x, lm[23].x, lm[24].x]
                y_coords = [lm[11].y, lm[12].y, lm[23].y, lm[24].y]
                self.last_bbox_area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
                self.last_seen = time.time()
        except (IndexError, AttributeError):
            pass

    def is_expired(self) -> bool:
        """Check if we've lost the target for too long"""
        return time.time() - self.last_seen > CONFIG.identity_memory_timeout

    def reset(self):
        """Forget the current target"""
        self.histogram = None
        self.last_position = None
        self.last_bbox_area = 0.0
        self.last_seen = 0.0
        self.locked = False


# =============================================================================
# PERSON FOLLOWER
# =============================================================================

class PersonFollower:
    """
    Follows a specific person using MediaPipe Pose + appearance identity.
    
    Flow:
    1. On start → lock onto nearest person (build PersonIdentity)
    2. Each frame → find matching person among all detected poses
    3. Drive toward matched person, maintaining safe distance
    4. If lost → search using last known position for a few seconds
    """

    def __init__(self):
        self.target = PersonIdentity()
        self.pose_detector = None
        self._init_pose()

    def _init_pose(self):
        """Initialize MediaPipe Pose"""
        if not HAS_MEDIAPIPE:
            log.warning("MediaPipe not available — person following disabled")
            return

        self.pose_detector = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,        # 0=lite for Pi performance
            min_detection_confidence=CONFIG.pose_confidence,
            min_tracking_confidence=CONFIG.pose_confidence,
        )

    def lock_target(self, frame: np.ndarray) -> bool:
        """
        Lock onto the nearest person in the frame.
        Returns True if a target was successfully locked.
        """
        if self.pose_detector is None:
            return False

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(rgb)

        if not results.pose_landmarks:
            log.info("👤 No person detected to lock onto")
            return False

        # Single detection — lock it
        success = self.target.build_from_frame(frame, results.pose_landmarks, w, h)
        if success:
            log.info("🔒 Target person locked — following")
        return success

    def process_frame(self, frame: np.ndarray) -> Tuple[str, int]:
        """
        Process a frame and return motor command (direction, speed).
        Returns ('S', 0) if no target found.
        """
        if self.pose_detector is None:
            return ('S', 0)

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(rgb)

        if not results.pose_landmarks:
            # No person detected
            if self.target.is_expired():
                return ('S', 0)  # Lost target
            else:
                # Use last known position briefly
                return self._drive_toward_last_known()

        # We have a detection — check if it matches our target
        if self.target.locked:
            score = self.target.match(frame, results.pose_landmarks, w, h)

            if score >= CONFIG.identity_match_threshold:
                # Match found — update and follow
                self.target.update_position(results.pose_landmarks, w, h)
                return self._drive_toward_target()
            else:
                # No match — might be wrong person
                if self.target.is_expired():
                    log.info("👤 Target lost — stopping")
                    return ('S', 0)
                return self._drive_toward_last_known()
        else:
            # No target locked yet — lock nearest
            self.lock_target(frame)
            return ('S', 0)

    def _drive_toward_target(self) -> Tuple[str, int]:
        """Generate motor command to drive toward the tracked target"""
        if self.target.last_position is None:
            return ('S', 0)

        cx, _ = self.target.last_position
        area = self.target.last_bbox_area

        # Horizontal steering
        offset = cx - 0.5   # negative = person is left, positive = right

        if abs(offset) > CONFIG.follow_center_deadzone:
            if offset < 0:
                return ('L', CONFIG.follow_speed)
            else:
                return ('R', CONFIG.follow_speed)

        # Distance control
        if area > CONFIG.follow_distance_min:
            # Too close — back up
            return ('B', CONFIG.follow_speed // 2)
        elif area < CONFIG.follow_distance_max:
            # Too far — move closer
            return ('F', CONFIG.follow_speed)
        else:
            # Good distance — stop
            return ('S', 0)

    def _drive_toward_last_known(self) -> Tuple[str, int]:
        """Drive toward last known position (brief memory)"""
        if self.target.last_position is None:
            return ('S', 0)

        cx, _ = self.target.last_position
        offset = cx - 0.5

        if abs(offset) > CONFIG.follow_center_deadzone:
            direction = 'L' if offset < 0 else 'R'
            return (direction, CONFIG.follow_speed // 2)

        # Was centered — move forward slowly
        return ('F', CONFIG.follow_speed // 3)

    def reset(self):
        """Reset the follower — forget target"""
        self.target.reset()

    def close(self):
        """Clean up"""
        if self.pose_detector:
            self.pose_detector.close()


# =============================================================================
# GESTURE NAVIGATOR
# =============================================================================

class GestureNavigator:
    """
    Maps hand gestures from kenza_gesture.py to motor commands.
    
    Gesture mappings:
    - Point Up → Forward
    - Point Down → Backward
    - Point Left → Turn Left
    - Point Right → Turn Right
    - Closed Fist → Emergency Stop
    - Open Palm → Pause/Resume
    """

    def __init__(self):
        self.paused = False
        self.tracker = None
        self._init_tracker()

    def _init_tracker(self):
        """Try to import and initialize gesture tracker"""
        try:
            from kenza_gesture import GestureTracker, GestureType
            self.tracker = GestureTracker()
            self.gesture_types = GestureType
            log.info("✋ Gesture navigation ready")
        except ImportError:
            log.warning("kenza_gesture not available — gesture nav disabled")

    def process_frame(self, frame: np.ndarray) -> Tuple[str, int]:
        """
        Process a frame for gesture commands.
        Returns (direction, speed) or ('S', 0) for no gesture.
        """
        if self.tracker is None:
            return ('S', 0)

        result = self.tracker.process_frame(frame)

        if result is None or result.gesture == self.gesture_types.NONE:
            return ('S', 0)

        gesture = result.gesture

        # Closed Fist → Emergency Stop
        if gesture == self.gesture_types.CLOSED_FIST:
            self.paused = True
            return ('S', 0)

        # Open Palm → Toggle pause
        if gesture == self.gesture_types.OPEN_PALM:
            self.paused = not self.paused
            return ('S', 0)

        if self.paused:
            return ('S', 0)

        # Point directions
        if gesture == self.gesture_types.POINT_UP:
            return ('F', CONFIG.gesture_speed)
        elif gesture == self.gesture_types.POINT_DOWN:
            return ('B', CONFIG.gesture_speed)
        elif gesture == self.gesture_types.POINT_LEFT:
            return ('L', CONFIG.gesture_speed)
        elif gesture == self.gesture_types.POINT_RIGHT:
            return ('R', CONFIG.gesture_speed)

        return ('S', 0)

    def reset(self):
        self.paused = False

    def close(self):
        if self.tracker:
            self.tracker.close()


# =============================================================================
# SENTRY MODE (Motion Detection)
# =============================================================================

class SentryDetector:
    """
    Detects motion using frame differencing.
    Used for security patrol mode.
    """

    def __init__(self):
        self.prev_frame = None
        self.motion_threshold = 5000    # minimum contour area
        self.motion_detected = False

    def process_frame(self, frame: np.ndarray) -> bool:
        """Returns True if significant motion is detected"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return False

        delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(delta, 30, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        self.motion_detected = any(
            cv2.contourArea(c) > self.motion_threshold for c in contours
        )

        self.prev_frame = gray
        return self.motion_detected

    def reset(self):
        self.prev_frame = None
        self.motion_detected = False


# =============================================================================
# AUTONOMY ENGINE (Master Orchestrator)
# =============================================================================

class AutonomyEngine:
    """
    Master orchestrator for all autonomous behavior.
    Runs in its own daemon thread.
    
    Args:
        motor_callback: function(direction: str, speed: int) to control motors
        status_callback: function(status: dict) for real-time status updates
        camera: optional camera capture object (needs .read() method)
    """

    def __init__(
        self,
        motor_callback: Callable[[str, int], None] = None,
        status_callback: Callable[[dict], None] = None,
        camera=None,
    ):
        # Callbacks
        self.motor_callback = motor_callback or (lambda d, s: None)
        self.status_callback = status_callback or (lambda s: None)

        # Camera
        self.camera = camera
        self._own_camera = False

        # Components
        self.detector = ObjectDetector()
        self.collision = CollisionAvoidance()
        self.planner = PathPlanner()
        self.follower = PersonFollower()
        self.gesture_nav = GestureNavigator()
        self.sentry = SentryDetector()

        # State
        self.mode = AutonomyMode.NONE
        self.state = AutonomyState.IDLE
        self.collision_avoidance_enabled = True
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Status tracking
        self._last_direction = 'S'
        self._last_speed = 0
        self._obstacle_count = 0

    def set_camera(self, camera):
        """Set camera source. Must have a .read() method returning (success, frame)."""
        self.camera = camera

    def _ensure_camera(self) -> bool:
        """Ensure camera is available"""
        if self.camera is not None:
            return True

        if not HAS_CV2:
            log.error("OpenCV not available — cannot open camera")
            return False

        try:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.frame_height)
            self._own_camera = True
            if self.camera.isOpened():
                log.info("📷 Camera opened for autonomy")
                return True
            else:
                log.error("Failed to open camera")
                return False
        except Exception as e:
            log.error(f"Camera error: {e}")
            return False

    # ======================== PUBLIC API ========================

    def start_explore(self):
        """Start autonomous exploration mode"""
        with self._lock:
            self.mode = AutonomyMode.EXPLORE
            self.planner.reset()
        self._start_thread()
        log.info("🧭 Autonomous exploration started")

    def start_follow(self):
        """Start person following mode"""
        with self._lock:
            self.mode = AutonomyMode.FOLLOW
            self.follower.reset()
            self.state = AutonomyState.FOLLOWING_SEARCHING
        self._start_thread()
        log.info("🚶 Person following started — looking for target...")

    def start_sentry(self):
        """Start sentry/security patrol mode"""
        with self._lock:
            self.mode = AutonomyMode.SENTRY
            self.sentry.reset()
        self._start_thread()
        log.info("🔒 Sentry mode started")

    def start_gesture_nav(self):
        """Start gesture navigation mode"""
        with self._lock:
            self.mode = AutonomyMode.GESTURE
            self.gesture_nav.reset()
        self._start_thread()
        log.info("✋ Gesture navigation started")

    def stop(self):
        """Stop all autonomous behavior"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        with self._lock:
            self.mode = AutonomyMode.NONE
            self.state = AutonomyState.IDLE

        # Stop motors
        self.motor_callback('S', 0)
        self._broadcast_status()
        log.info("⏹ Autonomy stopped")

    def emergency_stop(self):
        """Immediate motor stop (can be called from any thread)"""
        self.motor_callback('S', 0)
        self._last_direction = 'S'
        self._last_speed = 0
        log.info("🛑 Emergency stop!")

    def set_collision_avoidance(self, enabled: bool):
        """Enable/disable collision avoidance"""
        self.collision.enabled = enabled
        self.collision_avoidance_enabled = enabled
        log.info(f"🛡️ Collision avoidance {'enabled' if enabled else 'disabled'}")

    def get_status(self) -> dict:
        """Get current autonomy status"""
        with self._lock:
            return {
                'mode': self.mode.value,
                'state': self.state.value,
                'collision_avoidance': self.collision_avoidance_enabled,
                'direction': self._last_direction,
                'speed': self._last_speed,
                'obstacle_count': self._obstacle_count,
                'following_locked': self.follower.target.locked,
                'following_lost': self.follower.target.is_expired() if self.follower.target.locked else False,
            }

    # ======================== THREAD MANAGEMENT ========================

    def _start_thread(self):
        """Start the autonomy loop thread (or restart if already running)"""
        if self._running:
            # Mode changed but thread is already running — it will pick up the change
            return

        if not self._ensure_camera():
            log.error("Cannot start autonomy — no camera available")
            return

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="autonomy")
        self._thread.start()

    def _loop(self):
        """Main autonomy loop — runs in daemon thread"""
        frame_interval = 1.0 / CONFIG.detection_fps
        status_interval = 0.5
        last_status_time = 0

        log.info("🔄 Autonomy loop started")

        while self._running:
            loop_start = time.time()

            try:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                # Process based on current mode
                direction, speed = 'S', 0

                with self._lock:
                    current_mode = self.mode

                if current_mode == AutonomyMode.EXPLORE:
                    direction, speed = self._process_explore(frame)
                elif current_mode == AutonomyMode.FOLLOW:
                    direction, speed = self._process_follow(frame)
                elif current_mode == AutonomyMode.SENTRY:
                    direction, speed = self._process_sentry(frame)
                elif current_mode == AutonomyMode.GESTURE:
                    direction, speed = self._process_gesture(frame)
                else:
                    direction, speed = 'S', 0

                # Apply collision avoidance override (except in sentry mode)
                if self.collision_avoidance_enabled and current_mode != AutonomyMode.SENTRY:
                    direction, speed = self._apply_collision_check(frame, direction, speed)

                # Send motor command
                # Only send if mode is active, or if we need to stop (transition to idle)
                if current_mode != AutonomyMode.NONE:
                    self._last_direction = direction
                    self._last_speed = speed
                    self.motor_callback(direction, speed)
                elif self._last_direction != 'S' or self._last_speed != 0:
                    # Send one final stop command when entering idle state
                    self._last_direction = 'S'
                    self._last_speed = 0
                    self.motor_callback('S', 0)
                
                # Broadcast status periodically
                if time.time() - last_status_time > status_interval:
                    self._broadcast_status()
                    last_status_time = time.time()

            except Exception as e:
                log.error(f"Autonomy loop error: {e}")
                self.motor_callback('S', 0)

            # Frame rate control
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Cleanup
        if self._own_camera and self.camera:
            self.camera.release()
            self.camera = None
            self._own_camera = False

        log.info("🔄 Autonomy loop stopped")

    # ======================== MODE PROCESSORS ========================

    def _process_explore(self, frame: np.ndarray) -> Tuple[str, int]:
        """Process frame for exploration mode"""
        detections = self.detector.detect(frame)
        collision = self.collision.analyze(
            detections, CONFIG.frame_width, CONFIG.frame_height
        )
        self._obstacle_count = collision.obstacle_count

        direction, speed = self.planner.plan(collision)

        with self._lock:
            self.state = self.planner.state

        return direction, speed

    def _process_follow(self, frame: np.ndarray) -> Tuple[str, int]:
        """Process frame for person following mode"""
        if not self.follower.target.locked:
            # Try to lock onto someone
            with self._lock:
                self.state = AutonomyState.FOLLOWING_SEARCHING

            if self.follower.lock_target(frame):
                with self._lock:
                    self.state = AutonomyState.FOLLOWING
            return ('S', 0)

        direction, speed = self.follower.process_frame(frame)

        with self._lock:
            if self.follower.target.is_expired():
                self.state = AutonomyState.FOLLOWING_SEARCHING
            else:
                self.state = AutonomyState.FOLLOWING

        return direction, speed

    def _process_sentry(self, frame: np.ndarray) -> Tuple[str, int]:
        """Process frame for sentry mode — detect motion, report alerts"""
        motion = self.sentry.process_frame(frame)

        with self._lock:
            self.state = AutonomyState.SENTRY

        if motion:
            # Broadcast alert
            self.status_callback({
                'type': 'sentry_alert',
                'data': {
                    'motion_detected': True,
                    'timestamp': int(time.time()),
                }
            })

        # Sentry mode doesn't drive — just monitors
        return ('S', 0)

    def _process_gesture(self, frame: np.ndarray) -> Tuple[str, int]:
        """Process frame for gesture navigation"""
        direction, speed = self.gesture_nav.process_frame(frame)

        with self._lock:
            self.state = AutonomyState.GESTURE_NAV

        return direction, speed

    def _apply_collision_check(
        self, frame: np.ndarray, intended_dir: str, intended_speed: int
    ) -> Tuple[str, int]:
        """
        Collision avoidance override — if we're about to drive into something, stop or reroute.
        Only blocks forward/lateral movement; backward is always allowed.
        """
        if intended_dir in ('S', 'B') or intended_speed == 0:
            return intended_dir, intended_speed

        detections = self.detector.detect(frame)
        collision = self.collision.analyze(
            detections, CONFIG.frame_width, CONFIG.frame_height
        )
        self._obstacle_count = collision.obstacle_count

        if collision.emergency_stop:
            log.warning("🛑 Collision avoidance: emergency stop!")
            return ('S', 0)

        if intended_dir == 'F' and collision.blocked_center:
            log.info(f"🛡️ Collision avoidance: center blocked, turning {collision.clearest_zone}")
            if collision.clearest_zone == "left":
                return ('L', intended_speed // 2)
            else:
                return ('R', intended_speed // 2)

        if intended_dir == 'L' and collision.blocked_left:
            if not collision.blocked_center:
                return ('F', intended_speed)
            return ('S', 0)

        if intended_dir == 'R' and collision.blocked_right:
            if not collision.blocked_center:
                return ('F', intended_speed)
            return ('S', 0)

        return intended_dir, intended_speed

    # ======================== STATUS ========================

    def _broadcast_status(self):
        """Send current status via callback"""
        try:
            status = self.get_status()
            self.status_callback({
                'type': 'autonomy_status',
                'data': status,
            })
        except Exception as e:
            log.debug(f"Status broadcast error: {e}")

    def close(self):
        """Full cleanup"""
        self.stop()
        self.follower.close()
        self.gesture_nav.close()

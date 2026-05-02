"""
Gesture Control — Hand and Body Gesture Interface

Real gesture control using MediaPipe for hand tracking when available,
with a working simulated demo mode. Maps detected gestures to AETHERIS
commands: swipe to navigate, pinch to select, fist to stop, etc.

Supports: hand landmark tracking, gesture classification, command mapping,
and real-time feedback. Falls back gracefully to simulation when camera
or MediaPipe is not available.
"""

import time
import math
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import deque


class GestureType(Enum):
    """Recognized gesture types."""
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    PINCH = "pinch"
    SPREAD = "spread"
    ROTATE_CW = "rotate_cw"
    ROTATE_CCW = "rotate_ccw"
    POINT = "point"
    FIST = "fist"
    OPEN_HAND = "open_hand"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    PEACE = "peace"
    OK = "ok"


@dataclass
class GestureEvent:
    """Container for a detected gesture event."""
    gesture: GestureType
    confidence: float
    position: Tuple[float, float, float]
    timestamp: float
    hand_label: str = "unknown"  # "left" or "right"
    metadata: Dict[str, Any] = field(default_factory=dict)


class GestureController:
    """
    Gesture control interface for AETHERIS.

    Features:
    - Real hand tracking via MediaPipe (when camera available)
    - Working simulated mode for testing without camera
    - Gesture recognition with confidence scoring
    - Custom gesture-to-command mapping
    - Real-time interactive loop with visual feedback
    """

    # Default gesture to command mapping
    DEFAULT_GESTURE_MAP = {
        GestureType.SWIPE_LEFT: {"command": "previous", "description": "Previous model/view"},
        GestureType.SWIPE_RIGHT: {"command": "next", "description": "Next model/view"},
        GestureType.SWIPE_UP: {"command": "up", "description": "Increase value"},
        GestureType.SWIPE_DOWN: {"command": "down", "description": "Decrease value"},
        GestureType.PINCH: {"command": "select", "description": "Select/confirm action"},
        GestureType.SPREAD: {"command": "zoom_out", "description": "Zoom out/expand"},
        GestureType.FIST: {"command": "stop", "description": "Stop/cancel operation"},
        GestureType.OPEN_HAND: {"command": "start", "description": "Start operation"},
        GestureType.THUMBS_UP: {"command": "confirm", "description": "Confirm liberation"},
        GestureType.THUMBS_DOWN: {"command": "cancel", "description": "Cancel/undo"},
        GestureType.PEACE: {"command": "peace", "description": "Pause/toggle"},
        GestureType.OK: {"command": "ok", "description": "Accept results"},
        GestureType.POINT: {"command": "point", "description": "Point at target"},
        GestureType.ROTATE_CW: {"command": "rotate_right", "description": "Rotate clockwise"},
        GestureType.ROTATE_CCW: {"command": "rotate_left", "description": "Rotate counter-clockwise"},
    }

    def __init__(self, use_camera: bool = True, camera_id: int = 0):
        """
        Initialize gesture controller.

        Args:
            use_camera: Whether to use camera (False = simulation mode)
            camera_id: Camera device ID for MediaPipe
        """
        self.use_camera = use_camera
        self.camera_id = camera_id
        self._active = False
        self._gesture_map = dict(self.DEFAULT_GESTURE_MAP)
        self._gesture_history: deque = deque(maxlen=200)
        self._last_position: Optional[Tuple[float, float, float]] = None
        self._mediapipe_hands = None
        self._cap = None

    def start_tracking(self) -> Dict[str, Any]:
        """
        Start gesture tracking.

        Initializes MediaPipe if camera is available, otherwise enters
        simulation mode.

        Returns:
            Tracking startup status
        """
        if not self.use_camera:
            self._active = True
            return {
                "success": True,
                "mode": "simulation",
                "message": "Gesture tracking started in simulation mode. Use detect_gesture() to test.",
            }

        try:
            import mediapipe as mp
            import cv2

            mp_hands = mp.solutions.hands
            self._mediapipe_hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )

            self._cap = cv2.VideoCapture(self.camera_id)
            if not self._cap.isOpened():
                self._cleanup_camera()
                self._active = True
                return {
                    "success": True,
                    "mode": "simulation",
                    "message": f"Camera {self.camera_id} not accessible. Falling back to simulation mode.",
                }

            self._active = True
            return {
                "success": True,
                "mode": "camera",
                "backend": "mediapipe",
                "message": "Gesture tracking started with camera. Perform hand gestures.",
            }

        except ImportError:
            self._active = True
            return {
                "success": True,
                "mode": "simulation",
                "message": "MediaPipe/OpenCV not installed. Simulation mode. Install: pip install mediapipe opencv-python",
            }
        except Exception as e:
            self._active = True
            return {
                "success": True,
                "mode": "simulation",
                "message": f"Camera error: {e}. Falling back to simulation mode.",
            }

    def stop_tracking(self) -> Dict[str, Any]:
        """Stop gesture tracking and release resources."""
        self._active = False
        self._cleanup_camera()
        return {"success": True, "message": "Gesture tracking stopped", "gestures_logged": len(self._gesture_history)}

    def _cleanup_camera(self) -> None:
        """Release camera resources."""
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._mediapipe_hands = None

    def detect_gesture(self) -> Optional[GestureEvent]:
        """
        Detect current hand gesture.

        Uses MediaPipe camera feed when available, simulation otherwise.

        Returns:
            GestureEvent if gesture detected, None otherwise
        """
        if not self._active:
            return None

        if self._mediapipe_hands and self._cap:
            return self._detect_from_camera()
        else:
            return self._detect_simulated()

    def _detect_from_camera(self) -> Optional[GestureEvent]:
        """Detect gesture from camera using MediaPipe."""
        import cv2

        ret, frame = self._cap.read()
        if not ret:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mediapipe_hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label.lower()

            # Extract key landmarks
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

            # Classify gesture
            gesture, confidence = self._classify_landmarks(landmarks)

            if gesture and confidence > 0.6:
                # Compute position (wrist/center)
                wrist = landmarks[0]
                position = (wrist[0] * 2 - 1, -(wrist[1] * 2 - 1), wrist[2])

                event = GestureEvent(
                    gesture=gesture,
                    confidence=confidence,
                    position=position,
                    timestamp=time.time(),
                    hand_label=handedness,
                )
                self._gesture_history.append(event)
                self._last_position = position
                return event

        return None

    def _detect_simulated(self) -> Optional[GestureEvent]:
        """Generate simulated gesture for testing."""
        import random

        # 30% chance of a gesture per call
        if random.random() > 0.3:
            return None

        gestures = list(GestureType)
        gesture = random.choice(gestures)

        # Realistic position based on gesture type
        if "left" in gesture.value:
            pos = (-0.5 + random.uniform(-0.3, 0.3), random.uniform(-0.5, 0.5), random.uniform(0, 1))
        elif "right" in gesture.value:
            pos = (0.5 + random.uniform(-0.3, 0.3), random.uniform(-0.5, 0.5), random.uniform(0, 1))
        elif "up" in gesture.value:
            pos = (random.uniform(-0.5, 0.5), 0.5 + random.uniform(0, 0.3), random.uniform(0, 1))
        elif "down" in gesture.value:
            pos = (random.uniform(-0.5, 0.5), -0.5 + random.uniform(-0.3, 0.3), random.uniform(0, 1))
        else:
            pos = (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(0, 1))

        event = GestureEvent(
            gesture=gesture,
            confidence=0.65 + random.random() * 0.3,
            position=pos,
            timestamp=time.time(),
            hand_label=random.choice(["left", "right"]),
        )
        self._gesture_history.append(event)
        self._last_position = pos
        return event

    def _classify_landmarks(self, landmarks: List[Tuple[float, float, float]]) -> Tuple[Optional[GestureType], float]:
        """
        Classify hand landmarks into a gesture type.

        Uses relative finger positions and angles.

        Args:
            landmarks: 21 hand landmarks from MediaPipe

        Returns:
            (GestureType, confidence) tuple
        """
        # Extract key points
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        # Finger MCP joints (base)
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        ring_mcp = landmarks[13]
        pinky_mcp = landmarks[17]

        def is_extended(tip, mcp):
            return tip[1] < mcp[1]  # y is inverted — tip above MCP means extended

        def is_folded(tip, mcp):
            return tip[1] > mcp[1]  # tip below MCP

        index_up = is_extended(index_tip, index_mcp)
        middle_up = is_extended(middle_tip, middle_mcp)
        ring_up = is_extended(ring_tip, ring_mcp)
        pinky_up = is_extended(pinky_tip, pinky_mcp)
        thumb_up = is_extended(thumb_tip, wrist)

        # Classification
        fingers_extended = sum([index_up, middle_up, ring_up, pinky_up])

        if fingers_extended == 0 and not thumb_up:
            return GestureType.FIST, 0.95
        elif fingers_extended == 4 and thumb_up:
            return GestureType.OPEN_HAND, 0.95
        elif index_up and not middle_up and not ring_up and not pinky_up:
            return GestureType.POINT, 0.90
        elif index_up and middle_up and not ring_up and not pinky_up:
            return GestureType.PEACE, 0.90
        elif thumb_up and not index_up:
            # Check thumb direction for thumbs up vs OK
            thumb_angle = abs(thumb_tip[0] - wrist[0])
            if thumb_angle > 0.1:
                return GestureType.THUMBS_UP, 0.85
            else:
                # Thumb up vs OK: check index-thumb distance
                dist = math.sqrt((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2)
                if dist < 0.05:
                    return GestureType.OK, 0.80
                return GestureType.THUMBS_UP, 0.75
        elif not index_up and not middle_up:
            return GestureType.THUMBS_DOWN, 0.70 if not thumb_up else 0.0

        # Pinch detection (thumb-index proximity)
        pinch_dist = math.sqrt((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2)
        if pinch_dist < 0.04:
            return GestureType.PINCH, 0.90
        elif pinch_dist > 0.4 and index_up and pinky_up:
            return GestureType.SPREAD, 0.70

        return None, 0.0

    def get_command(self, gesture: GestureType) -> Optional[str]:
        """
        Get AETHERIS command associated with a gesture.

        Args:
            gesture: Detected gesture type

        Returns:
            Command string or None
        """
        entry = self._gesture_map.get(gesture)
        return entry["command"] if entry else None

    def map_gesture(self, gesture: str, command: str) -> Dict[str, Any]:
        """
        Map a gesture to a custom command.

        Args:
            gesture: Gesture name (e.g., "fist", "peace")
            command: Command string to map to

        Returns:
            Mapping result
        """
        try:
            g = GestureType(gesture)
            self._gesture_map[g] = {"command": command, "description": f"Custom: {command}"}
            return {"success": True, "gesture": gesture, "command": command}
        except ValueError:
            return {
                "success": False,
                "error": f"Unknown gesture: {gesture}",
                "valid_gestures": [g.value for g in GestureType],
            }

    def map_gestures_batch(self, custom_map: Dict[str, str]) -> Dict[str, Any]:
        """Map multiple gestures at once."""
        results = {}
        for gesture, command in custom_map.items():
            results[gesture] = self.map_gesture(gesture, command)
        return {"success": True, "results": results}

    def get_current_mapping(self) -> Dict[str, str]:
        """Get current gesture-to-command mapping."""
        return {g.value: entry["command"] for g, entry in self._gesture_map.items()}

    def get_gesture_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent gesture history."""
        history = list(self._gesture_history)[-limit:]
        return [
            {
                "gesture": e.gesture.value,
                "confidence": round(e.confidence, 2),
                "position": e.position,
                "hand": e.hand_label,
                "timestamp": e.timestamp,
            }
            for e in history
        ]

    def interactive_mode(self, duration: float = 30.0) -> None:
        """
        Run interactive gesture control loop with visual output.

        Args:
            duration: Maximum runtime in seconds (0 = indefinite)
        """
        print("=" * 55)
        print(" AETHERIS Gesture Control Active")
        print(f" Mode: {'Camera (MediaPipe)' if self._mediapipe_hands else 'Simulation'}")
        print(" Perform gestures to see commands")
        print(" Press Ctrl+C to exit")
        print("=" * 55)
        print()
        print(" Gesture Map:")
        for g, entry in self.DEFAULT_GESTURE_MAP.items():
            print(f"   {g.value:15s} -> {entry['command']}")
        print()

        self.start_tracking()
        start_time = time.time()

        try:
            gesture_count = 0
            while self._active:
                if duration > 0 and time.time() - start_time > duration:
                    break

                event = self.detect_gesture()
                if event:
                    command = self.get_command(event.gesture)
                    gesture_count += 1
                    bar = "#" * int(event.confidence * 20)
                    print(f"  [{gesture_count:3d}] {event.gesture.value:15s} "
                          f"| confidence: {event.confidence:.0%} [{bar:<20}] "
                          f"| hand: {event.hand_label:5s} "
                          f"| command: {command or 'none'}")
                time.sleep(0.08)

        except KeyboardInterrupt:
            print("\n\nGesture control interrupted by user.")
        finally:
            self.stop_tracking()
            print(f"\nTracked {gesture_count if 'gesture_count' in dir() else 0} gestures.")

    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """Check which gesture dependencies are available."""
        deps = {"mediapipe": False, "opencv": False, "camera": False}
        try:
            import mediapipe  # noqa: F401
            deps["mediapipe"] = True
        except ImportError:
            pass
        try:
            import cv2  # noqa: F401
            deps["opencv"] = True
            # Test camera
            cap = cv2.VideoCapture(0)
            deps["camera"] = cap.isOpened()
            cap.release()
        except ImportError:
            pass
        return deps

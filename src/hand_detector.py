"""Wrapper around the MediaPipe Tasks HandLandmarker.

Keeps all MediaPipe-specific code in one place and exposes a small, clean
interface to the rest of the application. The pre-trained hand landmark model
is downloaded automatically on first use.
"""

from __future__ import annotations

import os
import urllib.request
from typing import List, Optional, Tuple

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

Point = Tuple[float, float]

# Google-hosted pre-trained hand landmark model.
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task")

# Standard MediaPipe hand skeleton: pairs of landmark indices to connect.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]


def _ensure_model(path: str = _MODEL_PATH) -> str:
    """Return the model path, downloading the file once if it is missing."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("Downloading hand landmark model (first run only)...")
        urllib.request.urlretrieve(_MODEL_URL, path)
    return path


class HandDetector:
    """Detects a single hand and returns its landmarks in pixel coordinates."""

    def __init__(self, max_hands: int = 1, detection_confidence: float = 0.5) -> None:
        # Load the model as a byte buffer rather than passing a file path:
        # MediaPipe's native loader cannot open paths that contain non-ASCII
        # characters (common in Windows user folders), whereas Python can.
        with open(_ensure_model(), "rb") as model_file:
            model_data = model_file.read()

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_data),
            running_mode=RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def find_hand(self, rgb_frame) -> Optional[Tuple[List[Point], str]]:
        """Return ``(landmarks, handedness)`` for the first detected hand.

        ``landmarks`` is a list of 21 ``(x, y)`` pixel coordinates and
        ``handedness`` is ``"Right"`` or ``"Left"``. Returns ``None`` when no
        hand is found.
        """
        height, width = rgb_frame.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None

        hand = result.hand_landmarks[0]
        landmarks = [(lm.x * width, lm.y * height) for lm in hand]

        handedness = "Right"
        if result.handedness:
            handedness = result.handedness[0][0].category_name
        return landmarks, handedness

    def close(self) -> None:
        self._landmarker.close()

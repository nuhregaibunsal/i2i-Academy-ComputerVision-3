"""Geometric logic that decides how many fingers are open.

This module is intentionally independent of MediaPipe and OpenCV: it works on
plain ``(x, y)`` landmark coordinates so it can be unit-tested without a camera.

MediaPipe returns 21 hand landmarks. The indices used here follow the official
hand landmark model:

    Thumb : 1  2  3  4        (tip = 4)
    Index : 5  6  7  8        (tip = 8,  pip = 6)
    Middle: 9  10 11 12       (tip = 12, pip = 10)
    Ring  : 13 14 15 16       (tip = 16, pip = 14)
    Pinky : 17 18 19 20       (tip = 20, pip = 18)

Image coordinates grow downward, so a finger pointing up has its tip *above*
(smaller y than) its middle joint.
"""

from __future__ import annotations

from typing import Sequence, Tuple

Point = Tuple[float, float]

# Landmark indices for the four "vertical" fingers
FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_PIPS = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}

# Thumb landmarks (compared horizontally, not vertically)
THUMB_TIP = 4
THUMB_IP = 3


def _is_finger_open(tip: Point, pip: Point) -> bool:
    """A non-thumb finger is open when its tip is higher than its PIP joint."""
    return tip[1] < pip[1]


def _is_thumb_open(tip: Point, ip: Point, handedness: str) -> bool:
    """The thumb opens sideways, so compare x-coordinates.

    The direction depends on which hand is detected: for a right hand the thumb
    tip sits to the right of its joint, and vice versa for a left hand.
    """
    if handedness.lower().startswith("r"):
        return tip[0] > ip[0]
    return tip[0] < ip[0]


def count_fingers(landmarks: Sequence[Point], handedness: str = "Right") -> int:
    """Count how many fingers are open.

    Args:
        landmarks: 21 ``(x, y)`` points for a single hand.
        handedness: ``"Right"`` or ``"Left"`` as reported by MediaPipe.

    Returns:
        The number of open fingers, between 0 and 5.
    """
    if len(landmarks) != 21:
        raise ValueError(f"expected 21 landmarks, got {len(landmarks)}")

    open_count = 0
    for name, tip_idx in FINGER_TIPS.items():
        if _is_finger_open(landmarks[tip_idx], landmarks[FINGER_PIPS[name]]):
            open_count += 1

    if _is_thumb_open(landmarks[THUMB_TIP], landmarks[THUMB_IP], handedness):
        open_count += 1

    return open_count

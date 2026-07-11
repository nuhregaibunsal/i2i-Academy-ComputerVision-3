"""Unit tests for the finger-counting logic (no camera / MediaPipe needed)."""

import pytest

from src.finger_counter import count_fingers


def _base_hand():
    """A neutral right hand with all four fingers folded and thumb tucked in.

    Landmarks are ``(x, y)`` with y growing downward. For folded fingers the
    tip sits *below* (larger y than) the PIP joint.
    """
    hand = [(0.5, 0.9)] * 21  # start every point low on the image
    # PIP joints of the four fingers, placed high (small y)
    for pip in (6, 10, 14, 18):
        hand[pip] = (0.5, 0.4)
    # Their tips folded down (larger y than the PIP) -> closed
    for tip in (8, 12, 16, 20):
        hand[tip] = (0.5, 0.5)
    # Thumb tucked to the left of its IP joint -> closed for a right hand
    hand[3] = (0.5, 0.6)
    hand[4] = (0.4, 0.6)
    return hand


def test_closed_fist_counts_zero():
    assert count_fingers(_base_hand(), "Right") == 0


def test_single_finger_open():
    hand = _base_hand()
    hand[8] = (0.5, 0.2)  # index tip well above its PIP -> open
    assert count_fingers(hand, "Right") == 1


def test_all_fingers_open():
    hand = _base_hand()
    for tip in (8, 12, 16, 20):
        hand[tip] = (0.5, 0.2)   # four fingers up
    hand[4] = (0.7, 0.6)          # thumb out to the right -> open
    assert count_fingers(hand, "Right") == 5


def test_left_hand_thumb_direction():
    hand = _base_hand()
    hand[4] = (0.3, 0.6)  # thumb to the left -> open for a LEFT hand
    assert count_fingers(hand, "Left") == 1
    assert count_fingers(hand, "Right") == 0


def test_invalid_landmark_count_raises():
    with pytest.raises(ValueError):
        count_fingers([(0.0, 0.0)] * 5)

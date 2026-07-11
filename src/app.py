"""Real-time webcam loop: detect a hand, count fingers, overlay the result."""

from __future__ import annotations

import cv2

from .finger_counter import count_fingers
from .hand_detector import HAND_CONNECTIONS, HandDetector


def _draw_landmarks(frame, landmarks) -> None:
    """Draw the hand skeleton (bones + joints) onto a BGR frame."""
    for start, end in HAND_CONNECTIONS:
        p1 = (int(landmarks[start][0]), int(landmarks[start][1]))
        p2 = (int(landmarks[end][0]), int(landmarks[end][1]))
        cv2.line(frame, p1, p2, (255, 255, 255), 2)
    for x, y in landmarks:
        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), cv2.FILLED)


def _draw_count(frame, count: int) -> None:
    """Draw the open-finger count as large text inside a banner."""
    cv2.rectangle(frame, (0, 0), (170, 90), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, "fingers", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, str(count), (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)


def run(camera_index: int = 0) -> None:
    """Open the webcam and run the finger-counting loop until 'q' is pressed."""
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError("Could not open the webcam")

    detector = HandDetector(max_hands=1)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            # Mirror the frame so it behaves like a mirror for the user.
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            detection = detector.find_hand(rgb)
            count = 0
            if detection is not None:
                landmarks, handedness = detection
                count = count_fingers(landmarks, handedness)
                _draw_landmarks(frame, landmarks)

            _draw_count(frame, count)
            cv2.imshow("Finger Counter", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        detector.close()
        cv2.destroyAllWindows()

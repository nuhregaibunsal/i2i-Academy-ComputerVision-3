# Real-Time Finger Counter

A real-time computer vision application that detects a human hand from the
webcam feed and counts how many fingers are held up, drawing the live count
on the video window.

The project uses [MediaPipe Hands](https://developers.google.com/mediapipe)
for hand landmark detection and [OpenCV](https://opencv.org/) for capturing
and displaying the video stream.

## How it works

1. OpenCV captures the webcam feed frame by frame.
2. MediaPipe detects the hand and returns 21 landmark points (finger joints).
3. For each finger a simple geometric rule decides whether it is open or closed.
4. The number of open fingers is drawn on the frame as live text.

## Project structure

```
.
├── main.py                     # entry point
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── hand_detector.py        # MediaPipe Hands wrapper
│   ├── finger_counter.py       # open/closed finger logic
│   └── app.py                  # webcam loop + overlay
└── tests/
    └── test_finger_counter.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the application (press `q` to quit):

```bash
python main.py
```

## Running tests

The finger-counting logic is decoupled from the camera and MediaPipe, so the
unit tests run without any hardware:

```bash
pytest -q
```

## License

MIT

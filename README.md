# Virtual Piano with Computer Vision 🎹

Play piano using nothing but your hands and a webcam. Computer vision tracks your fingers above any desk — tap downward to trigger notes in real-time.

[▶️ Demo](https://www.youtube.com/shorts/ZdxvdxJ5CQM)

---

## How It Works

| Step | Description |
|------|-------------|
| **1. Desk Detection** | Runs edge detection (Hough Line Transform) to find the desk boundary as your virtual keyboard surface |
| **2. Hand Tracking** | Mediapipe tracks all 10 fingertips with landmarks |
| **3. Velocity Press** | Detects downward finger speed — if fast enough, a "key press" fires |
| **4. Note Mapping** | Each finger is mapped to a note (C4–E5 across both hands) |
| **5. Audio Output** | Plays the corresponding `.mp3` via Pygame mixer |

**Visual feedback**: fingertip circles (green = pressing, red = idle), note labels, desk edge overlay.

---

## Note Mapping

| Left Hand | Right Hand |
|-----------|------------|
| C4 | A4 |
| D4 | B4 |
| E4 | C5 |
| F4 | D5 |
| G4 | E5 |

---

## Setup

### Prerequisites
- Python 3.8+
- Webcam (built-in or external)

### Install

```bash
git clone https://github.com/0xindra/Virtual_Piano_With_Computer_Vision.git
cd Virtual_Piano_With_Computer_Vision
pip install -r requirements.txt
```

### Run

```bash
# 1. Calibrate desk edge (one-time per setup)
python src/surface_calibration.py

# 2. Play!
python src/Hybrid_test_2.py
```

> Calibration data is saved to `desk_edge_calibration.json` and `surface_calibration.json`.

---

## Project Structure

```
├── src/
│   ├── Hybrid_test_2.py          # ✅ Best working version — run this
│   ├── Hybrid_test.py            # Earlier iteration
│   ├── main.py                   # Initial prototype
│   ├── hand_tracking.py          # Mediapipe hand/fingertip tracking
│   ├── desk_edge_detection.py    # Desk edge detection logic
│   ├── interaction_detection.py  # Finger press detection
│   ├── surface_calibration.py    # Calibration routine
│   ├── play_notes.py             # Pygame audio playback
│   ├── song_model.py             # Song data model
│   ├── touch_state_machine.py    # Press/release state tracking
│   └── screen_segmentation.py    # Visual region mapping
├── resources/sounds/             # Note MP3 files (C4–E5)
├── tests/                        # Unit tests
├── requirements.txt
└── README.md
```

---

## Features

- **Multi-hand** — both hands, all 10 fingers simultaneously
- **Velocity-based** — natural pressing feel, no hardware needed
- **Auto-calibration** — adapts to any desk height and camera angle
- **Real-time UI** — see note names and press states on video feed
- **Chords supported** — multiple fingers can press at the same time

---

## Future Ideas

- Chord-aware detection
- Custom note mapping
- AR keyboard overlay on desk
- Gesture controls for octave switching

---

Built with [Mediapipe](https://github.com/google/mediapipe) + [OpenCV](https://opencv.org/) + [Pygame](https://www.pygame.org/).

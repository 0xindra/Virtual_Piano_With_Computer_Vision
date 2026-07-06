import cv2
import mediapipe as mp
import numpy as np
from play_notes import SoundPlayer
from touch_state_machine import TouchStateMachine
from desk_edge_detection import find_horizontal_edge_y

# Notes assigned to fingertips for left and right hands
LEFT_HAND_NOTES = ['C4', 'D4', 'E4', 'F4', 'G4']
RIGHT_HAND_NOTES = ['A4', 'B4', 'C5', 'D5', 'E5']
ALL_NOTES = LEFT_HAND_NOTES + RIGHT_HAND_NOTES  # 10 notes, indices 0-9

# Chord mapping: number of raised fingers -> list of note indices to play
# Using available notes: C4(0), D4(1), E4(2), F4(3), G4(4), A4(5), B4(6), C5(7), D5(8), E5(9)
CHORD_MAP = {
    0: [],                # Silence
    1: [0, 2, 4],         # C major (C4 E4 G4)
    2: [1, 3, 5],         # D minor (D4 F4 A4)
    3: [2, 4, 6],         # E minor (E4 G4 B4)
    4: [3, 5, 7],         # F major (F4 A4 C5)
    5: [4, 6, 8],         # G major (G4 B4 D5)
    6: [5, 7, 9],         # A minor (A4 C5 E5)
    7: [6, 8, 3],         # B diminished (B4 D5 F4)
    8: [0, 2, 4, 6],      # Cmaj7 (C4 E4 G4 B4)
    9: [4, 6, 8, 3],      # G7 (G4 B4 D5 F4)
    10: [0, 2, 4, 5, 7, 9]  # C major pentatonic spread
}

CHORD_NAMES = {
    0: "Silence", 1: "C Maj", 2: "D min", 3: "E min", 4: "F Maj",
    5: "G Maj", 6: "A min", 7: "B dim", 8: "Cmaj7", 9: "G7", 10: "C pentatonic"
}

# Webcam settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Mediapipe hand tracking with support for two hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.9)

# For drawing hand skeleton
mp_drawing = mp.solutions.drawing_utils

# Sound player
player = SoundPlayer(ALL_NOTES)

# Tracking state
touch_states_left = [TouchStateMachine() for _ in range(5)]
touch_states_right = [TouchStateMachine() for _ in range(5)]
previous_positions_left = [None] * 5
previous_positions_right = [None] * 5
desk_edge_y = None

# Gesture mode state
gesture_mode = False
last_chord_count = -1  # Track last played chord to avoid replaying same chord
chord_display = ""     # Current chord name to display
finger_count_text = "" # Finger count text

# Exponential smoothing function
def smooth_position(current, previous, alpha=0.3):
    if previous is None:
        return current
    return tuple(alpha * c + (1 - alpha) * p for c, p in zip(current, previous))

# Detect desk edge using Hough Line Transform
def detect_desk_edge(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    return find_horizontal_edge_y(lines)

def count_raised_fingers(hand_landmarks, hand_label, frame_shape):
    """
    Count how many fingers are raised for one hand.
    Standard Mediapipe counting:
    - Thumb: compare tip.x to IP.x (varies by hand)
    - Other fingers: tip.y < PIP.y means finger is raised
    """
    h, w, _ = frame_shape
    lm = hand_landmarks.landmark
    count = 0
    is_left = (hand_label == "Left")

    # Thumb: compare tip (4) x to IP (3) x
    # Right hand: tip.x > IP.x = raised; Left hand: tip.x < IP.x = raised
    thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = lm[mp_hands.HandLandmark.THUMB_IP]
    if is_left:
        if thumb_tip.x < thumb_ip.x:
            count += 1
    else:
        if thumb_tip.x > thumb_ip.x:
            count += 1

    # Other 4 fingers: tip.y < pip.y = raised (pointing up)
    finger_tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]
    finger_pips = [
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP,
    ]

    for tip_idx, pip_idx in zip(finger_tips, finger_pips):
        if lm[tip_idx].y < lm[pip_idx].y:
            count += 1

    return count


def play_chord_by_finger_count(total_fingers):
    """Play a chord based on total raised finger count."""
    if total_fingers in CHORD_MAP and len(CHORD_MAP[total_fingers]) > 0:
        player.play_chord_by_indices(CHORD_MAP[total_fingers])


# Main loop
mode_display = ""
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the video feed
    frame = cv2.flip(frame, 1)

    if gesture_mode:
        # ====== GESTURE MODE: count raised fingers -> play chords ======
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        total_fingers = 0
        hand_labels = []

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = results.multi_handedness[hand_idx].classification[0].label
                hand_labels.append(hand_label)

                # Draw hand skeleton
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Count raised fingers for this hand
                fingers_up = count_raised_fingers(hand_landmarks, hand_label, frame.shape)
                total_fingers += fingers_up

                # Draw finger count near each hand
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                h, w, _ = frame.shape
                wx, wy = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(frame, f"{fingers_up} up", (wx, wy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Clamp to 0-10
        total_fingers = min(total_fingers, 10)

        # Play chord when count changes (debounce)
        if total_fingers != last_chord_count:
            last_chord_count = total_fingers
            play_chord_by_finger_count(total_fingers)

        # Display info
        finger_count_text = f"Fingers: {total_fingers}/10"
        chord_name = CHORD_NAMES.get(total_fingers, "")
        chord_display = f"Chord: {chord_name}" if chord_name else "No chord"

        # Draw mode indicator
        mode_display = "MODE: CHORDS (raised fingers)"
        cv2.rectangle(frame, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.putText(frame, mode_display, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, finger_count_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, chord_display, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("Virtual Piano", frame)
        cv2.putText(frame, "Press m for Touch Mode", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    else:
        # ====== TOUCH MODE: existing keyboard detection ======
        # Desk edge detection
        if desk_edge_y is None:
            desk_edge_y = detect_desk_edge(frame)
        if desk_edge_y is None:
            cv2.putText(frame, "Desk Edge Not Detected. Please ensure the desk is visible.", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Virtual Piano", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Hand tracking
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = results.multi_handedness[hand_idx].classification[0].label
                hand_type = "LEFT" if hand_label == "Left" else "RIGHT"
                hand_notes = LEFT_HAND_NOTES if hand_type == "LEFT" else RIGHT_HAND_NOTES
                touch_states = touch_states_left if hand_type == "LEFT" else touch_states_right
                previous_positions = previous_positions_left if hand_type == "LEFT" else previous_positions_right

                h, w, _ = frame.shape
                fingertip_indices = [
                    mp_hands.HandLandmark.THUMB_TIP,
                    mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp_hands.HandLandmark.RING_FINGER_TIP,
                    mp_hands.HandLandmark.PINKY_TIP,
                ]
                fingertips = []
                for idx, landmark_idx in enumerate(fingertip_indices):
                    landmark = hand_landmarks.landmark[landmark_idx]
                    fingertips.append((int(landmark.x * w), int(landmark.y * h)))

                for i, pos in enumerate(fingertips):
                    smoothed_pos = smooth_position(pos, previous_positions[i])
                    previous_positions[i] = smoothed_pos
                    event = touch_states[i].update(smoothed_pos[1], desk_edge_y)

                    if event == "note_on":
                        note_idx = LEFT_HAND_NOTES.index(hand_notes[i]) if hand_type == "LEFT" else 5 + RIGHT_HAND_NOTES.index(hand_notes[i])
                        player.play_note_by_index(note_idx)

                    color = (0, 255, 0) if touch_states[i].is_pressed else (0, 0, 255)
                    cv2.circle(frame, (int(smoothed_pos[0]), int(smoothed_pos[1])), 10, color, -1)
                    cv2.putText(frame, hand_notes[i], (int(smoothed_pos[0]) + 10, int(smoothed_pos[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw desk edge
        cv2.line(frame, (0, desk_edge_y), (frame.shape[1], desk_edge_y), (255, 255, 0), 2)

        # Display mode indicator
        mode_display = "MODE: TOUCH (desk press)"
        cv2.putText(frame, mode_display, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.rectangle(frame, (10, 10), (320, 50), (0, 0, 0), -1)
        cv2.imshow("Virtual Piano", frame)
        cv2.putText(frame, "Press m for Chord Mode", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Key handlers
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        gesture_mode = not gesture_mode
        last_chord_count = -1  # Reset chord tracking on mode switch
        desk_edge_y = None if gesture_mode else None  # Re-detect desk edge when switching back
        # Reset touch states when switching modes
        touch_states_left = [TouchStateMachine() for _ in range(5)]
        touch_states_right = [TouchStateMachine() for _ in range(5)]

cap.release()
cv2.destroyAllWindows()

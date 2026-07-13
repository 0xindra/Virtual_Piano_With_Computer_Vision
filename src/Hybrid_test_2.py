import cv2
import mediapipe as mp
import numpy as np
import pygame
from play_notes import SoundPlayer
from touch_state_machine import TouchStateMachine
from desk_edge_detection import find_horizontal_edge_y

LEFT_HAND_NOTES = ['C4', 'D4', 'E4', 'F4', 'G4']
RIGHT_HAND_NOTES = ['A4', 'B4', 'C5', 'D5', 'E5']
ALL_NOTES = LEFT_HAND_NOTES + RIGHT_HAND_NOTES
NOTE_INDEX_MAP = {note: i for i, note in enumerate(ALL_NOTES)}

CHORD_MAP = {
    0: [], 1: [0, 2, 4], 2: [1, 3, 5], 3: [2, 4, 6],
    4: [3, 5, 7], 5: [4, 6, 8], 6: [5, 7, 9],
    7: [6, 8, 3], 8: [0, 2, 4, 6], 9: [4, 6, 8, 3],
    10: [0, 2, 4, 5, 7, 9],
}
CHORD_NAMES = {
    0: "Silence", 1: "C Maj", 2: "D min", 3: "E min", 4: "F Maj",
    5: "G Maj", 6: "A min", 7: "B dim", 8: "Cmaj7", 9: "G7", 10: "C pentatonic"
}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                        min_detection_confidence=0.6, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pygame.mixer.set_num_channels(32)
player = SoundPlayer(ALL_NOTES)

touch_states_left = [TouchStateMachine() for _ in range(5)]
touch_states_right = [TouchStateMachine() for _ in range(5)]
previous_positions_left = [None] * 5
previous_positions_right = [None] * 5
desk_edge_y = None

gesture_mode = False
last_played_chord = -1
chord_display = ""
finger_count_text = ""
chord_cooldown = 0
CHORD_COOLDOWN_MAX = 8

FINGER_LABELS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
FINGERTIP_INDICES = [
    mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP,
]


def smooth_position(current, previous, alpha=0.3):
    if previous is None:
        return current
    return tuple(alpha * c + (1 - alpha) * p for c, p in zip(current, previous))


def detect_desk_edge(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    return find_horizontal_edge_y(lines)


def count_raised_fingers(hand_landmarks, hand_label):
    lm = hand_landmarks.landmark
    is_left = (hand_label == "Left")
    raised = []
    thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = lm[mp_hands.HandLandmark.THUMB_IP]
    if (is_left and thumb_tip.x < thumb_ip.x) or (not is_left and thumb_tip.x > thumb_ip.x):
        raised.append(0)
    finger_parts = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP),
    ]
    for finger_idx, (tip, pip) in enumerate(finger_parts, start=1):
        if lm[tip].y < lm[pip].y:
            raised.append(finger_idx)
    return len(raised), raised


def draw_raised_finger_labels(frame, hand_landmarks, raised_indices):
    h, w, _ = frame.shape
    lm = hand_landmarks.landmark
    for fi in range(5):
        landmark_idx = FINGERTIP_INDICES[fi]
        fx = int(lm[landmark_idx].x * w)
        fy = int(lm[landmark_idx].y * h)
        is_up = fi in raised_indices
        color = (0, 255, 0) if is_up else (80, 80, 80)
        cv2.circle(frame, (fx, fy), 7, color, -1)
        cv2.putText(frame, FINGER_LABELS[fi][0], (fx + 10, fy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def play_chord(total_fingers):
    if total_fingers in CHORD_MAP and len(CHORD_MAP[total_fingers]) > 0:
        player.play_chord_by_indices(CHORD_MAP[total_fingers])
        return True
    return False


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    if gesture_mode:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        total_fingers = 0
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = results.multi_handedness[hand_idx].classification[0].label
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                fingers_up, raised_idx = count_raised_fingers(hand_landmarks, hand_label)
                total_fingers += fingers_up
                draw_raised_finger_labels(frame, hand_landmarks, raised_idx)
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                h_f, w_f, _ = frame.shape
                wx, wy = int(wrist.x * w_f), int(wrist.y * h_f)
                cv2.putText(frame, f"{hand_label}: {fingers_up}", (wx, wy - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        total_fingers = min(max(total_fingers, 0), 10)
        if chord_cooldown > 0:
            chord_cooldown -= 1
        if total_fingers != last_played_chord and chord_cooldown == 0:
            last_played_chord = total_fingers
            played = play_chord(total_fingers)
            chord_cooldown = CHORD_COOLDOWN_MAX
            if played:
                cn = CHORD_NAMES.get(total_fingers, "?")
                notes_str = " + ".join(ALL_NOTES[i] for i in CHORD_MAP.get(total_fingers, []))
                print(f"Chord: {cn} [{notes_str}]")
        finger_count_text = f"Fingers: {total_fingers}/10"
        chord_name = CHORD_NAMES.get(total_fingers, "")
        chord_display = f"Chord: {chord_name}" if chord_name else "Wait for fingers..."
        chord_notes_line = ""
        if total_fingers in CHORD_MAP and len(CHORD_MAP[total_fingers]) > 0:
            chord_notes_line = chr(9829) + " " + " ".join(ALL_NOTES[i] for i in CHORD_MAP[total_fingers])
        panel_h = 160 if chord_notes_line else 130
        cv2.rectangle(frame, (10, 10), (420, panel_h), (0, 0, 0), -1)
        cv2.putText(frame, "MODE: CHORDS (raised fingers)", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, finger_count_text, (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, chord_display, (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        if chord_notes_line:
            cv2.putText(frame, chord_notes_line, (20, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
        cv2.putText(frame, "Press M for Touch Mode", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    else:
        if desk_edge_y is None:
            desk_edge_y = detect_desk_edge(frame)
        if desk_edge_y is None:
            cv2.putText(frame, "Desk Edge Not Detected.", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Virtual Piano", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = results.multi_handedness[hand_idx].classification[0].label
                is_left_hand = (hand_label == "Left")
                hand_notes = LEFT_HAND_NOTES if is_left_hand else RIGHT_HAND_NOTES
                touch_states = touch_states_left if is_left_hand else touch_states_right
                prev_positions = previous_positions_left if is_left_hand else previous_positions_right
                h, w, _ = frame.shape
                fingertips = []
                for landmark_idx in FINGERTIP_INDICES:
                    landmark = hand_landmarks.landmark[landmark_idx]
                    fingertips.append((int(landmark.x * w), int(landmark.y * h)))
                for i, pos in enumerate(fingertips):
                    smoothed = smooth_position(pos, prev_positions[i])
                    prev_positions[i] = smoothed
                    event = touch_states[i].update(smoothed[1], desk_edge_y)
                    if event == "note_on":
                        note_name = hand_notes[i]
                        note_idx = NOTE_INDEX_MAP[note_name]
                        player.play_note_by_index(note_idx)
                    color = (0, 255, 0) if touch_states[i].is_pressed else (0, 0, 255)
                    cv2.circle(frame, (int(smoothed[0]), int(smoothed[1])), 10, color, -1)
                    cv2.putText(frame, hand_notes[i], (int(smoothed[0]) + 12, int(smoothed[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.line(frame, (0, desk_edge_y), (frame.shape[1], desk_edge_y), (255, 255, 0), 2)
        cv2.rectangle(frame, (10, 10), (320, 50), (0, 0, 0), -1)
        cv2.putText(frame, "MODE: TOUCH (desk press)", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press M for Chord Mode", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Virtual Piano", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        gesture_mode = not gesture_mode
        last_played_chord = -1
        chord_cooldown = 0
        touch_states_left = [TouchStateMachine() for _ in range(5)]
        touch_states_right = [TouchStateMachine() for _ in range(5)]
        if not gesture_mode:
            desk_edge_y = None
        print(f"Switched to {'GESTURE' if gesture_mode else 'TOUCH'} mode")

cap.release()
cv2.destroyAllWindows()

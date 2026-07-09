import cv2
from hand_tracking import HandTracker
from play_notes import SoundPlayer

# Notes assigned to fingertips
LEFT_HAND_NOTES = ['C4', 'D4', 'E4', 'F4', 'G4']
RIGHT_HAND_NOTES = ['A4', 'B4', 'C5', 'D5', 'E5']
NOTES = LEFT_HAND_NOTES + RIGHT_HAND_NOTES

# Virtual desk edge = top of piano keyboard overlay
edge_y = None  # set dynamically each frame

# Webcam settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Hand tracker and sound player
hand_tracker = HandTracker(max_hands=2, model_complexity=1)
player = SoundPlayer(NOTES)

# Press detection settings
PRESS_THRESHOLD = 5
finger_pressed = {i: False for i in range(10)}

# ─── Piano Keyboard Overlay ───────────────────────────────────

# Keyboard layout: 10 white keys, note names below
# Left hand (fingers 0-4):  C4  D4  E4  F4  G4
# Right hand (fingers 5-9): A4  B4  C5  D5  E5
#
# On a real piano these would be:
# C4 - D4 - E4 - F4 - G4 - A4 - B4 - C5 - D5 - E5
# So the black keys would appear between:
#  C4-D4, D4-E4, F4-G4, G4-A4, A4-B4

# Positions of black keys relative to white keys (index offset + height)
# Black key sits between white key [i] and [i+1]
# Given our 10-key layout, black keys would be at positions:
#   C#4 between C4(idx0) & D4(idx1) → at white key 0
#   D#4 between D4(idx1) & E4(idx2) → at white key 1
#   F#4 between F4(idx3) & G4(idx4) → at white key 3
#   G#4 between G4(idx4) & A4(idx5) → at white key 4
#   A#4 between A4(idx5) & B4(idx6) → at white key 5
#   C#5 between C5(idx7) & D5(idx8) → at white key 7
#   D#5 between D5(idx8) & E5(idx9) → at white key 8
BLACK_KEY_POSITIONS = [0, 1, 3, 4, 5, 7, 8]  # white key indices where a black key sits to the right

KEYBOARD_HEIGHT = 110
TOTAL_WHITE_KEYS = 10

def draw_piano_overlay(frame, pressed_flags):
    h, w = frame.shape[:2]
    key_w = w // TOTAL_WHITE_KEYS
    key_h = KEYBOARD_HEIGHT
    y_start = h - key_h

    # ── 1. White keys ──
    for i in range(TOTAL_WHITE_KEYS):
        x = i * key_w
        is_pressed = pressed_flags.get(i, False)

        # Color: white (normal) or green glow (pressed)
        if is_pressed:
            color_bg = (100, 255, 100)   # bright green
            border_color = (0, 200, 0)
        else:
            color_bg = (230, 230, 230)    # off-white
            border_color = (180, 180, 180)

        # Fill key
        cv2.rectangle(frame, (x, y_start), (x + key_w - 1, h - 1), color_bg, -1)
        cv2.rectangle(frame, (x, y_start), (x + key_w - 1, h - 1), border_color, 1)

        # Note name
        note = NOTES[i]
        text_size = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        tx = x + (key_w - text_size[0]) // 2
        ty = h - 15
        text_color = (0, 180, 0) if is_pressed else (50, 50, 50)
        cv2.putText(frame, note, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    # ── 2. Black keys (visual only — highlight if adjacent white key is pressed) ──
    black_key_w = int(key_w * 0.6)
    black_key_h = int(key_h * 0.6)

    for wi in BLACK_KEY_POSITIONS:
        if wi + 1 >= TOTAL_WHITE_KEYS:
            continue
        # Position the black key at the boundary between wi and wi+1
        bx = (wi + 1) * key_w - black_key_w // 2
        by = y_start

        # Check if either adjacent white key is pressed
        is_pressed = pressed_flags.get(wi, False) or pressed_flags.get(wi + 1, False)
        color = (40, 180, 40) if is_pressed else (30, 30, 30)

        cv2.rectangle(frame, (bx, by), (bx + black_key_w, by + black_key_h), color, -1)
        cv2.rectangle(frame, (bx, by), (bx + black_key_w, by + black_key_h), (10, 10, 10), 1)

    # ── 3. Label section ──
    # Current instrument label
    instr_name = player.current_instrument.upper()
    cv2.putText(frame, f"[{instr_name}]", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.putText(frame, "Virtual Piano - q:quit i:instrument", (10, y_start - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


# ─── Main Loop ─────────────────────────────────────────────────

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the video feed
    frame = cv2.flip(frame, 1)

    # Hand tracking
    hands_data = hand_tracker.get_all_fingertips(frame)
    if hands_data:
        hands_data.sort(key=lambda h: 0 if h['hand'] == 'left' else 1)

    # Extract fingertip positions
    fingertips_ordered = []
    for hand_info in hands_data:
        fingertips_2d = [p[:2] for p in hand_info['fingertips_3d']]
        fingertips_ordered.extend(fingertips_2d)

    # Ensure exactly 10 fingers
    while len(fingertips_ordered) < 10:
        fingertips_ordered.append(None)
    fingertips_ordered = fingertips_ordered[:10]

    # Calculate virtual desk edge = top of keyboard overlay
    h = frame.shape[0]
    edge_y = h // 2 # fixed at center

    # Press detection
    for i, pos in enumerate(fingertips_ordered):
        if pos is None:
            finger_pressed[i] = False
            continue

        distance_to_edge = abs(pos[1] - edge_y)
        if distance_to_edge <= PRESS_THRESHOLD and not finger_pressed[i]:
            finger_pressed[i] = True
            player.play_note_by_index(i)
        elif distance_to_edge > PRESS_THRESHOLD:
            finger_pressed[i] = False

    # Draw piano keyboard overlay
    draw_piano_overlay(frame, finger_pressed)

    # Visualization — fingertip dots
    for i, pos in enumerate(fingertips_ordered):
        if pos is not None:
            color = (0, 255, 0) if finger_pressed[i] else (0, 0, 255)
            cv2.circle(frame, (int(pos[0]), int(pos[1])), 10, color, -1)

    # Keyboard boundary line
    cv2.line(frame, (0, edge_y), (frame.shape[1], edge_y), (255, 255, 0), 2)

    cv2.imshow("Virtual Piano", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('i'):
        instr = player.switch_instrument()
        print(f"🔊 Switched to: {instr}")

cap.release()
cv2.destroyAllWindows()

import cv2
from hand_tracking import HandTracker
from play_notes import SoundPlayer, INSTRUMENT_NAMES

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

# Instrument menu popup state
show_instrument_menu = False
menu_timer = 0
MENU_DISPLAY_DURATION = 90  # ~3 seconds at 30fps

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
BLACK_KEY_POSITIONS = [0, 1, 3, 4, 5, 7, 8]

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

        if is_pressed:
            color_bg = (100, 255, 100)
            border_color = (0, 200, 0)
        else:
            color_bg = (230, 230, 230)
            border_color = (180, 180, 180)

        cv2.rectangle(frame, (x, y_start), (x + key_w - 1, h - 1), color_bg, -1)
        cv2.rectangle(frame, (x, y_start), (x + key_w - 1, h - 1), border_color, 1)

        note = NOTES[i]
        text_size = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        tx = x + (key_w - text_size[0]) // 2
        ty = h - 15
        text_color = (0, 180, 0) if is_pressed else (50, 50, 50)
        cv2.putText(frame, note, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    # ── 2. Black keys ──
    black_key_w = int(key_w * 0.6)
    black_key_h = int(key_h * 0.6)

    for wi in BLACK_KEY_POSITIONS:
        if wi + 1 >= TOTAL_WHITE_KEYS:
            continue
        bx = (wi + 1) * key_w - black_key_w // 2
        by = y_start
        is_pressed = pressed_flags.get(wi, False) or pressed_flags.get(wi + 1, False)
        color = (40, 180, 40) if is_pressed else (30, 30, 30)
        cv2.rectangle(frame, (bx, by), (bx + black_key_w, by + black_key_h), color, -1)
        cv2.rectangle(frame, (bx, by), (bx + black_key_w, by + black_key_h), (10, 10, 10), 1)


def draw_instrument_menu(frame):
    """Draw popup overlay showing all instruments with quick-select numbers."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    menu_w, menu_h = 380, 300
    mx = (w - menu_w) // 2
    my = (h - menu_h) // 2

    # Background box
    cv2.rectangle(overlay, (mx, my), (mx + menu_w, my + menu_h), (20, 20, 20), -1)
    cv2.rectangle(overlay, (mx, my), (mx + menu_w, my + menu_h), (0, 200, 255), 2)

    # Blend overlay
    alpha = 0.85
    frame[:] = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

    # Title
    cv2.putText(frame, "INSTRUMENTS", (mx + 20, my + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)

    # List instruments
    current = player.current_index
    for idx, name in enumerate(INSTRUMENT_NAMES):
        y_pos = my + 80 + idx * 40
        prefix = f"[{idx + 1}]"
        text = f"{prefix}  {name.upper()}"

        if idx == current:
            cv2.rectangle(frame, (mx + 15, y_pos - 22), (mx + menu_w - 15, y_pos + 5), (0, 200, 255), -1)
            text_color = (0, 0, 0)
        else:
            text_color = (220, 220, 220)

        cv2.putText(frame, text, (mx + 25, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    # Footer
    cv2.putText(frame, "Press 1-5 to select  |  i to close", (mx + 30, my + menu_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)


# ─── Main Loop ─────────────────────────────────────────────────

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror
    frame = cv2.flip(frame, 1)

    # Hand tracking
    hands_data = hand_tracker.get_all_fingertips(frame)
    if hands_data:
        hands_data.sort(key=lambda h: 0 if h['hand'] == 'left' else 1)

    # Fingertip positions
    fingertips_ordered = []
    for hand_info in hands_data:
        fingertips_2d = [p[:2] for p in hand_info['fingertips_3d']]
        fingertips_ordered.extend(fingertips_2d)

    while len(fingertips_ordered) < 10:
        fingertips_ordered.append(None)
    fingertips_ordered = fingertips_ordered[:10]

    h = frame.shape[0]
    edge_y = h // 2

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

    # Draw keyboard
    draw_piano_overlay(frame, finger_pressed)

    # Fingertip dots
    for i, pos in enumerate(fingertips_ordered):
        if pos is not None:
            color = (0, 255, 0) if finger_pressed[i] else (0, 0, 255)
            cv2.circle(frame, (int(pos[0]), int(pos[1])), 10, color, -1)

    # Boundary line
    cv2.line(frame, (0, edge_y), (frame.shape[1], edge_y), (255, 255, 0), 2)

    # ── Instrument name & controls ──
    instr_name = player.current_instrument.upper()

    # Bigger centered instrument name
    text_size = cv2.getTextSize(f"[ {instr_name} ]", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    tx = (frame.shape[1] - text_size[0]) // 2
    cv2.putText(frame, f"[ {instr_name} ]", (tx, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(frame, "q:quit  i:instruments", (10, h - KEYBOARD_HEIGHT - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Instrument menu popup
    if show_instrument_menu:
        draw_instrument_menu(frame)
        menu_timer += 1
        if menu_timer > MENU_DISPLAY_DURATION:
            show_instrument_menu = False
            menu_timer = 0

    cv2.imshow("Virtual Piano", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('i'):
        show_instrument_menu = not show_instrument_menu
        menu_timer = 0
        print(f"Instrument menu: {'shown' if show_instrument_menu else 'hidden'}")
    elif show_instrument_menu and ord('1') <= key <= ord('9'):
        idx = key - ord('1')
        if idx < len(INSTRUMENT_NAMES):
            player.switch_instrument(idx)
            print(f"Switched to: {player.current_instrument}")
            menu_timer = 0  # keep menu open, reset timer

cap.release()
cv2.destroyAllWindows()

class TouchStateMachine:
    """Track one fingertip against a surface and emit touch edge events."""

    def __init__(self, release_hysteresis=10, press_hysteresis=0, dropout_tolerance=1):
        self.release_hysteresis = release_hysteresis
        self.press_hysteresis = press_hysteresis
        self.dropout_tolerance = dropout_tolerance
        self.is_pressed = False
        self._previous_y = None
        self._dropout_frames = 0

    def update(self, y, surface_y, tracked=True):
        """Return note_on, note_off, or None for this frame."""
        if not tracked or y is None:
            self._dropout_frames += 1
            if self._dropout_frames <= self.dropout_tolerance:
                return None

            self._previous_y = None
            if self.is_pressed:
                self.is_pressed = False
                return "note_off"
            return None

        self._dropout_frames = 0
        press_y = surface_y + self.press_hysteresis
        release_y = surface_y - self.release_hysteresis
        event = None

        if self.is_pressed:
            if y < release_y:
                self.is_pressed = False
                event = "note_off"
        elif self._previous_y is not None and self._previous_y < press_y <= y:
            self.is_pressed = True
            event = "note_on"

        self._previous_y = y
        return event

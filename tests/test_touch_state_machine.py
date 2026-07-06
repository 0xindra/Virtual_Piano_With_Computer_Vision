import unittest

from src.touch_state_machine import TouchStateMachine


class TouchStateMachineTests(unittest.TestCase):
    SURFACE_Y = 100
    RELEASE_HYSTERESIS = 10

    def make_machine(self):
        return TouchStateMachine(release_hysteresis=self.RELEASE_HYSTERESIS)

    def events_for(self, machine, ys):
        return [
            event
            for y in ys
            if (event := machine.update(y, self.SURFACE_Y)) is not None
        ]

    def test_downward_movement_far_from_surface_never_note_on(self):
        machine = self.make_machine()

        events = self.events_for(machine, [20, 40, 60, 80])

        self.assertNotIn("note_on", events)
        self.assertEqual([], events)

    def test_surface_entry_emits_exactly_one_note_on(self):
        machine = self.make_machine()

        events = self.events_for(machine, [70, 90, 101, 108, 115])

        self.assertEqual(["note_on"], events)

    def test_hold_and_near_threshold_jitter_do_not_retrigger(self):
        machine = self.make_machine()

        events = self.events_for(machine, [80, 95, 101, 104, 99, 102, 98, 103])

        self.assertEqual(1, events.count("note_on"))
        self.assertEqual(["note_on"], events)

    def test_release_beyond_hysteresis_allows_a_second_press(self):
        machine = self.make_machine()

        events = self.events_for(machine, [80, 95, 101, 99, 96, 89, 95, 102])

        self.assertEqual(["note_on", "note_off", "note_on"], events)

    def test_one_frame_tracking_dropout_does_not_reset_or_retrigger(self):
        machine = self.make_machine()
        events = self.events_for(machine, [80, 95, 101])

        dropout_event = machine.update(None, self.SURFACE_Y, tracked=False)
        events.extend(self.events_for(machine, [103, 105]))

        self.assertIsNone(dropout_event)
        self.assertEqual(["note_on"], events)


if __name__ == "__main__":
    unittest.main()

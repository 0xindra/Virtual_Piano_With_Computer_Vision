import pygame
import os
from sound_generator import generate_instrument, GENERATORS

INSTRUMENT_NAMES = ['piano'] + sorted(GENERATORS.keys())


class SoundPlayer:
    def __init__(self, notes):
        pygame.mixer.init()
        self.notes = notes
        self.current_index = 0  # piano by default
        self.banks = {}

        # Load piano sounds from MP3s
        self.banks['piano'] = {}
        for note in self.notes:
            sound_path = os.path.join("resources", "sounds", f"{note}.mp3")
            if os.path.exists(sound_path):
                self.banks['piano'][note] = pygame.mixer.Sound(sound_path)
            else:
                print(f"Warning: {sound_path} not found.")

        # Pre-generate synthesized instruments
        for name in GENERATORS:
            try:
                self.banks[name] = generate_instrument(name, self.notes)
                print(f"  ✓ Instrument '{name}' loaded ({len(self.banks[name])} notes)")
            except Exception as e:
                print(f"  ✗ Instrument '{name}' failed: {e}")
                self.banks[name] = {}

    @property
    def current_instrument(self):
        return INSTRUMENT_NAMES[self.current_index]

    def switch_instrument(self, idx=None):
        """Switch instrument. If idx=None, cycle to next."""
        if idx is not None:
            self.current_index = idx % len(INSTRUMENT_NAMES)
        else:
            self.current_index = (self.current_index + 1) % len(INSTRUMENT_NAMES)
        return self.current_instrument

    def get_sound(self, note):
        bank = self.banks.get(self.current_instrument, {})
        return bank.get(note)

    def play_note_by_index(self, i):
        if 0 <= i < len(self.notes):
            note = self.notes[i]
            sound = self.get_sound(note)
            if sound:
                sound.play()

    def play_chord_by_indices(self, indices):
        """Play multiple notes simultaneously as a chord."""
        for i in indices:
            if 0 <= i < len(self.notes):
                note = self.notes[i]
                sound = self.get_sound(note)
                if sound:
                    sound.play()

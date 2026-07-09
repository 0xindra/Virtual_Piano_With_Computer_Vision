"""
Sound generator for Virtual Piano.
Synthesizes multiple instrument sounds using numpy + pygame.sndarray.
"""

import numpy as np
import pygame

SAMPLE_RATE = 44100
AMPLITUDE = 0.8

NOTE_FREQ = {
    'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23, 'G4': 392.00,
    'A4': 440.00, 'B4': 493.88, 'C5': 523.25, 'D5': 587.33, 'E5': 659.25,
}


def _normalize(wave):
    max_val = np.max(np.abs(wave))
    if max_val > 0:
        wave = wave / max_val * AMPLITUDE
    return np.int16(wave * 32767)


def _t(duration):
    return np.linspace(0, duration, int(SAMPLE_RATE * duration), False)


def gen_organ(freq, duration=1.5):
    """Warm organ — sine + odd harmonics + slight tremolo."""
    t = _t(duration)
    wave = (np.sin(2 * np.pi * freq * t) * 0.5 +
            np.sin(2 * np.pi * 3 * freq * t) * 0.25 +
            np.sin(2 * np.pi * 5 * freq * t) * 0.125 +
            np.sin(2 * np.pi * 7 * freq * t) * 0.06 +
            np.sin(2 * np.pi * 9 * freq * t) * 0.03)
    # Sustain envelope with light tremolo
    env = 0.9 * np.exp(-t * 0.8) + 0.1 * (1 + 0.05 * np.sin(2 * np.pi * 5.5 * t))
    return _normalize(wave * env)


def gen_synth(freq, duration=1.0):
    """Retro synth — saw + square blend with saturation."""
    t = _t(duration)
    saw = 2 * (freq * t - np.floor(freq * t + 0.5))
    square = np.sign(np.sin(2 * np.pi * freq * t))
    wave = saw * 0.5 + square * 0.3
    env = np.exp(-t * 2.5) * 0.6 + 0.4 * np.exp(-t * 1.5)
    wave = np.tanh(wave * env * 1.2)
    return _normalize(wave)


def gen_strings(freq, duration=2.0):
    """Warm strings — layered detuned saws with slow attack."""
    t = _t(duration)
    wave = np.zeros_like(t)
    for df in [0.997, 1.0, 1.003, 2.0, 3.0]:
        saw = 2 * (freq * df * t - np.floor(freq * df * t + 0.5))
        wave += saw * 0.2
    attack = np.minimum(t / 0.15, 1.0)
    env = attack * np.exp(-t * 0.6)
    return _normalize(wave * env)


def gen_bell(freq, duration=1.0):
    """Bell/music box — harmonic series with fast decay."""
    t = _t(duration)
    wave = np.zeros_like(t)
    for h, amp in [(1, 1.0), (2, 0.5), (3, 0.3), (4, 0.15), (5, 0.08)]:
        wave += amp * np.sin(2 * np.pi * freq * h * t)
    # Higher harmonics decay faster
    decay = np.exp(-t * 5)
    return _normalize(wave * decay)


# Registry of synthesizers (piano uses MP3s separately)
GENERATORS = {
    'organ': gen_organ,
    'synth': gen_synth,
    'strings': gen_strings,
    'bell': gen_bell,
}


def generate_instrument(name, notes):
    """Generate pygame Sound objects for all notes for the given instrument."""
    if name not in GENERATORS:
        raise ValueError(f"Unknown instrument: {name}")
    gen_func = GENERATORS[name]
    sounds = {}
    for note in notes:
        freq = NOTE_FREQ[note]
        wave = gen_func(freq)
        sounds[note] = pygame.sndarray.make_sound(wave)
    return sounds

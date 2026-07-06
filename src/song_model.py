"""
Song/Lesson data model for the Chord Runner game.

Songs are stored as JSON files with note sequences, chords, and timing info.
This module handles loading, validating, and representing song data.
"""

import json
import os


# Available notes mapped to finger indices
LEFT_HAND_NOTES = ['C4', 'D4', 'E4', 'F4', 'G4']
RIGHT_HAND_NOTES = ['A4', 'B4', 'C5', 'D5', 'E5']
ALL_NOTES = LEFT_HAND_NOTES + RIGHT_HAND_NOTES

# Note to finger index mapping
NOTE_TO_INDEX = {note: i for i, note in enumerate(ALL_NOTES)}

# Common chord definitions (notes that make up each chord)
CHORD_DEFINITIONS = {
    'C_major': ['C4', 'E4', 'G4'],
    'D_minor': ['D4', 'F4', 'A4'],
    'E_minor': ['E4', 'G4', 'B4'],
    'F_major': ['F4', 'A4', 'C5'],
    'G_major': ['G4', 'B4', 'D5'],
    'A_minor': ['A4', 'C5', 'E5'],
    'C_major_scale': ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5'],
}


class NoteEvent:
    """Represents a single note or chord event in a song."""

    def __init__(self, time_ms, notes, duration_ms=500, is_chord=False, chord_name=None):
        """
        Args:
            time_ms: When this event should be played (ms from song start)
            notes: List of note names (e.g. ['C4', 'E4', 'G4'])
            duration_ms: How long the note/chord should be held
            is_chord: Whether this is a chord (multiple simultaneous notes)
            chord_name: Optional display name for the chord
        """
        self.time_ms = time_ms
        self.notes = notes
        self.duration_ms = duration_ms
        self.is_chord = is_chord
        self.chord_name = chord_name
        self.finger_indices = [NOTE_TO_INDEX[n] for n in notes if n in NOTE_TO_INDEX]

    def to_dict(self):
        d = {
            'time_ms': self.time_ms,
            'notes': self.notes,
            'duration_ms': self.duration_ms,
        }
        if self.is_chord:
            d['is_chord'] = True
        if self.chord_name:
            d['chord_name'] = self.chord_name
        return d

    @classmethod
    def from_dict(cls, data):
        return cls(
            time_ms=data['time_ms'],
            notes=data['notes'],
            duration_ms=data.get('duration_ms', 500),
            is_chord=data.get('is_chord', len(data['notes']) > 1),
            chord_name=data.get('chord_name', None),
        )


class Song:
    """Represents a complete song/lesson for the Chord Runner game."""

    def __init__(self, title, artist, difficulty, bpm, events, description=""):
        """
        Args:
            title: Song/lesson title
            artist: Creator or 'Lesson'
            difficulty: 1-5 difficulty rating
            bpm: Beats per minute (used for visual scrolling speed)
            events: List of NoteEvent objects
            description: Optional description or instructions
        """
        self.title = title
        self.artist = artist
        self.difficulty = difficulty
        self.bpm = bpm
        self.events = events
        self.description = description
        self.duration_ms = max(e.time_ms + e.duration_ms for e in events) if events else 0

    def to_dict(self):
        return {
            'title': self.title,
            'artist': self.artist,
            'difficulty': self.difficulty,
            'bpm': self.bpm,
            'description': self.description,
            'events': [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data):
        events = [NoteEvent.from_dict(e) for e in data['events']]
        return cls(
            title=data['title'],
            artist=data.get('artist', 'Unknown'),
            difficulty=data.get('difficulty', 1),
            bpm=data.get('bpm', 120),
            events=events,
            description=data.get('description', ''),
        )

    @classmethod
    def load(cls, filepath):
        """Load a song from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, filepath):
        """Save the song to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def get_available_songs(songs_dir):
    """Scan a directory for available song JSON files."""
    songs = []
    if not os.path.exists(songs_dir):
        return songs
    for filename in sorted(os.listdir(songs_dir)):
        if filename.endswith('.json'):
            filepath = os.path.join(songs_dir, filename)
            try:
                song = Song.load(filepath)
                songs.append((filepath, song))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load {filename}: {e}")
    return songs

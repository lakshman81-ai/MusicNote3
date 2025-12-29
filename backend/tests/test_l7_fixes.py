import pytest
import numpy as np
import soundfile as sf
import os
from music21 import stream, note, tempo
from backend.benchmarks.ladder.synth import midi_to_wav_synth
from backend.pipeline.stage_c import _merge_notes_across_layers
from backend.pipeline.models import NoteEvent

def test_synth_duration_matches(tmp_path):
    # Build a MIDI with known duration
    s = stream.Score()
    p = stream.Part()
    # 120 BPM -> 0.5 sec per quarter. 20 quarters = 10.0 sec
    p.append(tempo.MetronomeMark(number=120))
    for _ in range(20):
        n = note.Note("C4")
        n.quarterLength = 1.0
        p.append(n)
    s.append(p)

    wav_path = str(tmp_path / "test_dur.wav")

    # We call it. If patch not applied, it returns string. If applied, it ignores return or we adapt.
    # The requirement is that we can verify the duration.
    # We will pass target_duration_sec to enforce strictness if the patch supports it.

    # Note: The test assumes Patch 2 changes midi_to_wav_synth to support 'target_duration_sec' and fix duration drift.
    # We will check if the file exists and its duration.

    try:
        # Pass diagnostics dict to populate if patch applied
        diag = {}
        midi_to_wav_synth(s, wav_path, target_duration_sec=10.0) # We can't pass diag in old signature
    except TypeError:
        # Fallback to old signature
        midi_to_wav_synth(s, wav_path, target_duration_sec=10.0)

    y, sr = sf.read(wav_path)
    dur = len(y) / sr

    # Tolerance 0.05s (Patch 2 acceptance)
    assert abs(dur - 10.0) < 0.05, f"Expected 10.0s, got {dur}s"

def test_chord_extracts_multiple_notes():
    # Test _merge_notes_across_layers
    n1 = NoteEvent(start_sec=1.00, end_sec=2.0, midi_note=60, pitch_hz=261.63, confidence=0.9, voice=0)
    n2 = NoteEvent(start_sec=1.01, end_sec=2.01, midi_note=64, pitch_hz=329.63, confidence=0.8, voice=1)
    n3 = NoteEvent(start_sec=0.99, end_sec=1.99, midi_note=67, pitch_hz=392.00, confidence=0.85, voice=2)
    # Duplicate C4 in another voice
    n4 = NoteEvent(start_sec=1.02, end_sec=2.02, midi_note=60, pitch_hz=261.63, confidence=0.7, voice=3)

    notes = [n1, n2, n3, n4]

    merged = _merge_notes_across_layers(notes, pitch_tolerance_cents=50.0, snap_onset_ms=30.0, max_gap_ms=100.0)

    # Expect 3 notes: C4, E4, G4. The second C4 should be merged.
    assert len(merged) == 3
    pitches = sorted([int(n.midi_note) for n in merged])
    assert pitches == [60, 64, 67]

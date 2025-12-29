
import pytest
import numpy as np
import os
import json
import soundfile as sf
from unittest.mock import MagicMock
from music21 import stream, note, tempo

from backend.benchmarks.ladder.synth import midi_to_wav_synth, _time_stretch_to_target

def test_time_stretch_resample():
    sr = 1000
    # Create 1 second of audio
    t = np.linspace(0, 1.0, sr)
    audio = np.sin(2 * np.pi * 10 * t).astype(np.float32)

    # Target 1.5 seconds
    target_sec = 1.5
    target_samples = int(target_sec * sr)

    out, diag = _time_stretch_to_target(audio, sr, target_sec)

    assert len(out) == target_samples
    assert diag["target_samples"] == target_samples
    assert diag["original_samples"] == 1000
    # Should prefer resample_poly if scipy available
    try:
        import scipy.signal
        assert diag["method"] == "resample_poly"
    except ImportError:
        assert diag["method"] == "interp"

def test_synth_duration_enforcement(tmp_path):
    # Create a simple score that is nominally 1 beat at 60 BPM = 1 second
    s = stream.Score()
    p = stream.Part()
    m = stream.Measure()
    m.append(tempo.MetronomeMark(number=60))
    n = note.Note("C4", quarterLength=1.0)
    m.append(n)
    p.append(m)
    s.append(p)

    wav_path = str(tmp_path / "test_synth.wav")
    sr = 22050

    # Case 1: No enforcement (target 0)
    midi_to_wav_synth(s, wav_path, sr=sr, target_duration_sec=0.0)
    assert os.path.exists(wav_path)
    assert not os.path.exists(wav_path + ".diagnostics.json")

    # Case 2: Enforcement required
    # Score is 1 sec (+ tail). Let's force target to be exactly 2.0 sec.
    target = 2.0
    midi_to_wav_synth(s, wav_path, sr=sr, target_duration_sec=target)

    assert os.path.exists(wav_path)
    assert os.path.exists(wav_path + ".diagnostics.json")

    # Check duration
    y, _ = sf.read(wav_path)
    # Allow 1 sample tolerance?
    assert abs(len(y) - int(target * sr)) <= 1

    # Check diagnostics
    with open(wav_path + ".diagnostics.json") as f:
        d = json.load(f)
        assert d["enforcement_applied"] is True
        assert d["target_duration_sec"] == target

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.wav2xml.pipeline import WavToXmlPipeline


def _synth_poly_mix(path: Path, sr: int = 44100, dur: float = 2.0) -> None:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    freqs = [261.63, 329.63, 392.0]  # C4, E4, G4
    waves = [0.33 * np.sin(2 * math.pi * f * t) for f in freqs]
    audio = np.sum(np.stack(waves, axis=0), axis=0)
    # Stereo copy for pipeline stereo handling tests
    stereo = np.stack([audio, audio], axis=0).T
    sf.write(path, stereo, sr)


def test_poly_mix_pipeline(tmp_path: Path):
    wav_path = tmp_path / "poly_mix.wav"
    _synth_poly_mix(wav_path)

    pipeline = WavToXmlPipeline(config_dir="backend/config")
    artifacts = pipeline.run(str(wav_path), workdir=str(tmp_path / "workdir"))

    # Notes and beat map should be present
    assert artifacts.notes, "Pipeline should emit at least one note for poly mix"
    assert artifacts.beat_map is not None

    # Artifacts should have been written to disk
    for rel in ["output.musicxml", "artifacts/notes_raw.json", "artifacts/timeline.json", "artifacts/meta.json"]:
        assert (tmp_path / "workdir" / rel).exists(), f"Missing artifact: {rel}"

    # Validate JSON schema subset for notes_raw
    with open(tmp_path / "workdir" / "artifacts" / "notes_raw.json", "r", encoding="utf-8") as f:
        notes = json.load(f)
    assert isinstance(notes, list)
    assert all("pitch" in n and "start" in n for n in notes)

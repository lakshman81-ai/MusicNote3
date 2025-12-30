from __future__ import annotations

from typing import List, Tuple

import librosa
import numpy as np

from .config_struct import BeatsConfig
from .models import BeatMap


def estimate_beats(audio: np.ndarray, sr: int, backend_choice: str, cfg: BeatsConfig, fallback_bpm: float) -> BeatMap:
    if backend_choice in {"madmom_py", "sonic_cli", "auto"}:
        try:
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            beat_times = librosa.frames_to_time(beats, sr=sr).tolist()
            if len(beat_times) >= cfg.min_beats:
                return BeatMap(beat_times, float(tempo), backend_choice)
        except Exception:
            pass

    # Fallback constant grid
    period = 60.0 / float(fallback_bpm)
    duration = audio.shape[-1] / float(sr)
    beat_times = list(np.arange(0, duration + 1e-6, period))
    return BeatMap(beat_times, float(fallback_bpm), "constant", metadata={"fallback": True})


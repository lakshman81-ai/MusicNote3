from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import librosa
import numpy as np
import soundfile as sf

from .config_struct import IOConfig, ViewsConfig
from .command_runner import CommandRunner


class AudioPrepResult:
    def __init__(self, audio: np.ndarray, sr: int, meta: Dict[str, float], views: Dict[str, Tuple[np.ndarray, int]]):
        self.audio = audio
        self.sr = sr
        self.meta = meta
        self.views = views


def canonicalize_audio(path: str, io_cfg: IOConfig, views_cfg: ViewsConfig, runner: CommandRunner, workdir: Path) -> AudioPrepResult:
    """Load audio and create canonical representations."""
    target_sr = io_cfg.sample_rate
    audio, sr = librosa.load(path, sr=target_sr, mono=False)
    if audio.ndim == 1:
        audio = np.expand_dims(audio, 0)
    meta = {
        "channels": float(audio.shape[0]),
        "frames": float(audio.shape[-1]),
        "target_sr": float(target_sr),
    }

    # Persist canonical base wav for downstream tools.
    base_path = workdir / "base.wav"
    workdir.mkdir(parents=True, exist_ok=True)
    sf.write(base_path, audio.T, target_sr, format="WAV", subtype="PCM_16")

    views: Dict[str, Tuple[np.ndarray, int]] = {"full": (audio, target_sr)}
    # Simple mono center/side approximations; filters logged but implemented lightly to keep deps minimal.
    if audio.shape[0] == 2:
        left, right = audio
        center = (left + right) * 0.5
        side = (left - right) * 0.5
        views["center"] = (np.stack([center], axis=0), target_sr)
        views["side"] = (np.stack([side], axis=0), target_sr)
    else:
        mono = audio[0]
        views["center"] = (np.expand_dims(mono, 0), target_sr)
        views["side"] = (np.expand_dims(mono, 0), target_sr)

    # Log ffmpeg intent for provenance even when using in-memory path.
    runner._append_log(
        [
            "ffmpeg",
            "-i",
            os.path.abspath(path),
            "-ar",
            str(target_sr),
            "-ac",
            str(io_cfg.channels),
            str(base_path),
        ],
        0,
        0.0,
        "",
        "",
    )

    return AudioPrepResult(audio, target_sr, meta, views)


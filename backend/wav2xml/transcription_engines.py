from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import librosa
import numpy as np

from .models import NoteEvent


@dataclass
class TranscriptionOutput:
    notes: List[NoteEvent]
    confidence: float
    backend: str


def _detect_onsets(y: np.ndarray, sr: int) -> List[float]:
    try:
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
        return [float(t) for t in onsets]
    except Exception:
        return []


def _simple_pitch_estimates(y: np.ndarray, sr: int, onsets: Iterable[float]) -> List[NoteEvent]:
    notes: List[NoteEvent] = []
    onset_list = list(onsets)
    if not onset_list:
        duration = y.shape[-1] / float(sr)
        midi = 60
        notes.append(NoteEvent(pitch=midi, start=0.0, duration=duration, velocity=64))
        return notes
    for idx, start in enumerate(onset_list):
        end = onset_list[idx + 1] if idx + 1 < len(onset_list) else y.shape[-1] / float(sr)
        frame = y[int(start * sr) : int(min(end, y.shape[-1] / float(sr)) * sr)]
        if frame.size == 0:
            continue
        try:
            f0, voiced_flag, _ = librosa.pyin(frame, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
            voiced = f0[voiced_flag]
            hz = float(np.median(voiced)) if voiced.size else librosa.note_to_hz("C4")
        except Exception:
            hz = librosa.note_to_hz("C4")
        midi = int(round(librosa.hz_to_midi(hz)))
        notes.append(
            NoteEvent(
                pitch=midi,
                start=float(start),
                duration=max(0.05, float(end - start)),
                velocity=80.0,
                confidence=0.6,
            )
        )
    return notes


class LiteTranscriber:
    def transcribe(self, y: np.ndarray, sr: int, stem: str) -> TranscriptionOutput:
        # Lightweight polyphonic approximation using onsets + pyin median
        mono = np.mean(y, axis=0) if y.ndim > 1 else y
        onsets = _detect_onsets(mono, sr)
        notes = _simple_pitch_estimates(mono, sr, onsets)
        for n in notes:
            n.stem = stem
            n.provenance["backend"] = "lite"
        return TranscriptionOutput(notes=notes, confidence=0.4, backend="lite")


class ProTranscriber:
    """
    Stub for PRO backend. In this implementation we reuse the lite logic but tag provenance,
    keeping the interface ready for pluggable model-backed engines.
    """

    def transcribe(self, y: np.ndarray, sr: int, stem: str) -> TranscriptionOutput:
        mono = np.mean(y, axis=0) if y.ndim > 1 else y
        onsets = _detect_onsets(mono, sr)
        notes = _simple_pitch_estimates(mono, sr, onsets)
        for n in notes:
            n.stem = stem
            n.provenance["backend"] = "pro"
        return TranscriptionOutput(notes=notes, confidence=0.6, backend="pro")


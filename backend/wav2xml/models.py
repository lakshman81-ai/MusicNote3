from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class NoteEvent:
    """Canonical representation of a note used across stages."""

    pitch: int
    start: float
    duration: float
    velocity: float = 64.0
    confidence: float = 1.0
    stem: str = "mix"
    channel: int = 0
    pedal: bool = False
    bends: Optional[List[Tuple[float, float]]] = None
    provenance: Dict[str, str] = field(default_factory=dict)

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class BeatMap:
    beats: List[float]
    tempo_bpm: float
    backend: str
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class BackendResolution:
    feature: str
    requested: Optional[str]
    resolved: str
    available: bool
    healthy: bool
    reason: str
    caps: Dict[str, str] = field(default_factory=dict)


@dataclass
class DecisionTrace:
    resolutions: List[BackendResolution] = field(default_factory=list)
    provenance: Dict[str, str] = field(default_factory=dict)
    effective_params: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class Candidate:
    notes: List[NoteEvent]
    beat_map: BeatMap
    score: float
    quantized: bool
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineArtifacts:
    musicxml: str
    notes_raw_path: str
    timeline_path: str
    meta_path: str
    commands_log: str
    notes: List[NoteEvent] = field(default_factory=list)
    beat_map: Optional[BeatMap] = None

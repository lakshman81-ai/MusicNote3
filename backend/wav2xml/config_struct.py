from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class IOConfig:
    sample_rate: int = 44100
    sample_fmt: str = "pcm_s16le"
    channels: int = 2
    time_signature: str = "4/4"
    divisions: int = 960
    fallback_bpm: float = 120.0


@dataclass
class CacheConfig:
    enabled: bool = True
    schema_version: int = 2
    reuse_policy: str = "warn"
    float_mode: str = "g12"


@dataclass
class LoggingConfig:
    commands_tail_bytes: int = 4096


@dataclass
class BackendPolicyConfig:
    fallback_mode: str = "auto"  # auto | strict | never
    allow_off: bool = True


@dataclass
class BackendPriorityConfig:
    separation: List[str] = field(default_factory=lambda: ["demucs_py", "demucs_cli", "none"])
    piano_tx: List[str] = field(default_factory=lambda: ["bytedance_py", "piano_cli", "none"])
    vocal_tx: List[str] = field(default_factory=lambda: ["basic_pitch_py", "basic_pitch_cli", "none"])
    beats: List[str] = field(default_factory=lambda: ["madmom_py", "sonic_cli", "off"])
    quantize: List[str] = field(default_factory=lambda: ["music21_py", "dp_cost_lite", "grid_lite"])
    midi_parser: List[str] = field(default_factory=lambda: ["mido_py", "minimal"])
    duration_fix: List[str] = field(default_factory=lambda: ["scipy_resample_poly", "interp_linear", "none"])


@dataclass
class ToolsConfig:
    ffmpeg: str = "ffmpeg"
    demucs: str = "demucs"
    sonic_annotator: str = "sonic-annotator"


@dataclass
class ViewFiltersConfig:
    full: str = "anull"
    low: str = "lowpass=f=250"
    mid: str = "bandpass=f=1000:w=2"
    center: str = "pan=mono|c0=0.5*c0+0.5*c1"
    side: str = "pan=mono|c0=0.5*c0-0.5*c1"


@dataclass
class ViewsConfig:
    normalize_mode: str = "peak"
    filters: ViewFiltersConfig = field(default_factory=ViewFiltersConfig)


@dataclass
class DemucsStemConfig:
    piano_stems: List[str] = field(default_factory=lambda: ["other", "bass"])
    skip_stems: List[str] = field(default_factory=lambda: ["drums"])
    vocal_stem: str = "vocals"
    also_transcribe_no_vocals: bool = False


@dataclass
class DemucsConfig:
    enabled: bool = True
    mode: str = "4stem"
    shifts_fast: int = 0
    shifts_final: int = 2
    overlap: float = 0.25
    stems: DemucsStemConfig = field(default_factory=DemucsStemConfig)


@dataclass
class SeparationConfig:
    demucs: DemucsConfig = field(default_factory=DemucsConfig)


@dataclass
class BeatsConfig:
    backend: str = "auto"
    min_beats: int = 8
    max_gap_s: float = 1.5


@dataclass
class MergeWeightsConfig:
    conf: float = 1.0
    dur: float = 0.2
    vel: float = 0.1
    stem: Dict[str, float] = field(default_factory=lambda: {"other": 1.0, "bass": 0.8, "no_vocals": 0.9})


@dataclass
class MergeConfig:
    enabled: bool = True
    onset_snap_ms: float = 20.0
    gap_merge_ms: float = 30.0
    dedupe_overlap_iou: float = 0.65
    prefer_longer: bool = True
    weights: MergeWeightsConfig = field(default_factory=MergeWeightsConfig)


@dataclass
class CleanupConfig:
    min_dur_s: float = 0.06
    merge_gap_s: float = 0.03
    dedup_eps_s: float = 0.02
    clamp_to_audio: bool = True


@dataclass
class QuantizeConfig:
    allowed_grids: List[int] = field(default_factory=lambda: [16, 12, 24])
    max_tuplet_level: int = 3
    soft_snap_tol_div: int = 24
    max_snap_div: int = 12


@dataclass
class EngraveConfig:
    staff_split_midi: int = 60
    max_voices_per_staff: int = 2
    tie_across_barlines: bool = True
    protect_melody: bool = True
    melody_top_percentile: float = 0.15
    drop_policy_max_chord_size: int = 6


@dataclass
class Wav2XmlConfig:
    io: IOConfig = field(default_factory=IOConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    backend_policy: BackendPolicyConfig = field(default_factory=BackendPolicyConfig)
    backend_priority: BackendPriorityConfig = field(default_factory=BackendPriorityConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    views: ViewsConfig = field(default_factory=ViewsConfig)
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    beats: BeatsConfig = field(default_factory=BeatsConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    quantize: QuantizeConfig = field(default_factory=QuantizeConfig)
    engrave: EngraveConfig = field(default_factory=EngraveConfig)


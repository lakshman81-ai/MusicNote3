import numpy as np
import pytest

from backend.pipeline.transcribe import _build_resolved_params
from backend.pipeline.models import MetaData, StageAOutput, StageBOutput, Stem, FramePitch, NoteEvent, AnalysisData, AudioType
from backend.pipeline.config import PipelineConfig
from backend.pipeline.stage_c import apply_theory


def _meta(sample_rate=48000, hop_length=240):
    return MetaData(sample_rate=sample_rate, hop_length=hop_length, window_size=1024)


def test_resolved_params_prefers_time_grid_over_meta():
    meta = _meta()
    stage_a_out = StageAOutput(
        stems={"mix": Stem(audio=np.zeros(16, dtype=np.float32), sr=meta.sample_rate, type="mix")},
        meta=meta,
        audio_type=AudioType.POLYPHONIC,
    )

    tg = np.array([0.0, 0.02, 0.04], dtype=np.float32)
    stage_b_out = StageBOutput(
        time_grid=tg,
        f0_main=np.zeros_like(tg),
        f0_layers=[],
        per_detector={},
        stem_timelines={},
        meta=meta,
        diagnostics={"decision_trace": {"routing_reasons": ["R1"], "rule_hits": []}},
        timeline=[],
    )

    cfg = PipelineConfig()
    resolved = _build_resolved_params(
        stage_a_out,
        cfg,
        stage_b_out=stage_b_out,
        decision_trace=stage_b_out.diagnostics["decision_trace"],
        timeline_source="synth_from_time_grid",
        frame_hop_seconds=0.1,
        frame_hop_source="meta",
        cand_score=0.0,
        cand_metrics={},
        candidate_id="classic_melody",
        quality_gate_cfg={"enabled": True, "threshold": 0.5},
    )

    tb = resolved["timebase"]
    assert tb["frame_hop_seconds_source"] == "time_grid"
    assert tb["frame_hop_seconds"] == pytest.approx(0.02)
    assert tb["timeline_source"] == "synth_from_time_grid"


def test_apply_theory_uses_resolved_timebase_hint():
    cfg = PipelineConfig()
    cfg.stage_c.segmentation_method = {"method": "threshold"}
    cfg.stage_c.confidence_threshold = 0.05
    cfg.stage_c.confidence_hysteresis = {"start": 0.05, "end": 0.05}
    cfg.stage_c.quantize = {"enabled": False}

    timeline = [
        FramePitch(time=0.0, pitch_hz=440.0, midi=69, confidence=0.9, rms=0.1),
        FramePitch(time=0.01, pitch_hz=440.0, midi=69, confidence=0.9, rms=0.1),
        FramePitch(time=0.02, pitch_hz=440.0, midi=69, confidence=0.9, rms=0.1),
    ]

    meta = _meta()
    analysis = AnalysisData(
        meta=meta,
        stem_timelines={"mix": timeline},
        diagnostics={
            "resolved_params": {
                "timebase": {
                    "frame_hop_seconds": 0.02,
                    "frame_hop_seconds_source": "config",
                }
            },
            "decision_trace": {"routing_reasons": ["R4"], "rule_hits": []},
            "timeline_source": "stage_b_timeline",
            "frame_hop_seconds_source": "config",
        },
    )

    notes = apply_theory(analysis, cfg)

    assert len(notes) == 1
    # hop hint 0.02 => end is last frame time + 0.02
    assert notes[0].end_sec == pytest.approx(0.04)

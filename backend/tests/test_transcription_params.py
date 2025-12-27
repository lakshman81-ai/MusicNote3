# backend/tests/test_transcription_params.py
import pytest
from unittest.mock import MagicMock
import numpy as np
from backend.pipeline.transcribe import _assemble_resolved_params
from backend.pipeline.stage_c import apply_theory
from backend.pipeline.models import AnalysisData, MetaData, FramePitch
from backend.pipeline.config import PipelineConfig

def test_assemble_resolved_params_structure():
    """Verify resolved_params has the correct keys and structure."""
    cfg = PipelineConfig()
    meta = MetaData(sample_rate=44100, hop_length=512)
    trace = {"rule_hits": [{"rule_id": "R_Test", "passed": True}]}
    
    resolved = _assemble_resolved_params(
        cand_cfg=cfg,
        meta=meta,
        decision_trace=trace,
        candidate_id="test_cand",
        cand_score=0.8,
        cand_metrics={"note_count": 10},
        quality_gate_cfg={"enabled": True},
        hop_sec=0.01,
        tb_source="test_source",
        timeline_source="test_timeline",
        time_grid=[0.0, 0.01]
    )
    
    assert "timebase" in resolved
    assert "preprocessing" in resolved
    assert "detectors" in resolved
    assert "segmentation" in resolved
    assert "post_processing" in resolved
    assert "scoring_routing" in resolved
    
    # Check specific values
    assert resolved["timebase"]["frame_hop_seconds"] == 0.01
    assert resolved["timebase"]["frame_hop_seconds_source"] == "test_source"
    assert resolved["scoring_routing"]["candidate_id"] == "test_cand"
    assert "R_Test" in resolved["scoring_routing"]["routing_reasons"]

def test_transcribe_prefers_time_grid():
    """Verify logic that time_grid presence results in time_grid_available=True."""
    # This logic is inside the assembler we just tested, but let's double check the bool conversion
    cfg = PipelineConfig()
    meta = MetaData()
    resolved = _assemble_resolved_params(
        cfg, meta, {}, "id", 0.0, {}, {}, 0.01, "src", "src",
        time_grid=[0.0, 0.1, 0.2]
    )
    assert resolved["timebase"]["time_grid_available"] is True
    
    resolved_empty = _assemble_resolved_params(
        cfg, meta, {}, "id", 0.0, {}, {}, 0.01, "src", "src",
        time_grid=None
    )
    assert resolved_empty["timebase"]["time_grid_available"] is False

def test_stage_c_respects_hop_hint():
    """Verify Stage C prioritizes resolved_params hop hint over timeline estimation."""
    # Create a timeline that suggests a hop of ~1.0s (crazy, but distinct)
    timeline = [
        FramePitch(time=0.0, pitch_hz=440.0, midi=69, confidence=1.0, rms=0.1),
        FramePitch(time=1.0, pitch_hz=440.0, midi=69, confidence=1.0, rms=0.1),
        FramePitch(time=2.0, pitch_hz=440.0, midi=69, confidence=1.0, rms=0.1),
    ]
    
    analysis = AnalysisData(
        meta=MetaData(),
        stem_timelines={"mix": timeline},
        diagnostics={
            "resolved_params": {
                "timebase": {
                    "frame_hop_seconds": 0.01 # Hint says 10ms
                }
            }
        }
    )
    
    # Run apply_theory
    # logic in stage_c should grab 0.01 for hop_s
    # We can check the diagnostics output which should log the used hop
    
    apply_theory(analysis, PipelineConfig())
    
    # Check voice details
    assert "voice_details" in analysis.diagnostics
    vd = analysis.diagnostics["voice_details"][0]
    
    # If it used the timeline, hop would be ~1.0. If hint, 0.01.
    assert vd["hop_seconds"] == 0.01

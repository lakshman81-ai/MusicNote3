
import pytest
import numpy as np
from backend.pipeline.stage_b import compute_decision_trace
from backend.pipeline.models import StageAOutput, MetaData, AudioType

# Mock StageAOutput
def make_stage_a_output(mode="classic", duration=10.0):
    return StageAOutput(
        stems={},
        audio_type=AudioType.MONOPHONIC,
        meta=MetaData(
            sample_rate=22050,
            duration_sec=duration,
            processing_mode=mode,
        ),
        diagnostics={}
    )

class MockConfig:
    def __init__(self, mode="auto", sep="auto", profile="piano"):
        self.stage_b = type("StageBConfig", (), {})()
        self.stage_b.transcription_mode = mode
        self.stage_b.separation = {"mode": sep, "enabled": True}
        self.stage_b.instrument = profile
        self.stage_b.routing = {}
        self.stage_b.detectors = {}
        self.stage_b.onsets_and_frames = {"enabled": False} # Force fallback from e2e_oaf
        self.device = "cpu"

    def get_profile(self, *args):
        return None

def test_decision_trace_requested_auto():
    # Setup: config requests "auto"
    config = MockConfig(mode="auto")

    stage_a = make_stage_a_output()

    # Act
    trace = compute_decision_trace(
        stage_a,
        config,
        requested_mode="auto",
        requested_profile="piano"
    )

    # Assert
    assert trace["requested"]["transcription_mode"] == "auto"
    assert trace["resolved"]["transcription_mode"] != "auto"

def test_decision_trace_requested_manual_override():
    # Setup: manual override "classic_song"
    config = MockConfig(mode="auto") # Config says auto
    stage_a = make_stage_a_output()

    # Act: Pass explicit requested_mode
    trace = compute_decision_trace(
        stage_a,
        config,
        requested_mode="classic_song",
        requested_profile="piano"
    )

    # Assert
    assert trace["requested"]["transcription_mode"] == "classic_song"
    assert trace["resolved"]["transcription_mode"] == "classic_song"

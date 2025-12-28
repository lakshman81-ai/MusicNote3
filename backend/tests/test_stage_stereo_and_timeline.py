import numpy as np
import tempfile
from pathlib import Path
from scipy.io import wavfile

from backend.pipeline.stage_a import load_and_preprocess
from backend.pipeline.config import StageAConfig
from backend.pipeline.models import MetaData, StageAOutput, StageBOutput, FramePitch, AudioType
import backend.transcription as transcription
import backend.pipeline.stage_b as stage_b


def test_stage_a_stereo_alignment_shared_trim():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    left = 0.2 * np.sin(2 * np.pi * 220 * t)
    right = 0.2 * np.sin(2 * np.pi * 330 * t)
    stereo = np.stack([left, right], axis=0)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wavfile.write(tmp.name, sr, (stereo.T).astype(np.float32))
        tmp_path = tmp.name

    try:
        cfg = StageAConfig(channel_handling="stereo")
        stage_a_out = load_and_preprocess(tmp_path, config=cfg)
        assert stage_a_out.meta.processed_n_channels == 2
        assert stage_a_out.meta.downmix_applied is False
        mix_audio = stage_a_out.stems["mix"].audio
        # mix_audio shape is (samples, channels) for stereo
        assert mix_audio.shape[0] == stage_a_out.meta.target_sr  # ~1s
        assert mix_audio.shape[1] == 2
        assert stage_a_out.diagnostics["trim"]["trim_end"] > stage_a_out.diagnostics["trim"]["trim_start"]
        assert (stage_a_out.diagnostics["trim"]["trim_end"] - stage_a_out.diagnostics["trim"]["trim_start"]) > int(0.1 * stage_a_out.meta.target_sr)
        assert mix_audio.shape[0] == stage_a_out.diagnostics["trim"]["trim_end"] - stage_a_out.diagnostics["trim"]["trim_start"]
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_timeline_selector_avoids_empty_vocals(monkeypatch):
    # Build a fake StageA output and StageB output
    meta = MetaData(
        sample_rate=16000,
        target_sr=16000,
        duration_sec=1.0,
        original_duration_sec=1.0,
        n_channels=1,
        original_n_channels=1,
        processed_n_channels=1,
        downmix_applied=True,
    )
    mix_audio = np.zeros(int(meta.target_sr), dtype=np.float32)
    stage_a_out = StageAOutput(stems={"mix": stage_b.Stem(audio=mix_audio, sr=16000, type="mix")}, meta=meta, audio_type=AudioType.MONOPHONIC)

    mix_timeline = [
        FramePitch(time=0.0, pitch_hz=440.0, midi=69, confidence=0.9),
        FramePitch(time=0.1, pitch_hz=441.0, midi=69, confidence=0.8),
    ]
    vocals_timeline: list[FramePitch] = []

    stage_b_out = StageBOutput(
        time_grid=np.array([0.0, 0.1], dtype=np.float32),
        f0_main=np.array([440.0, 441.0], dtype=np.float32),
        f0_layers=[],
        per_detector={},
        stem_timelines={"vocals": vocals_timeline, "mix": mix_timeline},
        meta=meta,
        diagnostics={},
        timeline=[],
    )

    def fake_stage_a(path: str, config=None, target_sr=None, start_offset=0.0, max_duration=None, pipeline_logger=None, **kwargs):
        return stage_a_out

    def fake_stage_b(sa_out, config=None, use_crepe=False, confidence_threshold=0.5, min_duration_ms=0.0):
        return stage_b_out

    def fake_apply_theory(analysis_data, config=None):
        return []

    class DummyResult(stage_b.TranscriptionResult if hasattr(stage_b, "TranscriptionResult") else object):
        pass

    def fake_quantize(render_notes, analysis_data, config=None):
        from backend.pipeline.models import TranscriptionResult

        return TranscriptionResult(musicxml="<score-partwise/>", analysis_data=analysis_data, midi_bytes=b"")

    monkeypatch.setattr(transcription, "load_and_preprocess", fake_stage_a)
    monkeypatch.setattr(transcription, "extract_features", fake_stage_b)
    monkeypatch.setattr(transcription, "apply_theory", fake_apply_theory)
    monkeypatch.setattr(transcription, "quantize_and_render", fake_quantize)

    res = transcription.transcribe_audio_pipeline("dummy.wav", use_mock=False)
    assert res.analysis_data.timeline == mix_timeline
    assert len(res.analysis_data.timeline) > 0

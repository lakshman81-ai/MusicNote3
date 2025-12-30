from typing import Optional
import os

from .pipeline.models import AnalysisData, TranscriptionResult, MetaData
from .wav2xml.pipeline import WavToXmlPipeline


def transcribe_audio_pipeline(
    audio_path: str,
    *,
    stereo_mode: Optional[str] = None,  # maintained for compatibility
    use_mock: bool = False,
    start_offset: Optional[float] = None,
    max_duration: Optional[float] = None,
    **kwargs,
) -> TranscriptionResult:
    """
    New modular pipeline aligned to WI v2.4.
    Produces MusicXML + artifacts + provenance with dual LITE/PRO backends.
    """
    del stereo_mode, start_offset, max_duration, kwargs  # handled by config/pipeline internally

    if use_mock:
        mock_xml_path = os.path.join(os.path.dirname(__file__), "mock_data", "happy_birthday.xml")
        if os.path.exists(mock_xml_path):
            with open(mock_xml_path, "r", encoding="utf-8") as f:
                musicxml_str = f.read()
        else:
            musicxml_str = "<?xml version='1.0' encoding='utf-8'?><score-partwise><part><measure><note><rest/></note></measure></part></score-partwise>"
        return TranscriptionResult(musicxml=musicxml_str, analysis_data=AnalysisData(meta=MetaData(sample_rate=22050)), midi_bytes=b"")

    pipeline = WavToXmlPipeline()
    midi_input = audio_path.lower().endswith((".mid", ".midi"))
    run_result = pipeline.run(audio_path, midi_input=midi_input)

    # Minimal AnalysisData wiring to satisfy existing consumers.
    meta = MetaData(sample_rate=pipeline.config.io.sample_rate, tempo_bpm=run_result.beat_map.tempo_bpm if run_result.beat_map else pipeline.config.io.fallback_bpm)
    analysis = AnalysisData(meta=meta, beats=run_result.beat_map.beats if run_result.beat_map else [])

    return TranscriptionResult(musicxml=run_result.musicxml, analysis_data=analysis, midi_bytes=b"")


def transcribe_audio(
    file_path: str,
    use_mock: bool = False,
    stereo_mode: bool = False,
) -> str:
    """
    Legacy entry point.
    """
    res = transcribe_audio_pipeline(file_path, use_mock=use_mock, stereo_mode=str(stereo_mode))
    return res.musicxml

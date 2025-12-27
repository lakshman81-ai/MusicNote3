
import os
import sys
import json
import logging
import numpy as np
from backend.benchmarks.run_real_songs import load_ground_truth, synthesize_audio
from backend.pipeline.config import PipelineConfig
from backend.pipeline.models import (
    StageAOutput, MetaData, Stem, AudioType, AudioQuality, AnalysisData
)
from backend.pipeline.stage_b import extract_features
from backend.pipeline.stage_c import apply_theory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_l4")

def debug_l4_song(song="happy_birthday"):
    logger.info(f"Debugging L4 song: {song}")
    
    # 1. Load GT and Synthesize
    gt = load_ground_truth(song)
    notes_dur = [(n['midi_note'], n['end_sec'] - n['start_sec']) for n in gt]
    sr = 44100
    audio = synthesize_audio(notes_dur, sr=sr)
    
    # Check audio stats
    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))
    logger.info(f"Audio Stats: RMS={rms:.4f}, Peak={peak:.4f}, Duration={len(audio)/sr:.2f}s")

    # 2. Config similar to run_real_songs.py
    config = PipelineConfig()
    config.stage_b.separation['enabled'] = False
    
    # Force enable YIN and CREEP for debugging
    config.stage_b.detectors['yin']['enabled'] = True
    config.stage_b.detectors['crepe'] = {"enabled": True, "model_capacity": "small", "confidence_threshold": 0.3} # match benchmark
    
    # Tune Stage C from original script
    config.stage_c.velocity_map['min_db'] = -15.0
    
    # 3. Manual Stage A construction
    meta = MetaData(
        tuning_offset=0.0,
        detected_key="C",
        lufs=-23.0,
        processing_mode="monophonic",
        audio_type=AudioType.MONOPHONIC,
        audio_quality=AudioQuality.LOSSLESS,
        snr=0.0,
        window_size=2048,
        hop_length=512,
        sample_rate=sr,
        tempo_bpm=120.0,
        time_signature="4/4",
        original_sr=sr,
        target_sr=sr,
        duration_sec=float(len(audio) / sr),
        beats=[],
        audio_path=None,
        n_channels=1,
        normalization_gain_db=0.0,
        rms_db=-20.0,
        noise_floor_rms=0.0,
        noise_floor_db=-80.0,
        pipeline_version="bench",
    )
    stage_a_out = StageAOutput(
        stems={"mix": Stem(audio=audio, sr=sr, type="mix")},
        meta=meta,
        audio_type=AudioType.MONOPHONIC,
    )
    
    # 4. Run Stage B
    logger.info("Running Stage B...")
    stage_b_out = extract_features(stage_a_out, config=config)
    
    # Inspect Stage B Diagnostics
    diag = stage_b_out.diagnostics
    logger.info(f"Stage B Diagnostics Keys: {list(diag.keys())}")
    
    # Check pitch track
    if "mix" in stage_b_out.stem_timelines:
        timeline = stage_b_out.stem_timelines["mix"]
        logger.info(f"Timeline frames: {len(timeline)}")
        voiced_frames = sum(1 for f in timeline if f.pitch_hz > 0)
        logger.info(f"Voiced frames: {voiced_frames} / {len(timeline)}")
        if voiced_frames > 0:
            avg_pitch = np.mean([f.pitch_hz for f in timeline if f.pitch_hz > 0])
            logger.info(f"Avg Pitch (Hz): {avg_pitch:.2f}")
    else:
        logger.warning("No 'mix' timeline in Stage B output!")

    # 5. Run Stage C
    logger.info("Running Stage C...")
    analysis = AnalysisData(meta=meta)
    analysis.stem_timelines = stage_b_out.stem_timelines
    notes_pred = apply_theory(analysis, config=config)
    
    logger.info(f"Predicted Notes: {len(notes_pred)}")
    for n in notes_pred[:5]:
        logger.info(f"Note: {n.midi_note} ({n.start_sec:.2f}-{n.end_sec:.2f}s)")

if __name__ == "__main__":
    debug_l4_song("happy_birthday")

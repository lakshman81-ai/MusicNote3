from __future__ import annotations

import copy
import time
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import PipelineConfig
from .instrumentation import PipelineLogger
from .models import AnalysisData, NoteEvent, TranscriptionResult, FramePitch
from .stage_a import load_and_preprocess
from .stage_b import compute_decision_trace, extract_features
from .stage_c import apply_theory
from .stage_d import quantize_and_render
from .neural_transcription import transcribe_onsets_frames


def _quality_metrics(
    notes: List[NoteEvent],
    duration_sec: float,
    *,
    timeline_source: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    duration_sec = float(max(1e-6, duration_sec or 0.0))
    note_count = int(len(notes or []))
    if note_count == 0:
        return {
            "voiced_ratio": 0.0,
            "note_count": 0,
            "notes_per_sec": 0.0,
            "median_note_dur_ms": 0.0,
            "fragmentation_lt_80ms": 0.0,
        }

    durs = []
    tiny = 0
    total_note_dur = 0.0
    for n in notes:
        dur = float(getattr(n, "end_sec", 0.0)) - float(getattr(n, "start_sec", 0.0))
        dur = max(0.0, dur)
        durs.append(dur)
        total_note_dur += dur
        if dur < 0.08:
            tiny += 1

    durs_sorted = sorted(durs)
    mid = len(durs_sorted) // 2
    median_dur = durs_sorted[mid] if durs_sorted else 0.0

    # voiced_ratio:
    # - prefer detector timeline if available (classic path)
    # - else fallback to coverage ratio (E2E path)
    voiced_ratio: Optional[float] = None
    if timeline_source is not None:
        total_frames = max(1, len(timeline_source))
        voiced_frames = 0
        frames_with_active_attr = 0
        for fr in timeline_source:
            ap = getattr(fr, "active_pitches", None)
            if ap is None:
                continue
            frames_with_active_attr += 1
            if len(ap) > 0:
                voiced_frames += 1
        if frames_with_active_attr > 0:
            voiced_ratio = float(voiced_frames / total_frames)

    if voiced_ratio is None:
        voiced_ratio = float(min(1.0, total_note_dur / duration_sec))

    return {
        "voiced_ratio": float(max(0.0, min(1.0, voiced_ratio))),
        "note_count": note_count,
        "notes_per_sec": float(note_count / duration_sec),
        "median_note_dur_ms": float(1000.0 * median_dur),
        "fragmentation_lt_80ms": float(tiny / max(1, note_count)),
    }


def _quality_score(metrics: Dict[str, Any]) -> float:
    """
    Deterministic scalar score in [0, 1].
    Intended to reject: (a) empty output, (b) extreme fragmentation, (c) crazy note rate.
    """
    if int(metrics.get("note_count", 0) or 0) <= 0:
        return 0.0

    voiced = float(metrics.get("voiced_ratio", 0.0) or 0.0)
    nps = float(metrics.get("notes_per_sec", 0.0) or 0.0)
    med_ms = float(metrics.get("median_note_dur_ms", 0.0) or 0.0)
    frag = float(metrics.get("fragmentation_lt_80ms", 0.0) or 0.0)

    # normalize median duration: 0ms..200ms -> 0..1 (cap)
    dur_term = max(0.0, min(1.0, med_ms / 200.0))

    # note rate penalty (prefer <= ~12 notes/sec)
    rate_term = 1.0 - max(0.0, min(1.0, (nps - 6.0) / 12.0))  # 6->1.0, 18->0.0

    # fragmentation penalty
    frag_term = 1.0 - max(0.0, min(1.0, frag))

    score = 0.45 * voiced + 0.25 * dur_term + 0.20 * rate_term + 0.10 * frag_term
    return float(max(0.0, min(1.0, score)))


def _candidate_order(routed_mode: str) -> List[str]:
    """
    Deterministic fallback chain.
    (We don’t run everything; we run until we find an accepted candidate.)

    NOTE: These mode ids must match your routing + Stage B logic.
    """
    all_modes = [
        "e2e_onsets_frames",
        "e2e_basic_pitch",
        "classic_piano_poly",
        "classic_song",
        "classic_melody",
    ]
    routed_mode = routed_mode if routed_mode in all_modes else "classic_melody"
    rest = [m for m in all_modes if m != routed_mode]
    return [routed_mode] + rest


def _safe_trace(x: Any) -> Dict[str, Any]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    return {"raw": x}


def _extract_routing_reasons(decision_trace: Dict[str, Any]) -> List[str]:
    reasons = []
    try:
        for item in decision_trace.get("rule_hits", []):
            if item.get("passed"):
                reasons.append(str(item.get("rule_id")))
    except Exception:
        pass
    try:
        extra = decision_trace.get("routing_reasons", [])
        for rid in extra:
            if rid not in reasons:
                reasons.append(str(rid))
    except Exception:
        pass
    return reasons


def _build_resolved_params(
    stage_a_out,
    cand_cfg: PipelineConfig,
    *,
    stage_b_out=None,
    decision_trace: Optional[Dict[str, Any]] = None,
    timeline_source: str = "unknown",
    frame_hop_seconds: float = 0.0,
    frame_hop_source: str = "unknown",
    cand_score: float = 0.0,
    cand_metrics: Optional[Dict[str, Any]] = None,
    candidate_id: str = "",
    quality_gate_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cand_metrics = cand_metrics or {}
    quality_gate_cfg = quality_gate_cfg or {}
    dt = _safe_trace(decision_trace)
    time_grid = getattr(stage_b_out, "time_grid", None)

    allowed_timeline_sources = {
        "stage_b_timeline",
        "stems",
        "synth_from_time_grid",
        "e2e_notes",
        "unknown",
    }

    tb_source = frame_hop_source
    hop_sec = float(frame_hop_seconds or 0.0)
    if time_grid is not None and len(time_grid) >= 2:
        hop_sec = float(np.median(np.diff(time_grid)))
        tb_source = "stage_b_time_grid"
    elif hop_sec <= 0.0:
        try:
            sr = float(stage_a_out.meta.sample_rate)
            hop = float(stage_a_out.meta.hop_length)
            hop_sec = hop / max(sr, 1e-9)
            tb_source = "stage_a_meta"
        except Exception:
            hop_sec = 0.0

    stage_a_conf = getattr(cand_cfg, "stage_a", None)
    stage_b_conf = getattr(cand_cfg, "stage_b", None)
    stage_c_conf = getattr(cand_cfg, "stage_c", None)

    timeline_source_valid = str(timeline_source) if str(timeline_source) in allowed_timeline_sources else "unknown"

    sources = {
        "stage_b.transcription_mode": "config",
        "stage_b.separation.enabled": "config",
        "stage_c.segmentation_method": "config",
        "stage_c.confidence_threshold": "config",
    }

    return {
        "param_snapshot_version": 1,
        "timebase": {
            "sample_rate": float(getattr(stage_a_out.meta, "sample_rate", 0.0) or 0.0),
            "hop_length": float(getattr(stage_a_out.meta, "hop_length", 0.0) or 0.0),
            "window_size": float(getattr(stage_a_out.meta, "window_size", 0.0) or 0.0),
            "frame_hop_seconds": hop_sec,
            "frame_hop_seconds_source": tb_source,
            "timeline_source": timeline_source_valid,
            "time_grid_available": bool(time_grid is not None and len(time_grid) > 0),
        },
        "preprocessing": {
            "target_sample_rate": getattr(stage_a_conf, "target_sample_rate", None),
            "channel_handling": getattr(stage_a_conf, "channel_handling", None),
            "silence_trimming": getattr(stage_a_conf, "silence_trimming", None),
            "loudness_normalization": getattr(stage_a_conf, "loudness_normalization", None),
            "high_pass_filter": getattr(stage_a_conf, "high_pass_filter", None),
            "peak_limiter": getattr(stage_a_conf, "peak_limiter", None),
            "separation": getattr(stage_b_conf, "separation", None),
        },
        "detectors": {
            "confidence_voicing_threshold": getattr(stage_b_conf, "confidence_voicing_threshold", None) if stage_b_conf else None,
            "confidence_priority_floor": getattr(stage_b_conf, "confidence_priority_floor", None) if stage_b_conf else None,
            "ensemble_weights": dict(getattr(stage_b_conf, "ensemble_weights", {}) or {}),
            "detectors": dict(getattr(stage_b_conf, "detectors", {}) or {}),
            "smoothing_method": getattr(stage_b_conf, "smoothing_method", None) if stage_b_conf else None,
            "viterbi": {
                "transition_smoothness": getattr(stage_b_conf, "viterbi_transition_smoothness", None) if stage_b_conf else None,
                "jump_penalty": getattr(stage_b_conf, "viterbi_jump_penalty", None) if stage_b_conf else None,
            },
        },
        "segmentation": {
            "method": _safe_trace(getattr(stage_c_conf, "segmentation_method", {})).get("method", None) if stage_c_conf else None,
            "min_note_duration_ms": getattr(stage_c_conf, "min_note_duration_ms", None) if stage_c_conf else None,
            "min_note_duration_ms_poly": getattr(stage_c_conf, "min_note_duration_ms_poly", None) if stage_c_conf else None,
            "gap_tolerance_s": getattr(stage_c_conf, "gap_tolerance_s", None) if stage_c_conf else None,
            "confidence_threshold": getattr(stage_c_conf, "confidence_threshold", None) if stage_c_conf else None,
            "confidence_hysteresis": getattr(stage_c_conf, "confidence_hysteresis", None) if stage_c_conf else None,
        },
        "post_processing": {
            "chord_onset_snap_ms": getattr(stage_c_conf, "chord_onset_snap_ms", None) if stage_c_conf else None,
            "post_merge": getattr(stage_c_conf, "post_merge", None) if stage_c_conf else None,
            "gap_filling": getattr(stage_c_conf, "gap_filling", None) if stage_c_conf else None,
        },
        "scoring_routing": {
            "candidate_id": str(candidate_id),
            "quality_score": float(cand_score),
            "quality_metrics": dict(cand_metrics),
            "quality_gate_threshold": float(quality_gate_cfg.get("threshold", 0.0)),
            "quality_gate_enabled": bool(quality_gate_cfg.get("enabled", True)),
            "routing_reasons": _extract_routing_reasons(dt),
            "decision_trace": dt,
        },
        "sources": sources,
    }


def transcribe(
    audio_path: str,
    config: Optional[PipelineConfig] = None,
    pipeline_logger: Optional[PipelineLogger] = None,
    device: str = "cpu",
    *,
    # explicit caller overrides
    requested_mode: Optional[str] = None,
    requested_profile: Optional[str] = None,
    requested_separation_mode: Optional[str] = None,
) -> TranscriptionResult:
    """
    High-level transcription entry point with unified quality gate.

    Callers can force intended mode explicitly:
        transcribe(..., requested_mode="classic_song")
    """
    if config is None:
        config = PipelineConfig()
    if pipeline_logger is None:
        pipeline_logger = PipelineLogger()

    t0 = time.time()

    # ---------------- Stage A ----------------
    stage_a_out = load_and_preprocess(audio_path, config, pipeline_logger=pipeline_logger)
    duration_sec = float(getattr(getattr(stage_a_out, "meta", None), "duration_sec", 0.0) or 0.0)

    # Resolve caller intent (robust)
    cfg_mode = "auto"
    try:
        cfg_mode = getattr(config.stage_b, "transcription_mode", "auto")
    except Exception:
        cfg_mode = "auto"

    req_mode = str(requested_mode) if requested_mode is not None else str(cfg_mode)

    # B1: Preserve Stage A texture if caller passed "quality"/"fast"
    requested_quality_mode = None
    meta_processing_mode_preserved = False
    meta_processing_mode_val = None

    if requested_mode in ("quality", "fast"):
        requested_quality_mode = requested_mode
        # Ensure we don't clobber meta.processing_mode
        meta_processing_mode_preserved = True
        try:
            meta_processing_mode_val = stage_a_out.meta.processing_mode
        except Exception:
            pass

    meta_profile = str(getattr(getattr(stage_a_out, "meta", None), "instrument", None) or "unknown")
    req_profile = str(requested_profile) if requested_profile is not None else meta_profile

    # ---------------- Routing (Stage B trace only) ----------------
    base_trace = compute_decision_trace(
        stage_a_out,
        config,
        requested_mode=req_mode,
        requested_profile=req_profile,
        requested_separation_mode=requested_separation_mode,
        pipeline_logger=pipeline_logger,
    )

    if req_mode and req_mode != "auto":
        routed_mode = req_mode
    else:
        routed_mode = str(_safe_trace(base_trace).get("resolved", {}).get("transcription_mode", "classic_melody"))

    candidate_ids = _candidate_order(routed_mode)

    # ---------------- Unified Quality Gate ----------------
    qcfg = getattr(config, "quality_gate", None)
    if not isinstance(qcfg, dict):
        qcfg = {}
    q_enabled = bool(qcfg.get("enabled", True))
    q_threshold = float(qcfg.get("threshold", 0.45))
    q_max_candidates = int(qcfg.get("max_candidates", 3))

    candidates: List[Dict[str, Any]] = []
    fallbacks_triggered: List[str] = []

    best: Optional[Tuple[str, float, AnalysisData, PipelineConfig]] = None

    # Mix audio for E2E paths
    mix_audio = None
    try:
        mix_audio = stage_a_out.stems["mix"].audio
    except Exception:
        mix_audio = None

    for idx, cand_id in enumerate(candidate_ids):
        if idx >= q_max_candidates:
            break

        cand_score = 0.0
        cand_metrics: Dict[str, Any] = {}
        cand_analysis: Optional[AnalysisData] = None
        cand_trace: Dict[str, Any] = {}
        stage_b_out = None

        try:
            cand_cfg = copy.deepcopy(config)

            # Force candidate mode
            try:
                cand_cfg.stage_b.transcription_mode = cand_id
            except Exception:
                pass

            if cand_id == "e2e_onsets_frames":
                enabled = bool(getattr(cand_cfg.stage_b, "onsets_and_frames", {}).get("enabled", False))
                if not enabled:
                    raise RuntimeError("onsets_frames_disabled")
                if mix_audio is None:
                    raise RuntimeError("missing_mix_audio")

                notes_res, diag_res = transcribe_onsets_frames(
                    mix_audio,
                    int(getattr(stage_a_out.meta, "sample_rate", 44100)),
                    cand_cfg
                )
                notes = notes_res
                cand_trace.update(diag_res)

                cand_trace = _safe_trace(
                    compute_decision_trace(
                        stage_a_out,
                        cand_cfg,
                        requested_mode="e2e_onsets_frames",
                        requested_profile=req_profile,
                        requested_separation_mode=requested_separation_mode,
                        pipeline_logger=pipeline_logger,
                    )
                )

                cand_analysis = AnalysisData(
                    meta=stage_a_out.meta,
                    timeline=[],
                    stem_timelines={},
                    notes=list(notes),
                    notes_before_quantization=list(notes),
                    chords=[],
                    diagnostics={
                        "decision_trace": cand_trace,
                        "timeline_source": "e2e_notes",
                        "frame_hop_seconds_source": "meta",
                        "frame_hop_seconds": float(getattr(stage_a_out.meta, "hop_length", 512)) / float(getattr(stage_a_out.meta, "sample_rate", 44100)),
                    },
                )
                try:
                    cand_analysis.diagnostics["resolved_params"] = _build_resolved_params(
                        stage_a_out,
                        cand_cfg,
                        stage_b_out=None,
                        decision_trace=cand_trace,
                        timeline_source="e2e_notes",
                        frame_hop_seconds=float(cand_analysis.diagnostics["frame_hop_seconds"]),
                        frame_hop_source="meta",
                        cand_score=0.0,
                        cand_metrics={},
                        candidate_id=cand_id,
                        quality_gate_cfg=qcfg,
                    )
                except Exception:
                    pass

            else:
                # Classic + BasicPitch go through Stage B -> Stage C
                stage_b_out = extract_features(
                    stage_a_out, config=cand_cfg, pipeline_logger=pipeline_logger, device=device
                )

                sb_diag = getattr(stage_b_out, "diagnostics", None) or {}
                cand_trace = _safe_trace(sb_diag.get("decision_trace", {}))

                # B2: Timeline fallback logic
                sb_timeline = getattr(stage_b_out, "timeline", None) or []
                sb_stem_timelines = getattr(stage_b_out, "stem_timelines", None) or {}

                timeline_source = "empty"
                final_timeline = []
                timeline_synth_diag = {}

                if sb_stem_timelines:
                    timeline_source = "stems"
                    final_timeline = sb_timeline # Use existing if present, or let AnalysisData handle
                elif sb_timeline:
                    timeline_source = "stage_b_timeline"
                    final_timeline = sb_timeline
                else:
                    # Fallback synthesis
                    time_grid = getattr(stage_b_out, "time_grid", None)
                    f0_main = getattr(stage_b_out, "f0_main", None)
                    if time_grid is not None and f0_main is not None and len(time_grid) == len(f0_main):
                        timeline_source = "synth_from_time_grid"
                        timeline_synth_diag = {"used_time_grid": True, "used_f0_main": True, "confidence": "assumed_1.0"}

                        synths = []
                        for t_idx, t_val in enumerate(time_grid):
                            f0 = float(f0_main[t_idx])
                            voiced = f0 > 0.0

                            midi_val = None
                            if voiced:
                                try:
                                    midi_val = int(round(69.0 + 12.0 * math.log2(f0 / 440.0)))
                                except Exception:
                                    pass

                            synths.append(FramePitch(
                                time=float(t_val),
                                pitch_hz=f0,
                                midi=midi_val,
                                confidence=1.0 if voiced else 0.0,
                                rms=0.0,
                                active_pitches=[(f0, 1.0)] if voiced else []
                            ))
                        final_timeline = synths

                # B3: Frame hop seconds calculation
                frame_hop_seconds = 0.0
                frame_hop_source = "unknown"

                time_grid_ref = getattr(stage_b_out, "time_grid", None)
                if time_grid_ref is not None and len(time_grid_ref) >= 2:
                    frame_hop_seconds = float(np.median(np.diff(time_grid_ref)))
                    frame_hop_source = "stage_b_time_grid"
                else:
                    try:
                        frame_hop_seconds = float(stage_b_out.meta.hop_length) / float(stage_b_out.meta.sample_rate)
                        frame_hop_source = "meta"
                    except Exception:
                        pass

                cand_analysis = AnalysisData(
                    meta=stage_b_out.meta,
                    timeline=final_timeline,
                    stem_timelines=sb_stem_timelines,
                    diagnostics=sb_diag,
                    precalculated_notes=getattr(stage_b_out, "precalculated_notes", None),
                    n_frames=len(final_timeline),
                    frame_hop_seconds=frame_hop_seconds
                )

                # Restore processing_mode if needed (B1 fix)
                if requested_quality_mode and meta_processing_mode_val is not None:
                    try:
                        cand_analysis.meta.processing_mode = meta_processing_mode_val
                    except Exception:
                        pass

                cand_analysis.diagnostics = cand_analysis.diagnostics or {}
                cand_analysis.diagnostics["timeline_source"] = timeline_source
                cand_analysis.diagnostics["timeline_frames"] = len(final_timeline)
                if timeline_synth_diag:
                    cand_analysis.diagnostics["timeline_synth"] = timeline_synth_diag
                cand_analysis.diagnostics["frame_hop_seconds_source"] = frame_hop_source
                try:
                    pre_resolved = _build_resolved_params(
                        stage_a_out,
                        cand_cfg,
                        stage_b_out=stage_b_out,
                        decision_trace=cand_trace,
                        timeline_source=cand_analysis.diagnostics.get("timeline_source", "unknown"),
                        frame_hop_seconds=float(getattr(cand_analysis, "frame_hop_seconds", 0.0) or 0.0),
                        frame_hop_source=cand_analysis.diagnostics.get("frame_hop_seconds_source", "unknown"),
                        cand_score=0.0,
                        cand_metrics={},
                        candidate_id=cand_id,
                        quality_gate_cfg=qcfg,
                    )
                    cand_analysis.diagnostics["resolved_params"] = pre_resolved
                except Exception:
                    pass

                # Stage C — IMPORTANT: capture returned notes
                notes_pred = apply_theory(cand_analysis, cand_cfg)

                # Ensure notes are actually attached (handles both “returns list” and “mutates in place” implementations)
                if notes_pred is not None:
                    try:
                        cand_analysis.notes = list(notes_pred)
                    except Exception:
                        pass

                if not getattr(cand_analysis, "notes_before_quantization", None):
                    try:
                        cand_analysis.notes_before_quantization = list(getattr(cand_analysis, "notes", []) or [])
                    except Exception:
                        pass

                # Ensure decision trace visible at top-level
                try:
                    cand_analysis.diagnostics = cand_analysis.diagnostics or {}
                    cand_analysis.diagnostics["decision_trace"] = cand_trace
                except Exception:
                    pass

            # Score candidate
            notes_raw = list(cand_analysis.notes_before_quantization or cand_analysis.notes or [])
            timeline_src = cand_analysis.timeline if (cand_analysis.timeline and len(cand_analysis.timeline) > 0) else None
            cand_metrics = _quality_metrics(notes_raw, duration_sec, timeline_source=timeline_src)
            cand_score = _quality_score(cand_metrics)

            try:
                resolved_params = _build_resolved_params(
                    stage_a_out,
                    cand_cfg,
                    stage_b_out=stage_b_out,
                    decision_trace=cand_trace,
                    timeline_source=cand_analysis.diagnostics.get("timeline_source", "unknown") if hasattr(cand_analysis, "diagnostics") else "unknown",
                    frame_hop_seconds=float(getattr(cand_analysis, "frame_hop_seconds", 0.0) or 0.0),
                    frame_hop_source=cand_analysis.diagnostics.get("frame_hop_seconds_source", "unknown") if hasattr(cand_analysis, "diagnostics") else "unknown",
                    cand_score=cand_score,
                    cand_metrics=cand_metrics,
                    candidate_id=cand_id,
                    quality_gate_cfg=qcfg,
                )
                if hasattr(cand_analysis, "diagnostics"):
                    cand_analysis.diagnostics = cand_analysis.diagnostics or {}
                    cand_analysis.diagnostics["resolved_params"] = resolved_params
                    cand_analysis.diagnostics.setdefault("routing_reasons", _extract_routing_reasons(cand_trace))
                    if stage_b_out is not None and getattr(stage_b_out, "diagnostics", None) is not None:
                        stage_b_out.diagnostics["resolved_params"] = resolved_params
            except Exception:
                pass

            high_density = cand_metrics.get("notes_per_sec", 0.0) is not None and float(cand_metrics.get("notes_per_sec", 0.0)) > 12.0
            accepted = (not q_enabled) or (
                int(cand_metrics.get("note_count", 0)) > 0 and cand_score >= q_threshold and (not high_density)
            )
            cand_decision = "accepted" if accepted else "rejected"
            if high_density:
                cand_reason = "note_density_high"
            else:
                cand_reason = "ok" if accepted else "below_threshold"

            if best is None or cand_score > best[1]:
                best = (cand_id, cand_score, cand_analysis, cand_cfg)

            candidates.append(
                {
                    "candidate_id": cand_id,
                    "score": float(cand_score),
                    "metrics": dict(cand_metrics),
                    "decision": cand_decision,
                    "reason": cand_reason,
                }
            )

            if accepted:
                break

            fallbacks_triggered.append(f"fallback_from_{cand_id}")

        except Exception as e:
            fallbacks_triggered.append(f"error_{cand_id}")
            candidates.append(
                {
                    "candidate_id": cand_id,
                    "score": 0.0,
                    "metrics": {
                        "voiced_ratio": 0.0,
                        "note_count": 0,
                        "notes_per_sec": 0.0,
                        "median_note_dur_ms": 0.0,
                        "fragmentation_lt_80ms": 0.0,
                    },
                    "decision": "rejected",
                    "reason": f"error:{type(e).__name__}",
                }
            )

    if best is None:
        analysis_data = AnalysisData(
            meta=stage_a_out.meta,
            timeline=[],
            stem_timelines={},
            notes=[],
            notes_before_quantization=[],
            chords=[],
            diagnostics={},
        )
        selected_id = "none"
        selected_score = 0.0
        selected_cfg = config
    else:
        selected_id, selected_score, analysis_data, selected_cfg = best

    # Mark selected
    for c in candidates:
        if c.get("candidate_id") == selected_id:
            c["decision"] = "selected"

    analysis_data.diagnostics = analysis_data.diagnostics or {}
    analysis_data.diagnostics["quality_gate"] = {
        "enabled": bool(q_enabled),
        "threshold": float(q_threshold),
        "candidates": list(candidates),
        "selected_candidate_id": str(selected_id),
        "fallbacks_triggered": list(fallbacks_triggered),
        "threshold_config": {"threshold": float(q_threshold)},
        "metrics_measured": {c["candidate_id"]: c.get("metrics", {}) for c in candidates if "candidate_id" in c},
        "decision_reason_codes": fallbacks_triggered,
        "invariants_failed": bool(analysis_data.diagnostics.get("health_flags") and "invariants_failed" in analysis_data.diagnostics.get("health_flags", [])),
    }
    # Invariants and sanity checks
    invariants = []
    tb = analysis_data.diagnostics.get("resolved_params", {}).get("timebase", {}) if analysis_data.diagnostics else {}
    time_grid_present = tb.get("time_grid_available", False)
    hop_source = tb.get("frame_hop_seconds_source", "")
    if time_grid_present and hop_source != "stage_b_time_grid":
        invariants.append({"id": "inv_timegrid_precedence", "ok": False, "details": hop_source})
    if tb.get("timeline_source", "unknown") == "unknown":
        invariants.append({"id": "inv_timeline_enum", "ok": False, "details": "unknown_timeline_source"})
    analysis_data.diagnostics["invariants"] = invariants
    invariants_failed = any(not inv.get("ok", True) for inv in invariants)

    # Routing sanity checks / health flags
    health_flags = analysis_data.diagnostics.get("health_flags", [])
    routing_features = analysis_data.diagnostics.get("decision_trace", {}).get("routing_features", {}) if analysis_data.diagnostics else {}
    poly_mean = routing_features.get("polyphony_mean", 0.0) or 0.0
    resolved_mode = analysis_data.diagnostics.get("decision_trace", {}).get("resolved", {}).get("transcription_mode", "")
    if poly_mean >= 2.5 and resolved_mode == "classic_melody":
        health_flags.append("routing_poly_mismatch")
    try:
        selected_metrics = next((c["metrics"] for c in candidates if c.get("candidate_id") == selected_id), {})
    except Exception:
        selected_metrics = {}
    if selected_metrics:
        if selected_metrics.get("voiced_ratio", 1.0) is not None and selected_metrics.get("voiced_ratio", 1.0) < 0.25:
            health_flags.append("voiced_ratio_low")
        if selected_metrics.get("fragmentation_lt_80ms", 0.0) is not None and selected_metrics.get("fragmentation_lt_80ms", 0.0) > 0.55:
            health_flags.append("fragmentation_high")
        if selected_metrics.get("notes_per_sec", 0.0) is not None and selected_metrics.get("notes_per_sec", 0.0) > 12.0:
            health_flags.append("note_density_high")
    if invariants_failed:
        health_flags.append("invariants_failed")
    analysis_data.diagnostics["health_flags"] = health_flags

    if requested_quality_mode:
        analysis_data.diagnostics["requested_quality_mode"] = requested_quality_mode
        analysis_data.diagnostics["meta_processing_mode_preserved"] = meta_processing_mode_preserved
        if meta_processing_mode_val is not None:
            analysis_data.diagnostics["meta_processing_mode_value"] = meta_processing_mode_val

    # B4: Persist beat grid
    if hasattr(analysis_data.meta, "beats") and analysis_data.meta.beats:
        analysis_data.diagnostics["beats"] = list(analysis_data.meta.beats)

    if "decision_trace" not in analysis_data.diagnostics:
        analysis_data.diagnostics["decision_trace"] = _safe_trace(base_trace)

    analysis_data.diagnostics["timing"] = {
        "total_sec": float(time.time() - t0),
        "selected_candidate_score": float(selected_score),
        "selected_candidate_id": str(selected_id),
    }

    # ---------------- Stage D ----------------
    # IMPORTANT: render using the selected candidate config (mode-specific quantize/render knobs may differ later)
    tr = quantize_and_render(analysis_data.notes, analysis_data, selected_cfg, pipeline_logger=pipeline_logger)
    return TranscriptionResult(musicxml=tr.musicxml, analysis_data=analysis_data, midi_bytes=tr.midi_bytes)

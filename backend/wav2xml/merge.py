from __future__ import annotations

from typing import Dict, List

from .config_struct import MergeConfig
from .models import NoteEvent


def _snap_onsets(notes: List[NoteEvent], snap_ms: float) -> None:
    snap_s = snap_ms / 1000.0
    by_pitch: Dict[int, List[NoteEvent]] = {}
    for n in notes:
        by_pitch.setdefault(n.pitch, []).append(n)
    for pitch, bucket in by_pitch.items():
        bucket.sort(key=lambda n: n.start)
        for n in bucket:
            snapped = round(n.start / snap_s) * snap_s
            n.start = snapped


def _iou(a: NoteEvent, b: NoteEvent) -> float:
    start = max(a.start, b.start)
    end = min(a.end, b.end)
    if end <= start:
        return 0.0
    inter = end - start
    union = (a.end - a.start) + (b.end - b.start) - inter
    return inter / union if union > 0 else 0.0


def _dedupe(notes: List[NoteEvent], iou_thresh: float, prefer_longer: bool, weights: Dict[str, float]) -> List[NoteEvent]:
    kept: List[NoteEvent] = []
    notes = sorted(notes, key=lambda n: (n.pitch, n.start))
    for n in notes:
        duplicate = False
        for k in kept:
            if k.pitch != n.pitch:
                continue
            if _iou(k, n) >= iou_thresh:
                s_k = _score_note(k, weights)
                s_n = _score_note(n, weights)
                if prefer_longer and n.duration > k.duration:
                    kept.remove(k)
                    kept.append(n)
                elif s_n > s_k:
                    kept.remove(k)
                    kept.append(n)
                duplicate = True
                break
        if not duplicate:
            kept.append(n)
    return kept


def _gap_merge(notes: List[NoteEvent], gap_ms: float) -> List[NoteEvent]:
    gap_s = gap_ms / 1000.0
    merged: List[NoteEvent] = []
    by_pitch: Dict[int, List[NoteEvent]] = {}
    for n in notes:
        by_pitch.setdefault(n.pitch, []).append(n)
    for pitch, bucket in by_pitch.items():
        bucket.sort(key=lambda n: n.start)
        acc: List[NoteEvent] = []
        for n in bucket:
            if not acc:
                acc.append(n)
                continue
            last = acc[-1]
            if n.start - last.end <= gap_s:
                last.duration = max(n.end, last.end) - last.start
                last.velocity = max(last.velocity, n.velocity)
                last.confidence = max(last.confidence, n.confidence)
            else:
                acc.append(n)
        merged.extend(acc)
    return merged


def _score_note(note: NoteEvent, weights: Dict[str, float]) -> float:
    stem_weight = weights.get(f"stem.{note.stem}", 1.0)
    return (
        weights.get("conf", 1.0) * note.confidence
        + weights.get("dur", 0.0) * note.duration
        + weights.get("vel", 0.0) * note.velocity / 127.0
    ) * stem_weight


def merge_notes(notes: List[NoteEvent], cfg: MergeConfig) -> List[NoteEvent]:
    working = [n for n in notes]
    if not cfg.enabled:
        return working
    _snap_onsets(working, cfg.onset_snap_ms)
    working = _dedupe(working, cfg.dedupe_overlap_iou, cfg.prefer_longer, {"conf": cfg.weights.conf, "dur": cfg.weights.dur, "vel": cfg.weights.vel, **{f"stem.{k}": v for k, v in cfg.weights.stem.items()}})
    working = _gap_merge(working, cfg.gap_merge_ms)
    return sorted(working, key=lambda n: (n.start, n.pitch))


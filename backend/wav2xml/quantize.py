from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .config_struct import QuantizeConfig
from .models import NoteEvent


def _compute_grid_cost(note: NoteEvent, grid_div: int, quarter_note_s: float, cfg: QuantizeConfig) -> Tuple[float, float]:
    ticks_per_second = grid_div / quarter_note_s
    ideal_start = round(note.start * ticks_per_second) / ticks_per_second
    ideal_end = round((note.start + note.duration) * ticks_per_second) / ticks_per_second
    snap_error = abs(note.start - ideal_start) + abs(note.end - ideal_end)
    complexity_penalty = grid_div / max(cfg.allowed_grids)
    return snap_error, complexity_penalty


def quantize_notes(notes: List[NoteEvent], cfg: QuantizeConfig, tempo_bpm: float) -> List[NoteEvent]:
    if not notes:
        return []
    quarter_note_s = 60.0 / float(tempo_bpm)
    quantized: List[NoteEvent] = []
    for n in notes:
        best = None
        best_cost = float("inf")
        for grid in cfg.allowed_grids:
            snap_err, comp_pen = _compute_grid_cost(n, grid, quarter_note_s, cfg)
            cost = snap_err + 0.1 * comp_pen
            if cost < best_cost:
                best_cost = cost
                best = grid
        if best is None:
            best = cfg.allowed_grids[0]
        ticks_per_second = best / quarter_note_s
        start = round(n.start * ticks_per_second) / ticks_per_second
        end = round((n.start + n.duration) * ticks_per_second) / ticks_per_second
        quantized.append(
            NoteEvent(
                pitch=n.pitch,
                start=start,
                duration=max(end - start, 1e-3),
                velocity=n.velocity,
                confidence=n.confidence,
                stem=n.stem,
                channel=n.channel,
                pedal=n.pedal,
                bends=n.bends,
                provenance=dict(n.provenance, quant_grid=str(best)),
            )
        )
    return quantized


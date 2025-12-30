from __future__ import annotations

from typing import List
import importlib
import importlib.util

mido = importlib.import_module("mido") if importlib.util.find_spec("mido") else None

from .models import NoteEvent


def parse_midi(path: str) -> List[NoteEvent]:
    if mido is None:
        raise RuntimeError("mido is required for midi parsing backend 'mido_py'")
    midi = mido.MidiFile(path)
    notes: List[NoteEvent] = []
    time_acc = 0.0
    tempo = 500000  # default 120 bpm
    tempo_stack = [tempo]
    ticks_per_beat = midi.ticks_per_beat
    on_map = {}
    for msg in midi:
        time_acc += mido.tick2second(msg.time, ticks_per_beat, tempo_stack[-1])
        if msg.type == "set_tempo":
            tempo_stack.append(msg.tempo)
        if msg.type == "note_on" and msg.velocity > 0:
            on_map.setdefault((msg.channel, msg.note), []).append((time_acc, msg.velocity))
        if msg.type in {"note_off"} or (msg.type == "note_on" and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if key in on_map and on_map[key]:
                start, vel = on_map[key].pop(0)
                duration = max(time_acc - start, 1e-4)
                notes.append(
                    NoteEvent(
                        pitch=msg.note,
                        start=start,
                        duration=duration,
                        velocity=float(vel),
                        channel=msg.channel,
                        provenance={"source": "midi"},
                    )
                )
    return notes

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import List

from .config_struct import EngraveConfig
from .models import NoteEvent


def _note_to_xml(note: NoteEvent, divisions: int, quarter_length: float) -> ET.Element:
    el = ET.Element("note")
    pitch = ET.SubElement(el, "pitch")
    step_names = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"]
    step = step_names[note.pitch % 12]
    alter = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0][note.pitch % 12]
    octave = note.pitch // 12 - 1
    ET.SubElement(pitch, "step").text = step
    if alter != 0:
        ET.SubElement(pitch, "alter").text = str(alter)
    ET.SubElement(pitch, "octave").text = str(octave)
    duration_divs = int(round((note.duration / quarter_length) * divisions))
    if duration_divs <= 0:
        duration_divs = 1
    ET.SubElement(el, "duration").text = str(duration_divs)
    ET.SubElement(el, "voice").text = "1"
    ET.SubElement(el, "type").text = "quarter"
    dynamics = ET.SubElement(el, "velocity")
    dynamics.text = str(int(note.velocity))
    return el


def notes_to_musicxml(notes: List[NoteEvent], cfg: EngraveConfig, divisions: int, tempo_bpm: float) -> str:
    score = ET.Element("score-partwise", version="3.1")
    part_list = ET.SubElement(score, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Transcription"
    part = ET.SubElement(score, "part", id="P1")

    measure = ET.SubElement(part, "measure", number="1")
    attributes = ET.SubElement(measure, "attributes")
    ET.SubElement(attributes, "divisions").text = str(divisions)
    key = ET.SubElement(attributes, "key")
    ET.SubElement(key, "fifths").text = "0"
    time = ET.SubElement(attributes, "time")
    ET.SubElement(time, "beats").text = "4"
    ET.SubElement(time, "beat-type").text = "4"
    clef = ET.SubElement(attributes, "clef")
    ET.SubElement(clef, "sign").text = "G"
    ET.SubElement(clef, "line").text = "2"
    sound = ET.SubElement(measure, "sound")
    sound.set("tempo", f"{tempo_bpm:.2f}")

    quarter_length = 60.0 / float(tempo_bpm)
    running_measure_time = 0.0
    measure_duration = quarter_length * 4
    current_measure = measure
    sorted_notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    for note in sorted_notes:
        while note.start >= running_measure_time + measure_duration:
            # close current measure and create a new one
            running_measure_time += measure_duration
            current_measure = ET.SubElement(part, "measure", number=str(len(part) + 1))
        current_measure.append(_note_to_xml(note, divisions, quarter_length))

    xml_str = ET.tostring(score, encoding="utf-8", method="xml")
    return xml_str.decode("utf-8")

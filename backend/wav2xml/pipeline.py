from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .audio import canonicalize_audio
from .backend_resolver import BackendResolver
from .beats import estimate_beats
from .command_runner import CommandRunner
from .config_loader import ConfigLoader
from .config_struct import Wav2XmlConfig
from .merge import merge_notes
from .midi_parser import parse_midi
from .models import Candidate, DecisionTrace, NoteEvent, PipelineArtifacts
from .musicxml import notes_to_musicxml
from .quantize import quantize_notes
from .separation import separate_stems
from .transcription_engines import LiteTranscriber, ProTranscriber, TranscriptionOutput


class WavToXmlPipeline:
    def __init__(self, config_dir: str | Path = "backend/config", preset: Optional[str] = None) -> None:
        loader = ConfigLoader()
        self.config: Wav2XmlConfig = loader.load(str(config_dir), preset=preset)
        self.config_provenance = loader.provenance

    def run(self, audio_path: str, *, midi_input: bool = False, workdir: Optional[str] = None) -> PipelineArtifacts:
        candidate_id = self._candidate_id(audio_path)
        workdir_path = Path(workdir or Path("outputs") / "workdir" / candidate_id)
        cache_dir = workdir_path / "cache"
        artifacts_dir = workdir_path / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        runner = CommandRunner(workdir_path / "commands.jsonl")

        resolver = BackendResolver(self.config.backend_policy, self.config.backend_priority, self.config.tools)
        resolved = resolver.resolve()
        decision_trace = DecisionTrace(resolutions=resolver.trace, provenance=self.config_provenance)

        if midi_input:
            notes = parse_midi(audio_path)
            beat_map = estimate_beats(np.zeros(1), self.config.io.sample_rate, resolved["beats"], self.config.beats, self.config.io.fallback_bpm)
        else:
            stage_a = canonicalize_audio(audio_path, self.config.io, self.config.views, runner, workdir_path)
            sep = separate_stems(stage_a.audio, stage_a.sr, self.config.separation.demucs, resolved["separation"], runner, workdir_path)

            notes: List[NoteEvent] = []
            # Piano stems
            piano_backend = ProTranscriber() if resolved["piano_tx"].endswith("py") else LiteTranscriber()
            for stem_name in self.config.separation.demucs.stems.piano_stems:
                if stem_name in sep.stems:
                    y, sr = sep.stems[stem_name]
                    notes.extend(piano_backend.transcribe(y, sr, stem_name).notes)
            # Optional vocal stem
            if self.config.separation.demucs.stems.vocal_stem in sep.stems:
                y, sr = sep.stems[self.config.separation.demucs.stems.vocal_stem]
                vocal_backend = ProTranscriber() if resolved["vocal_tx"].endswith("py") else LiteTranscriber()
                notes.extend(vocal_backend.transcribe(y, sr, "vocals").notes)
            if not notes and "mix" in sep.stems:
                y, sr = sep.stems["mix"]
                notes.extend(LiteTranscriber().transcribe(y, sr, "mix").notes)
            beat_map = estimate_beats(stage_a.audio[0], stage_a.sr, resolved["beats"], self.config.beats, self.config.io.fallback_bpm)

        merged_notes = merge_notes(notes, self.config.merge)
        quantized_notes = quantize_notes(merged_notes, self.config.quantize, beat_map.tempo_bpm)

        musicxml = notes_to_musicxml(quantized_notes, self.config.engrave, self.config.io.divisions, beat_map.tempo_bpm)

        # Candidate scoring (simplified)
        score = self._score_candidate(merged_notes, beat_map, quantized_notes)
        candidate = Candidate(notes=quantized_notes, beat_map=beat_map, score=score, quantized=True, metadata={"backend_quantize": resolved["quantize"]})
        decision_trace.effective_params["quantize"] = {"allowed_grids": float(len(self.config.quantize.allowed_grids))}

        # Artifacts
        notes_path = artifacts_dir / "notes_raw.json"
        timeline_path = artifacts_dir / "timeline.json"
        meta_path = artifacts_dir / "meta.json"

        with notes_path.open("w", encoding="utf-8") as f:
            json.dump([note.__dict__ for note in merged_notes], f, indent=2)

        with timeline_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "beats": beat_map.beats,
                    "tempo_bpm": beat_map.tempo_bpm,
                    "backend": beat_map.backend,
                },
                f,
                indent=2,
            )

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "decision_trace": [r.__dict__ for r in decision_trace.resolutions],
                    "provenance": decision_trace.provenance,
                    "candidate_score": candidate.score,
                    "resolved_backends": resolved,
                    "candidate_id": candidate_id,
                },
                f,
                indent=2,
            )

        output_musicxml_path = workdir_path / "output.musicxml"
        output_musicxml_path.write_text(musicxml, encoding="utf-8")

        return PipelineArtifacts(
            musicxml=musicxml,
            notes_raw_path=str(notes_path),
            timeline_path=str(timeline_path),
            meta_path=str(meta_path),
            commands_log=str(workdir_path / "commands.jsonl"),
            notes=quantized_notes,
            beat_map=beat_map,
        )

    def _score_candidate(self, notes: List[NoteEvent], beat_map, quantized: List[NoteEvent]) -> float:
        if not notes:
            return 1e6
        micro_ratio = sum(1 for n in notes if n.duration < 0.05) / len(notes)
        chordiness = len({(n.start, n.pitch) for n in notes}) / max(len(notes), 1)
        quant_shift = np.mean([abs(q.start - n.start) for q, n in zip(quantized, notes[: len(quantized)])]) if quantized else 0.0
        return float(micro_ratio * 5 + (1 - chordiness) + quant_shift)

    def _candidate_id(self, audio_path: str) -> str:
        h = hashlib.sha256()
        h.update(Path(audio_path).resolve().as_posix().encode("utf-8"))
        try:
            with open(audio_path, "rb") as f:
                h.update(f.read(4096))
        except Exception:
            pass
        h.update(json.dumps(self.config_provenance, sort_keys=True).encode("utf-8"))
        return h.hexdigest()[:16]

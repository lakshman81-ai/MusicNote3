# WAV → MusicXML pipeline (LITE + PRO)

This module implements the two-path pipeline described in the WI: a dependency-light **LITE** flow and an optional **PRO** flow selected via the backend resolver. The API surface is intentionally small:

* `WavToXmlPipeline.run(audio_path, midi_input=False)` → returns artifacts + MusicXML
* Config files live in `backend/config/*.toml` (default + presets)
* Artifacts are written to `outputs/workdir/<candidate_id>/`

## Key pieces

* **Strict config loader** (`config_loader.py`): merges `default.toml` + presets + overrides, errors on unknown keys, tracks provenance for meta logging.
* **Backend resolver** (`backend_resolver.py`): classifies backends as `available` / `healthy` and chooses according to policy + priority lists. Decisions are recorded in `meta.json`.
* **Command logging** (`command_runner.py`): every external call (ffmpeg/demucs placeholder) is appended to `commands.jsonl`.
* **Canonical audio prep** (`audio.py`): ffmpeg-aligned canonical WAV + stereo views. Writes `base.wav` and logs argv.
* **Separation** (`separation.py`): demucs stub with alignment invariant—stems are length-matched to the mix.
* **Transcription engines** (`transcription_engines.py`): LITE (dependency-light) + PRO (placeholder) share a common interface; both rely on deterministic librosa onsets/pyin for now.
* **Merge + quantize** (`merge.py`, `quantize.py`): implements onset snap → IoU dedupe → gap merge, then cost-based grid selection.
* **Beats** (`beats.py`): prefers tracker when available, falls back to constant BPM.
* **MusicXML writer** (`musicxml.py`): dependency-free MusicXML export tuned for readability.

## Outputs (per run)

* `output.musicxml` – readable MusicXML
* `artifacts/notes_raw.json` – merged, pre-quantization notes (seconds)
* `artifacts/timeline.json` – tempo/beat backend + beat times
* `artifacts/meta.json` – decision trace, provenance, candidate id
* `cache/` – deterministic cache directory ready for future reuse
* `commands.jsonl` – tail of external invocations

The design leaves hooks for true PRO backends (demucs/python models, music21, etc.) to be slotted in without API changes. Tests should exercise resolver behavior, merge stability, and quantization bounds as outlined in the WI checklist.

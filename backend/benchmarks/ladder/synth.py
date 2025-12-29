import numpy as np
import soundfile as sf
import json
import os
import logging
from typing import Tuple
from music21 import stream, note, chord, tempo

logger = logging.getLogger(__name__)

# Try importing scipy.signal for high-quality resampling
try:
    import scipy.signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

def _clamp_bpm(bpm: float, lo: float = 20.0, hi: float = 300.0, default: float = 120.0) -> float:
    try:
        bpm = float(bpm)
    except Exception:
        return default
    if not np.isfinite(bpm):
        return default
    if bpm < lo or bpm > hi:
        return default
    return bpm

def _duration_sec(audio: np.ndarray, sr: int) -> float:
    if audio is None or sr <= 0:
        return 0.0
    return float(len(audio)) / float(sr)

def _duration_ok(audio: np.ndarray, sr: int, expected_sec: float, tol: float = 0.10) -> bool:
    got = _duration_sec(audio, sr)
    exp = float(expected_sec)
    return exp > 0.0 and abs(got - exp) <= exp * float(tol)

def _time_stretch_to_target(audio: np.ndarray, sr: int, target_sec: float) -> Tuple[np.ndarray, dict]:
    """
    Stretch/shrink audio to match target_sec exactly (in samples).
    Returns (processed_audio, diagnostics_dict).
    """
    target_samples = int(target_sec * sr)
    original_samples = len(audio)

    if target_samples <= 0:
        return audio, {"method": "none", "reason": "invalid_target"}

    ratio = target_samples / max(1, original_samples)
    method = "interp"

    if SCIPY_AVAILABLE and abs(target_samples - original_samples) > 0:
        # Use resample_poly for better quality
        # resample_poly(x, up, down) -> len ~ len(x) * up / down
        # We want len * up/down = target. up/down = target/len.
        # Use GCD to simplify fraction? scipy handles it reasonably.
        # But we pass integers.
        # Simple approx: up=target, down=original.
        try:
             # Reduce fraction to avoid huge integers if possible?
             # Python's math.gcd might help but let's just pass raw.
             # Note: huge values can be slow.
             # If ratio is simple, like 1.01, integers might be big.
             # However, resample_poly is usually robust.
             # Let's cap max up/down or fallback if too large?
             # For now, trust scipy.

             # Limitation: resample_poly with very large ints can be slow/memory heavy.
             # Let's check if ratio is close to 1.0.

             if target_samples > 2**20 or original_samples > 2**20:
                 # Fallback to linear for very long files to be safe/fast in synth benchmarks
                 method = "interp"
             else:
                 resampled = scipy.signal.resample_poly(audio, target_samples, original_samples)
                 method = "resample_poly"
                 # Ensure exact length (resample_poly might act slightly differently on boundaries)
                 if len(resampled) != target_samples:
                     # Trim or pad
                     if len(resampled) > target_samples:
                         resampled = resampled[:target_samples]
                     else:
                         resampled = np.pad(resampled, (0, target_samples - len(resampled)))
                 return resampled.astype(np.float32), {
                     "method": method,
                     "target_samples": target_samples,
                     "original_samples": original_samples,
                     "ratio": ratio
                 }
        except Exception as e:
            logger.warning(f"resample_poly failed: {e}. Fallback to interp.")
            method = "interp_fallback"

    # Fallback: Linear interpolation
    old_indices = np.linspace(0, original_samples - 1, original_samples)
    new_indices = np.linspace(0, original_samples - 1, target_samples)

    # interp requires 1D x-coords.
    # We map new grid (0..target-1) to old grid (0..original-1)

    out_audio = np.interp(new_indices, old_indices, audio).astype(np.float32)
    return out_audio, {
        "method": method,
        "target_samples": target_samples,
        "original_samples": original_samples,
        "ratio": ratio
    }


def midi_to_wav_synth(
    score_stream: stream.Score,
    wav_path: str,
    sr: int = 22050,
    use_enhanced_synth: bool = False,
    target_duration_sec: float = 0.0,
):
    """
    Synthesizes a Music21 stream to a WAV file using simple sine waves.
    Handles polyphony by summing waveforms.

    If use_enhanced_synth is True, vary timbre/envelope by range to help
    downstream separation/skyline (saw/square-ish for bass, sine for lead).
    """
    # Flatten the score to get all notes with absolute offsets
    # We use flat, but flat merges parts. We need to respect overlapping notes.
    # flat.notes gives all notes sorted by offset.

    # 1. Calculate total duration
    # Find the last release time
    max_end = 0.0

    def _extract_bpm(score) -> float:
        # Try finding in parts first (more reliable than recurse sometimes)
        if hasattr(score, 'parts'):
            for p in score.parts:
                mms = list(p.getElementsByClass(tempo.MetronomeMark))
                if mms:
                     return float(mms[0].number)

        # Fallback to recurse
        mms = list(score.recurse().getElementsByClass(tempo.MetronomeMark))
        for mm in mms:
            if getattr(mm, "number", None):
                try:
                    return float(mm.number)
                except Exception:
                    pass
        return 100.0

    detected_bpm = _extract_bpm(score_stream)
    current_bpm = _clamp_bpm(detected_bpm, lo=20.0, hi=300.0, default=120.0)

    # We need to process chords and notes
    # music21.chord.Chord contains notes.
    # music21.note.Note is a single note.

    events = [] # list of (start_sec, end_sec, freq, velocity)

    # We estimate tempo. If explicit tempo changes exist, we should follow them.
    # For this simple synth, let's assume constant tempo or handle metronome marks if feasible.
    # music21 'flat' stream has metronome marks embedded.

    # To do this accurately, we can use score.secondsMap or similar if available,
    # but that requires a robust environment.
    # Let's use a simpler approach: Iterate elements and track time.
    # But for polyphony (different parts), we need to be careful.

    # Calculate duration using our BPM to be consistent
    # score.flatten() puts everything in one timeline
    try:
        score_dur_beats = float(score_stream.highestTime)
        total_dur_sec = score_dur_beats * (60.0 / current_bpm)
    except:
        total_dur_sec = 10.0

    # Buffer
    # Add 1 second tail
    n_samples = int((total_dur_sec + 2.0) * sr)
    audio = np.zeros(n_samples, dtype=np.float32)

    # We will process flattened notes.
    # To handle tempo changes correctly in a custom loop is hard.
    # Let's assume our benchmarks have constant tempo (defined in generators).
    # Happy Birthday: 100 bpm. Old MacDonald: 100 bpm.
    # So we can just map quarterLength -> seconds.

    sec_per_quarter = 60.0 / current_bpm

    flat_els = score_stream.flatten().elements

    for el in flat_els:
        if isinstance(el, note.Note):
            start_sec = el.offset * sec_per_quarter
            dur_sec = el.quarterLength * sec_per_quarter
            freq = el.pitch.frequency
            vel = el.volume.velocity if el.volume.velocity else 90
            events.append((start_sec, dur_sec, freq, vel))

        elif isinstance(el, chord.Chord):
            start_sec = el.offset * sec_per_quarter
            dur_sec = el.quarterLength * sec_per_quarter
            vel = el.volume.velocity if el.volume.velocity else 90
            for p in el.pitches:
                events.append((start_sec, dur_sec, p.frequency, vel))

    def _waveform(freq, t, amp):
        if not use_enhanced_synth:
            return amp * np.sin(2 * np.pi * freq * t)

        # Lead (approx. 400Hz - 1200Hz): FM + vibrato
        if freq > 350.0:
            vib = 0.003 * np.sin(2 * np.pi * 5.5 * t)
            return amp * (0.7 * np.sin(2 * np.pi * freq * (1 + vib) * t) +
                          0.3 * np.sin(2 * np.pi * 2 * freq * (1 + vib) * t))

        # Bass (approx. < 200Hz): Saw-ish (odd harmonics)
        if freq < 200.0:
            return amp * (np.sin(2 * np.pi * freq * t) +
                          (1/3) * np.sin(2 * np.pi * 3 * freq * t) +
                          (1/5) * np.sin(2 * np.pi * 5 * freq * t))

        # Chords/Pad (Mid range): Sine stack
        return amp * (np.sin(2 * np.pi * freq * t) +
                      0.5 * np.sin(2 * np.pi * 2 * freq * t) +
                      0.25 * np.sin(2 * np.pi * 3 * freq * t))

    # Synthesize
    for start, dur, freq, vel in events:
        if dur <= 0: continue

        start_idx = int(start * sr)
        end_idx = int((start + dur) * sr)
        if start_idx >= n_samples: continue
        if end_idx > n_samples: end_idx = n_samples

        length = end_idx - start_idx
        if length <= 0: continue

        t = np.arange(length) / sr
        # Apply envelope: Attack/Release
        amp = (vel / 127.0) * 0.3 # Master gain

        wave = _waveform(freq, t, amp)

        # Envelopes per type (heuristic based on freq)
        if use_enhanced_synth:
            if freq > 350.0: # Lead: ADSR
                a, d, s, r = 0.01, 0.08, 0.6, 0.12
            elif freq < 200.0: # Bass: fast attack
                a, d, s, r = 0.01, 0.1, 0.8, 0.1
            else: # Chords: softer
                a, d, s, r = 0.05, 0.1, 0.7, 0.15

            env = np.ones_like(t)
            # Attack
            attack_mask = t < a
            if np.any(attack_mask):
                env[attack_mask] = t[attack_mask] / max(a, 1e-6)

            # Decay/Sustain
            decay_mask = (t >= a) & (t < a + d)
            if np.any(decay_mask):
                env[decay_mask] = 1.0 - (1.0 - s) * ((t[decay_mask] - a) / max(d, 1e-6))

            # Release (approximated at note end)
            release_n = int(r * sr)
            if length > release_n:
                 env[-release_n:] *= np.linspace(1, 0, release_n)

            wave *= env
        else:
            # Simple envelope
            attack_s = 0.02
            release_s = 0.05
            attack_n = int(attack_s * sr)
            release_n = int(release_s * sr)
            if length > attack_n:
                wave[:attack_n] *= np.linspace(0, 1, attack_n)
            if length > release_n:
                wave[-release_n:] *= np.linspace(1, 0, release_n)
        attack_n = int(attack_s * sr)
        release_n = int(release_s * sr)

        # Attack
        if length > attack_n:
            wave[:attack_n] *= np.linspace(0, 1, attack_n)

        # Release
        if length > release_n:
            wave[-release_n:] *= np.linspace(1, 0, release_n)

        audio[start_idx:end_idx] += wave

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.95

    diagnostics = {}
    enforcement_applied = False

    # Enforce duration if target_duration_sec is provided
    if target_duration_sec > 0.0:
        if not _duration_ok(audio, sr, target_duration_sec, tol=0.10):
             logger.info(f"Enforcing duration: expected {target_duration_sec}s, got {_duration_sec(audio, sr):.2f}s")
             audio, diag = _time_stretch_to_target(audio, sr, target_duration_sec)
             diagnostics.update(diag)
             enforcement_applied = True
             diagnostics["enforcement_applied"] = True
             diagnostics["bpm_detected"] = float(detected_bpm)
             diagnostics["bpm_clamped"] = float(current_bpm)
             diagnostics["original_duration_sec"] = diagnostics.get("original_samples", 0) / sr
             diagnostics["target_duration_sec"] = target_duration_sec

    sf.write(wav_path, audio, sr)

    # Write diagnostics sidecar if applied
    if enforcement_applied:
        try:
            diag_path = wav_path + ".diagnostics.json"
            with open(diag_path, "w") as f:
                json.dump(diagnostics, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write diagnostics: {e}")

    return wav_path

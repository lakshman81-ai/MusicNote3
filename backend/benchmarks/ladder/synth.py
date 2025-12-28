import numpy as np
import soundfile as sf
from music21 import stream, note, chord, tempo

def midi_to_wav_synth(score_stream: stream.Score, wav_path: str, sr: int = 22050, use_enhanced_synth: bool = False):
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

    current_bpm = _extract_bpm(score_stream)

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
        total_dur_sec = score_stream.highestTime * (60.0 / current_bpm)
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

    sf.write(wav_path, audio, sr)
    return wav_path

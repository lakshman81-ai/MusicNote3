
export interface NoteLabel {
  display: string;
  isAccidental: boolean;
  octave?: number;
  meta?: {
    solfegeFallback?: boolean;
    reason?: string;
  };
}

export interface SolfegeSettings {
  format: 'scientific' | 'note_only' | 'solfege';
  accidentalStyle: 'sharp' | 'flat' | 'double_sharp';
  showOctave: boolean;
  solfegeMode?: 'fixed' | 'movable';
  keyContext?: {
    tonic: string; // e.g., 'C', 'F#', 'Bb'
    mode?: 'major' | 'minor' | string;
  };
}

// Helpers for solfege
const PITCH_CLASS_MAP: { [key: string]: number } = {
  'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
  'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8,
  'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
};

// Chromatic syllables
// Index 0..11
const SOLFEGE_FIXED_SHARP = ['Do', 'Di', 'Re', 'Ri', 'Mi', 'Fa', 'Fi', 'Sol', 'Si', 'La', 'Li', 'Ti'];
const SOLFEGE_FIXED_FLAT  = ['Do', 'Ra', 'Re', 'Me', 'Mi', 'Fa', 'Se', 'Sol', 'Le', 'La', 'Te', 'Ti'];

function getPitchClass(noteName: string): number {
  return PITCH_CLASS_MAP[noteName] ?? 0;
}

export const formatPitch = (
  midiPitch: number,
  settings: SolfegeSettings
): NoteLabel => {
  const noteNamesSharp = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const noteNamesFlat = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'];
  
  if (!Number.isFinite(midiPitch)) {
    return { display: '?', isAccidental: false };
  }

  const roundedPitch = Math.round(midiPitch);
  const octave = Math.floor(roundedPitch / 12) - 1;
  const semitone = ((roundedPitch % 12) + 12) % 12;

  let baseName = '';
  let isAccidental = false;
  let meta: NoteLabel['meta'] = undefined;

  if (settings.format === 'solfege') {
    const mode = settings.solfegeMode || 'fixed';
    const useFlat = settings.accidentalStyle === 'flat';
    const syllables = useFlat ? SOLFEGE_FIXED_FLAT : SOLFEGE_FIXED_SHARP;

    if (mode === 'movable') {
      if (!settings.keyContext || !settings.keyContext.tonic) {
        // Fallback
        baseName = syllables[semitone];
        isAccidental = useFlat ? baseName.endsWith('e') || baseName.endsWith('a') : baseName.endsWith('i');
        // Heuristic: Do, Re, Mi, Fa, Sol, La, Ti are diatonic (mostly).
        // Ra, Me, Se, Le, Te, Di, Ri, Fi, Si, Li are chromatic.
        // Actually, let's just use the fixed list.
        meta = { solfegeFallback: true, reason: 'missing_context' };
      } else {
        // Movable logic
        const tonicPC = getPitchClass(settings.keyContext.tonic);
        let interval = (semitone - tonicPC + 12) % 12;

        // Handle Minor Mode (La-based)
        // If minor, Tonic matches La. So relative major tonic is Tonic + 3 semitones.
        // e.g. A Minor. Tonic=A(9). Relative Major=C(0).
        // Note A(9). Interval from C = 9. 9 corresponds to La.
        if (settings.keyContext.mode === 'minor') {
          // Shift interval to be relative to the Relative Major
          // In La-based minor, the tonic is La (9).
          // So we need to shift such that 0 (Tonic) becomes 9.
          // (0 + 9) % 12 = 9.
          interval = (interval + 9) % 12;
        }

        baseName = syllables[interval];
      }
    } else {
      // Fixed Do
      baseName = syllables[semitone];
    }

    // Determine accidental status for solfege
    // Diatonic: Do, Re, Mi, Fa, Sol, La, Ti
    const diatonic = new Set(['Do', 'Re', 'Mi', 'Fa', 'Sol', 'La', 'Ti']);
    isAccidental = !diatonic.has(baseName);

  } else {
    // Scientific / Note Only
    const useSharps = settings.accidentalStyle !== 'flat';
    const names = useSharps ? noteNamesSharp : noteNamesFlat;
    baseName = names[semitone];
    
    if (!baseName) {
        baseName = '?';
    } else {
        isAccidental = baseName.includes('#') || baseName.includes('b');
        
        if (settings.accidentalStyle === 'double_sharp' && isAccidental && useSharps) {
            if (baseName.includes('#')) baseName = baseName.replace('#', 'x');
        } else if (settings.accidentalStyle === 'flat' && isAccidental && !useSharps) {
           if (baseName.includes('b')) baseName = baseName.replace('b', '♭');
        } else if (settings.accidentalStyle === 'sharp' && isAccidental && useSharps) {
           if (baseName.includes('#')) baseName = baseName.replace('#', '♯');
        }
    }
  }

  let display = baseName;
  if (settings.showOctave && settings.format !== 'note_only') {
    display += octave;
  }

  const result: NoteLabel = { display, isAccidental, octave };
  if (meta) {
    result.meta = meta;
  }
  return result;
};

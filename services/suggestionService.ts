import { NoteEvent } from '../types';
import { STYLES, VOICES } from '../components/constants';

export interface SuggestedSettings {
  voice: string;
  style: string;
  bpm: number;
}

export const SuggestionService = {
  generateSuggestions(notes: NoteEvent[]): SuggestedSettings | null {
    if (notes.length < 10) {
      return null;
    }

    const validVoices = new Set(VOICES.map(v => v.id));
    const validStyles = new Set(STYLES.map(s => s.id));
    const mapVoice = (voice: string) => {
      if (validVoices.has(voice)) return voice;
      if (voice === 'synth_bass') return 'synth_lead';
      if (voice === 'grand_piano') return 'piano';
      return 'piano';
    };

    const mapStyle = (style: string) => {
      if (validStyles.has(style)) return style;
      if (style === 'funk') return 'r_n_b';
      if (style === 'pop') return 'beat_16';
      if (style === 'techno') return 'dance';
      return 'none';
    };

    const clampToValid = (settings: SuggestedSettings): SuggestedSettings => ({
      voice: mapVoice(settings.voice),
      style: mapStyle(settings.style),
      bpm: settings.bpm,
    });

    const averagePitch = notes.reduce((sum, note) => sum + note.midi_pitch, 0) / notes.length;
    const averageDuration = notes.reduce((sum, note) => sum + note.duration, 0) / notes.length;
    const span = notes[notes.length - 1].start_time - notes[0].start_time;
    const rawDensity = span > 0 ? notes.length / span : 0;
    const noteDensity = Number.isFinite(rawDensity) ? rawDensity : 0;

    if (averagePitch < 48 && averageDuration > 0.4) {
      return clampToValid({
        voice: 'synth_bass',
        style: 'funk',
        bpm: 95,
      });
    } else if (averagePitch > 65 && noteDensity > 5) {
      return clampToValid({
        voice: 'piano',
        style: 'pop',
        bpm: 125,
      });
    } else if (noteDensity > 8 && averageDuration < 0.2) {
        return clampToValid({
          voice: 'synth_lead',
          style: 'techno',
          bpm: 145,
        });
    } else {
      return clampToValid({
        voice: 'grand_piano',
        style: 'ballad',
        bpm: 80,
      });
    }
  },
};

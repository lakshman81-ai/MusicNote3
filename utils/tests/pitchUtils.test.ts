
import { test, expect } from '@playwright/test';
import { formatPitch } from '../pitchUtils';

test.describe('formatPitch Solfege', () => {

  test('Fixed-Do basic mapping', () => {
    // C4 (60) -> Do
    const c4 = formatPitch(60, { format: 'solfege', accidentalStyle: 'sharp', showOctave: true, solfegeMode: 'fixed' });
    expect(c4.display).toBe('Do4');

    // C#4 (61) -> Di
    const cs4 = formatPitch(61, { format: 'solfege', accidentalStyle: 'sharp', showOctave: true, solfegeMode: 'fixed' });
    expect(cs4.display).toBe('Di4');

    // Db4 (61) -> Ra (if implemented) or fallback?
    // The current fixed-do logic in plan says: "If flat preference: Do/Ra/Re/Me..."
    const db4 = formatPitch(61, { format: 'solfege', accidentalStyle: 'flat', showOctave: true, solfegeMode: 'fixed' });
    expect(db4.display).toBe('Ra4');
  });

  test('Movable-Do Major', () => {
    // Key D Major. D is Do.
    const context = { tonic: 'D', mode: 'major' };

    // D4 (62) -> Do
    const d4 = formatPitch(62, { format: 'solfege', accidentalStyle: 'sharp', showOctave: false, solfegeMode: 'movable', keyContext: context });
    expect(d4.display).toBe('Do');

    // F#4 (66) -> Mi (D + 4 semi = F#)
    const fs4 = formatPitch(66, { format: 'solfege', accidentalStyle: 'sharp', showOctave: false, solfegeMode: 'movable', keyContext: context });
    expect(fs4.display).toBe('Mi');
  });

  test('Movable-Do Minor (La-based)', () => {
    // A Minor. Relative Major is C Major.
    // A (57) -> La.
    const context = { tonic: 'A', mode: 'minor' };

    // A3 (57)
    const a3 = formatPitch(57, { format: 'solfege', accidentalStyle: 'sharp', showOctave: false, solfegeMode: 'movable', keyContext: context });
    expect(a3.display).toBe('La');

    // C4 (60) -> Do
    const c4 = formatPitch(60, { format: 'solfege', accidentalStyle: 'sharp', showOctave: false, solfegeMode: 'movable', keyContext: context });
    expect(c4.display).toBe('Do');
  });

  test('Fallback to Fixed-Do', () => {
    // Movable requested but no context
    const res = formatPitch(60, { format: 'solfege', accidentalStyle: 'sharp', showOctave: false, solfegeMode: 'movable' });
    expect(res.display).toBe('Do'); // C -> Do in fixed
    expect(res.meta?.solfegeFallback).toBe(true);
  });
});

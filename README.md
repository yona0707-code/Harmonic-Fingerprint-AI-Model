# Harmonic Fingerprint AI Ver2

This is an isolated, pre-training audio workspace for the finalized four-category experiment. The working Streamlit/Supabase project at `/Users/yonainoue/Emotions` is read-only and is not imported as code.

## Active source experiment

The four copied references are:

| Class | Original WAV | Formula/voicing |
| --- | --- | --- |
| major | `kaeru_basic_major.wav` | C3-E3-G3, F3-A3-C4, G3-B3-D4 |
| minor | `kaeru_basic_minor.wav` | C3-Eb3-G3, F3-Ab3-C4, G3-Bb3-D4 |
| ninth | `kaeru_ninth_rich_major.wav` | The exact rich major voicings from `generate_kaeru.py`: major/seventh chord plus the specified ninth, including Cmaj7+D4, E7+F#4, Am7+B3, C7+D4, F+G4, Dm7+E4, G+A4, C+D4 |
| diminished | `kaeru_diminished_seventh.wav` | C3-Eb3-Gb3-A3, F3-Ab3-B3-D4, G3-Bb3-Db4-E4 |

All references and generated audio use 44,100 Hz and 17.142857 seconds (32 beats at 112 BPM). The generator uses the same GeneralUser-GS acoustic-grand SoundFont, stereo FluidSynth rendering, shared attack/rest timing, and overlapping gain variation. It does not train a model or create a feature CSV.

## Commands

```bash
python3 scripts/generate_training_audio.py
python3 scripts/generate_training_audio.py --validate
python3 scripts/compare_domains.py
```

Generated WAVs are under `data/synthetic/{major,minor,ninth,diminished}`. Diagnostics are under `reports/`.
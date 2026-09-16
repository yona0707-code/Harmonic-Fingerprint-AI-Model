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

## Development Journey: From Version 1 to Version 2

This was my first time building a machine-learning model from scratch, and the process involved much more trial and error than I initially expected.

My first goal was to train an AI model to identify chord categories directly from audio. I originally used seven classes:

- Basic major
- Basic minor
- Seventh major
- Seventh minor
- Ninth major
- Ninth minor
- Diminished seventh

### Version 1: A Model That Worked on Paper, but Failed in Reality

For Version 1, I generated a synthetic training dataset and trained a Random Forest classifier.

The result initially looked promising:

**Synthetic held-out accuracy: 89.76%**

However, when I tested the model on the actual WAV files used in my Harmonic Fingerprint experiment, it correctly classified only:

**1 out of 7 real audio files**

Most of the real sounds were incorrectly predicted as `ninth_minor`.

At first, I thought the problem might be the model itself or that some chord categories were too similar. Instead of immediately retraining it, I compared the synthetic training audio with the real experiment audio.

That revealed a much bigger problem: the two datasets came from very different audio domains.

| Feature | Real experiment audio | Synthetic V1 audio |
| --- | ---: | ---: |
| Sample rate | 44,100 Hz | 22,050 Hz |
| Duration | ~17.14 s | ~5-7 s |
| RMS / loudness | ~0.011-0.015 | ~0.22-0.28 |
| Spectral centroid | ~818-940 Hz | ~228-620 Hz |
| Spectral bandwidth | ~1,426-1,634 Hz | ~215-325 Hz |

The synthetic files were shorter, much louder, and spectrally very different from the actual experiment recordings.

The model had therefore learned patterns from a synthetic environment that did not represent the sounds it was later asked to classify.

This was my first major lesson from the project:

> High validation accuracy does not necessarily mean a model will work on real data.

### Version 2: Rebuilding the Pipeline

Rather than continuing to modify Version 1, I restarted the AI project from scratch.

I also simplified the classification problem to the four chord categories used in the final experiment:

- Major
- Minor
- Ninth
- Diminished

This time, I designed the synthetic dataset around the actual experiment audio.

The new training files use:

- 44,100 Hz audio
- the same ~17.14 second duration
- the same FluidSynth piano rendering system
- the same timing and silence structure
- similar amplitude ranges
- controlled variations in root, octave, voicing, velocity, and gain

I generated:

**100 variations x 4 chord classes = 400 training WAV files**

Before training anything, I compared the synthetic and real audio domains.

I also focused the AI features more strongly on harmonic information such as:

- Chroma
- CQT pitch-class energy
- Root-normalized harmonic structure

rather than allowing the model to rely on irrelevant properties such as file duration or overall volume.

### Another Failure: Invalid Synthetic Chords

During validation, I discovered another problem.

My generator was forcing individual MIDI notes into a fixed piano range. In some transpositions, one note of a chord would be shifted independently from the others.

This accidentally changed the interval structure of the chord.

Out of 400 generated files:

**103 contained invalid voicings.**

Instead of ignoring them, I changed the generator so that complete chord voicings are shifted together by whole octaves. Every chord is now musically validated before it is rendered.

After regeneration:

**Invalid generated voicings: 0 / 400**

I then performed transposition-invariant chroma and CQT comparisons. For all four real reference sounds, the intended synthetic chord class ranked first.

### Final Version

I trained a Random Forest classifier using 40 primarily harmonic audio features.

#### Synthetic validation

**Accuracy: 100%**

| True / Predicted | Major | Minor | Ninth | Diminished |
| --- | ---: | ---: | ---: | ---: |
| Major | 20 | 0 | 0 | 0 |
| Minor | 0 | 20 | 0 | 0 |
| Ninth | 0 | 0 | 20 | 0 |
| Diminished | 0 | 0 | 0 | 20 |

#### Experiment reference audio

The frozen model was then tested on the four original WAV files used in the Harmonic Fingerprint listening experiment.

| Actual | Predicted |
| --- | --- |
| Major | Major (correct) |
| Minor | Minor (correct) |
| Ninth | Ninth (correct) |
| Diminished | Diminished (correct) |

**4 / 4 reference sounds classified correctly.**

Because these four reference recordings were also used while checking and designing the synthetic-data domain, I do not treat this result as proof that the model will generalize to every possible recording of these chords. Testing on completely independent recordings would be the next step.

### What I Learned

The most important part of this project was not the final accuracy.

Version 1 taught me that a machine-learning model can achieve a high test score while learning from data that does not represent the real problem.

Version 2 taught me to check the data before trusting the model:

**generate -> inspect -> compare -> diagnose -> fix -> validate -> train**

As my first machine-learning project, much of the work involved finding mistakes in my own assumptions. The final model succeeded not because I immediately found the correct algorithm, but because each failed test showed me what needed to be redesigned.

## Commands

```bash
python3 scripts/generate_training_audio.py
python3 scripts/generate_training_audio.py --validate
python3 scripts/compare_domains.py
```

Generated WAVs are under `data/synthetic/{major,minor,ninth,diminished}`. Diagnostics are under `reports/`.
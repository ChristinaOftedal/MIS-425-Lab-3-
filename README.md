# MIS-425-Lab-3-

MIS 425 Lab 3 Speech and Emotion

Option 1: run a different (stronger) pre-trained Hugging Face SER model on RAVDESS
and compare it against the baseline wav2vec2 model we used in class.

## Setup

```bash
pip install -r requirements.txt
```

## Audio files

The RAVDESS `.wav` files live in the `Actor_01`, `Actor_02`, `Actor_04` and
`Actor_07` folders in this repo. That is 4 actors x 60 clips = 240 clips.

## How to reproduce our results

Run the baseline model from class:

```bash
python run_emotion_model.py --model ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition --out baseline_results.csv
```

Run the second model (put the model you chose here):

```bash
python run_emotion_model.py --model YOUR_MODEL_NAME_HERE --out new_model_results.csv
```

Score both and build the comparison table:

```bash
python evaluate_results.py --results baseline_results.csv new_model_results.csv --names Baseline NewModel
```

That last command prints accuracy, precision, recall and F1, prints the most
common mix-ups, prints accuracy per actor, prints example wrong predictions for
the error analysis, and writes these files:

- `confusion_matrix_Baseline.png`
- `confusion_matrix_NewModel.png`
- `comparison_table.csv`

## Dataset split

There is no train / validation / test split, because Option 1 does not train
anything. Both models are pre-trained and we only run them in inference mode
(zero-shot), so every clip is a test clip.

- Test set: all 240 clips (Actor_01, Actor_02, Actor_04, Actor_07)
- Training set: none
- Validation set: none

Our earlier results only covered Actor_01, Actor_02 and Actor_04 (180 clips) and
left out Actor_07 by mistake. The scripts here pick up every `Actor_*` folder, so
all 240 clips get scored.

## Results we got

| Model | Accuracy | Average Confidence | Clips Scored |
|---|---|---|---|
| Baseline (`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`) | 0.0292 | 0.1406 | 240 |
| `superb/wav2vec2-base-superb-er` | 0.2042 | 0.7633 | 240 |

Random guessing on 8 emotions would be 0.125.

## Important: the baseline model does not load correctly

When the baseline model loads, transformers prints a LOAD REPORT that looks like
this:

```
classifier.dense.bias    | UNEXPECTED
classifier.output.weight | UNEXPECTED
classifier.bias          | MISSING
projector.weight         | MISSING
classifier.weight        | MISSING
```

That checkpoint saves its final classifier layer under different names than the
standard `Wav2Vec2ForSequenceClassification` class expects. So the trained
classifier weights get thrown away (UNEXPECTED) and a brand new random classifier
is created in their place (MISSING).

This means the baseline is guessing with a random final layer. That is why its
confidence sits at about 0.14 on every single clip (1 out of 8 is 0.125) and why
its accuracy of 0.0292 is actually *below* random. It is not that the baseline
generalizes badly - it never loaded.

`evaluate_results.py` prints a warning when average confidence is this close to
random guessing, so this problem is easy to spot.

## Note about our first attempt

Our first version of this code loaded audio with `soundfile` and sent it straight
to the model. Two things were wrong with that:

1. RAVDESS files are recorded at 48000 Hz, but wav2vec2 expects 16000 Hz. Because
   we never resampled, the model heard every clip at 3x its real length and much
   lower in pitch.
2. Two of the clips are stereo (`Actor_01/03-01-02-01-01-02-01.wav` and
   `Actor_01/03-01-08-01-02-02-01.wav`), and the model only accepts mono, so those
   two files were silently skipped by a `try/except` that hid the error.

`run_emotion_model.py` fixes both inside `load_audio_for_model()`, and prints the
number of files found next to the number of rows saved so a silent skip cannot
happen again.

We also used to report a "Match Rate" as if it were accuracy. It was not - it only
measured how often the model repeated its most common answer, so a completely
wrong prediction could still show as 100%. `evaluate_results.py` compares the
prediction to the true label instead.

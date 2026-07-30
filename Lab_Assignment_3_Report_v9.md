**Lab Assignment 3: Speech Emotion Recognition Systems**

MIS 425 — Spring 2026 — Dr. Mary Pourebadi

*Group Members: Christina Oftedal, \[Name 2\], \[Name 3\], \[Name 4\]*

***\[NOTE TO FILL IN:** Replace the placeholder names above with all
group members exactly as required by the syllabus before
submitting.**\]***

# 1. Logistics

This report is submitted in fulfillment of Lab Assignment 3 (10 points),
evaluating two pre-trained Speech Emotion Recognition (SER) models on a
240-clip RAVDESS subset.

|                           |                                                                                                             |
| ------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Item**                  | **Detail**                                                                                                  |
| Group size                | 4 students (names above)                                                                                    |
| Total points              | 10                                                                                                          |
| Submission                | One PDF per the group's final report, submitted individually by each member to Canvas                       |
| Code repository           | https://github.com/ChristinaOftedal/MIS-425-Lab-3-                                                          |
| Environment               | Python 3.9, PyTorch 2.6. PyTorch 2.0.1 did not work in this environment — see Appendix A.                   |
| Reproduction instructions | See Sections 4–5 and Appendix A for the exact steps to reproduce the CSV outputs and figures in this report |

# 2. Objective

The goal of this lab is to evaluate the robustness of Speech Emotion
Recognition (SER) systems built on pre-trained wav2vec2-based emotion
classifiers. In this assignment we designed an evaluation pipeline to
test two independently pre-trained SER models on the same RAVDESS
evaluation set, and we critically analyze both models' behavior rather
than assuming a stronger label implies stronger performance.

# 3. Task Overview

**Option Selected: Option 1 — Use a Stronger Pre-Trained SER Model**

We selected Option 1. Rather than fine-tuning wav2vec2 ourselves (Option
2) or building a classical feature-based classifier from scratch (Option
3), we researched and tested two independently pre-trained Hugging Face
SER models against a 240-clip RAVDESS subset:

|         |                                             |                                                           |
| ------- | ------------------------------------------- | --------------------------------------------------------- |
|         | **Model**                                   | **Hugging Face ID**                                       |
| Model 1 | wav2vec2-lg-xlsr fine-tuned for English SER | ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition |
| Model 2 | wav2vec (English) fine-tuned for SER        | r-f/wav2vec-english-speech-emotion-recognition            |

Both are wav2vec2-family transformer encoders fine-tuned for speech
emotion classification and were run zero-shot (no additional
fine-tuning), one clip at a time, using identical inference methodology
for both models — this keeps the comparison between them
apples-to-apples.

Justification: both models are wav2vec2-derived encoders fine-tuned
specifically for speech emotion recognition rather than general audio
classification, making them directly comparable to each other. We
selected two candidates so we could compare their failure patterns
against one another directly.

**What the data itself tells us about the two models:** Model 1 predicts
all 8 RAVDESS emotion labels at least once, while Model 2 never predicts
“calm” — evidence the two models were fine-tuned on different label
taxonomies rather than being the same model re-run. Model 1 also
achieves a real, if narrow, pocket of accuracy on “sad” clips (discussed
in Section 6), while Model 2 does not show an equivalent strength on any
class.

***Data-integrity note:** Two of the files provided alongside this data
("comparison\_table (1).csv" and "baseline\_results.csv") report a
"Baseline" vs. "SuperbER" comparison with a much higher accuracy (20.4%)
and confidence (0.76) than either ehcalabres or r-f produced. That third
model is not ehcalabres or r-f and is out of scope for this report's
two-model comparison — we did not include it here. If your group tested
a third candidate model, it should get its own clearly-labeled section
rather than being mixed into the Model 1 vs. Model 2 tables.*

# 4. Dataset Description

## 4.1 Dataset

Both models were evaluated on the RAVDESS (Ryerson Audio-Visual Database
of Emotional Speech and Song) audio-speech corpus. The evaluation set is
a 240-clip subset drawn from 4 actors present in the repository's audio/
folder — Actor\_01, Actor\_02, Actor\_04, and Actor\_07 — 60 clips each.
This is a non-sequential subset of RAVDESS's 24 total actors; the
remaining 20 actors (including Actor\_03, Actor\_05, Actor\_06, etc.)
are not part of this evaluation.

|                            |                               |
| -------------------------- | ----------------------------- |
| **Emotion (RAVDESS code)** | **Samples in evaluation set** |
| 01 – neutral               | 16                            |
| 02 – calm                  | 32                            |
| 03 – happy                 | 32                            |
| 04 – sad                   | 32                            |
| 05 – angry                 | 32                            |
| 06 – fearful               | 32                            |
| 07 – disgust               | 32                            |
| 08 – surprised             | 32                            |
| Total                      | 240                           |

The class imbalance for “neutral” (16 vs. 32 for every other class) is
inherent to RAVDESS: the neutral category only has a “normal” intensity
level, while every other emotion is recorded at both “normal” and
“strong” intensity.

## 4.2 Train / Validation / Test Split

No training occurred for Option 1 — both models were evaluated zero-shot
(inference only) on the full 240-clip set. run\_emotion\_model.py
iterates over every .wav file found under the audio/Actor\_\* folders
and scores it directly; there is no train/validation/test split, holdout
set, or cross-validation in this pipeline, consistent with Option 1's
“test an existing pre-trained model” scope rather than Option 2's
fine-tuning scope.

## 4.3 Preprocessing

Unlike a naive approach that hands each model a raw file path,
run\_emotion\_model.py explicitly preprocesses every clip before
inference: it reads the .wav file, averages stereo channels down to mono
where needed, and resamples from RAVDESS's native 48,000 Hz to each
model's own required sample rate (read directly from that model's
\`feature\_extractor.sampling\_rate\`, rather than assumed) using the
soxr resampler, before casting to float32 and passing the array directly
to the Hugging Face pipeline. This matters: RAVDESS audio played at the
wrong sample rate is effectively pitch-shifted and time-stretched, and
would otherwise degrade predictions independently of the model's real
emotion-recognition ability.

# 5. Methodology

## 5.1 Feature Extraction

Both models take raw (resampled, mono, float32) waveform input, which
each model's own wav2vec2-based feature extractor converts internally
into learned embeddings — no hand-crafted features (MFCCs, spectrograms,
etc.) are computed by our pipeline. Both models are called identically
(same preprocessing function, same Hugging Face \`audio-classification\`
pipeline interface, \`top\_k=None\` to capture every class's
probability), isolating the comparison to differences between the models
themselves rather than differences in a custom feature pipeline.

## 5.2 Model Architecture

**Model 1 (ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition):**
a wav2vec2-large-xlsr transformer encoder fine-tuned for English speech
emotion classification, used here unmodified as one of the two models
compared in this report.

**Model 2 (r-f/wav2vec-english-speech-emotion-recognition):** a
wav2vec2-based English SER model with a smaller output label set (never
predicts “calm” on this dataset), substituted in with no additional
fine-tuning.

Neither model was fine-tuned in this submission (Option 1 is zero-shot
evaluation), so there are no training hyperparameters (epochs, learning
rate, batch size) to report — both models are evaluated purely at
inference time, one .wav file per pipeline call, using each model's
default inference settings (\`top\_k=None\`, taking the highest-scoring
label and its confidence).

***Data-integrity note:** Environment: this pipeline requires Python 3.9
and PyTorch 2.6 to run correctly. An earlier attempt using PyTorch 2.0.1
in the same environment did not work. Document the specific failure
(import error, CUDA mismatch, etc.) if you want this fully explained in
your submission — we don't have the original error message, so this note
only records that the downgrade was attempted and failed, not why.*

# 6. Evaluation

## 6.1 Quantitative Metrics

Accuracy was computed by comparing each model's predicted label directly
against the ground-truth emotion derived from the RAVDESS filename (the
3rd numeric field), after normalizing label spelling differences
(“fear”→“fearful”, “surprise”→“surprised”) so both models' outputs
compare fairly against the same ground truth.

|                                                |                          |                           |
| ---------------------------------------------- | ------------------------ | ------------------------- |
| **Metric**                                     | **Model 1 (ehcalabres)** | **Model 2 (r-f)**         |
| Total samples evaluated                        | 240                      | 240                       |
| Correct predictions                            | 20                       | 6                         |
| Overall accuracy                               | 8.33%                    | 2.50%                     |
| Chance-level accuracy (1/8 classes)            | 12.5%                    | 12.5%                     |
| Mean prediction confidence                     | 0.1387                   | 0.1580                    |
| Std. dev. of confidence                        | 0.0042                   | 0.0050                    |
| Confidence range (min–max)                     | 0.1280 – 0.1491          | 0.1456 – 0.1653           |
| Distinct predicted labels used (of 8 possible) | 8 (all labels used)      | 7 (never predicts “calm”) |

![](report_images/figure1_model1_confusion_matrix_v9.png)

*Figure 1. Model 1 (ehcalabres) — predicted vs. actual emotion,
confusion matrix over all 240 clips.*

![](report_images/figure2_model2_confusion_matrix_v9.png)

*Figure 2. Model 2 (r-f) — predicted vs. actual emotion, confusion
matrix over all 240 clips.*

## 6.2 Baseline Comparison

This section compares the two candidate models directly against each
other, using Model 1 as the point of reference for Model 2.

|                                                |                                                                                                                                                                                                                                                                               |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Question**                                   | **Finding**                                                                                                                                                                                                                                                                   |
| Did performance improve (Model 2 vs. Model 1)? | No. Model 1 (8.33%) meaningfully outperforms Model 2 (2.50%) — both are still far below the 12.5% chance level for 8-way classification, but Model 1's edge is real and driven almost entirely by one class (“sad” — see 6.3), not a general improvement across all emotions. |
| Did confidence values increase?                | No. Mean confidence is 0.139 (Model 1) vs. 0.158 (Model 2) — both sit barely above the 0.125 uniform-chance baseline for 8 classes. Model 2 is slightly more confident on average despite being less accurate, which is a bad sign for its calibration, not a good one.       |
| Were predictions more stable?                  | No. Both models exhibit heavy mode collapse: several classes are predicted almost entirely as one or two labels (e.g. Model 1's disgust→happy at 32/32, Model 2's fearful→surprised at 32/32), regardless of the clip's actual acoustic content.                              |

![](report_images/figure3_confidence_comparison_v9.png)

*Figure 3. Confidence score distributions for both models, relative to
the 1/8 chance level.*

![](report_images/figure4_per_emotion_accuracy_v9.png)

*Figure 4. Per-emotion true accuracy for both models.*

## 6.3 Performance Analysis

Easiest to classify: “sad” is Model 1's only real strength — 20 of 32
actual “sad” clips (62.5%) were correctly labeled. Model 2's best class
is “happy” at 5 of 32 (15.6%), with a further 1 of 32 “surprised” clips
(3.1%) correct. Every other class scored 0% for both models.

**A caution on that “sad” number:** Model 1 predicted “sad” for 80 of
the 240 clips overall (33%) — far more than the 32 clips (13%) that are
actually sad. So while its recall on sad is 62.5% (20/32), its precision
on sad is only 25% (20/80): three out of every four “sad” predictions it
makes are wrong. The high recall looks like a strength in isolation, but
it is at least partly a symptom of the model defaulting to “sad”
unusually often, not of it specifically understanding sadness. The same
caution applies more mildly to Model 2's “happy” predictions (54 of 240
clips, 22.5% of the dataset, for a class that's actually only 13% of
it).

Most frequently confused: Model 1 collapses “disgust” to “happy”
(32/32), “fearful” and “calm” both largely to “sad” (28/32 and 30/32),
and “surprised” mostly to “disgust” (25/32). Model 2 collapses “angry”
to “fearful” (32/32), “calm” to “happy” (32/32), “disgust” to “sad”
(32/32), and “fearful” to “surprised” (32/32) — every one of these four
classes for Model 2 is a complete, deterministic swap to a single wrong
label.

Speaker effects: Model 1's accuracy ranges from 3/60 (Actor\_07) to 7/60
(Actor\_02); Model 2's ranges from 0/60 (Actor\_07) to 3/60 (Actor\_02).
Actor\_02 is the strongest speaker for both models and Actor\_07 the
weakest for both, but the gap is small relative to how close both models
are to zero overall — this isn't a strong, reliable speaker effect so
much as noise around a near-floor accuracy.

Overall confidence: both models' softmax outputs sit in a narrow band
only slightly above the 0.125 uniform-chance value for 8 classes (Model
1: 0.128–0.149; Model 2: 0.146–0.165), meaning neither model is
confidently distinguishing between emotion classes for this dataset.

# 7. Results & Analysis

## 7.1 Error Analysis

Three representative misclassifications from each model, drawn from the
full prediction logs:

|           |                          |           |                |                     |                |
| --------- | ------------------------ | --------- | -------------- | ------------------- | -------------- |
| **Model** | **File**                 | **Actor** | **True Label** | **Predicted Label** | **Confidence** |
| Model 1   | 03-01-04-01-01-02-04.wav | Actor\_04 | sad            | fearful             | 0.132          |
| Model 1   | 03-01-07-02-02-02-04.wav | Actor\_04 | disgust        | happy               | 0.146          |
| Model 1   | 03-01-05-02-01-01-04.wav | Actor\_04 | angry          | neutral             | 0.137          |
| Model 2   | 03-01-01-01-02-01-02.wav | Actor\_02 | neutral        | happy               | 0.162          |
| Model 2   | 03-01-03-01-02-02-01.wav | Actor\_01 | happy          | disgust             | 0.146          |
| Model 2   | 03-01-08-01-01-02-02.wav | Actor\_02 | surprised      | angry               | 0.159          |

**disgust → happy (Model 1, 32/32 of the class):** disgust and happy sit
on opposite ends of valence (negative vs. positive), so this is not a
plausible acoustic mix-up — every single disgust clip getting the same
wrong label points to a fixed class-level bias in the model's output
layer for this dataset, not sample-by-sample acoustic confusion.

**angry → neutral (Model 1):** 28 of 32 angry clips were labeled
“neutral” — angry is normally one of the more acoustically
distinctive, high-energy classes, so a near-total collapse to the
calmest possible label is a strong sign the model isn't using
arousal-related acoustic cues at all for this dataset.

**surprised → angry (Model 2, 31/32 of the class):** both are
high-arousal emotions, so this is the more acoustically defensible
confusion of the six examples — but the fact that it happens to 31 of 32
clips, rather than a handful, still points to class-level collapse
rather than genuine per-clip acoustic reasoning.

## 7.2 Model Behavior & Robustness

Model 1 showed slightly better — though still weak — robustness than
Model 2, and it is the only one of the two with a real (if narrow and
precision-poor) foothold on a specific class. Both models still show the
defining symptom of a system that is not extracting usable emotional
information from most of this dataset: several ground-truth classes are
almost perfectly determined by a single fixed predicted label (28–32 of
32 clips in a class getting the same wrong answer), which is not what
genuine, sample-dependent acoustic confusion looks like. Swapping
between the two candidate models changed which fixed label each class
collapses to and slightly changed the confidence distribution, but did
not meaningfully reduce the collapse behavior in either model.

Classical ML behavior was not evaluated in this submission since Option
1 was selected; a classical baseline (Option 3) would be a useful
robustness comparison point for future work (see Section 8.3).

Speaker sensitivity: accuracy varies modestly by actor for both models
(Section 6.3), with Actor\_02 consistently doing best and Actor\_07
consistently doing worst, but the variation is small in absolute terms
given how close both models are to the floor overall.

## 7.3 Interpretation of Results

**Dataset size / domain mismatch:** this evaluation used 240 clips from
4 of RAVDESS's 24 actors, and the near-chance confidence scores (both
means within \~0.03 of the 0.125 uniform baseline) suggest the deeper
issue is not sample size but domain mismatch — both pre-trained models
appear to have been fine-tuned on data whose acoustic and/or label
distribution differs enough from RAVDESS that neither can reliably map
RAVDESS recordings onto RAVDESS's own emotion taxonomy.

**Model capacity vs. class-prior bias:** the extreme class-level mode
collapse (a single predicted label capturing 28–32 of 32 clips in
several ground-truth classes) is a classic symptom of a classifier whose
output distribution is dominated by learned class priors from its own
fine-tuning data rather than by the acoustic input. Model 1's “sad”
recall/precision gap (62.5% vs. 25%) is a direct illustration of this:
the headline recall number is inflated by the model simply guessing
“sad” unusually often.

**Overfitting / underfitting:** since neither model was fine-tuned on
RAVDESS in this submission (Option 1 is zero-shot evaluation), classic
overfitting to RAVDESS is not applicable; the pattern instead looks like
underfitting to this specific dataset's acoustic distribution,
consistent with pre-trained wav2vec2-style SER models not generalizing
well to new speakers or datasets without dataset-specific fine-tuning.

**Label-taxonomy mismatch:** Model 2 never predicts “calm” on this
dataset. Since RAVDESS's ground truth includes it, any clip whose true
label is “calm” is guaranteed to be misclassified by Model 2 regardless
of audio quality — a structural ceiling on accuracy that has nothing to
do with model capability on the clips it can, in principle, label
correctly. Model 1 does not have this specific gap on this run (it used
all 8 labels at least once), which is itself a change from an earlier,
smaller evaluation run where it never predicted “calm” either.

# 8. Limitations & Future Improvements

## 8.1 Identified Limitations

  - Limited and non-representative dataset: only 240 clips from 4 of 24
    RAVDESS actors (Actor\_01, 02, 04, 07) were evaluated, which is too
    small and too specific a sample to draw conclusions about how either
    model performs across RAVDESS's full speaker diversity.

  - Recall without precision overstates “success”: Model 1's headline
    strength (62.5% recall on “sad”) comes with only 25% precision on
    that same class, because the model predicts “sad” for a third of all
    clips regardless of true label. Any report of per-class accuracy
    should be read alongside precision, not recall alone.

  - Domain / label-taxonomy mismatch: both models' confidence scores
    cluster tightly just above the 8-class chance level, and Model 2's
    output vocabulary doesn't cover all 8 RAVDESS emotions (never
    predicts “calm”) — evidence that neither model's fine-tuning
    distribution matches RAVDESS's acoustic or label distribution well
    enough to transfer cleanly.

  - Pipeline history: earlier drafts of this pipeline had a model-ID bug
    (Model 1's script loading the wrong Hugging Face model) and summary
    report files that didn't match their underlying CSVs. Both are
    corrected in the current run\_emotion\_model.py /
    evaluate\_results.py scripts used to produce this report (see
    Appendix A); older files still present in the repository
    (baseline\_results.csv, the old "terminal output" and
    "emotion\_results" CSVs, and the .md summary reports) are stale and
    should be removed or clearly archived before final submission to
    avoid confusion.

## 8.2 Generalization Concerns

Based on this evidence, neither model would be expected to generalize
well to new speakers, different accents, different recording
environments, or real-world audio. Even with a larger, 4-actor
evaluation set than before, both models remain far below chance-level
accuracy on speakers drawn from the same curated, studio-recorded,
professionally-acted RAVDESS corpus — a best-case scenario for audio
quality and label clarity. Real-world audio, with background noise,
casual speech, and unscripted emotional expression, would present a
substantially harder distribution shift than what is reflected here, so
we would expect performance to degrade further, not improve, outside
RAVDESS.

## 8.3 Future Improvements

  - Fine-tune on RAVDESS (Option 2): given that both zero-shot models
    collapse to fixed per-class labels, task-specific fine-tuning on a
    training split of RAVDESS (with a held-out speaker-disjoint test
    split) is the most direct way to address the domain-mismatch pattern
    documented in Section 7.3.

  - Add a classical ML baseline (Option 3): training an SVM or Random
    Forest on hand-crafted features (e.g., MFCCs, pitch, energy) would
    clarify whether the failure is specific to these transformer models
    or reflects a harder property of this evaluation subset.

  - Expand the evaluation set: testing all 24 RAVDESS actors (1,440
    clips) instead of 4 would validate whether the current per-actor and
    per-class patterns generalize or are specific to this particular
    4-actor subset.

  - Report precision alongside recall for every class: as shown by Model
    1's “sad” results, recall alone can make a class-prior bias look
    like a genuine strength — always pair it with precision (and ideally
    F1) before calling a class a model's strength.

# Appendix A: Reproducing These Results

## A.1 Environment

  - Python 3.9.

  - PyTorch 2.6. PyTorch 2.0.1 was tried in this same environment and
    did not work — use 2.6 (or re-test and document the exact failure if
    you need to explain why 2.0.1 failed).

  - Install dependencies: torch, transformers, soundfile, soxr, numpy,
    pandas, scikit-learn, matplotlib.

## A.2 Setup

  - Place the RAVDESS actor folders (Actor\_01, Actor\_02, Actor\_04,
    Actor\_07, ...) under an audio/ folder in the repository, e.g.
    /workspaces/MIS-425-Lab-3-/audio/Actor\_01/\*.wav.

  - Run: python run\_emotion\_model.py --model
    ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition --out
    results/model1\_results.csv

  - Run: python run\_emotion\_model.py --model
    r-f/wav2vec-english-speech-emotion-recognition --out
    results/model2\_results.csv

  - Run: python evaluate\_results.py --results
    results/model1\_results.csv results/model2\_results.csv --names
    "Model 1" "Model 2" — this prints accuracy/precision/recall/F1 for
    each model, saves a confusion matrix image per model under images/,
    and writes results/comparison\_table.csv.

## A.3 Data provenance for this report

Model 1's results in this report come from model1\_results.csv (240
rows, columns: Actor, File Name, Emotion, Confidence Score, plus one
\*\_probability column per label). Model 2's results come from
model2\_results.csv in the same format. Both were verified by
recomputing accuracy and mean confidence directly from every row and
confirming an exact match against results/comparison\_table.csv (Model
1: 8.33% / 0.1387; Model 2: 2.50% / 0.1580) and against the two
"Prediction Patterns Matrix" confusion-matrix images included with the
data.

***Data-integrity note:** Two other files in this data batch —
baseline\_results.csv and "comparison\_table (1).csv" — describe a
different comparison ("Baseline" at 2.92%/0.1406 vs. a model called
"SuperbER" at 20.42%/0.7633, on the same 240 clips). "SuperbER" is not
ehcalabres or r-f and was not included in this report, since the
assignment scope here is specifically the two named models. If that
third-model comparison is meant to be part of your submission, it needs
its own section with its own justification for why that model was chosen
(Section 3) rather than being folded into this Model 1 vs. Model 2
report.*

A separate, smaller 180-clip, 3-actor run (Actor\_01, Actor\_02,
Actor\_04 — no Actor\_07) also exists in the project's files. Earlier
versions of that run's pipeline scripts had real bugs (a shared
hardcoded input path that let one model's script silently overwrite the
other's data, and a predicted-column name mismatch), so its numbers were
not trustworthy enough to include here. Those scripts have since been
fixed and cross-verified — evaluate\_results.py and both per-model
analysis scripts now independently agree on the same confusion matrices
and accuracy figures for this 180-clip set. It remains a smaller,
separate sample from this report's main 240-clip comparison and is
presented as supplementary evidence in Appendix D, not merged into the
tables above.

# Appendix B: Full Inference Code

Full source for both pipeline scripts follows, satisfying the
assignment's code-submission requirement (Section 1). These listings are
supplementary and are not intended to count against the 8-page report
body limit — confirm this with your syllabus/instructor before
submitting.

## B.1 run\_emotion\_model.py

"""

Runs a Hugging Face speech emotion recognition model on the RAVDESS
audio files

and saves every prediction to a CSV file.

Lab 3, Option 1: we run TWO models with this same script (the baseline
model from

class, and a different model we picked) so the two CSV files can be
compared fairly.

Example:

python run\_emotion\_model.py --model
ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition --out
results/baseline\_results.csv

"""

import argparse

from pathlib import Path

import numpy as np

import pandas as pd

import soundfile as sf

import soxr

import torch

from transformers import pipeline

def load\_audio\_for\_model(wav\_path, target\_sample\_rate):

"""

Reads one .wav file and gets it into the shape the model expects.

RAVDESS clips are recorded at 48000 Hz and a couple of them are stereo,

but wav2vec2 models want 16000 Hz mono audio. If we skip these two steps

the model hears the clip stretched to 3x its real length and pitched way

down, and the predictions come out worse than random guessing.

"""

audio, sample\_rate = sf.read(wav\_path)

\# Step 1: stereo -\> mono. A stereo file comes back with 2 columns,

\# so we average them into a single channel.

if audio.ndim \> 1:

audio = audio.mean(axis=1)

\# Step 2: resample to whatever rate the model was trained on (16000
Hz).

if sample\_rate \!= target\_sample\_rate:

audio = soxr.resample(audio, sample\_rate, target\_sample\_rate)

\# The model wants 32-bit floats.

return audio.astype(np.float32)

def main():

parser = argparse.ArgumentParser()

parser.add\_argument("--model", required=True,

help="Hugging Face model name")

\# FIX: the Actor\_01, Actor\_02, ... folders live inside an "audio"
subfolder on GitHub

\# (https://github.com/ChristinaOftedal/MIS-425-Lab-3-/tree/main/audio),
not at the repo

\# root. This repo gets cloned into /workspaces/MIS-425-Lab-3-/ in the
Codespace, so the

\# actor folders end up at /workspaces/MIS-425-Lab-3-/audio/Actor\_01,
etc. Default to

\# that path so you don't have to pass --audio\_dir every time; override
it if you're

\# running somewhere else (e.g. a local clone).

parser.add\_argument("--audio\_dir",
default="/workspaces/MIS-425-Lab-3-/audio",

help="Folder that contains the Actor\_01, Actor\_02, ... folders "

"(defaults to the repo's audio/ folder as cloned in the Codespace)")

parser.add\_argument("--out", required=True,

help="Where to save the CSV file")

parser.add\_argument("--checkpoint\_every", type=int, default=25,

help="Write partial results to --out every N files, so a crash "

"partway through doesn't lose everything already processed")

args = parser.parse\_args()

\# FIX: the script's own example (\`--out
results/baseline\_results.csv\`) crashes with

\# FileNotFoundError on a fresh checkout, because pandas won't create
missing parent

\# directories for you. Create the output folder up front.

out\_path = Path(args.out)

out\_path.parent.mkdir(parents=True, exist\_ok=True)

print("Loading model:", args.model)

\# FIX: use a GPU automatically if one is available instead of always
running on CPU.

device = 0 if torch.cuda.is\_available() else -1

classifier = pipeline("audio-classification", model=args.model,
device=device)

\# Ask the model what sample rate it needs instead of guessing.

target\_sample\_rate = classifier.feature\_extractor.sampling\_rate

print("This model expects audio at", target\_sample\_rate, "Hz")

\# Find every .wav file inside the Actor\_\* folders.

\# FIX: glob("Actor\_\*/\*.wav") is case-sensitive and only looks one
folder deep, so it

\# silently misses files saved as ".WAV" or nested in a subfolder. rglob
+ a

\# case-insensitive suffix check catches those too.

audio\_dir = Path(args.audio\_dir)

wav\_files = sorted(

p for p in audio\_dir.glob("Actor\_\*/\*\*/\*") if p.suffix.lower() ==
".wav" and p.is\_file()

)

if len(wav\_files) == 0:

\# FIX: print exactly where it looked (and whether that folder even
exists), instead

\# of a generic message — this is what makes a wrong --audio\_dir
instantly obvious

\# rather than a mystery, which is what happened here.

print(f"No .wav files found under: {audio\_dir.resolve()}")

if not audio\_dir.exists():

print(" -\> That folder does not exist. Check --audio\_dir.")

else:

subfolders = sorted(p.name for p in audio\_dir.iterdir() if p.is\_dir())

print(f" -\> Folder exists. Subfolders found there: {subfolders or
'(none)'}")

print("Expected a structure like: \<audio\_dir\>/Actor\_01/\*.wav,
\<audio\_dir\>/Actor\_02/\*.wav, ...")

return

print("Found", len(wav\_files), "audio files")

rows = \[\]

skipped = \[\]

for i, wav\_path in enumerate(wav\_files, start=1):

\# FIX: a single unreadable/corrupted file used to crash the whole run
and throw

\# away every prediction gathered so far, since nothing was saved until
the very

\# end. Now we log the failure, skip just that file, and keep going.

try:

audio = load\_audio\_for\_model(wav\_path, target\_sample\_rate)

\# top\_k=None gives us the score for every emotion, not just the best
one.

scores = classifier(audio, top\_k=None)

best = max(scores, key=lambda s: s\["score"\])

row = {

"Actor": wav\_path.parent.name,

"File Name": wav\_path.name,

"Emotion": best\["label"\],

"Confidence Score": round(best\["score"\], 4),

}

\# Save all the probabilities too, so we can look at them later.

for s in scores:

row\[s\["label"\] + "\_probability"\] = round(s\["score"\], 4)

rows.append(row)

except Exception as file\_error:

print(f" \[SKIPPED\] {wav\_path.name}: {file\_error}")

skipped.append(wav\_path.name)

if i % 20 == 0 or i == len(wav\_files):

print(" processed", i, "of", len(wav\_files))

\# FIX: checkpoint progress periodically so a crash near the end of a
long run

\# (e.g. 1,440 files across all 24 RAVDESS actors) doesn't lose
everything.

if rows and (i % args.checkpoint\_every == 0 or i == len(wav\_files)):

pd.DataFrame(rows).to\_csv(out\_path, index=False)

df = pd.DataFrame(rows)

df.to\_csv(out\_path, index=False)

\# These two numbers should match minus anything skipped. If they don't,
something

\# other than a skipped/corrupted file caused rows to go missing.

print("Saved", len(df), "rows to", args.out)

print("Files found was", len(wav\_files), "-", len(skipped), "skipped
-",

len(wav\_files) - len(skipped), "expected rows.")

if skipped:

print("Skipped files:", ", ".join(skipped))

if \_\_name\_\_ == "\_\_main\_\_":

main()

## B.2 evaluate\_results.py

"""

Scores the CSV files made by run\_emotion\_model.py against the real
RAVDESS labels.

This prints accuracy, precision, recall and F1, saves a confusion matrix
picture,

lists the emotions that get mixed up the most, and prints some wrong
predictions

we can write about in the error analysis section.

Example (one model):

python evaluate\_results.py --results results/baseline\_results.csv
--names Baseline

Example (comparing both models, which is what Option 1 asks for):

python evaluate\_results.py --results results/baseline\_results.csv
results/new\_model\_results.csv --names Baseline NewModel

"""

import argparse

from pathlib import Path

import matplotlib

matplotlib.use("Agg") \# save pictures to files instead of opening a
window

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.metrics import accuracy\_score, classification\_report,
confusion\_matrix

\# The 3rd number in a RAVDESS file name is the emotion.

\# Example: 03-01-05-01-02-01-12.wav -\> "05" -\> angry

RAVDESS\_EMOTIONS = {

"01": "neutral",

"02": "calm",

"03": "happy",

"04": "sad",

"05": "angry",

"06": "fearful",

"07": "disgust",

"08": "surprised",

}

\# Different models spell the same emotion differently, so we make them
match.

LABEL\_FIXES = {

"fear": "fearful",

"surprise": "surprised",

"ang": "angry",

"hap": "happy",

"neu": "neutral",

"sad": "sad",

}

def get\_true\_emotion(file\_name):

"""Reads the real emotion out of a RAVDESS file name."""

parts = file\_name.replace(".wav", "").split("-")

if len(parts) \< 3:

return "unknown"

return RAVDESS\_EMOTIONS.get(parts\[2\], "unknown")

def tidy\_label(label):

"""Makes label spelling consistent, e.g. 'fear' becomes 'fearful'."""

label = str(label).strip().lower()

return LABEL\_FIXES.get(label, label)

def score\_one\_model(csv\_path, model\_name):

"""Scores one results CSV and prints everything we need for the
report."""

print()

print("=" \* 60)

print("RESULTS FOR:", model\_name, " (file:", csv\_path, ")")

print("=" \* 60)

\# FIX: a missing/misspelled --results path used to crash the whole
comparison run

\# with a raw pandas traceback and abandon every other model in the same
run. Now it

\# prints a clear message and lets the remaining models still get
scored.

if not Path(csv\_path).exists():

print(f"ERROR: file not found: {csv\_path}")

print(" Check the path — this model will be skipped, the others will
still run.")

return None

df = pd.read\_csv(csv\_path)

df\["true\_emotion"\] = df\["File Name"\].apply(get\_true\_emotion)

df\["predicted\_emotion"\] = df\["Emotion"\].apply(tidy\_label)

y\_true = df\["true\_emotion"\]

y\_pred = df\["predicted\_emotion"\]

\# --- Accuracy
---------------------------------------------------------

\# This is REAL accuracy: it compares the prediction to the true label.

accuracy = accuracy\_score(y\_true, y\_pred)

n\_classes = y\_true.nunique()

print()

print("Total clips scored:", len(df))

print("Accuracy: %.4f (%d out of %d correct)"

% (accuracy, (y\_true == y\_pred).sum(), len(df)))

print("Random guessing would be about %.4f (1 out of %d classes)"

% (1 / n\_classes, n\_classes))

average\_confidence = df\["Confidence Score"\].mean()

print("Average confidence: %.4f" % average\_confidence)

\# Sanity check. If the model is barely more confident than a coin flip
across

\# every single clip, it usually means the model did not load properly
rather

\# than that the task is hard. When we loaded the class baseline model,

\# transformers printed a LOAD REPORT saying the classifier layer was
MISSING

\# and got randomly initialized, and every confidence came out near 1/8.

\#

\# FIX: this used to compare against 1 / (number of DIFFERENT labels
this model

\# happened to predict), not 1 / (number of real RAVDESS classes).
That's backwards:

\# a model that mode-collapses onto one label with 100% confidence would
get

\# n\_predicted\_classes = 1, threshold = 1.5, and 1.0 \< 1.5 would
\*wrongly\* fire this

\# "looks like random guessing" warning even though it's the opposite
problem

\# (maximally overconfident on one class, not underconfident like a coin
flip).

\# Compare against the same n\_classes used in the "random guessing"
line above instead.

if average\_confidence \< 1.5 \* (1 / n\_classes):

print()

print("WARNING: the average confidence is very close to random
guessing.")

print(" Scroll up to the LOAD REPORT that transformers printed when")

print(" the model loaded. If any layer says MISSING, the model's")

print(" classifier was randomly initialized and these numbers are")

print(" not meaningful.")

\# Warn if a model simply does not have one of the RAVDESS emotions.

missing = sorted(set(y\_true.unique()) - set(y\_pred.unique()))

if missing:

print()

print("NOTE: this model never predicted:", ", ".join(missing))

print(" If the model has no such class, those clips can never be
correct.")

\# --- Precision / recall / F1
------------------------------------------

print()

print("Precision, recall and F1 for each emotion:")

print(classification\_report(y\_true, y\_pred, zero\_division=0))

\# --- Confusion matrix
-------------------------------------------------

labels = sorted(set(y\_true) | set(y\_pred))

matrix = confusion\_matrix(y\_true, y\_pred, labels=labels)

fig, ax = plt.subplots(figsize=(9, 8))

picture = ax.imshow(matrix, cmap="Blues")

fig.colorbar(picture, label="Number of clips")

ax.set\_xticks(range(len(labels)))

ax.set\_xticklabels(labels, rotation=45, ha="right")

ax.set\_yticks(range(len(labels)))

ax.set\_yticklabels(labels)

ax.set\_xlabel("Predicted emotion")

ax.set\_ylabel("True emotion")

ax.set\_title("Confusion Matrix - " + model\_name)

for r in range(len(labels)):

for c in range(len(labels)):

ax.text(c, r, matrix\[r\]\[c\], ha="center", va="center")

fig.tight\_layout()

Path("images").mkdir(exist\_ok=True)

picture\_name = Path("images") / ("confusion\_matrix\_"

\+ model\_name.replace(" ", "\_") + ".png")

fig.savefig(picture\_name, dpi=150)

plt.close(fig)

print("Saved confusion matrix picture:", picture\_name)

\# --- Which emotions get mixed up the most
-----------------------------

wrong = df\[df\["true\_emotion"\] \!= df\["predicted\_emotion"\]\]

print()

print("Most common mix-ups (true -\> predicted):")

mixups = wrong.groupby(\["true\_emotion", "predicted\_emotion"\]).size()

mixups = mixups.sort\_values(ascending=False).head(5)

for (true\_label, pred\_label), count in mixups.items():

print(" %-10s -\> %-10s %d times" % (true\_label, pred\_label, count))

\# --- Accuracy for each actor (speaker)
--------------------------------

print()

print("Accuracy for each actor:")

for actor, group in df.groupby("Actor"):

actor\_acc = (group\["true\_emotion"\] ==
group\["predicted\_emotion"\]).mean()

print(" %-10s %.4f (%d clips)" % (actor, actor\_acc, len(group)))

\# --- Example wrong predictions for the error analysis section
---------

print()

print("Example wrong predictions (the report needs at least 3):")

for \_, row in wrong.head(5).iterrows():

print(" %s | true: %-10s predicted: %-10s confidence: %.4f"

% (row\["File Name"\], row\["true\_emotion"\],

row\["predicted\_emotion"\], row\["Confidence Score"\]))

return {

"Model": model\_name,

"Accuracy": round(accuracy, 4),

"Average Confidence": round(df\["Confidence Score"\].mean(), 4),

"Clips Scored": len(df),

}

def main():

parser = argparse.ArgumentParser()

parser.add\_argument("--results", nargs="+", required=True,

help="One or more results CSV files")

parser.add\_argument("--names", nargs="+", default=None,

help="A short name for each CSV file")

args = parser.parse\_args()

names = args.names

if names is None:

names = \[Path(p).stem for p in args.results\]

if len(names) \!= len(args.results):

print("Please give one name for each results file.")

return

summary\_rows = \[\]

for csv\_path, model\_name in zip(args.results, names):

\# FIX: score\_one\_model() can now return None (missing file) instead
of crashing;

\# skip it out of the comparison table rather than let a None row
through.

result = score\_one\_model(csv\_path, model\_name)

if result is not None:

summary\_rows.append(result)

if not summary\_rows:

print()

print("No results files could be scored — nothing to compare.")

return

\# --- Comparison table
-------------------------------------------------

\# This is the table the assignment asks for in section 6.2.

summary = pd.DataFrame(summary\_rows)

print()

print("=" \* 60)

print("COMPARISON TABLE")

print("=" \* 60)

print(summary.to\_string(index=False))

Path("results").mkdir(exist\_ok=True)

summary.to\_csv(Path("results") / "comparison\_table.csv", index=False)

print()

print("Saved results/comparison\_table.csv")

if \_\_name\_\_ == "\_\_main\_\_":

main()

# Appendix C: RAVDESS Summary Reports

Full per-class summary reports for both models, regenerated directly
from the verified 240-clip CSVs (model1\_results.csv /
model2\_results.csv). These follow the same “Match Rate” format as the
project's original ravdess\_summary\_report files, with a true-accuracy
line and a per-class true-accuracy table added, since “Match Rate” alone
is not accuracy (see the note inside each report).

## C.1 ravdess\_summary\_report\_Model\_1.txt (ehcalabres)

\============================================================

ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition (Model 1)

RAVDESS EMOTION ANALYSIS SUMMARY REPORT

\============================================================

Total Audio Samples Analyzed: 240

Actors included: Actor\_01, Actor\_02, Actor\_04, Actor\_07

TRUE ACCURACY (predicted label == ground-truth label): 8.33% (20/240)

Mean confidence: 0.1387 Chance level (1/8 classes): 12.50%

NOTE: 'Match Rate' below is the percentage of a ground-truth class's
clips

that received that class's single MOST COMMON prediction. It is NOT
accuracy --

a class can have a high match rate while that most-common prediction is
wrong

for every clip in the class. Use the per-class accuracy table further
down for

true correctness.

Match Rate / Most Common Model Predictions:

\--------------------------------------------------

Ground Truth \[ANGRY\]:

\-\> Most Predicted: 'neutral'

\-\> Match Rate: 28/32 samples (87.5%)

Ground Truth \[CALM\]:

\-\> Most Predicted: 'sad'

\-\> Match Rate: 30/32 samples (93.8%)

Ground Truth \[DISGUST\]:

\-\> Most Predicted: 'happy'

\-\> Match Rate: 32/32 samples (100.0%)

Ground Truth \[FEARFUL\]:

\-\> Most Predicted: 'sad'

\-\> Match Rate: 28/32 samples (87.5%)

Ground Truth \[HAPPY\]:

\-\> Most Predicted: 'calm'

\-\> Match Rate: 28/32 samples (87.5%)

Ground Truth \[NEUTRAL\]:

\-\> Most Predicted: 'surprised'

\-\> Match Rate: 14/16 samples (87.5%)

Ground Truth \[SAD\]:

\-\> Most Predicted: 'sad'

\-\> Match Rate: 20/32 samples (62.5%)

Ground Truth \[SURPRISED\]:

\-\> Most Predicted: 'disgust'

\-\> Match Rate: 25/32 samples (78.1%)

Per-Class TRUE Accuracy (predicted == actual for that class):

\--------------------------------------------------

angry 0/32 (0.0%)

calm 0/32 (0.0%)

disgust 0/32 (0.0%)

fearful 0/32 (0.0%)

happy 0/32 (0.0%)

neutral 0/16 (0.0%)

sad 20/32 (62.5%)

surprised 0/32 (0.0%)

Predicted Emotion Frequency Distribution:

\--------------------------------------------------

sad 80 (33.3% of all predictions)

calm 41 (17.1% of all predictions)

happy 39 (16.2% of all predictions)

neutral 32 (13.3% of all predictions)

disgust 25 (10.4% of all predictions)

surprised 16 (6.7% of all predictions)

fearful 5 (2.1% of all predictions)

angry 2 (0.8% of all predictions)

Accuracy by Actor:

\--------------------------------------------------

Actor\_01 4/60 (6.7%)

Actor\_02 7/60 (11.7%)

Actor\_04 6/60 (10.0%)

Actor\_07 3/60 (5.0%)

Report generated: 2026-07-29 | Source: verified row-by-row from the raw
results CSV

## C.2 ravdess\_summary\_report\_Model\_2.txt (r-f)

\============================================================

r-f/wav2vec-english-speech-emotion-recognition (Model 2)

RAVDESS EMOTION ANALYSIS SUMMARY REPORT

\============================================================

Total Audio Samples Analyzed: 240

Actors included: Actor\_01, Actor\_02, Actor\_04, Actor\_07

TRUE ACCURACY (predicted label == ground-truth label): 2.50% (6/240)

Mean confidence: 0.1580 Chance level (1/8 classes): 12.50%

NOTE: 'Match Rate' below is the percentage of a ground-truth class's
clips

that received that class's single MOST COMMON prediction. It is NOT
accuracy --

a class can have a high match rate while that most-common prediction is
wrong

for every clip in the class. Use the per-class accuracy table further
down for

true correctness.

Match Rate / Most Common Model Predictions:

\--------------------------------------------------

Ground Truth \[ANGRY\]:

\-\> Most Predicted: 'fearful'

\-\> Match Rate: 32/32 samples (100.0%)

Ground Truth \[CALM\]:

\-\> Most Predicted: 'happy'

\-\> Match Rate: 32/32 samples (100.0%)

Ground Truth \[DISGUST\]:

\-\> Most Predicted: 'sad'

\-\> Match Rate: 32/32 samples (100.0%)

Ground Truth \[FEARFUL\]:

\-\> Most Predicted: 'surprised'

\-\> Match Rate: 32/32 samples (100.0%)

Ground Truth \[HAPPY\]:

\-\> Most Predicted: 'sad'

\-\> Match Rate: 13/32 samples (40.6%)

Ground Truth \[NEUTRAL\]:

\-\> Most Predicted: 'happy'

\-\> Match Rate: 16/16 samples (100.0%)

Ground Truth \[SAD\]:

\-\> Most Predicted: 'disgust'

\-\> Match Rate: 31/32 samples (96.9%)

Ground Truth \[SURPRISED\]:

\-\> Most Predicted: 'angry'

\-\> Match Rate: 31/32 samples (96.9%)

Per-Class TRUE Accuracy (predicted == actual for that class):

\--------------------------------------------------

angry 0/32 (0.0%)

calm 0/32 (0.0%)

disgust 0/32 (0.0%)

fearful 0/32 (0.0%)

happy 5/32 (15.6%)

neutral 0/16 (0.0%)

sad 0/32 (0.0%)

surprised 1/32 (3.1%)

Predicted Emotion Frequency Distribution:

\--------------------------------------------------

happy 54 (22.5% of all predictions)

sad 45 (18.8% of all predictions)

disgust 34 (14.2% of all predictions)

surprised 33 (13.8% of all predictions)

fearful 32 (13.3% of all predictions)

angry 32 (13.3% of all predictions)

neutral 10 (4.2% of all predictions)

NOTE: this model never predicted: calm

Accuracy by Actor:

\--------------------------------------------------

Actor\_01 1/60 (1.7%)

Actor\_02 3/60 (5.0%)

Actor\_04 2/60 (3.3%)

Actor\_07 0/60 (0.0%)

Report generated: 2026-07-29 | Source: verified row-by-row from the raw
results CSV

# Appendix D: Supplementary 180-Clip Evaluation

This appendix presents a second, independent evaluation run on a smaller
subset — 180 clips from 3 actors (Actor\_01, Actor\_02, Actor\_04),
versus the 240-clip, 4-actor set used everywhere else in this report. It
is included as supplementary evidence, not as a replacement for or a
merge into the main 240-clip comparison. Both models' pipeline scripts
for this run (evaluate\_results.py and the two per-model analysis
notebooks) were fixed for a shared-file-path bug and a
predicted-column-name mismatch, then re-run and cross-checked against
each other; all three now agree on the same confusion matrices and
accuracy figures reported below.

## D.1 ravdess\_summary\_report\_Model\_1.txt (ehcalabres, 180-clip run)

\==================================================

ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition (Model 1)

RAVDESS EMOTION ANALYSIS SUMMARY REPORT

(Supplementary 180-clip, 3-actor run)

\==================================================

Total Audio Samples Analyzed: 180

Actors included: Actor\_01, Actor\_02, Actor\_04

TRUE ACCURACY (predicted label == ground-truth label): 13.89% (25/180)

Mean confidence: 0.1395 Chance level (1/8 classes): 12.50%

NOTE: 'Match Rate' below is the percentage of a ground-truth class's
clips

that received that class's single MOST COMMON prediction. It is NOT
accuracy --

a class can have a high match rate while that most-common prediction is
wrong

for every clip in the class. Use the per-class accuracy table further
down for

true correctness.

Match Rate / Most Common Model Predictions:

\--------------------------------------------------

Ground Truth \[ANGRY\]:

\-\> Most Predicted: 'happy'

\-\> Match Rate: 18/24 samples (75.0%)

Ground Truth \[CALM\]:

\-\> Most Predicted: 'disgust'

\-\> Match Rate: 22/24 samples (91.7%)

Ground Truth \[DISGUST\]:

\-\> Most Predicted: 'sad'

\-\> Match Rate: 24/24 samples (100.0%)

Ground Truth \[FEARFUL\]:

\-\> Most Predicted: 'fearful'

\-\> Match Rate: 23/24 samples (95.8%)

Ground Truth \[HAPPY\]:

\-\> Most Predicted: 'sad'

\-\> Match Rate: 13/24 samples (54.2%)

Ground Truth \[NEUTRAL\]:

\-\> Most Predicted: 'calm'

\-\> Match Rate: 10/12 samples (83.3%)

Ground Truth \[SAD\]:

\-\> Most Predicted: 'surprised'

\-\> Match Rate: 14/24 samples (58.3%)

Ground Truth \[SURPRISED\]:

\-\> Most Predicted: 'happy'

\-\> Match Rate: 12/24 samples (50.0%)

Per-Class TRUE Accuracy (predicted == actual for that class):

\--------------------------------------------------

angry 0/24 (0.0%)

calm 0/24 (0.0%)

disgust 0/24 (0.0%)

fearful 23/24 (95.8%)

happy 0/24 (0.0%)

neutral 0/12 (0.0%)

sad 2/24 (8.3%)

surprised 0/24 (0.0%)

Predicted Emotion Frequency Distribution:

\--------------------------------------------------

sad 41 (22.8% of all predictions)

fearful 34 (18.9% of all predictions)

happy 31 (17.2% of all predictions)

disgust 27 (15.0% of all predictions)

neutral 19 (10.6% of all predictions)

surprised 16 (8.9% of all predictions)

calm 10 (5.6% of all predictions)

angry 2 (1.1% of all predictions)

Accuracy by Actor:

\--------------------------------------------------

Actor\_01 10/60 (16.7%)

Actor\_02 8/60 (13.3%)

Actor\_04 7/60 (11.7%)

Report generated: 2026-07-30 | Source: verified row-by-row from

emotion\_results\_Model\_1.csv, cross-checked against
evaluate\_results.py's

independent scoring (same 13.89% accuracy, same confusion matrix).

## D.2 ravdess\_summary\_report\_Model\_2.txt (r-f, 180-clip run)

\==================================================

RAVDESS EMOTION ANALYSIS SUMMARY REPORT

(Supplementary 180-clip, 3-actor run)

model=r-f/wav2vec-english-speech-emotion-recognition (Model 2)

\==================================================

Total Audio Samples Analyzed: 180

Actors included: Actor\_01, Actor\_02, Actor\_04

TRUE ACCURACY (predicted label == ground-truth label): 1.67% (3/180)

Mean confidence: 0.1541 Chance level (1/8 classes): 12.50%

NOTE: 'Match Rate' below is the percentage of a ground-truth class's
clips

that received that class's single MOST COMMON prediction. It is NOT
accuracy --

a class can have a high match rate while that most-common prediction is
wrong

for every clip in the class. Use the per-class accuracy table further
down for

true correctness.

Match Rate / Most Common Model Predictions:

\--------------------------------------------------

Ground Truth \[ANGRY\]:

\-\> Most Predicted: 'disgust'

\-\> Match Rate: 12/24 samples (50.0%)

Ground Truth \[CALM\]:

\-\> Most Predicted: 'happy'

\-\> Match Rate: 16/24 samples (66.7%)

Ground Truth \[DISGUST\]:

\-\> Most Predicted: 'happy'

\-\> Match Rate: 23/24 samples (95.8%)

Ground Truth \[FEARFUL\]:

\-\> Most Predicted: 'angry'

\-\> Match Rate: 18/24 samples (75.0%)

Ground Truth \[HAPPY\]:

\-\> Most Predicted: 'fearful'

\-\> Match Rate: 24/24 samples (100.0%)

Ground Truth \[NEUTRAL\]:

\-\> Most Predicted: 'happy'

\-\> Match Rate: 8/12 samples (66.7%)

Ground Truth \[SAD\]:

\-\> Most Predicted: 'happy'

\-\> Match Rate: 24/24 samples (100.0%)

Ground Truth \[SURPRISED\]:

\-\> Most Predicted: 'happy'

\-\> Match Rate: 23/24 samples (95.8%)

Per-Class TRUE Accuracy (predicted == actual for that class):

\--------------------------------------------------

angry 0/24 (0.0%)

calm 0/24 (0.0%)

disgust 0/24 (0.0%)

fearful 3/24 (12.5%)

happy 0/24 (0.0%)

neutral 0/12 (0.0%)

sad 0/24 (0.0%)

surprised 0/24 (0.0%)

Predicted Emotion Frequency Distribution:

\--------------------------------------------------

happy 97 (53.9% of all predictions)

fearful 28 (15.6% of all predictions)

disgust 24 (13.3% of all predictions)

angry 19 (10.6% of all predictions)

neutral 12 (6.7% of all predictions)

NOTE: this model never predicted: calm, sad, surprised

Accuracy by Actor:

\--------------------------------------------------

Actor\_01 2/60 (3.3%)

Actor\_02 0/60 (0.0%)

Actor\_04 1/60 (1.7%)

Report generated: 2026-07-30 | Source: verified row-by-row from

emotion\_results\_Model\_2.csv, cross-checked against
evaluate\_results.py's

independent scoring (same 1.67% accuracy, same confusion matrix).

## D.3 Evaluation of This Supplementary Run

True accuracy on this 180-clip set is 13.89% (25/180) for Model 1 and
1.67% (3/180) for Model 2 — consistent in direction with the main
240-clip result (8.33% and 2.50%), and again far below the 82% accuracy
ehcalabres' own model card reports for Model 1 on RAVDESS.

***Data-integrity note:** The 'Match Rate' figures in D.1/D.2 are
dramatically misleading here and illustrate exactly why this report
treats Match Rate as untrustworthy. For example, Model 1's DISGUST class
shows a 100% Match Rate (24/24 clips got the class's single most common
prediction), but that most-common prediction is 'sad' — true accuracy
for disgust is 0%. Model 2 is worse: HAPPY, SAD, and NEUTRAL all show
Match Rates of 66–100%, and every one of those 'high match rate' classes
has 0% true accuracy, because the model is consistently and confidently
guessing the wrong label. A reader who only saw the Match Rate table,
without the true-accuracy table beside it, would conclude both models
are performing reasonably well on most classes; they are not.*

The one genuine exception is Model 1's FEARFUL class, where Match Rate
(95.8%) and true accuracy (95.8%) agree — the model is actually,
consistently correct here, not just self-consistent. This is worth
noting precisely because it argues against writing off Model 1's low
overall score as pure noise: a model producing meaningless output would
not be expected to land on the correct label this reliably for one
specific emotion.

Confidence scores on this run stay tightly clustered near the 1-in-8
chance level for both models (mean 0.1395 for Model 1, 0.1541 for Model
2), matching the pattern already discussed for the 240-clip run and
consistent with the classifier-head weight-loading issue investigated
separately for these checkpoints (both models' fine-tuned classification
layers use a wider intermediate projection size than this version of the
transformers library defaults to, causing that layer's trained weights
to be dropped and randomly reinitialized unless explicitly corrected).
If that diagnosis fully explains the low scores, the near-chance
performance on 7 of 8 classes would reflect a broken weight-loading path
rather than a genuine failure of either model to transfer to RAVDESS.
Model 1's strong, consistent FEARFUL performance is a real complication
for that story, though: a fully randomly-initialized classification head
would not be expected to reliably land on one specific correct class.
The two findings sit alongside each other rather than fully resolving
one another.

Because this is a smaller, 3-actor sample evaluated with since-fixed
scripts, the group should treat it as corroborating rather than
conclusive. The most direct way to settle the open question above would
be to re-run the full 240-clip, 4-actor evaluation after applying the
classifier\_proj\_size fix to the model-loading code, and compare the
resulting accuracy and confidence distribution against both runs
presented in this report.

"""
Runs a Hugging Face speech emotion recognition model on the RAVDESS audio files
and saves every prediction to a CSV file.

Lab 3, Option 1: we run TWO models with this same script (the baseline model from
class, and a different model we picked) so the two CSV files can be compared fairly.

Example:
    python run_emotion_model.py --model ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition --out results/baseline_results.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import soxr
from transformers import pipeline


def load_audio_for_model(wav_path, target_sample_rate):
    """
    Reads one .wav file and gets it into the shape the model expects.

    RAVDESS clips are recorded at 48000 Hz and a couple of them are stereo,
    but wav2vec2 models want 16000 Hz mono audio. If we skip these two steps
    the model hears the clip stretched to 3x its real length and pitched way
    down, and the predictions come out worse than random guessing.
    """
    audio, sample_rate = sf.read(wav_path)

    # Step 1: stereo -> mono. A stereo file comes back with 2 columns,
    # so we average them into a single channel.
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Step 2: resample to whatever rate the model was trained on (16000 Hz).
    if sample_rate != target_sample_rate:
        audio = soxr.resample(audio, sample_rate, target_sample_rate)

    # The model wants 32-bit floats.
    return audio.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Hugging Face model name")
    parser.add_argument("--audio_dir", default="audio",
                        help="Folder that contains the Actor_01, Actor_02, ... folders")
    parser.add_argument("--out", required=True,
                        help="Where to save the CSV file")
    args = parser.parse_args()

    print("Loading model:", args.model)
    classifier = pipeline("audio-classification", model=args.model)

    # Ask the model what sample rate it needs instead of guessing.
    target_sample_rate = classifier.feature_extractor.sampling_rate
    print("This model expects audio at", target_sample_rate, "Hz")

    # Find every .wav file inside the Actor_* folders.
    audio_dir = Path(args.audio_dir)
    wav_files = sorted(audio_dir.glob("Actor_*/*.wav"))
    if len(wav_files) == 0:
        print("No .wav files found. Check that --audio_dir points at the folder")
        print("that holds Actor_01, Actor_02, etc.")
        return
    print("Found", len(wav_files), "audio files")

    rows = []
    for i, wav_path in enumerate(wav_files, start=1):
        audio = load_audio_for_model(wav_path, target_sample_rate)

        # top_k=None gives us the score for every emotion, not just the best one.
        scores = classifier(audio, top_k=None)
        best = max(scores, key=lambda s: s["score"])

        row = {
            "Actor": wav_path.parent.name,
            "File Name": wav_path.name,
            "Emotion": best["label"],
            "Confidence Score": round(best["score"], 4),
        }
        # Save all the probabilities too, so we can look at them later.
        for s in scores:
            row[s["label"] + "_probability"] = round(s["score"], 4)
        rows.append(row)

        if i % 20 == 0 or i == len(wav_files):
            print("  processed", i, "of", len(wav_files))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    # These two numbers should match. If they do not, some files were skipped.
    print("Saved", len(df), "rows to", args.out)
    print("Files found was", len(wav_files), "- these should be the same number.")


if __name__ == "__main__":
    main()

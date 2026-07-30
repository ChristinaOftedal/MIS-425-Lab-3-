import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import soxr
from transformers import pipeline


def load_audio(wav_path, target_sr):
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = soxr.resample(audio, sr, target_sr)
    return audio.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio_dir", default="audio")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    classifier = pipeline("audio-classification", model=args.model)
    target_sr = classifier.feature_extractor.sampling_rate

    wav_files = sorted(Path(args.audio_dir).glob("Actor_*/*.wav"))
    if not wav_files:
        print("No wav files found in", args.audio_dir)
        return
    print("Found", len(wav_files), "files")

    rows = []
    for i, wav_path in enumerate(wav_files, start=1):
        audio = load_audio(wav_path, target_sr)
        scores = classifier(audio, top_k=None)
        best = max(scores, key=lambda s: s["score"])

        row = {
            "Actor": wav_path.parent.name,
            "File Name": wav_path.name,
            "Emotion": best["label"],
            "Confidence Score": round(best["score"], 4),
        }
        for s in scores:
            row[s["label"] + "_probability"] = round(s["score"], 4)
        rows.append(row)

        if i % 20 == 0 or i == len(wav_files):
            print(i, "of", len(wav_files))

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("Saved", len(rows), "rows to", args.out)


if __name__ == "__main__":
    main()

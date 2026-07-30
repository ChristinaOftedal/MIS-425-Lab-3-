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
import torch
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
    # FIX: the Actor_01, Actor_02, ... folders live inside an "audio" subfolder on GitHub
    # (https://github.com/ChristinaOftedal/MIS-425-Lab-3-/tree/main/audio), not at the repo
    # root. This repo gets cloned into /workspaces/MIS-425-Lab-3-/ in the Codespace, so the
    # actor folders end up at /workspaces/MIS-425-Lab-3-/audio/Actor_01, etc. Default to
    # that path so you don't have to pass --audio_dir every time; override it if you're
    # running somewhere else (e.g. a local clone).
    parser.add_argument("--audio_dir", default="/workspaces/MIS-425-Lab-3-/audio",
                        help="Folder that contains the Actor_01, Actor_02, ... folders "
                             "(defaults to the repo's audio/ folder as cloned in the Codespace)")
    parser.add_argument("--out", required=True,
                        help="Where to save the CSV file")
    parser.add_argument("--checkpoint_every", type=int, default=25,
                        help="Write partial results to --out every N files, so a crash "
                             "partway through doesn't lose everything already processed")
    args = parser.parse_args()

    # FIX: the script's own example (`--out results/baseline_results.csv`) crashes with
    # FileNotFoundError on a fresh checkout, because pandas won't create missing parent
    # directories for you. Create the output folder up front.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading model:", args.model)
    # FIX: use a GPU automatically if one is available instead of always running on CPU.
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("audio-classification", model=args.model, device=device)

    # Ask the model what sample rate it needs instead of guessing.
    target_sample_rate = classifier.feature_extractor.sampling_rate
    print("This model expects audio at", target_sample_rate, "Hz")

    # Find every .wav file inside the Actor_* folders.
    # FIX: glob("Actor_*/*.wav") is case-sensitive and only looks one folder deep, so it
    # silently misses files saved as ".WAV" or nested in a subfolder. rglob + a
    # case-insensitive suffix check catches those too.
    audio_dir = Path(args.audio_dir)
    wav_files = sorted(
        p for p in audio_dir.glob("Actor_*/**/*") if p.suffix.lower() == ".wav" and p.is_file()
    )
    if len(wav_files) == 0:
        # FIX: print exactly where it looked (and whether that folder even exists), instead
        # of a generic message ‚Äî this is what makes a wrong --audio_dir instantly obvious
        # rather than a mystery, which is what happened here.
        print(f"No .wav files found under: {audio_dir.resolve()}")
        if not audio_dir.exists():
            print("  -> That folder does not exist. Check --audio_dir.")
        else:
            subfolders = sorted(p.name for p in audio_dir.iterdir() if p.is_dir())
            print(f"  -> Folder exists. Subfolders found there: {subfolders or '(none)'}")
        print("Expected a structure like: <audio_dir>/Actor_01/*.wav, <audio_dir>/Actor_02/*.wav, ...")
        return
    print("Found", len(wav_files), "audio files")

    rows = []
    skipped = []
    for i, wav_path in enumerate(wav_files, start=1):
        # FIX: a single unreadable/corrupted file used to crash the whole run and throw
        # away every prediction gathered so far, since nothing was saved until the very
        # end. Now we log the failure, skip just that file, and keep going.
        try:
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

        except Exception as file_error:
            print(f"  [SKIPPED] {wav_path.name}: {file_error}")
            skipped.append(wav_path.name)

        if i % 20 == 0 or i == len(wav_files):
            print("  processed", i, "of", len(wav_files))

        # FIX: checkpoint progress periodically so a crash near the end of a long run
        # (e.g. 1,440 files across all 24 RAVDESS actors) doesn't lose everything.
        if rows and (i % args.checkpoint_every == 0 or i == len(wav_files)):
            pd.DataFrame(rows).to_csv(out_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    # These two numbers should match minus anything skipped. If they don't, something
    # other than a skipped/corrupted file caused rows to go missing.
    print("Saved", len(df), "rows to", args.out)
    print("Files found was", len(wav_files), "-", len(skipped), "skipped -",
          len(wav_files) - len(skipped), "expected rows.")
    if skipped:
        print("Skipped files:", ", ".join(skipped))


if __name__ == "__main__":
    main()


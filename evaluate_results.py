"""
Scores the CSV files made by run_emotion_model.py against the real RAVDESS labels.

This prints accuracy, precision, recall and F1, saves a confusion matrix picture,
lists the emotions that get mixed up the most, and prints some wrong predictions
we can write about in the error analysis section.

Example (one model):
    python evaluate_results.py --results results/baseline_results.csv --names Baseline

Example (comparing both models, which is what Option 1 asks for):
    python evaluate_results.py --results results/baseline_results.csv results/new_model_results.csv --names Baseline NewModel
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save pictures to files instead of opening a window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# The 3rd number in a RAVDESS file name is the emotion.
# Example: 03-01-05-01-02-01-12.wav -> "05" -> angry
RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# Different models spell the same emotion differently, so we make them match.
LABEL_FIXES = {
    "fear": "fearful",
    "surprise": "surprised",
    "ang": "angry",
    "hap": "happy",
    "neu": "neutral",
    "sad": "sad",
}


def get_true_emotion(file_name):
    """Reads the real emotion out of a RAVDESS file name."""
    parts = file_name.replace(".wav", "").split("-")
    if len(parts) < 3:
        return "unknown"
    return RAVDESS_EMOTIONS.get(parts[2], "unknown")


def tidy_label(label):
    """Makes label spelling consistent, e.g. 'fear' becomes 'fearful'."""
    label = str(label).strip().lower()
    return LABEL_FIXES.get(label, label)


def score_one_model(csv_path, model_name):
    """Scores one results CSV and prints everything we need for the report."""
    print()
    print("=" * 60)
    print("RESULTS FOR:", model_name, " (file:", csv_path, ")")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    df["true_emotion"] = df["File Name"].apply(get_true_emotion)
    df["predicted_emotion"] = df["Emotion"].apply(tidy_label)

    y_true = df["true_emotion"]
    y_pred = df["predicted_emotion"]

    # --- Accuracy ---------------------------------------------------------
    # This is REAL accuracy: it compares the prediction to the true label.
    accuracy = accuracy_score(y_true, y_pred)
    n_classes = y_true.nunique()
    print()
    print("Total clips scored:", len(df))
    print("Accuracy:          %.4f  (%d out of %d correct)"
          % (accuracy, (y_true == y_pred).sum(), len(df)))
    print("Random guessing would be about %.4f (1 out of %d classes)"
          % (1 / n_classes, n_classes))

    average_confidence = df["Confidence Score"].mean()
    n_predicted_classes = y_pred.nunique()
    print("Average confidence: %.4f" % average_confidence)

    # Sanity check. If the model is barely more confident than a coin flip across
    # every single clip, it usually means the model did not load properly rather
    # than that the task is hard. When we loaded the class baseline model,
    # transformers printed a LOAD REPORT saying the classifier layer was MISSING
    # and got randomly initialized, and every confidence came out near 1/8.
    if average_confidence < 1.5 * (1 / max(n_predicted_classes, 1)):
        print()
        print("WARNING: the average confidence is very close to random guessing.")
        print("         Scroll up to the LOAD REPORT that transformers printed when")
        print("         the model loaded. If any layer says MISSING, the model's")
        print("         classifier was randomly initialized and these numbers are")
        print("         not meaningful.")

    # Warn if a model simply does not have one of the RAVDESS emotions.
    missing = sorted(set(y_true.unique()) - set(y_pred.unique()))
    if missing:
        print()
        print("NOTE: this model never predicted:", ", ".join(missing))
        print("      If the model has no such class, those clips can never be correct.")

    # --- Precision / recall / F1 ------------------------------------------
    print()
    print("Precision, recall and F1 for each emotion:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # --- Confusion matrix -------------------------------------------------
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(9, 8))
    picture = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(picture, label="Number of clips")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted emotion")
    ax.set_ylabel("True emotion")
    ax.set_title("Confusion Matrix - " + model_name)
    for r in range(len(labels)):
        for c in range(len(labels)):
            ax.text(c, r, matrix[r][c], ha="center", va="center")
    fig.tight_layout()

    Path("images").mkdir(exist_ok=True)
    picture_name = Path("images") / ("confusion_matrix_"
                                     + model_name.replace(" ", "_") + ".png")
    fig.savefig(picture_name, dpi=150)
    plt.close(fig)
    print("Saved confusion matrix picture:", picture_name)

    # --- Which emotions get mixed up the most -----------------------------
    wrong = df[df["true_emotion"] != df["predicted_emotion"]]
    print()
    print("Most common mix-ups (true -> predicted):")
    mixups = wrong.groupby(["true_emotion", "predicted_emotion"]).size()
    mixups = mixups.sort_values(ascending=False).head(5)
    for (true_label, pred_label), count in mixups.items():
        print("   %-10s -> %-10s  %d times" % (true_label, pred_label, count))

    # --- Accuracy for each actor (speaker) --------------------------------
    print()
    print("Accuracy for each actor:")
    for actor, group in df.groupby("Actor"):
        actor_acc = (group["true_emotion"] == group["predicted_emotion"]).mean()
        print("   %-10s %.4f  (%d clips)" % (actor, actor_acc, len(group)))

    # --- Example wrong predictions for the error analysis section ---------
    print()
    print("Example wrong predictions (the report needs at least 3):")
    for _, row in wrong.head(5).iterrows():
        print("   %s | true: %-10s predicted: %-10s confidence: %.4f"
              % (row["File Name"], row["true_emotion"],
                 row["predicted_emotion"], row["Confidence Score"]))

    return {
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Average Confidence": round(df["Confidence Score"].mean(), 4),
        "Clips Scored": len(df),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True,
                        help="One or more results CSV files")
    parser.add_argument("--names", nargs="+", default=None,
                        help="A short name for each CSV file")
    args = parser.parse_args()

    names = args.names
    if names is None:
        names = [Path(p).stem for p in args.results]
    if len(names) != len(args.results):
        print("Please give one name for each results file.")
        return

    summary_rows = []
    for csv_path, model_name in zip(args.results, names):
        summary_rows.append(score_one_model(csv_path, model_name))

    # --- Comparison table -------------------------------------------------
    # This is the table the assignment asks for in section 6.2.
    summary = pd.DataFrame(summary_rows)
    print()
    print("=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(summary.to_string(index=False))
    Path("results").mkdir(exist_ok=True)
    summary.to_csv(Path("results") / "comparison_table.csv", index=False)
    print()
    print("Saved results/comparison_table.csv")


if __name__ == "__main__":
    main()

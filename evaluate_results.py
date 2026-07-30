import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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

LABEL_FIXES = {
    "fear": "fearful",
    "surprise": "surprised",
    "ang": "angry",
    "hap": "happy",
    "neu": "neutral",
    "sad": "sad",
}


def get_true_emotion(file_name):
    parts = file_name.replace(".wav", "").split("-")
    if len(parts) < 3:
        return "unknown"
    return RAVDESS_EMOTIONS.get(parts[2], "unknown")


def tidy_label(label):
    return LABEL_FIXES.get(str(label).strip().lower(), str(label).strip().lower())


def score_one_model(csv_path, model_name):
    print()
    print(model_name, "-", csv_path)

    df = pd.read_csv(csv_path)
    df["true_emotion"] = df["File Name"].apply(get_true_emotion)
    df["predicted_emotion"] = df["Emotion"].apply(tidy_label)

    y_true = df["true_emotion"]
    y_pred = df["predicted_emotion"]

    accuracy = accuracy_score(y_true, y_pred)
    avg_confidence = df["Confidence Score"].mean()

    print("Clips scored:", len(df))
    print("Accuracy: %.4f (%d of %d)" % (accuracy, (y_true == y_pred).sum(), len(df)))
    print("Random guessing: %.4f" % (1 / y_true.nunique()))
    print("Average confidence: %.4f" % avg_confidence)

    if avg_confidence < 1.5 * (1 / max(y_pred.nunique(), 1)):
        print("Confidence is near random, the model may not have loaded properly.")

    missing = sorted(set(y_true) - set(y_pred))
    if missing:
        print("Never predicted:", ", ".join(missing))

    print()
    print(classification_report(y_true, y_pred, zero_division=0))

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
    picture_name = Path("images") / ("confusion_matrix_" + model_name.replace(" ", "_") + ".png")
    fig.savefig(picture_name, dpi=150)
    plt.close(fig)
    print("Saved", picture_name)

    wrong = df[df["true_emotion"] != df["predicted_emotion"]]

    print()
    print("Most common mix-ups:")
    mixups = wrong.groupby(["true_emotion", "predicted_emotion"]).size()
    for (true_label, pred_label), count in mixups.sort_values(ascending=False).head(5).items():
        print("   %-10s -> %-10s %d" % (true_label, pred_label, count))

    print()
    print("Accuracy by actor:")
    for actor, group in df.groupby("Actor"):
        actor_acc = (group["true_emotion"] == group["predicted_emotion"]).mean()
        print("   %-10s %.4f (%d clips)" % (actor, actor_acc, len(group)))

    print()
    print("Some wrong predictions:")
    for _, row in wrong.head(5).iterrows():
        print("   %s true: %-10s predicted: %-10s conf: %.4f"
              % (row["File Name"], row["true_emotion"],
                 row["predicted_emotion"], row["Confidence Score"]))

    return {
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Average Confidence": round(avg_confidence, 4),
        "Clips Scored": len(df),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", default=None)
    args = parser.parse_args()

    names = args.names
    if names is None:
        names = [Path(p).stem for p in args.results]
    if len(names) != len(args.results):
        print("Give one name for each results file.")
        return

    summary_rows = []
    for csv_path, model_name in zip(args.results, names):
        summary_rows.append(score_one_model(csv_path, model_name))

    summary = pd.DataFrame(summary_rows)
    print()
    print("Comparison")
    print(summary.to_string(index=False))

    Path("results").mkdir(exist_ok=True)
    summary.to_csv(Path("results") / "comparison_table.csv", index=False)
    print("Saved results/comparison_table.csv")


if __name__ == "__main__":
    main()

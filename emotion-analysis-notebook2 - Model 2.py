# python#!/usr/bin/env python
# coding: utf-8

# # Simple Emotion Prediction Analysis
# Exploring the relationships between predicted emotions, probabilities, and actual emotions in our dataset.

import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Basic plot settings
plt.rcParams['figure.figsize'] = [12, 6]

# Define path options based on your previous CSV creation step
path_option_1 = Path("/workspaces/MIS-425-Lab-3-/emotion_results2.csv")
path_option_2 = Path("results2.csv")

# Automatically pick whichever file exists on your workspace
if path_option_1.exists():
    csv_file_path = path_option_1
elif path_option_2.exists():
    csv_file_path = path_option_2
else:
    raise FileNotFoundError(
        f"Could not find your results spreadsheet.\n"
        f"Please verify that {path_option_1.resolve()} has been created by your main pipeline."
    )

# Load and prepare the data
df = pd.read_csv(csv_file_path)

# Quick data cleanup to match column naming differences across lab formats
if 'filename' in df.columns:
    file_col = 'filename'
elif 'File Name' in df.columns:
    file_col = 'File Name'
else:
    raise KeyError("Could not find a filename or File Name column in your CSV.")

if 'emotion' not in df.columns and 'Emotion' in df.columns:
    df['emotion'] = df['Emotion']

if 'confidence' not in df.columns and 'Confidence Score' in df.columns:
    df['confidence'] = pd.to_numeric(df['Confidence Score'], errors='coerce')

# RAVDESS EMOTION CODE DICTIONARY MAPPING
ravdess_emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def extract_ravdess_emotion(filename: str) -> str:
    """Extracts the 3rd numerical value from a RAVDESS file name and maps it."""
    name_without_ext = filename.replace(".wav", "").replace(".WAV", "")
    parts = name_without_ext.split("-")
    
    if len(parts) >= 3:
        emotion_code = parts[2]
        return ravdess_emotions.get(emotion_code, "unknown")
    return "unknown"

# Extract actual emotion using the RAVDESS mapping structure
df['actual_emotion'] = df[file_col].apply(extract_ravdess_emotion)

print("Dataset Overview:")
print(f"Total samples: {len(df)}\n")
print("Actual emotions in dataset (Extracted via RAVDESS codes):")
print(df['actual_emotion'].value_counts())
print("\nPredicted emotions:")
print(df['emotion'].value_counts())
df.head()


# ## Distribution of Predictions vs Actual Emotions
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
df['actual_emotion'].value_counts().plot(kind='bar')
plt.title('Distribution of Actual Emotions')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
df['emotion'].value_counts().plot(kind='bar')
plt.title('Distribution of Predicted Emotions')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ## Probability Distributions by Actual Emotion
prob_cols = [col for col in df.columns if col.endswith('_probability')]
if prob_cols:
    plt.figure(figsize=(15, 6))
    for emotion in df['actual_emotion'].unique():
        emotion_data = df[df['actual_emotion'] == emotion]
        plt.figure(figsize=(10, 4))
        probs = emotion_data[prob_cols].mean()
        probs.plot(kind='bar')
        plt.title(f'Average Probabilities for Actual Emotion: {emotion}')
        plt.xlabel('Predicted Emotion')
        plt.ylabel('Average Probability')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
else:
    print("\nSkipping probability distributions plots: No columns ending in '_probability' found.")


# ## Confidence Distribution
if 'confidence' in df.columns and not df['confidence'].isna().all():
    plt.figure(figsize=(12, 5))
    df.boxplot(column='confidence', by='actual_emotion')
    plt.title('Confidence Scores by Actual Emotion')
    plt.suptitle('')  
    plt.xlabel('Actual Emotion')
    plt.ylabel('Confidence')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print("\nAverage confidence by actual emotion:")
    print(df.groupby('actual_emotion')['confidence'].mean().sort_values(ascending=False))
else:
    print("\nSkipping confidence graphs: Confidence values are missing or invalid.")


# ## Predicted vs Actual Patterns (Heatmap Setup)
prediction_patterns = pd.crosstab(df['actual_emotion'], df['emotion'])

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(prediction_patterns, cmap='YlOrRd')
fig.colorbar(im, label='Count')

ax.set_xticks(range(len(prediction_patterns.columns)))
ax.set_xticklabels(prediction_patterns.columns, rotation=45)
ax.set_yticks(range(len(prediction_patterns.index)))
ax.set_yticklabels(prediction_patterns.index)
ax.set_xlabel('Predicted Emotion')
ax.set_ylabel('Actual Emotion')
ax.set_title('Prediction Patterns Matrix')

for i in range(len(prediction_patterns.index)):
    for j in range(len(prediction_patterns.columns)):
        ax.text(j, i, prediction_patterns.iloc[i, j], ha='center', va='center')

plt.tight_layout()

# NEW: Automatically export the generated Heatmap image
image_out_path = Path("/workspaces/MIS-425-Lab-3-/prediction_patterns_heatmap2.png")
plt.savefig(image_out_path, dpi=300)
print(f"\n[EXPORT COMPLETE] Visual Heatmap saved to: {image_out_path.resolve()}")
plt.show()


# =====================================================================
# NEW: EXPORT RAVDESS SUMMARY REPORT TO A FILE
# =====================================================================
report_path = Path("/workspaces/MIS-425-Lab-3-/ravdess_summary_report2.txt")

try:
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("      RAVDESS EMOTION ANALYSIS SUMMARY REPORT\n")
        f.write("==================================================\n\n")
        f.write(f"Total Audio Samples Analyzed: {len(df)}\n\n")
        f.write("Accuracy / Most Common Model Predictions:\n")
        f.write("-" * 50 + "\n")
        
        for emotion in sorted(df['actual_emotion'].unique()):
            subset = df[df['actual_emotion'] == emotion]
            if not subset.empty and 'emotion' in subset.columns:
                most_common = subset['emotion'].mode().iloc[0]
                count = len(subset[subset['emotion'] == most_common])
                total = len(subset)
                accuracy_pct = (count / total) * 100
                f.write(f"Ground Truth [{emotion.upper()}]:\n")
                f.write(f"  -> Most Predicted: '{most_common}'\n")
                f.write(f"  -> Match Rate:     {count}/{total} samples ({accuracy_pct:.1f}%)\n\n")
                
    print(f"[EXPORT COMPLETE] Text performance summary report saved to: {report_path.resolve()}\n")
except Exception as e:
    print(f"\n[WARNING] Could not save summary report file: {e}")
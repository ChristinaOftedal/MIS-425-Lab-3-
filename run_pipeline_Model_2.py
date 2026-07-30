import os  # NEW: Added to change environment settings
# NEW: Disables multi-threading inside OpenBLAS to prevent the loop warning/hanging
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import csv  
import logging
import sys
from pathlib import Path
import torch
from transformers import pipeline

# NEW imports for analytics, reports, and data visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# CONFIGURATION & LOGGING SETUP
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Absolute workspace file system path pointing directly to your audio directories
DATASET_BASE_DIRS = [Path(f"/workspaces/MIS-425-Lab-3-/Actor_{i:02d}") for i in range(1, 5)]

# Output file system paths for artifacts
CSV_OUTPUT_PATH = Path("/workspaces/MIS-425-Lab-3-/emotion_results_Model_2.csv")
REPORT_OUTPUT_PATH = Path("/workspaces/MIS-425-Lab-3-/emotion_summary_report_Model_2.md")
HEATMAP_OUTPUT_PATH = Path("/workspaces/MIS-425-Lab-3-/emotion_distribution_heatmap_Model_2.png")

# Mapping RAVDESS file codes (3rd numeric block) to match model label names for true metrics
RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",  # Note: The model uses 'neutral' or others; mapped for alignment
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
}

# =====================================================================
# MAIN ENTRY POINT
# =====================================================================
def main() -> None:
    # 1. Initialize the Hugging Face emotion recognition pipeline
    logger.info("Initializing Hugging Face audio classification model...")
    pipe = pipeline(
        "audio-classification", model="r-f/wav2vec-english-speech-emotion-recognition"
    )

    # Blank list to store all classification results data
    all_results = []

    # Loop through each of the actor directories
    for dataset_dir in DATASET_BASE_DIRS:
        logger.info(f"Scanning target drive directory: {dataset_dir.resolve()}")

        # 2. Check if the directory exists before proceeding
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            logger.warning(
                f"Directory path not found: {dataset_dir.resolve()}\n"
                f"Skipping this folder. Please ensure it exists inside your main project folder."
            )
            continue

        # 3. Gather all .wav files automatically (case-insensitive check)
        audio_files = [
            p for p in dataset_dir.rglob("*") if p.suffix.lower() == ".wav" and p.is_file()
        ]

        if not audio_files:
            logger.warning(f"Zero '.wav' files were found inside: {dataset_dir.resolve()}")
            continue 

        print(f"\nDiscovered {len(audio_files)} local files to evaluate in {dataset_dir.name}...")
        print("-" * 50)

        # 4. Iterate over every automatically discovered local file path
        for local_file_path in audio_files:
            print(f"Processing: {local_file_path.name}")
            print(f"Path: {local_file_path.resolve()}")

            try:
                # 5. Extract ground truth emotion from RAVDESS standard filename format (e.g., 03-01-05-...)
                # 3rd block represents the emotion index
                filename_parts = local_file_path.stem.split("-")
                ground_truth = "Unknown"
                if len(filename_parts) >= 3 and filename_parts[2] in RAVDESS_EMOTIONS:
                    ground_truth = RAVDESS_EMOTIONS[filename_parts[2]]

                # 6. Pass absolute file path string directly to the pipeline
                results = pipe(str(local_file_path.resolve()))

                # Variables to store what we find
                emotion_label = "Unknown"
                confidence_score = 0.0

                # 7. Format and display output safely
                if isinstance(results, list) and len(results) > 0:
                    top_prediction = results[0]
                    emotion_label = top_prediction['label']
                    confidence_score = top_prediction['score']
                    print(f"Result: {emotion_label} ({confidence_score:.4f}) | True: {ground_truth}")
                else:
                    print(f"Result: {results}")
                print("-" * 50)

                # Save this specific row's data into our master list
                all_results.append({
                    "Actor": dataset_dir.name,
                    "File Name": local_file_path.name,
                    "True Emotion": ground_truth,
                    "Predicted Emotion": emotion_label,
                    "Confidence Score": confidence_score
                })

            except Exception as file_error:
                logger.error(f"Error reading file {local_file_path.name}: {file_error}")
                print("-" * 50)

    # 8. Export CSV data and perform advanced reporting/heatmaps
    if all_results:
        logger.info(f"Writing results to CSV: {CSV_OUTPUT_PATH.resolve()}")
        try:
            df = pd.DataFrame(all_results)
            # Save raw dataset to CSV
            df.to_csv(CSV_OUTPUT_PATH, index=False)
            logger.info("CSV export completed successfully.")
            
            # --- GENERATING THE REPORT ---
            logger.info(f"Generating summary report at: {REPORT_OUTPUT_PATH.resolve()}")
            total_processed = len(df)
            avg_confidence = df["Confidence Score"].mean()
            
            # Calculate precision metrics where ground truth matches predictions
            valid_gt = df[df["True Emotion"] != "Unknown"]
            accuracy_str = "N/A (No valid RAVDESS labels found)"
            if not valid_gt.empty:
                correct_preds = (valid_gt["True Emotion"] == valid_gt["Predicted Emotion"]).sum()
                accuracy_str = f"{(correct_preds / len(valid_gt)) * 100:.2f}%"

            report_content = f"""# Speech Emotion Recognition Execution Report
## System Overview
- **Model Used**: `r-f/wav2vec-english-speech-emotion-recognition`
- **Total Audio Files Processed**: {total_processed}
- **Average Prediction Confidence**: {avg_confidence:.4f}
- **Subset Match Accuracy (True vs Predicted)**: {accuracy_str}

## Breakdown by Actor
{df.groupby("Actor")["Confidence Score"].agg(["count", "mean"]).rename(columns={"count": "Files Evaluated", "mean": "Avg Confidence"}).to_markdown()}

## Predicted Emotion Frequency Distributions
{df["Predicted Emotion"].value_counts().to_markdown()}
"""
            with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
                f.write(report_content)
                
            # --- GENERATING THE HEATMAP ---
            logger.info(f"Generating analytics visualization heatmap at: {HEATMAP_OUTPUT_PATH.resolve()}")
            plt.figure(figsize=(10, 8))
            
            # Check if we can build a proper Confusion Matrix, or fallback to an Actor-Emotion intensity map
            if not valid_gt.empty and len(valid_gt["True Emotion"].unique()) > 1:
                # Confusion Matrix Heatmap
                matrix_data = pd.crosstab(df["True Emotion"], df["Predicted Emotion"])
                sns.heatmap(matrix_data, annot=True, cmap="YlGnBu", fmt="d", cbar=True)
                plt.title("Confusion Matrix Heatmap (True Emotion vs Predicted)")
                plt.ylabel("Actual Emotion Label")
            else:
                # Fallback: Actor vs Predicted Emotion density distribution
                pivot_data = pd.crosstab(df["Actor"], df["Predicted Emotion"])
                sns.heatmap(pivot_data, annot=True, cmap="Purples", fmt="d", cbar=True)
                plt.title("Emotion Distributions Detected Across Tracked Actors")
                plt.ylabel("Target Folders")
                
            plt.xlabel("Predicted Classification Label")
            plt.tight_layout()
            plt.savefig(HEATMAP_OUTPUT_PATH, dpi=300)
            plt.close()
            
            logger.info("Report generation and heatmaps rendered completely.")

        except Exception as reporting_error:
            logger.error(f"Failed to compile report metrics/visuals: {reporting_error}")
    else:
        logger.warning("No data was collected, skipping reporting steps.")

    logger.info("Local path emotion classification pipeline execution complete.")

if __name__ == "__main__":
    main()
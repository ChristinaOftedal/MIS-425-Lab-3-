import os  # NEW: Added to change environment settings
# NEW: Disables multi-threading inside OpenBLAS to prevent the loop warning/hanging
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import csv  
import logging
import sys
from pathlib import Path
import torch
from transformers import pipeline

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

# NEW: Define where to save the final spreadsheet
CSV_OUTPUT_PATH = Path("/workspaces/MIS-425-Lab-3-/emotion_results.csv")

# =====================================================================
# MAIN ENTRY POINT
# =====================================================================
def main() -> None:
    # 1. Initialize the Hugging Face emotion recognition pipeline (UPDATED MODEL)
    logger.info("Initializing Hugging Face audio classification model...")
    pipe = pipeline(
        "audio-classification", model="r-f/wav2vec-english-speech-emotion-recognition"
    )

    # NEW: Create a blank list to store all classification results data
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
                # 5. FIXED: Pass the clean string path instead of raw bytes.
                # The pipeline requires a filepath to handle audio decoding and 16kHz resampling.
                audio_target = str(local_file_path.resolve())

                # 6. Pass target path directly to the model pipeline
                results = pipe(audio_target)

                # NEW: Variables to store what we find
                emotion_label = "Unknown"
                confidence_score = 0.0

                # 7. Format and display output safely
                if isinstance(results, list) and len(results) > 0:
                    top_prediction = results[0]
                    emotion_label = top_prediction['label']
                    confidence_score = top_prediction['score']
                    print(f"Result: {emotion_label} ({confidence_score:.4f})")
                else:
                    print(f"Result: {results}")
                print("-" * 50)

                # NEW: Save this specific row's data into our master list
                all_results.append({
                    "Actor": dataset_dir.name,
                    "File Name": local_file_path.name,
                    "Emotion": emotion_label,
                    "Confidence Score": f"{confidence_score:.4f}"
                })

            except Exception as file_error:
                logger.error(f"Error reading file {local_file_path.name}: {file_error}")
                print("-" * 50)

    # NEW: Write all gathered results out to the CSV file once loops finish
    if all_results:
        logger.info(f"Writing results to CSV: {CSV_OUTPUT_PATH.resolve()}")
        try:
            with open(CSV_OUTPUT_PATH, mode="w", newline="", encoding="utf-8") as csv_file:
                fieldnames = ["Actor", "File Name", "Emotion", "Confidence Score"]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                
                writer.writeheader()
                writer.writerows(all_results)
            logger.info("CSV export completed successfully.")
        except Exception as csv_error:
            logger.error(f"Failed to write CSV file: {csv_error}")
    else:
        logger.warning("No data was collected, skipping CSV export.")

    logger.info("Local path emotion classification pipeline execution complete.")

if __name__ == "__main__":
    main()
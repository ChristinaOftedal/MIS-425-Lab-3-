#!/usr/bin/env python3
"""Librosa Audio Feature Extraction Pipeline for GitHub Codespaces.

This script safely locates, loads, and processes audio files from the 
explicit absolute workspace path: /workspaces/MIS-425-Lab-3-/Actor_01
"""

import logging
import sys
from pathlib import Path
from typing import Generator
import librosa
import numpy as np

# =====================================================================
# CONFIGURATION & LOGGING SETUP
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Explicit absolute path to your GitHub Codespace audio directory
AUDIO_DATA_DIR = Path("/workspaces/MIS-425-Lab-3-/Actor_01")

# Librosa configuration parameters
SAMPLE_RATE = None  # None preserves the native sample rate of the audio file
NUM_MFCC = 13       # Number of Mel-Frequency Cepstral Coefficients to extract


# =====================================================================
# CORE PIPELINE FUNCTIONS
# =====================================================================
def get_audio_files(directory: Path, extension: str = "*.wav") -> Generator[Path, None, None]:
    

    Args:
        directory (Path): Path to the folder containing audio files.
        extension (str): File extension pattern. Defaults to "*.wav".

    Yields:
        Generator[Path, None, None]: Iterator over verified file paths.

    Raises:
        FileNotFoundError: If the directory does not exist or is placed incorrectly.
    """
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(
            f"Required directory '{directory.name}' not found at explicit path: {directory.resolve()}\n"
            f"Please verify that the 'Actor_01' folder is located exactly inside /workspaces/MIS-425-Lab-3-/"
        )
    yield from directory.glob(extension)


def extract_features(file_path: Path) -> dict:
    """Loads a single audio file via librosa and extracts foundational features.

    Args:
        file_path (Path): Path to the audio file.

    Returns:
        dict: Extracted features including audio duration and MFCC means.
    """
    logger.info(f"Loading file via Librosa: {file_path.name}")
    
    # 1. Load the audio file from Codespace storage
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    
    # 2. Extract Basic Metadata
    duration = librosa.get_duration(y=y, sr=sr)
    
    # 3. Extract MFCC Features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=NUM_MFCC)
    mfccs_scaled = np.mean(mfccs.T, axis=0) # Take the mean across time frames
    
    return {
        "filename": file_path.name,
        "sample_rate": sr,
        "duration_seconds": round(duration, 2),
        "mfcc_mean": mfccs_scaled.tolist()
    }


def process_single_audio_file(file_path: Path) -> None:
    """Wrapper function executing the feature extraction pipeline on a file.

    Args:
        file_path (Path): Absolute path to the audio file.
    """
    try:
        # Run extraction
        features = extract_features(file_path)
        
        # Log successful completion metrics
        logger.info(
            f"Success: {features['filename']} | "
            f"SR: {features['sample_rate']}Hz | "
            f"Length: {features['duration_seconds']}s"
        )
        
        # -----------------------------------------------------------
        # 
        # -----------------------------------------------------------

    except Exception as error:
        # Prevents one corrupted file from stopping your entire pipeline run
        logger.error(f"Skipping {file_path.name} due to a processing error: {error}", exc_info=True)


# =====================================================================
# ENTRY POINT
# =====================================================================
def main() -> None:
    """Main execution block to orchestrate the Codespace audio pipeline."""
    logger.info("Initializing Codespace Audio Pipeline...")
    logger.info(f"Targeting explicit path: {AUDIO_DATA_DIR.resolve()}")

    try:
        audio_files = list(get_audio_files(AUDIO_DATA_DIR, extension="*.wav"))
        
        if not audio_files:
            logger.warning(f"No '.wav' files found inside '{AUDIO_DATA_DIR.name}'. Pipeline exiting.")
            return

        logger.info(f"Discovered {len(audio_files)} target audio tracks. Commencing batch process...")

        for file_path in audio_files:
            process_single_audio_file(file_path)

        logger.info("Codespace processing pipeline batch run finished cleanly.")

    except FileNotFoundError as fnf_error:
        logger.critical(fnf_error)
        sys.exit(1)
    except Exception as unexpected_error:
        logger.critical(f"A catastrophic pipeline failure occurred: {unexpected_error}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
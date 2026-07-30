#!/usr/bin/env python3
import logging
import sys
from pathlib import Path
import librosa
import numpy as np

# Configure Logging to force visibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Base path
WORKSPACE_ROOT = Path("/workspaces/MIS-425-Lab-3-")

def main():
    logger.info("=== STARTING AGGRESIVE FILE SEARCH ===")
    
    # Check if the folder exists at all
    if not WORKSPACE_ROOT.exists():
        logger.error(f"CRITICAL: The workspace path '{WORKSPACE_ROOT}' does not exist!")
        return

    logger.info(f"Scanning workspace root: {WORKSPACE_ROOT.resolve()}")

    # 1. Broadly scan for ANY wav file anywhere in the workspace (ignores case)
    # This searches subfolders recursively
    audio_files = []
    for p in WORKSPACE_ROOT.rglob("*"):
        if p.suffix.lower() == ".wav" and p.is_file():
            audio_files.append(p)

    if not audio_files:
        logger.warning("CRITICAL FAILURE: Zero .wav or .WAV files were found anywhere in the directory tree.")
        logger.info("Printing directory contents to see what you actually have:")
        for item in WORKSPACE_ROOT.iterdir():
            logger.info(f" - Found item in root: {item.name} ({'Folder' if item.is_dir() else 'File'})")
        return

    logger.info(f"Success! Found {len(audio_files)} total audio files across your workspace.")
    logger.info("Commencing Librosa feature processing on first discovered tracks...")

    # 2. Process whatever it found
    for file_path in audio_files[:5]:  # Safety limit to first 5 for testing
        try:
            logger.info(f"Processing target: {file_path.relative_to(WORKSPACE_ROOT)}")
            y, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            logger.info(f" -> Load Success! Duration: {duration:.2f}s | Sample Rate: {sr}Hz")
        except Exception as e:
            logger.error(f" -> Error reading {file_path.name}: {e}")

if __name__ == "__main__":
    main()
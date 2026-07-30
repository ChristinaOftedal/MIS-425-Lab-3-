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
        logger.warning("CRITICAL FAILURE: Zero .
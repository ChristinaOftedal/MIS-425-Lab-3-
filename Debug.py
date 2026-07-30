import io
from pathlib import Path
import torch
from transformers import pipeline

# 1. Initialize pipeline on CPU
pipe = pipeline(
    "audio-classification", 
    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    device=-1
)

# 2. Correct Windows Root Syntax for Python
DATASET_BASE_DIR = Path("C:/RAVDESS/Audio_Speech_Actors_01-24") 

print(f"Checking directory: {DATASET_BASE_DIR.resolve()}")

# --- AUTOMATIC PATH DIAGNOSTIC ---
if not DATASET_BASE_DIR.exists():
    print("❌ ERROR: Python cannot find this path. Let's check what IS in C:/RAVDESS:")
    backup_root = Path("C:/RAVDESS")
    if backup_root.exists():
        for item in backup_root.iterdir():
            print(f"  Found folder/file: {item.name}")
    else:
         print("❌ ERROR: Even C:/RAVDESS does not exist. Check your spelling.")
else:
    print("✅ Path exists! Printing the immediate contents of this folder:")
    for item in DATASET_BASE_DIR.iterdir():
        print(f"  -> {item.name}")
# ---------------------------------

# 3. Scan the folder and gather all .wav files
audio_files = list(DATASET_BASE_DIR.rglob("*.wav"))

print(f"\nDiscovered {len(audio_files)} local files to evaluate...")
print("-" * 50)

# 4. Process files if found
for local_file_path in audio_files:
    print(f"Processing: {local_file_path.name}")
    try:
        with open(local_file_path, "rb") as f:
            audio_bytes = f.read()
            
        results = pipe(audio_bytes)
        
        if isinstance(results, list) and len(results) > 0:
            top_prediction = results[0]
            print(f"Result: {top_prediction['label']} ({top_prediction['score']:.4f})")
        else:
            print(f"Result: {results}")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error processing {local_file_path.name}: {e}")
        print("-" * 50)
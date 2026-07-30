import io
from pathlib import Path
import torch
from transformers import pipeline

# 1. Initialize the Hugging Face emotion recognition pipeline
pipe = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

# 2. Define the absolute path to your dataset folder on the C drive
DATASET_BASE_DIR = Path("C:/RAVDESS/Audio_Speech_Actors_01-24")

# 3. Scan the folder and gather all .wav files automatically
# rglob("*.wav") searches through the main directory and all subfolders (like Actor_01, Actor_02)
audio_files = list(DATASET_BASE_DIR.rglob("*.wav"))

print(f"\nDiscovered {len(audio_files)} local files to evaluate...")
print("-" * 50)

# 4. Iterate over every automatically discovered file path
for local_file_path in audio_files:
    print(f"Processing: {local_file_path.name}")
    print(f"Path: {local_file_path}")
    
    try:
        # 5. Read the local file as raw bytes directly into memory
        with open(local_file_path, "rb") as f:
            audio_bytes = f.read()
            
        # 6. Pass raw binary data array directly to the model pipeline
        results = pipe(audio_bytes)
        
        # 7. Format and display output safely
        if isinstance(results, list) and len(results) > 0:
            top_prediction = results[0]
            print(f"Result: {top_prediction['label']} ({top_prediction['score']:.4f})")
        else:
            print(f"Result: {results}")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error processing {local_file_path.name}: {e}")
        print("-" * 50)
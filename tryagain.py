import requests
from transformers import pipeline

# 1. Initialize the Hugging Face emotion recognition pipeline on CPU
print("Loading Wav2Vec2 model pipeline...")
pipe = pipeline(
    "audio-classification", 
    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    device=-1
)

# 2. Hardcoded list of files to loop through
audio_files = [
    "03-01-01-01-01-01-01.wav", 
    "03-01-01-01-01-02-01.wav", 
    "03-01-01-01-02-01-01.wav",
    "03-01-01-01-02-02-01.wav",
    "03-01-08-02-02-01-01.wav",
    "03-01-08-02-02-02-01.wav"
]

print(f"\nEvaluating {len(audio_files)} files dynamically over network...")
print("-" * 50)

# 3. Process each file
for file_name in audio_files:
    # HARDCODED FIX: The full, working URL path is typed directly out here
    stream_url = f"https://githubusercontent.com{file_name}"
    
    try:
        # Download raw bytes safely from your GitHub repo
        response = requests.get(stream_url, timeout=15)
        response.raise_for_status()
        
        # Pass raw binary data array directly to the model pipeline
        results = pipe(response.content)
        
        # Format and display output safely
        if isinstance(results, list) and len(results) > 0:
            top_prediction = results[0]
            print(f"{file_name} -> Result: {top_prediction['label']} ({top_prediction['score']:.4f})")
        else:
            print(f"{file_name} -> Result: {results}")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error handling {file_name}: {e}")
        print("-" * 50)
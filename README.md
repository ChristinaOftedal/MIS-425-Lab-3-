MIS 425 Lab 3 - Speech and Emotion

Setup:

pip install torch transformers soxr soundfile pandas numpy matplotlib scikit-learn

Run a model on the audio:

python run_emotion_model.py --model MODEL_NAME --out results/my_results.csv

Score the results:

python evaluate_results.py --results results/my_results.csv --names MyModel

To compare two models pass both files:

python evaluate_results.py --results results/a.csv results/b.csv --names ModelA ModelB

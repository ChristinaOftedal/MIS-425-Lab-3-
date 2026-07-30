MIS 425 Lab 3 - Speech and Emotion

Setup:

[Contribute to the Lab Report Template Google Doc](https://docs.google.com/document/d/1i43yxuwl4kXb0z1Qn4dh8r7SzrrLpc0e5On3Pn-qgBM/edit?usp=sharing)

## Setup

```bash
pip install torch transformers soxr soundfile pandas numpy matplotlib scikit-learn

Run a model on the audio:

python run_emotion_model.py --model MODEL_NAME --out results/my_results.csv

Score the results:

python evaluate_results.py --results results/my_results.csv --names MyModel

To compare two models pass both files:

python evaluate_results.py --results results/a.csv results/b.csv --names ModelA ModelB

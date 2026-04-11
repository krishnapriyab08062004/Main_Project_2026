import os
import sys
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))
from models_improved import AttentionLayer
from app import extract_features

# Paths
BASE_DIR = os.path.join(os.getcwd(), "src")
MODEL_PATH = os.path.join(BASE_DIR, "models", "checkpoints", "cnn_lstm_attention_finetuned_best.keras")
SCALER_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "scaler.pkl")
LABEL_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "label_encoder.pkl")

print("Loading model and dependencies for verification...")
model = load_model(MODEL_PATH, custom_objects={"AttentionLayer": AttentionLayer})
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
with open(LABEL_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# Find some existing files to verify
test_files = []
actor_01_path = "data/raw/audio_speech_actors_01-24/Actor_01"
if os.path.exists(actor_01_path):
    wavs = [os.path.join(actor_01_path, f) for f in os.listdir(actor_01_path) if f.endswith(".wav")]
    test_files.extend(wavs[:5])

# Add the 2 synthetic ones we managed to generate
synthetic_path = "data/synthetic_tests"
if os.path.exists(synthetic_path):
    s_wavs = [os.path.join(synthetic_path, f) for f in os.listdir(synthetic_path) if f.endswith(".wav")]
    test_files.extend(s_wavs)

def run_verification(files):
    print(f"\n{'File':<50} | {'Prediction':<15} | {'Confidence'}")
    print("-" * 80)
    for f in files:
        if not os.path.exists(f):
            continue
        
        try:
            features = extract_features(f)
            preds = model.predict(features, verbose=0)
            pred_class = np.argmax(preds)
            emotion = label_encoder.inverse_transform([pred_class])[0]
            confidence = np.max(preds) * 100
            print(f"{os.path.basename(f):<50} | {emotion:<15} | {confidence:>8.2f}%")
        except Exception as e:
            print(f"Error testing {f}: {e}")

if __name__ == "__main__":
    if not test_files:
        print("No files found to verify.")
    else:
        run_verification(test_files)

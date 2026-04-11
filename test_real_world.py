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
MODEL_PATH = os.path.join(BASE_DIR, "models", "checkpoints", "cnn_lstm_attention_best.keras")
SCALER_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "scaler.pkl")
LABEL_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "label_encoder.pkl")

print("Loading model and dependencies...")
model = load_model(MODEL_PATH, custom_objects={"AttentionLayer": AttentionLayer})
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
with open(LABEL_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# Files to test (Mix of uploads and data)
test_files = [
    "src/uploads/OAF_bought_sad.wav",
    "src/uploads/i_am_krishna_angry.wav",
    "src/uploads/angry_sample.wav",
    "src/uploads/1001_IOM_NEU_XX.wav",
    "src/uploads/1001_IEO_SAD_MD.wav",
    "src/uploads/1001_IEO_HAP_LO.wav",
    "src/uploads/1001_IEO_ANG_MD.wav",
    "src/uploads/1001_DFA_SAD_XX.wav",
    "src/uploads/1001_DFA_HAP_XX.wav",
    "src/uploads/1001_DFA_ANG_XX.wav",
]

# Add 10 more from RAVDESS actors if available
actor_dirs = [d for d in os.listdir("data/raw/audio_speech_actors_01-24") if d.startswith("Actor")]
for actor in actor_dirs[:2]: # Take from first two actors
    actor_path = os.path.join("data/raw/audio_speech_actors_01-24", actor)
    wavs = [f for f in os.listdir(actor_path) if f.endswith(".wav")]
    test_files.extend([os.path.join(actor_path, w) for w in wavs[:5]])

def run_test(files, use_trim=False):
    print(f"\nModel: {os.path.basename(MODEL_PATH)}")
    print(f"Trimming: {'ENABLED' if use_trim else 'DISABLED'}")
    print(f"\n{'File':<40} | {'Prediction':<15} | {'Confidence'}")
    print("-" * 75)
    for f in files:
        if not os.path.exists(f):
            # print(f"File not found: {f}")
            continue
        
        try:
            features = extract_features(f)
            preds = model.predict(features, verbose=0)
            pred_class = np.argmax(preds)
            emotion = label_encoder.inverse_transform([pred_class])[0]
            confidence = np.max(preds) * 100
            print(f"{os.path.basename(f):<40} | {emotion:<15} | {confidence:>8.2f}%")
        except Exception as e:
            print(f"Error testing {f}: {e}")

if __name__ == "__main__":
    print("\n--- TEST 3: STANDARD Model, No Trim ---")
    run_test(test_files[:30], use_trim=False)
    print("\n--- TEST 4: STANDARD Model, WITH Trim ---")
    run_test(test_files[:30], use_trim=True)

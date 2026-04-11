import os
import sys
import numpy as np
import librosa
import pickle
import glob
from tensorflow.keras.models import load_model

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))
from models_improved import AttentionLayer
from app import extract_features

# Configuration
CHECKPOINT_DIR = "src/models/checkpoints"
DATASET_PATH = "data/raw/audio_speech_actors_01-24" # Using RAVDESS for known labels
SCALER_PATH = "data/processed/scaler.pkl"
LABEL_PATH = "data/processed/label_encoder.pkl"

EMOTION_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def load_dependencies():
    print("Loading dependencies...")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(LABEL_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    return scaler, label_encoder

def get_actual_emotion(filename):
    parts = os.path.basename(filename).replace(".wav", "").split("-")
    if len(parts) >= 3:
        return EMOTION_MAP.get(parts[2], "unknown")
    return "unknown"

def evaluate_model(model_path, test_files):
    print(f"\nEvaluating Model: {os.path.basename(model_path)}")
    try:
        model = load_model(model_path, custom_objects={"AttentionLayer": AttentionLayer})
    except Exception as e:
        print(f"  Error loading model: {e}")
        return None

    correct = 0
    total = 0
    
    for f in test_files:
        actual = get_actual_emotion(f)
        if actual == "unknown": continue
        
        try:
            features = extract_features(f)
            preds = model.predict(features, verbose=0)
            pred_class = np.argmax(preds)
            predicted = label_encoder.inverse_transform([pred_class])[0]
            
            if predicted.lower() == actual.lower():
                correct += 1
            total += 1
        except Exception as e:
            print(f"  Error processing {f}: {e}")
            
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy

if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        sys.exit(1)

    scaler, label_encoder = load_dependencies()
    
    # Get a diverse set of test files (unseen by convention - taking last actors)
    all_wavs = glob.glob(os.path.join(DATASET_PATH, "Actor_2*", "*.wav")) # Actors 20-24
    if not all_wavs:
        all_wavs = glob.glob(os.path.join(DATASET_PATH, "**", "*.wav"), recursive=True)
    
    import random
    test_files = random.sample(all_wavs, min(50, len(all_wavs)))
    print(f"Testing with {len(test_files)} files...")

    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "*.keras"))
    results = {}
    
    for cp in checkpoints:
        acc = evaluate_model(cp, test_files)
        if acc is not None:
            results[os.path.basename(cp)] = acc

    print("\n" + "="*50)
    print("FINAL RANKING (Accuracy on Unseen Data)")
    print("="*50)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for name, acc in sorted_results:
        print(f"{name:<40} | {acc:>6.2f}%")

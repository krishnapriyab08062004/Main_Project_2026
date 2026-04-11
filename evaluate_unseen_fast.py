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
DATASET_PATH = "data/raw/audio_speech_actors_01-24" 
SCALER_PATH = "data/processed/scaler.pkl"
LABEL_PATH = "data/processed/label_encoder.pkl"

EMOTION_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def load_dependencies():
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

def evaluate_model_fast(model_path, test_files, label_encoder):
    print(f"\nEvaluating Model: {os.path.basename(model_path)}")
    try:
        model = load_model(model_path, custom_objects={"AttentionLayer": AttentionLayer})
    except Exception as e:
        print(f"  Error: {e}")
        return None

    all_features = []
    actual_labels = []
    
    for f in test_files:
        actual = get_actual_emotion(f)
        if actual == "unknown": continue
        
        try:
            features = extract_features(f)
            all_features.append(features)
            actual_labels.append(actual.lower())
        except:
            continue
            
    if not all_features: return 0
    
    # Predict in batch
    X = np.concatenate(all_features, axis=0) # (N, 128, 101)
    preds = model.predict(X, verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    predicted_labels = [label_encoder.classes_[p].lower() for p in pred_classes]
    
    correct = sum(1 for p, a in zip(predicted_labels, actual_labels) if p == a)
    accuracy = (correct / len(actual_labels)) * 100
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{len(actual_labels)})")
    return accuracy

if __name__ == "__main__":
    scaler, label_encoder = load_dependencies()
    
    # Focus on Actors 23 and 24 (likely unseen)
    test_files = glob.glob(os.path.join(DATASET_PATH, "Actor_2[34]", "*.wav"))
    if not test_files:
        test_files = glob.glob(os.path.join(DATASET_PATH, "**", "*.wav"), recursive=True)[:50]
        
    print(f"Testing with {len(test_files)} unseen files...")

    # Focus on the most promising models
    key_models = [
        "cnn_lstm_attention_best.keras",
        "cnn_lstm_v2_best.keras",
        "cnn_dnn_v2_best.keras",
        "cnn_lstm_attention_finetuned_best.keras"
    ]
    
    results = {}
    for name in key_models:
        path = os.path.join(CHECKPOINT_DIR, name)
        if os.path.exists(path):
            acc = evaluate_model_fast(path, test_files, label_encoder)
            if acc is not None:
                results[name] = acc

    print("\n" + "="*50)
    print("BEST MODELS FOR UNSEEN DATA")
    print("="*50)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:<40} | {acc:>6.2f}%")

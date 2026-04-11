import os
import sys
import numpy as np
import librosa
import pickle
import glob
import random
from tensorflow.keras.models import load_model

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))
from models_improved import AttentionLayer

# Configuration
CHECKPOINT_DIR = "src/models/checkpoints"
TEST_DATA_PATH = "data/raw/test" 
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

def extract_features_standardized(audio_path, scaler, max_len=128, n_mfcc=40, sr=22050, padding='trailing'):
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    audio = audio - np.mean(audio)
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T
    zcr = librosa.feature.zero_crossing_rate(audio).T
    rms = librosa.feature.rms(y=audio).T
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max).T
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr).T
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr).T

    features = np.hstack([mfcc, zcr, rms, mel_db, chroma, contrast])

    if features.shape[0] < max_len:
        pad_total = max_len - features.shape[0]
        if padding == 'center':
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            features = np.pad(features, ((pad_before, pad_after), (0, 0)), mode='constant')
        else: # trailing
            features = np.pad(features, ((0, pad_total), (0, 0)), mode='constant')
    else:
        features = features[:max_len, :]

    reshaped = features.reshape(-1, features.shape[-1])
    scaled = scaler.transform(reshaped)
    return scaled.reshape(1, max_len, -1)

def get_actual_emotion(filename):
    parts = os.path.basename(filename).replace(".wav", "").split("-")
    if len(parts) >= 3:
        return EMOTION_MAP.get(parts[2], "unknown")
    return "unknown"

def test_samples(num_samples=10):
    scaler, label_encoder = load_dependencies()
    
    model_path = os.path.join(CHECKPOINT_DIR, "cnn_lstm_attention_finetuned_best.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(CHECKPOINT_DIR, "cnn_lstm_attention_best.keras")
        
    print(f"Loading model: {model_path}")
    model = load_model(model_path, custom_objects={"AttentionLayer": AttentionLayer})

    test_files = glob.glob(os.path.join(TEST_DATA_PATH, "**", "*.wav"), recursive=True)
    if not test_files:
        print("No test files found!")
        return

    samples = random.sample(test_files, min(num_samples, len(test_files)))

    print("\n" + "="*90)
    print(f"{'Filename':<30} | {'Actual':<10} | {'Pred (Trailing)':<15} | {'Pred (Center)':<15}")
    print("-" * 90)

    for f in samples:
        filename = os.path.basename(f)
        actual = get_actual_emotion(f)
        
        # Test trailing
        f_trailing = extract_features_standardized(f, scaler, padding='trailing')
        p_trailing = label_encoder.classes_[np.argmax(model.predict(f_trailing, verbose=0))]
        
        # Test center
        f_center = extract_features_standardized(f, scaler, padding='center')
        p_center = label_encoder.classes_[np.argmax(model.predict(f_center, verbose=0))]
        
        status_t = "✓" if p_trailing.lower() == actual.lower() else "✗"
        status_c = "✓" if p_center.lower() == actual.lower() else "✗"
        
        print(f"{filename[:30]:<30} | {actual:<10} | {p_trailing:<10} {status_t} | {p_center:<10} {status_c}")

    print("="*90 + "\n")

if __name__ == "__main__":
    test_samples(10)

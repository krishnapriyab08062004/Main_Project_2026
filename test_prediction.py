import os
import sys
import numpy as np
import pickle
import librosa
from tensorflow.keras.models import load_model

sys.path.append(os.path.join(os.getcwd(), "src"))
from models_improved import AttentionLayer
from app import extract_features

BASE_DIR = os.path.join(os.getcwd(), "src")
MODEL_PATH = os.path.join(BASE_DIR, "models", "checkpoints", "cnn_lstm_attention_best.keras")
SCALER_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "scaler.pkl")
LABEL_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "label_encoder.pkl")

print("Loading model...")
model = load_model(
    MODEL_PATH,
    custom_objects={"AttentionLayer": AttentionLayer}
)

print("Loading scaler and label encoder...")
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
with open(LABEL_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# Find an angry file from RAVDESS
# Emotion code 05 is angry
data_dir = os.path.join(BASE_DIR, "..", "data", "raw", "audio_speech_actors_01-24")
angry_file = None
disgust_file = None
for root, dirs, files in os.walk(data_dir):
    for f in files:
        if f.endswith('.wav'):
            parts = f.split('-')
            if len(parts) >= 3 and parts[2] == '05':
                angry_file = os.path.join(root, f)
            if len(parts) >= 3 and parts[2] == '07':
                disgust_file = os.path.join(root, f)
        if angry_file and disgust_file:
            break
    if angry_file and disgust_file:
        break


def test_file(filepath):
    print(f"\\nTesting {filepath}")
    
    # Redefine extract_features locally to avoid variable scope issues with scaler
    audio, _ = librosa.load(filepath, sr=22050, mono=True)
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    mfcc = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=40).T
    zcr = librosa.feature.zero_crossing_rate(audio).T
    rms = librosa.feature.rms(y=audio).T
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=40)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).T
    chroma = librosa.feature.chroma_stft(y=audio, sr=22050).T
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=22050).T
    features = np.hstack([mfcc, zcr, rms, mel_spec_db, chroma, spectral_contrast])
    max_len = 128
    if features.shape[0] < max_len:
        features = np.pad(features, ((0, max_len - features.shape[0]), (0, 0)), mode='constant')
    else:
        features = features[:max_len, :]
    reshaped = features.reshape(-1, features.shape[-1])
    scaled = scaler.transform(reshaped)
    features = scaled.reshape(1, max_len, -1)
    
    preds = model.predict(features, verbose=0)
    pred_class = np.argmax(preds)
    emotion = label_encoder.inverse_transform([pred_class])[0]
    confidence = round(float(np.max(preds)) * 100, 2)
    print(f"Predicted: {emotion} ({confidence}%)")
    for i, prob in enumerate(preds[0]):
        print(f"  {label_encoder.classes_[i]}: {prob*100:.2f}%")

if angry_file:
    test_file(angry_file)
if disgust_file:
    test_file(disgust_file)

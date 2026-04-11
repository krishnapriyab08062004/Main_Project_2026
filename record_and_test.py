import os
import sys
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import pickle
from tensorflow.keras.models import load_model

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))
try:
    from models_improved import AttentionLayer
except ImportError:
    print("Warning: Could not import AttentionLayer from models_improved. Ensure you are in the project root.")

# --------------------------------------------------
# 🔥 PATH SETUP
# --------------------------------------------------
BASE_DIR = os.path.join(os.getcwd(), "src")
MODEL_PATH = os.path.join(BASE_DIR, "models", "checkpoints", "cnn_lstm_attention_finetuned_best.keras")
SCALER_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "scaler.pkl")
LABEL_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "label_encoder.pkl")

# --------------------------------------------------
# 🔥 LOAD MODEL & TOOLS
# --------------------------------------------------
print("Loading Model and Scaler...")
try:
    model = load_model(MODEL_PATH, custom_objects={"AttentionLayer": AttentionLayer})
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(LABEL_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    print("[SUCCESS]: System Ready for Live Testing.")
except Exception as e:
    print(f"[ERROR]: Failed to load assets: {e}")
    sys.exit(1)

# --------------------------------------------------
# 🔥 FEATURE EXTRACTION (Synced with app.py)
# --------------------------------------------------
def extract_features_live(audio, sr=22050, max_len=128):
    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=25)
    
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    # Features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T
    zcr = librosa.feature.zero_crossing_rate(audio).T
    rms = librosa.feature.rms(y=audio).T
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max).T
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr).T
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr).T

    features = np.hstack([mfcc, zcr, rms, mel_db, chroma, contrast])

    # Post-padding
    if features.shape[0] < max_len:
        pad_width = max_len - features.shape[0]
        features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
    else:
        features = features[:max_len, :]

    scaled = scaler.transform(features)
    return scaled.reshape(1, max_len, -1)

# --------------------------------------------------
# 🔥 RECORDING FUNCTION
# --------------------------------------------------
def record_and_predict(duration=3, sr=22050):
    print(f"\n🎤 RECORDING FOR {duration} SECONDS...")
    print("Speak now!")
    
    # Record audio
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait() # Wait until recording is finished
    
    print("✅ Recording complete. Analyzing...")
    
    # Flatten recording to 1D
    audio = recording.flatten()
    
    try:
        # Extract features
        features = extract_features_live(audio, sr=sr)
        
        # Predict
        preds = model.predict(features, verbose=0)
        pred_class = np.argmax(preds)
        emotion = label_encoder.inverse_transform([pred_class])[0]
        confidence = np.max(preds) * 100
        
        print("\n" + "="*30)
        print(f"RESULT: {emotion.upper()}")
        print(f"CONFIDENCE: {confidence:.2f}%")
        print("="*30)
        
        # Breakdown
        print("\nTop 3 Probabilities:")
        top_indices = np.argsort(preds[0])[::-1][:3]
        for i in top_indices:
            name = label_encoder.classes_[i]
            prob = preds[0][i] * 100
            print(f"- {name:<10}: {prob:>5.1f}%")

    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    while True:
        record_and_predict()
        cont = input("\nPress Enter to record again, or 'q' to quit: ")
        if cont.lower() == 'q':
            break

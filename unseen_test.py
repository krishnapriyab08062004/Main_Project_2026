import os
import sys
import numpy as np
import librosa
import joblib
import pickle
import glob
from tensorflow.keras.models import load_model

# Add src to sys.path to import local modules
sys.path.append(os.path.join(os.getcwd(), "src"))

from models_improved import AttentionLayer
from config import DATASET_PATH, PROCESSED_DATA_DIR, MAX_LEN, N_MFCC, CHECKPOINT_DIR

# Emotion mapping for RAVDESS (for verification)
EMOTION_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def extract_features_unseen(audio_path, max_len=128, n_mfcc=40, sr=22050):
    """
    Extract features matching the training pipeline for unseen files.
    """
    try:
        # Load audio file
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        
        # Normalize audio
        audio = audio / (np.max(np.abs(audio)) + 1e-6)
        
        # 1. MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T
        
        # 2. ZCR
        zcr = librosa.feature.zero_crossing_rate(audio).T
        
        # 3. RMS
        rms = librosa.feature.rms(y=audio).T
        
        # 4. Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).T
        
        # 5. Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr).T
        
        # 6. Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr).T
        
        # Combine all features (Total 101 features per timestep)
        features = np.hstack([mfcc, zcr, rms, mel_spec_db, chroma, spectral_contrast])
        
        # Pad or truncate
        if features.shape[0] < max_len:
            pad_width = max_len - features.shape[0]
            features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
        else:
            features = features[:max_len, :]
            
        return features
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None

def test_unseen_data():
    print("="*70)
    print("SER MODEL - UNSEEN DATA TEST")
    print("="*70)

    # 1. Load Paths
    scaler_path = os.path.join(PROCESSED_DATA_DIR, "scaler.pkl")
    label_path = os.path.join(PROCESSED_DATA_DIR, "label_encoder.pkl")
    
    # Try to find the best model
    model_path = os.path.join(CHECKPOINT_DIR, "cnn_dnn_v2_best.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(CHECKPOINT_DIR, "cnn_lstm_v2_best.keras")
    
    if not os.path.exists(model_path):
        print("Error: No model checkpoint found. Please train the model first.")
        return

    # 2. Load dependencies
    print("\n[OK] Loading Model, Scaler, and Label Encoder...")
    model = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_path)
    
    # 3. Find some wav files from raw data
    raw_audio_dir = DATASET_PATH
    all_wavs = glob.glob(os.path.join(raw_audio_dir, "**", "*.wav"), recursive=True)
    
    if not all_wavs:
        print(f"Error: No .wav files found in {raw_audio_dir}")
        return
    
    # Select a few random files for testing
    import random
    test_files = random.sample(all_wavs, min(5, len(all_wavs)))
    
    print(f"\n[OK] Found {len(all_wavs)} files. Testing with {len(test_files)} files...\n")
    print(f"{'File Name':<25} | {'Actual':<10} | {'Predicted':<10} | {'Confidence'}")
    print("-" * 70)

    for audio_file in test_files:
        filename = os.path.basename(audio_file)
        
        # Extract actual emotion from filename (RAVDESS format)
        parts = filename.replace(".wav", "").split("-")
        actual_emotion = "Unknown"
        if len(parts) >= 3:
            actual_emotion = EMOTION_MAP.get(parts[2], "Unknown")
        
        # Preprocess
        features = extract_features_unseen(audio_file)
        if features is None:
            continue
            
        # Standardize and Predict
        # Reshape to (time*frames, features) for scaling
        reshaped = features.reshape(-1, features.shape[-1])
        scaled = scaler.transform(reshaped)
        # Reshape back to (1, time, features)
        final_input = scaled.reshape(1, MAX_LEN, -1)
        
        preds = model.predict(final_input, verbose=0)
        pred_class = np.argmax(preds)
        predicted_emotion = label_encoder.inverse_transform([pred_class])[0]
        confidence = np.max(preds) * 100
        
        status = "✓" if actual_emotion.lower() == predicted_emotion.lower() else "✗"
        print(f"{filename[:25]:<25} | {actual_emotion:<10} | {predicted_emotion:<10} | {confidence:>8.2f}% {status}")

    print("\n" + "="*70)
    print("UNSEEN DATA TEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    test_unseen_data()

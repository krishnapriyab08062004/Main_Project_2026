import librosa
import numpy as np
import pickle

MAX_LEN = 128
N_MFCC = 40
SAMPLE_RATE = 22050

# Load scaler
with open("../data/processed/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def extract_features_for_prediction(audio_path):
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC).T

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(audio).T

    # RMS
    rms = librosa.feature.rms(y=audio).T

    features = np.hstack([mfcc, zcr, rms])

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
    mel = librosa.power_to_db(mel, ref=np.max).T

    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr).T

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr).T

    features = np.hstack([features, mel, chroma, contrast])

    # Pad / truncate
    if features.shape[0] < MAX_LEN:
        pad = MAX_LEN - features.shape[0]
        features = np.pad(features, ((0, pad), (0, 0)))
    else:
        features = features[:MAX_LEN]

    # Standardization (same scaler!)
    reshaped = features.reshape(-1, features.shape[-1])
    scaled = scaler.transform(reshaped)
    features = scaled.reshape(1, MAX_LEN, features.shape[-1])

    return features

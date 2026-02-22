import os
import numpy as np
import pickle
import librosa
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

# If using custom Attention layer
from models_improved import AttentionLayer

# --------------------------------------------------
# 🔥 PATH SETUP (NO RELATIVE PATH CONFUSION)
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "models", "checkpoints", "cnn_dnn_improved_best.keras")
SCALER_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "scaler.pkl")
LABEL_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "label_encoder.pkl")

# --------------------------------------------------
# 🔥 FLASK APP
# --------------------------------------------------

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --------------------------------------------------
# 🔥 LOAD MODEL + SCALER + LABEL ENCODER
# --------------------------------------------------

print("Loading model...")
model = load_model(
    MODEL_PATH,
    custom_objects={"AttentionLayer": AttentionLayer}
)

print("Loading scaler...")
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

print("Loading label encoder...")
with open(LABEL_PATH, "rb") as f:
    label_encoder = pickle.load(f)

print("Everything loaded successfully [OK]")
print("Model input shape:", model.input_shape)


# --------------------------------------------------
# 🔥 FEATURE EXTRACTION (SAME AS TRAINING)
# --------------------------------------------------

def extract_features(audio_path, max_len=128, n_mfcc=40, sr=22050):

    audio, _ = librosa.load(audio_path, sr=sr, mono=True)

    # Trim leading and trailing background silence to ensure the model focuses on speech
    audio, _ = librosa.effects.trim(audio, top_db=30)

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(audio).T

    # RMS
    rms = librosa.feature.rms(y=audio).T

    # Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).T

    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr).T

    # Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr).T

    # Combine all features
    features = np.hstack([mfcc, zcr, rms,
                          mel_spec_db,
                          chroma,
                          spectral_contrast])

    # Pad / Truncate (Center the audio for better alignment with RAVDESS training)
    if features.shape[0] < max_len:
        pad_width = max_len - features.shape[0]
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        features = np.pad(features, ((pad_left, pad_right), (0, 0)), mode='constant')
    else:
        # Extract the middle part instead of the very beginning
        start_idx = (features.shape[0] - max_len) // 2
        features = features[start_idx : start_idx + max_len, :]

    # Scale using saved scaler
    reshaped = features.reshape(-1, features.shape[-1])
    scaled = scaler.transform(reshaped)
    features = scaled.reshape(1, max_len, -1)

    return features


# --------------------------------------------------
# 🔥 MAIN ROUTE
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    confidence = None

    if request.method == "POST":

        if "audio" not in request.files:
            return render_template("index.html",
                                   prediction="No file uploaded")

        file = request.files["audio"]

        if file.filename == "":
            return render_template("index.html",
                                   prediction="No file selected")

        try:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            print("File saved:", filepath)

            # Extract features
            features = extract_features(filepath)

            print("Feature shape:", features.shape)

            # Predict
            preds = model.predict(features)
            pred_class = np.argmax(preds)

            emotion = label_encoder.inverse_transform([pred_class])[0]
            confidence = round(float(np.max(preds)) * 100, 2)

            prediction = emotion

            print("Prediction:", prediction, "| Confidence:", confidence)

        except Exception as e:
            print("ERROR:", e)
            prediction = "Error occurred during prediction"
            confidence = None

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence)


# --------------------------------------------------
# 🔥 RUN APP
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
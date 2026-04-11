import os
import numpy as np
import pickle
import librosa
import soundfile as sf

from flask import Flask, render_template, request, redirect
from flask_mysqldb import MySQL
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message
from tensorflow.keras.models import load_model
from models_improved import AttentionLayer

# --------------------------------------------------
# 🔥 PATH SETUP
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "models", "checkpoints", "cnn_dnn_improved_finetuned_best.keras")
SCALER_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "scaler.pkl")
LABEL_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "label_encoder.pkl")

# --------------------------------------------------
# 🔥 FLASK APP
# --------------------------------------------------

app = Flask(__name__)
app.secret_key = "secret123"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --------------------------------------------------
# 🔥 MYSQL CONFIG
# --------------------------------------------------

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'ser'

mysql = MySQL(app)

# --------------------------------------------------
# 🔥 MAIL CONFIG
# --------------------------------------------------

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'krishnapriyab864@gmail.com'
app.config['MAIL_PASSWORD'] = 'zmoayfsrlmirogge'
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

mail = Mail(app)

# --------------------------------------------------
# 🔥 LOGIN MANAGER
# --------------------------------------------------

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, username, email FROM users WHERE id=%s", (user_id,))
    user = cur.fetchone()
    cur.close()

    if user:
        return User(user[0], user[1], user[2])
    return None

# --------------------------------------------------
# 🔥 LOAD MODEL
# --------------------------------------------------

print("Checking dependencies...")

required_files = [MODEL_PATH, SCALER_PATH, LABEL_PATH]
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    print(f"[ERROR]: Missing required files: {missing_files}")
    # In production, you might want to raise an error instead of continuing
else:
    print("[SUCCESS]: All required files found")

try:
    print("Loading model...")
    model = load_model(MODEL_PATH, custom_objects={"AttentionLayer": AttentionLayer})

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    with open(LABEL_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    print("[SUCCESS]: Model Loaded Successfully")
except Exception as e:
    print(f"[ERROR] loading model/scaler: {e}")
    model = None
    scaler = None
    label_encoder = None

# --------------------------------------------------
# 🔥 FEATURE EXTRACTION
# --------------------------------------------------

def extract_features(audio_path, max_len=128, n_mfcc=40, sr=22050):

    if scaler is None:
        raise ValueError("Scaler is not loaded. Cannot extract features.")

    # 1. Load audio
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    
    # 2. Trim silence (Important for real-world audio)
    audio, _ = librosa.effects.trim(audio, top_db=25)
    
    # 3. Normalization (MATCH TRAINING: No DC offset removal here)
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    # 4. Extract Features (Sequence must match training exactly)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T
    zcr = librosa.feature.zero_crossing_rate(audio).T
    rms = librosa.feature.rms(y=audio).T
    
    # Advanced features
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max).T
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr).T
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr).T

    # Combine in SAME order as data_preprocessing_enhanced.py
    # Order: MFCC, ZCR, RMS, Mel, Chroma, Contrast
    features = np.hstack([mfcc, zcr, rms, mel_db, chroma, contrast])

    # 5. Padding (MATCH TRAINING: Post-padding, not center padding)
    if features.shape[0] < max_len:
        pad_width = max_len - features.shape[0]
        features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
    else:
        features = features[:max_len, :]

    # Standardization
    reshaped = features.reshape(-1, features.shape[-1])
    scaled = scaler.transform(reshaped)

    return scaled.reshape(1, max_len, -1)

# --------------------------------------------------
# 🔥 STRESS + ALERT
# --------------------------------------------------

NEGATIVE_EMOTIONS = ["angry", "fearful", "sad", "disgust"]

def calculate_stress(preds):
    if label_encoder is None:
        raise ValueError("Label encoder is not loaded. Cannot calculate stress.")
    stress = 0
    for i, prob in enumerate(preds[0]):
        emotion = label_encoder.inverse_transform([i])[0]
        if emotion in NEGATIVE_EMOTIONS:
            stress += prob
    return float(stress)

def send_email_alert(to_email, score):
    try:
        msg = Message("Stress Alert",
                      sender=app.config['MAIL_USERNAME'],
                      recipients=[to_email])

        msg.body = f"High stress detected! Score: {score:.2f}"
        mail.send(msg)

    except Exception as e:
        print("Email error:", e)

from datetime import datetime
from typing import Dict, Tuple

# In-memory dictionary to track when the last email alert was sent per user
# Format: user_id -> (last_alert_time, last_alert_level)
user_last_alert_time: Dict[int, Tuple[datetime, str]] = {}

def check_alert(user_id):
    cur = mysql.connection.cursor()

    cur.execute("""
        SELECT stress_score FROM emotion_logs
        WHERE user_id=%s ORDER BY created_at DESC LIMIT 5
    """, (user_id,))
    records = cur.fetchall()

    cur.execute("SELECT email FROM users WHERE id=%s", (user_id,))
    email = cur.fetchone()[0]

    cur.close()

    if len(records) < 5:
        return None

    scores = [r[0] for r in records]
    avg = sum(scores) / len(scores)

    # Multi-level alert thresholds
    level = None
    msg = ""
    should_email = False

    if avg >= 0.8:
        level = "Critical"
        msg = "🚨 CRITICAL: Sustained crucial stress levels detected. Please take immediate action and consider seeking support."
        should_email = True
    elif avg >= 0.6:
        level = "High"
        msg = "⚠️ HIGH: Sustained elevated stress detected. It's highly recommended to step away and take a break."
        should_email = True
    elif avg >= 0.4:
        level = "Mild"
        msg = "ℹ️ MILD: Noticeable stress building up. Remember to stay hydrated and take deep breaths."
        should_email = False

    if not level:
        return None

    # Cooldown mechanism for emails
    cooldown_seconds = {
        "Critical": 1800,  # 30 minutes
        "High": 3600       # 1 hour
    }

    alert_suffix = ""
    if should_email:
        now = datetime.now()
        last_time, last_level = user_last_alert_time.get(user_id, (None, None))

        can_send_email = True
        if last_time and last_level:
            time_diff = (now - last_time).total_seconds()
            req_cooldown = cooldown_seconds.get(level, 3600)
            
            # If the stress level escalated to Critical, we can override the High cooldown
            if level == "Critical" and last_level != "Critical":
                can_send_email = True
            elif time_diff < req_cooldown:
                can_send_email = False

        if can_send_email:
            send_email_alert(email, avg)
            user_last_alert_time[user_id] = (now, level)
            alert_suffix = " (Notification emailed)"
        else:
            alert_suffix = " (Email suppressed due to cooldown limit)"

    return msg + alert_suffix

# --------------------------------------------------
# 🔥 AUTH ROUTES
# --------------------------------------------------

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])

        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users(username,email,password) VALUES(%s,%s,%s)",
                    (username, email, password))
        mysql.connection.commit()
        cur.close()

        return redirect("/login")

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cur.fetchone()
        cur.close()

        if user and check_password_hash(user[3], password):
            login_user(User(user[0], user[1], user[2]))
            return redirect("/")

    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/login")

# --------------------------------------------------
# 🔥 MAIN ROUTE
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
@login_required
def index():

    prediction = None
    confidence = None
    alert = None
    stress_score = None
    top_emotions = None

    if request.method == "POST":
        prediction, confidence, stress_score, alert, top_emotions = None, None, None, None, None

        if "audio" not in request.files or request.files["audio"].filename == "":
            print("No audio file selected.")
            return render_template("index.html", 
                                   prediction=None, 
                                   confidence=None, 
                                   alert="Please select an audio file first.", 
                                   stress_score=None)

        file = request.files["audio"]
        filename = file.filename
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Convert to .wav if not already
        if not filename.lower().endswith(".wav"):
            try:
                # Convert to .wav using librosa + soundfile
                audio, sr = librosa.load(filepath, sr=22050)
                wav_filename = os.path.splitext(filename)[0] + ".wav"
                wav_filepath = os.path.join(app.config["UPLOAD_FOLDER"], wav_filename)
                sf.write(wav_filepath, audio, sr)
                filepath = wav_filepath
                print(f"Converted {filename} to {wav_filename}")
            except Exception as e:
                print(f"Audio conversion failed: {e}")
                return render_template("index.html", 
                                       prediction=None, 
                                       confidence=None, 
                                       alert=f"Audio conversion failed: {e}", 
                                       stress_score=None)

        if model is None or scaler is None or label_encoder is None:
            return render_template("index.html", 
                                   prediction=None, 
                                   confidence=None, 
                                   top_emotions=None,
                                   alert="Model, Scaler, or Label Encoder not loaded correctly. Please check server logs.", 
                                   stress_score=None)

        try:
            features = extract_features(filepath)

            preds = model.predict(features)
            
            # Primary prediction
            pred_class = np.argmax(preds)
            prediction = label_encoder.inverse_transform([pred_class])[0]
            confidence = round(float(np.max(preds)) * 100, 2)
            
            # Confidence Breakdown (Top 3)
            top_indices = np.argsort(preds[0])[::-1][:3]
            top_emotions = []
            for i in top_indices:
                name = label_encoder.classes_[i]
                prob = round(float(preds[0][i]) * 100, 1)
                top_emotions.append({"name": name, "prob": prob})

            # Stress calculation
            stress_score = calculate_stress(preds)
            
            # Save to DB
            cur = mysql.connection.cursor()
            cur.execute("""
                INSERT INTO emotion_logs(user_id, emotion, confidence, stress_score)
                VALUES (%s,%s,%s,%s)
            """, (current_user.id, prediction, confidence, stress_score))
            mysql.connection.commit()
            cur.close()

            alert = check_alert(current_user.id)
            print(f"Prediction: {prediction} ({confidence}%) - Stress: {stress_score:.2f}")
        except Exception as e:
            print(f"Prediction/DB Error: {e}")
            alert = f"An error occurred: {e}"

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           top_emotions=top_emotions,
                           alert=alert,
                           stress_score=stress_score)

# --------------------------------------------------
# 🔥 DASHBOARD
# --------------------------------------------------

@app.route("/dashboard")
@login_required
def dashboard():

    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT emotion, stress_score, created_at
        FROM emotion_logs
        WHERE user_id=%s ORDER BY created_at DESC LIMIT 10
    """, (current_user.id,))
    logs = cur.fetchall()
    cur.close()

    return render_template("dashboard.html", logs=logs)

# --------------------------------------------------
# 🔥 ANALYTICS
# --------------------------------------------------

@app.route("/analytics")
@login_required
def analytics():

    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT DATE(created_at), AVG(stress_score)
        FROM emotion_logs
        WHERE user_id=%s GROUP BY DATE(created_at)
    """, (current_user.id,))
    data = cur.fetchall()
    cur.close()

    dates = [str(d[0]) for d in data]
    scores = [float(d[1]) for d in data]

    return render_template("analytics.html", dates=dates, scores=scores)

# --------------------------------------------------
# 🔥 RUN
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
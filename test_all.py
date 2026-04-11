import os
import sys
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model

sys.path.append(os.path.join(os.getcwd(), "src"))
from models_improved import AttentionLayer
from app import extract_features

def test_all_models(filepath):
    print(f"\nTesting: {filepath}")
    
    # Paths
    checkpoints_dir = "src/models/checkpoints"
    le_path = "data/processed/label_encoder.pkl"
    
    with open(le_path, "rb") as f:
        le = pickle.load(f)
        
    models = [f for f in os.listdir(checkpoints_dir) if f.endswith('.keras')]
    
    for m_file in models:
        try:
            m_path = os.path.join(checkpoints_dir, m_file)
            print(f"\nModel: {m_file}")
            
            # Load model
            model = load_model(m_path, custom_objects={"AttentionLayer": AttentionLayer})
            
            # Extract and predict
            fts = extract_features(filepath)
            p = model.predict(fts, verbose=0)
            
            pred_class = np.argmax(p)
            emotion = le.inverse_transform([pred_class])[0]
            conf = np.max(p) * 100
            
            print(f"  Prediction: {emotion} ({conf:.2f}%)")
            # Show top 3
            top_indices = np.argsort(p[0])[::-1][:3]
            for i in top_indices:
                print(f"    - {le.classes_[i]}: {p[0][i]*100:.2f}%")
        except Exception as e:
            print(f"  Error loading/running {m_file}: {e}")

if __name__ == "__main__":
    # Find a RAVDESS angry file
    data_dir = "data/raw/audio_speech_actors_01-24"
    rav_angry = None
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if "-05-" in f:
                rav_angry = os.path.join(root, f)
                break
        if rav_angry: break
    
    user_file = "src/uploads/i_am_krishna_angry.wav"
    
    if rav_angry:
        print("\n" + "="*50)
        print("TESTING RAVDESS ANGRY (GROUND TRUTH: ANGRY)")
        print("="*50)
        test_all_models(rav_angry)
    
    if os.path.exists(user_file):
        print("\n" + "="*50)
        print("TESTING USER ANGRY (GROUND TRUTH: ANGRY)")
        print("="*50)
        test_all_models(user_file)
    else:
        print(f"User file not found: {user_file}")

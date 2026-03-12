import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from tensorflow.keras.models import load_model

# Import custom objects if needed
try:
    from models_improved import AttentionLayer
except ImportError:
    # If running from a different directory, try to adjust path
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from models_improved import AttentionLayer

def test_model(model_path, data_dir):
    """
    Evaluate a specific model on the enhanced test dataset.
    """
    print("="*70)
    print(f"TESTING MODEL: {os.path.basename(model_path)}")
    print("="*70)

    # 1. Load Test Data
    print("\n[1/4] Loading test data...")
    try:
        X_test = np.load(os.path.join(data_dir, 'X_test_enhanced.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test_enhanced.npy'))
        print(f"  - Loaded X_test: {X_test.shape}")
        print(f"  - Loaded y_test: {y_test.shape}")
    except FileNotFoundError:
        print(f"Error: Enhanced test data not found in {data_dir}")
        return

    # 2. Load Label Encoder for class names
    try:
        label_encoder = joblib.load(os.path.join(data_dir, "label_encoder.pkl"))
        class_names = label_encoder.classes_
    except FileNotFoundError:
        print("Warning: Label encoder not found. Using default indices.")
        class_names = [str(i) for i in range(len(np.unique(y_test)))]

    # 3. Load Model
    print("\n[2/4] Loading model...")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    model = load_model(
        model_path,
        custom_objects={'AttentionLayer': AttentionLayer}
    )
    print("  - Model loaded successfully")

    # 4. Predictions & Metrics
    print("\n[3/4] Performing predictions...")
    y_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nOVERALL RESULTS:")
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"  F1-Score      : {f1:.4f}")
    
    # --- PER-EMOTION ACCURACY BREAKDOWN ---
    print("\n" + "-"*30)
    print("PER-EMOTION ACCURACY")
    print("-"*30)
    print(f"{'Emotion':<12} | {'Accuracy':<10} | {'Correct/Total'}")
    print("-"*35)

    for i, emotion in enumerate(class_names):
        mask = (y_test == i)
        total_samples = np.sum(mask)
        if total_samples > 0:
            correct_preds = np.sum((y_pred[mask] == i))
            emotion_acc = (correct_preds / total_samples) * 100
            print(f"{emotion:<12} | {emotion_acc:>8.2f}% | {correct_preds}/{total_samples}")
    
    # --- SHOW SOME EXAMPLES ---
    print("\n" + "-"*30)
    print("SAMPLE TEST PREDICTIONS")
    print("-"*30)
    indices = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
    for idx in indices:
        true_label = class_names[int(y_test[idx])]
        pred_label = class_names[y_pred[idx]]
        status = "✓ CORRECT" if true_label == pred_label else "✗ WRONG"
        confidence = np.max(y_probs[idx]) * 100
        print(f"[{status}] True: {true_label:<10} | Pred: {pred_label:<10} | Conf: {confidence:>5.1f}%")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # 5. Confusion Matrix
    print("\n[4/4] Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"Confusion Matrix: {os.path.basename(model_path)}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    # Save the plot
    output_dir = "results/evaluation"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{os.path.basename(model_path)}_test_cm.png")
    plt.savefig(plot_path)
    print(f"  - Confusion matrix saved to: {plot_path}")
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)

if __name__ == "__main__":
    # Default paths (Update these if your files are named differently)
    from config import PROCESSED_DATA_DIR, CHECKPOINT_DIR
    
    # You can change the model path to evaluate different versions
    # e.g., 'models/checkpoints/cnn_dnn_v2_best.keras'
    target_model = os.path.join(CHECKPOINT_DIR, "cnn_dnn_v2_best.keras")
    
    if not os.path.exists(target_model):
        # Try fallbacks
        fallbacks = [
            "models/saved_models/cnn_dnn_v2_final.keras",
            "models/checkpoints/cnn_lstm_v2_best.keras"
        ]
        for fb in fallbacks:
            if os.path.exists(fb):
                target_model = fb
                break
    
    test_model(target_model, PROCESSED_DATA_DIR)

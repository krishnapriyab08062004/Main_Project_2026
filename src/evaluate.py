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
from config import PROCESSED_DATA_DIR

# Custom attention layer import
from models_improved import AttentionLayer

print("="*70)
print("EVALUATING SER MODEL PERFORMANCE")
print("="*70)

# ===============================
# LOAD TEST DATA
# ===============================

processed_dir = os.path.abspath(PROCESSED_DATA_DIR)

X_test = np.load(os.path.join(processed_dir, 'X_test_enhanced.npy'))
y_test = np.load(os.path.join(processed_dir, 'y_test_enhanced.npy'))

print("\n✓ Test data loaded")
print("  X_test shape:", X_test.shape)
print("  y_test shape:", y_test.shape)

# Load label encoder
label_encoder = joblib.load(os.path.join("../data/processed", "label_encoder.pkl"))
class_names = label_encoder.classes_

# ===============================
# LOAD MODELS
# ===============================

models_to_evaluate = {
    "CNN_DNN_Improved": "models/checkpoints/cnn_dnn_improved_best.keras",
    "CNN_LSTM_Attention": "models/checkpoints/cnn_lstm_attention_best.keras"
}

results = {}

for model_name, model_path in models_to_evaluate.items():

    print("\n" + "-"*60)
    print(f"Evaluating {model_name}")
    print("-"*60)

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        continue

    # Load model
    model = load_model(
        model_path,
        custom_objects={'AttentionLayer': AttentionLayer}
    )

    # Predict
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    results[model_name] = acc

    # ===============================
    # CONFUSION MATRIX
    # ===============================

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    os.makedirs("results/confusion_matrices", exist_ok=True)
    plt.savefig(f"results/confusion_matrices/{model_name}_cm.png")
    plt.close()

    print("✓ Confusion matrix saved")

    # ===============================
    # CLASSIFICATION REPORT
    # ===============================

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

# ===============================
# BEST MODEL SELECTION
# ===============================

print("\n" + "="*70)
print("BEST MODEL SELECTION")
print("="*70)

best_model = max(results, key=results.get)
best_accuracy = results[best_model]

print(f"\n🏆 Best Model: {best_model}")
print(f"🏆 Best Test Accuracy: {best_accuracy*100:.2f}%")
print("="*70)
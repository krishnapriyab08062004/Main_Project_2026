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
# LOAD TRAIN & TEST DATA
# ===============================

processed_dir = os.path.abspath(PROCESSED_DATA_DIR)

X_train = np.load(os.path.join(processed_dir, 'X_train_enhanced.npy'))
y_train = np.load(os.path.join(processed_dir, 'y_train_enhanced.npy'))

X_test = np.load(os.path.join(processed_dir, 'X_test_enhanced.npy'))
y_test = np.load(os.path.join(processed_dir, 'y_test_enhanced.npy'))

print("\n[OK] Train and Test data loaded")
print("  X_train shape:", X_train.shape)
print("  y_train shape:", y_train.shape)
print("  X_test shape :", X_test.shape)
print("  y_test shape :", y_test.shape)

# Load label encoder
label_encoder = joblib.load(os.path.join("../data/processed", "label_encoder.pkl"))
class_names = label_encoder.classes_

# ===============================
# LOAD MODELS
# ===============================

models_to_evaluate = {
    "CNN_DNN_Improved": "models/saved_models/cnn_dnn_improved_final.keras",
    "CNN_LSTM_Attention": "models/saved_models/cnn_lstm_attention_final.keras",
    "CNN_DNN_v2": "models/checkpoints/cnn_dnn_v2_best.keras",
    "CNN_LSTM_v2": "models/checkpoints/cnn_lstm_v2_best.keras"
}

results = {}

for model_name, model_path in models_to_evaluate.items():

    print("\n" + "-"*60)
    print(f"Evaluating {model_name}")
    print("-"*60)

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        continue

    # ===============================
    # LOAD MODEL
    # ===============================
    model = load_model(
        model_path,
        custom_objects={'AttentionLayer': AttentionLayer}
    )

    # ===============================
    # TRAIN ACCURACY
    # ===============================
    y_train_probs = model.predict(X_train)
    y_train_pred = np.argmax(y_train_probs, axis=1)

    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"\nTrain Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")

    # ===============================
    # TEST ACCURACY
    # ===============================
    y_test_probs = model.predict(X_test)
    y_test_pred = np.argmax(y_test_probs, axis=1)

    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')

    print(f"\nTest Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1-score      : {f1:.4f}")

    # ===============================
    # OVERFITTING CHECK
    # ===============================
    gap = train_acc - test_acc
    
    # Store results for final comparison
    results[model_name] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'gap': gap,
        'f1': f1
    }

    print(f"\nAccuracy Gap (Train - Test): {gap:.4f}")

    if gap > 0.10:
        print("WARNING: Model is likely OVERFITTING")
    elif gap > 0.05:
        print("CAUTION: Slight overfitting detected")
    else:
        print("OK: No significant overfitting - good generalization!")

    # ===============================
    # PER-CLASS ACCURACY
    # ===============================
    print("\nPer-Class Accuracy:")
    for i, emotion in enumerate(class_names):
        mask = (y_test == i)
        if mask.sum() > 0:
            class_acc = accuracy_score(y_test[mask], y_test_pred[mask])
            class_count = mask.sum()
            print(f"  {emotion:12s}: {class_acc:.4f} ({class_acc*100:.1f}%) [{class_count} samples]")

    # ===============================
    # CONFUSION MATRIX
    # ===============================
    cm = confusion_matrix(y_test, y_test_pred)

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

    print("[OK] Confusion matrix saved")

    # ===============================
    # CLASSIFICATION REPORT
    # ===============================
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=class_names))


# ===============================
# FINAL RECAP & RECOMMENDATION
# ===============================

print("\n" + "="*70)
print("FINAL MODEL RANKING & RECOMMENDATION")
print("="*70)

if results:
    # results is a dict: {model_name: {'test_acc': val, 'gap': val, 'f1': val}}
    # Print a summary table
    print(f"\n{'Model Name':<25} | {'Train Acc':<10} | {'Test Acc':<10} | {'Gap':<10} | {'F1-Score':<10}")
    print("-" * 80)
  
    
    # Sort models by Test Accuracy (primary) and Gap (secondary)
    sorted_models = sorted(
        results.items(), 
        key=lambda x: (x[1]['test_acc'], -x[1]['gap']), 
        reverse=True
    )
    
    for name, metrics in sorted_models:
        print(f"{name:<25} | {metrics['train_acc']*100:>8.2f}% | {metrics['test_acc']*100:>8.2f}% | {metrics['gap']:>8.4f} | {metrics['f1']:>8.4f}")
        

    best_name, best_metrics = sorted_models[0]
    
    print("\n" + "-"*40)
    print(f"RECOMMENDED MODEL: {best_name}")
    print(f"Accuracy: {best_metrics['test_acc']*100:.2f}%")
    print(f"Generalization Gap: {best_metrics['gap']:.4f}")
    print("-" * 40)
    
    if best_metrics['test_acc'] >= 0.90 and best_metrics['gap'] <= 0.05:
        print("\nCONCLUSION: EXCELLENT. This model meets both high accuracy and generalization requirements.")
    elif best_metrics['test_acc'] >= 0.85:
        print("\nCONCLUSION: GOOD. High performance, though further tuning may strictly hit 95%.")
    else:
        print("\nCONCLUSION: NEEDS IMPROVEMENT. Accuracy is below the 95% target.")
else:
    print("No models evaluated.")

print("="*70)

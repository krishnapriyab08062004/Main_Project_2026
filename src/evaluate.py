import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

# Paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

# Map finetuned eval names back to their base training log names
FINETUNED_TO_BASE = {
    'CNN_DNN_Finetuned': 'cnn_dnn_improved',
    'CNN_LSTM_Attention_Finetuned': 'cnn_lstm_attention',
}

def plot_training_history_from_logs(model_name, checkpoint_dir='models/checkpoints', save_dir='results/plots'):
    """
    Load and merge training logs into a SINGLE continuous graph.
    
    - For finetuned models: merges Stage 1 + Stage 2 into one 150-epoch graph
    - For base/other models: shows only their own training log
    """
    log_dir = os.path.join(BASE_DIR, 'models', 'checkpoints')
    is_finetuned = model_name in FINETUNED_TO_BASE
    
    logs = []
    stage1_epochs = 0
    
    if is_finetuned:
        # Finetuned model: load BOTH base Stage 1 + finetuned Stage 2
        base_name = FINETUNED_TO_BASE[model_name]
        log1_path = os.path.join(log_dir, f"{base_name}_training_log.csv")
        log2_path = os.path.join(log_dir, f"{base_name}_finetuned_training_log.csv")
        
        # Stage 1
        if os.path.exists(log1_path) and os.path.getsize(log1_path) > 0:
            try:
                df1 = pd.read_csv(log1_path)
                if not df1.empty:
                    logs.append(df1)
                    stage1_epochs = len(df1)
                    print(f"  Found Stage 1 log ({base_name}): {len(df1)} epochs")
            except pd.errors.EmptyDataError:
                print(f"  [!] Stage 1 log is empty ({log1_path})")
        
        # Stage 2
        if os.path.exists(log2_path) and os.path.getsize(log2_path) > 0:
            try:
                df2 = pd.read_csv(log2_path)
                if not df2.empty:
                    if logs:
                        df2['epoch'] = df2['epoch'] + logs[0]['epoch'].max() + 1
                    logs.append(df2)
                    print(f"  Found Stage 2 log (finetuned): {len(df2)} epochs")
            except pd.errors.EmptyDataError:
                print(f"  [!] Stage 2 log is empty ({log2_path})")
    else:
        # Base model: load only its own training log
        log_path = os.path.join(log_dir, f"{model_name.lower()}_training_log.csv")
        if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
            try:
                df = pd.read_csv(log_path)
                if not df.empty:
                    logs.append(df)
                    print(f"  Found training log: {len(df)} epochs")
            except pd.errors.EmptyDataError:
                print(f"  [!] Training log is empty ({log_path})")
    
    if not logs:
        print(f"  [!] No logs found for {model_name} in {log_dir}")
        return

    # Merge into one continuous dataframe
    full_df = pd.concat(logs, ignore_index=True)
    total_epochs = len(full_df)
    has_two_stages = len(logs) == 2
    
    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # --- Accuracy ---
    axes[0, 0].plot(full_df['accuracy'], label='Train Acc', color='#2E86AB', linewidth=2)
    axes[0, 0].plot(full_df['val_accuracy'], label='Val Acc', color='#A23B72', linewidth=2)
    if has_two_stages:
        axes[0, 0].axvline(x=stage1_epochs, color='#F18F01', linestyle='--', alpha=0.7, label='Fine-tuning Start')
    axes[0, 0].set_title(f"{model_name} - Accuracy ({total_epochs} Epochs)", fontweight='bold', fontsize=13)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # --- Loss ---
    axes[0, 1].plot(full_df['loss'], label='Train Loss', color='#2E86AB', linewidth=2)
    axes[0, 1].plot(full_df['val_loss'], label='Val Loss', color='#A23B72', linewidth=2)
    if has_two_stages:
        axes[0, 1].axvline(x=stage1_epochs, color='#F18F01', linestyle='--', alpha=0.7, label='Fine-tuning Start')
    axes[0, 1].set_title(f"{model_name} - Loss ({total_epochs} Epochs)", fontweight='bold', fontsize=13)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # --- Learning Rate ---
    if 'lr' in full_df.columns:
        axes[1, 0].plot(full_df['lr'], linewidth=2, color='#F18F01')
        if has_two_stages:
            axes[1, 0].axvline(x=stage1_epochs, color='red', linestyle='--', alpha=0.7, label='Fine-tuning Start')
        axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold', fontsize=13)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Learning Rate\nNot Logged',
                       ha='center', va='center', fontsize=14, color='gray')
        axes[1, 0].axis('off')
    
    # --- Accuracy Gap ---
    acc_gap = np.array(full_df['accuracy']) - np.array(full_df['val_accuracy'])
    axes[1, 1].plot(acc_gap, linewidth=2, color='#C73E1D')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    if has_two_stages:
        axes[1, 1].axvline(x=stage1_epochs, color='#F18F01', linestyle='--', alpha=0.7, label='Fine-tuning Start')
    axes[1, 1].set_title('Accuracy Gap (Train - Val)', fontweight='bold', fontsize=13)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Gap')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f"{model_name} \u2014 Full Training History ({total_epochs} Epochs)", 
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    # Save with _original suffix
    save_dir_abs = os.path.join(BASE_DIR, save_dir)
    os.makedirs(save_dir_abs, exist_ok=True)
    save_path = os.path.join(save_dir_abs, f"{model_name}_{total_epochs}_epochs_original.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Saved {total_epochs}-epoch plot: {save_path}")

# ===============================
# LOAD TRAIN & TEST DATA
# ===============================

processed_dir = os.path.abspath(PROCESSED_DATA_DIR)

# Check if files exist
required_files = [
    'X_train_enhanced.npy', 'y_train_enhanced.npy',
    'X_val_enhanced.npy', 'y_val_enhanced.npy',
    'X_test_enhanced.npy', 'y_test_enhanced.npy'
]

missing_files = [f for f in required_files if not os.path.exists(os.path.join(processed_dir, f))]
if missing_files:
    print(f"\n[!] Error: Missing processed data files in {processed_dir}:")
    for f in missing_files:
        print(f"  - {f}")
    print("\nPlease run 'python src/data_preprocessing_enhanced.py' first.")
    exit(1)

X_train = np.load(os.path.join(processed_dir, 'X_train_enhanced.npy'))
y_train = np.load(os.path.join(processed_dir, 'y_train_enhanced.npy'))

X_val = np.load(os.path.join(processed_dir, 'X_val_enhanced.npy'))
y_val = np.load(os.path.join(processed_dir, 'y_val_enhanced.npy'))

X_test = np.load(os.path.join(processed_dir, 'X_test_enhanced.npy'))
y_test = np.load(os.path.join(processed_dir, 'y_test_enhanced.npy'))

print("\n[OK] Train, Val, and Test data loaded")
print(f"  X_train shape: {X_train.shape}")
print(f"  X_val   shape: {X_val.shape}")
print(f"  X_test  shape: {X_test.shape}")

# Load label encoder
le_path = os.path.join(processed_dir, "label_encoder.pkl")
if not os.path.exists(le_path):
    # Fallback to the other possible location
    le_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "label_encoder.pkl"))

label_encoder = joblib.load(le_path)
class_names = label_encoder.classes_

# Construct absolute paths for models
# Prioritize root models directory over src directory
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_ROOT = os.path.join(PROJ_ROOT, "models", "checkpoints")

models_to_evaluate = {
    # Basic Models
    "CNN_DNN_v1": os.path.join(CHECKPOINT_ROOT, "cnn_dnn_v1_best.keras"),
    "CNN_LSTM_v1": os.path.join(CHECKPOINT_ROOT, "cnn_lstm_v1_best.keras"),
    # Improved Models
    # "CNN_DNN_Improved": os.path.join(CHECKPOINT_ROOT, "cnn_dnn_improved_best.keras"),
    # "CNN_LSTM_Attention": os.path.join(CHECKPOINT_ROOT, "cnn_lstm_attention_best.keras"),
    # Finetuned Models
    "CNN_DNN_Finetuned": os.path.join(CHECKPOINT_ROOT, "cnn_dnn_improved_finetuned_best.keras"),
    "CNN_LSTM_Attention_Finetuned": os.path.join(CHECKPOINT_ROOT, "cnn_lstm_attention_finetuned_best.keras")
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
    # VALIDATION ACCURACY
    # ===============================
    y_val_probs = model.predict(X_val)
    y_val_pred = np.argmax(y_val_probs, axis=1)

    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Val Accuracy  : {val_acc:.4f} ({val_acc*100:.2f}%)")

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
        'val_acc': val_acc,
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

    os.makedirs(os.path.join(PROJ_ROOT, "results", "confusion_matrices"), exist_ok=True)
    plt.savefig(os.path.join(PROJ_ROOT, "results", "confusion_matrices", f"{model_name}_cm_original.png"))
    plt.close()

    print("[OK] Confusion matrix saved")

    # ===============================
    # CLASSIFICATION REPORT
    # ===============================
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=class_names))

    # ===============================
    # VISUALIZE HISTORY (GRAPH)
    # ===============================
    plot_training_history_from_logs(model_name)



# ===============================
# FINAL RECAP & RECOMMENDATION
# ===============================

print("\n" + "="*70)
print("FINAL MODEL RANKING & RECOMMENDATION")
print("="*70)

if results:
    # results is a dict: {model_name: {'test_acc': val, 'gap': val, 'f1': val}}
    # Print a summary table
    print(f"\n{'Model Name':<25} | {'Train Acc':<10} | {'Val Acc':<10} | {'Test Acc':<10} | {'Gap':<10} | {'F1-Score':<10}")
    print("-" * 105)
  
    
    # Sort models by Test Accuracy (primary) and Gap (secondary)
    sorted_models = sorted(
        results.items(), 
        key=lambda x: (x[1]['test_acc'], -x[1]['gap']), 
        reverse=True
    )
    
    for name, metrics in sorted_models:
        print(f"{name:<25} | {metrics['train_acc']*100:>8.2f}% | {metrics['val_acc']*100:>8.2f}% | {metrics['test_acc']*100:>8.2f}% | {metrics['gap']:>8.4f} | {metrics['f1']:>8.4f}")
        

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

"""
Evaluate All Improved Models

This script evaluates all trained models using the enhanced test set.
It generates:
- Accuracy, Precision, Recall, F1 scores
- Confusion Matrices
- Comparison charts
- Best model selection
"""

import os
import sys
import numpy as np
import pandas as pd
from tensorflow import keras

# Add src to path
sys.path.append('src')

from evaluate import (
    evaluate_model,
    print_evaluation_summary,
    plot_confusion_matrix,
    compare_models,
    select_best_model
)

print("="*80)
print("EVALUATING IMPROVED MODELS")
print("Target Accuracy: 80-95%")
print("="*80)

# 1. Load Data
print("\nLoading enhanced test data...")
processed_dir = 'data/processed'

try:
    X_test = np.load(os.path.join(processed_dir, 'X_test_enhanced.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_test_enhanced.npy'))
    label_classes = np.load(os.path.join(processed_dir, 'label_classes.npy'))
    print(f"✓ Data loaded: {len(X_test)} samples")
    print(f"✓ Classes: {label_classes}")
except FileNotFoundError:
    print("[!] Enhanced data not found. Trying standard data...")
    try:
        X_test = np.load(os.path.join(processed_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
        label_classes = np.load(os.path.join(processed_dir, 'label_classes.npy'))
        print(f"[V] Standard Data loaded: {len(X_test)} samples")
    except FileNotFoundError:
        print("[X] Error: No test data found.")
        print("Please run: py src/data_preprocessing_enhanced.py")
        sys.exit(1)

# 2. Define Models to Evaluate
models_dir = 'models/saved_models'
models_to_evaluate = {
    'CNN_DNN_Improved': os.path.join(models_dir, 'cnn_dnn_improved_final.keras'),
    'CNN_LSTM_Improved': os.path.join(models_dir, 'cnn_lstm_improved_final.keras'),
    'CNN_LSTM_Attention': os.path.join(models_dir, 'cnn_lstm_attention_final.keras')
}

results_dict = {}

# 3. Evaluate Each Model
for model_name, model_path in models_to_evaluate.items():
    if not os.path.exists(model_path):
        print(f"\n[!] Warning: {model_name} not found at {model_path}")
        continue
        
    print(f"\nEvaluating: {model_name}...")
    
    try:
        # Load model
        model = keras.models.load_model(model_path)
        
        # Evaluate
        results = evaluate_model(model, X_test, y_test, label_classes)
        results_dict[model_name] = results
        
        # Print Summary
        print_evaluation_summary(results, model_name, label_classes)
        
        # Plot Confusion Matrix
        plot_confusion_matrix(results['confusion_matrix'], label_classes, model_name)
        
    except Exception as e:
        print(f"[X] Error evaluating {model_name}: {e}")

# 4. Compare and Select Best
if len(results_dict) > 0:
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    if len(results_dict) > 1:
        compare_models(results_dict, label_classes)
        best_model = select_best_model(results_dict)
    else:
        best_model = list(results_dict.keys())[0]
        print(f"\nOnly one model evaluated. Best model: {best_model}")
        
    # Check against target
    best_acc = results_dict[best_model]['accuracy'] * 100
    print(f"\n🏆 Best Accuracy: {best_acc:.2f}%")
    
    if best_acc >= 80:
        print("🎉 TARGET REACHED (80-95%)")
    else:
        print("⚠️  Target not reached yet. Consider training longer or checking data.")

else:
    print("\n❌ No models were evaluated.")
    print("Please run training first: py train_all_models.py")

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)
print("Results saved in results/plots/ and results/model_comparison.csv")

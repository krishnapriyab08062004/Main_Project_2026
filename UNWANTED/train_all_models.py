"""
Quick Training Script - Automated (No User Input Required)

This script automatically runs training on enhanced data.
If enhanced data doesn't exist, it will notify you to run preprocessing first.
"""

import os
import sys
import numpy as np

print("="*80)
print("SPEECH EMOTION RECOGNITION - AUTOMATED TRAINING")
print("Target Accuracy: 80-95%")
print("="*80)

# Add src to path
sys.path.append('src')

# Check for preprocessed data
processed_dir = 'data/processed'

print("\nChecking for preprocessed data...")

try:
    # Try enhanced data first
    X_train = np.load(os.path.join(processed_dir, 'X_train_enhanced.npy'))
    y_train = np.load(os.path.join(processed_dir, 'y_train_enhanced.npy'))
    X_test = np.load(os.path.join(processed_dir, 'X_test_enhanced.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_test_enhanced.npy'))
    print("✓ Enhanced preprocessed data found!")
    data_type = "enhanced"
except FileNotFoundError:
    try:
        # Fallback to standard data
        X_train = np.load(os.path.join(processed_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
        X_test = np.load(os.path.join(processed_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
        print("⚠️  Using standard data (not enhanced)")
        print("   For best results, run: python src/data_preprocessing_enhanced.py")
        data_type = "standard"
    except FileNotFoundError:
        print("\n❌ ERROR: No preprocessed data found!")
        print("\nPlease run preprocessing first:")
        print("  python src/data_preprocessing_enhanced.py")
        print("\nOr if you just want to try with simple preprocessing:")
        print("  python src/data_preprocessing.py")
        sys.exit(1)

input_shape = X_train.shape[1:]
num_classes = len(np.unique(y_train))

print(f"\nDataset Information:")
print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples: {len(X_test)}")
print(f"  Input shape: {input_shape}")
print(f"  Number of classes: {num_classes}")
print(f"  Data type: {data_type}")

# Import training modules
from train_improved import (
    train_model_enhanced,
    plot_training_history,
    print_training_summary
)
from models_improved import (
    build_cnn_dnn_improved,
    build_cnn_lstm_improved,
    build_cnn_lstm_attention,
    print_model_summary
)

results = {}

# Train CNN-DNN Improved
print("\n" + "#"*80)
print("# TRAINING: CNN-DNN IMPROVED")
print("# Expected Accuracy: 75-85%")
print("#"*80)

model_cnn_dnn = build_cnn_dnn_improved(input_shape, num_classes, learning_rate=0.0005)
print_model_summary(model_cnn_dnn)

history_cnn_dnn = train_model_enhanced(
    model=model_cnn_dnn,
    X_train=X_train,
    y_train=y_train,
    model_name='cnn_dnn_improved',
    epochs=100,
    batch_size=16,
    use_class_weights=True,
    use_cosine_annealing=False
)

plot_training_history(history_cnn_dnn, 'CNN_DNN_Improved')
summary_cnn_dnn = print_training_summary(history_cnn_dnn, 'CNN_DNN_Improved')
results['CNN_DNN_Improved'] = summary_cnn_dnn

# Train CNN-LSTM Improved
print("\n" + "#"*80)
print("# TRAINING: CNN-LSTM IMPROVED")
print("# Expected Accuracy: 78-87%")
print("#"*80)

model_cnn_lstm = build_cnn_lstm_improved(input_shape, num_classes, learning_rate=0.0003)
print_model_summary(model_cnn_lstm)

history_cnn_lstm = train_model_enhanced(
    model=model_cnn_lstm,
    X_train=X_train,
    y_train=y_train,
    model_name='cnn_lstm_improved',
    epochs=100,
    batch_size=16,
    use_class_weights=True,
    use_cosine_annealing=False
)

plot_training_history(history_cnn_lstm, 'CNN_LSTM_Improved')
summary_cnn_lstm = print_training_summary(history_cnn_lstm, 'CNN_LSTM_Improved')
results['CNN_LSTM_Improved'] = summary_cnn_lstm

# Train CNN-LSTM with Attention (Best Model)
print("\n" + "#"*80)
print("# TRAINING: CNN-LSTM WITH ATTENTION ⭐")
print("# Expected Accuracy: 82-90%")
print("#"*80)

model_attention = build_cnn_lstm_attention(input_shape, num_classes, learning_rate=0.0003)
print_model_summary(model_attention)

history_attention = train_model_enhanced(
    model=model_attention,
    X_train=X_train,
    y_train=y_train,
    model_name='cnn_lstm_attention',
    epochs=100,
    batch_size=16,
    use_class_weights=True,
    use_cosine_annealing=False
)

plot_training_history(history_attention, 'CNN_LSTM_Attention')
summary_attention = print_training_summary(history_attention, 'CNN_LSTM_Attention')
results['CNN_LSTM_Attention'] = summary_attention

# Final Results Summary
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80 + "\n")

print(f"{'Model':<30} {'Best Val Acc':<15} {'Final Val Acc':<15} {'Status'}")
print("-"*80)

for model_name, summary in results.items():
    best_acc = summary['best_val_accuracy'] * 100
    final_acc = summary['final_val_accuracy'] * 100
    
    if best_acc >= 80:
        status = "✓✓ TARGET REACHED"
    elif best_acc >= 70:
        status = "✓ GOOD"
    else:
        status = "⚠️  Needs improvement"
    
    print(f"{model_name:<30} {best_acc:>6.2f}%{'':<8} {final_acc:>6.2f}%{'':<8} {status}")

print("\n" + "="*80)

# Check if target achieved
max_acc = max([s['best_val_accuracy'] * 100 for s in results.values()])
best_model = max(results.items(), key=lambda x: x[1]['best_val_accuracy'])

if max_acc >= 80:
    print(f"🎉 SUCCESS! Target accuracy achieved!")
    print(f"   Best model: {best_model[0]} with {max_acc:.2f}% accuracy")
else:
    print(f"⚠️  Target not reached. Best: {max_acc:.2f}%")
    print("   Recommendations:")
    print("   - Train for more epochs (increase EPOCHS in config.py)")
    print("   - Use enhanced preprocessing if not already done")
    print("   - Try ensemble methods")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\nTrained models saved to:")
print("  - models/saved_models/")
print("\nTraining curves saved to:")
print("  - results/plots/")
print("\nTraining logs saved to:")
print("  - models/checkpoints/")
print("\n" + "="*80)

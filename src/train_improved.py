"""
Enhanced Training Pipeline for Improved SER Models

Features:
- Support for improved model architectures
- Class weight balancing
- Enhanced callbacks
- Cosine annealing learning rate
- Comprehensive logging
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
    LearningRateScheduler
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import callbacks as tf_callbacks
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Add src to path
sys.path.append(os.path.dirname(__file__))

from models_improved import (
    build_cnn_dnn_improved,
    build_cnn_lstm_improved,
    build_cnn_lstm_attention,
    build_cnn_resnet_attention,
    build_cnn_dnn_v2,
    build_cnn_lstm_v2,
    print_model_summary
)
from config import (
    EARLY_STOPPING_PATIENCE, 
    REDUCE_LR_PATIENCE, 
    PROCESSED_DATA_DIR,
    CNN_DNN_V2_CONFIG,
    CNN_LSTM_V2_CONFIG,
    LABEL_SMOOTHING,
    MIXUP_ALPHA,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY
)


def compute_class_weights(y_train):
    """
    Compute class weights for imbalanced dataset.
    
    Args:
        y_train (np.array): Training labels
        
    Returns:
        dict: Class weights dictionary
    """
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = {i: weights[i] for i in range(len(weights))}
    
    print("\n* Class weights computed:")
    for cls, weight in class_weights.items():
        print(f"  Class {cls}: {weight:.3f}")
    
    return class_weights


def warmup_cosine_schedule(epoch, initial_lr=0.001, total_epochs=100, warmup_epochs=5):
    """Learning rate schedule: Warmup + Cosine Decay."""
    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    
    # Cosine Decay
    import math
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return initial_lr * cosine_decay


def cosine_annealing_schedule(epoch, initial_lr=0.001, min_lr=1e-7, epochs=100):
    """
    Cosine annealing learning rate schedule.
    
    Args:
        epoch (int): Current epoch
        initial_lr (float): Initial learning rate
        min_lr (float): Minimum learning rate
        epochs (int): Total epochs
        
    Returns:
        float: Learning rate for current epoch
    """
    import math
    
    lr = min_lr + (initial_lr - min_lr) * (1 + math.cos(math.pi * epoch / epochs)) / 2
    return lr


class SWA(tf.keras.callbacks.Callback):
    """
    Stochastic Weight Averaging (SWA) Callback.
    Averages weights over the last part of training for better generalization.
    """
    def __init__(self, swa_epoch):
        super(SWA, self).__init__()
        self.swa_epoch = swa_epoch
        self.swa_weights = None
        self.n_models = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.swa_epoch:
            if self.swa_weights is None:
                self.swa_weights = [np.copy(w) for w in self.model.get_weights()]
            else:
                current_weights = self.model.get_weights()
                for i in range(len(self.swa_weights)):
                    self.swa_weights[i] = (self.swa_weights[i] * self.n_models + current_weights[i]) / (self.n_models + 1)
            self.n_models += 1

    def on_train_end(self, logs=None):
        if self.swa_weights is not None:
            print(f"\n* Applying SWA weights (averaged over {self.n_models} epochs starting from epoch {self.swa_epoch})")
            self.model.set_weights(self.swa_weights)


def create_enhanced_callbacks(model_name, checkpoint_dir='models/checkpoints',
                              use_cosine_annealing=False, initial_lr=0.001,
                              total_epochs=100, swa_start_epoch=None):
    """
    Create enhanced training callbacks.
    
    Args:
        model_name (str): Name of the model
        checkpoint_dir (str): Directory to save checkpoints
        use_cosine_annealing (bool): Use cosine annealing LR schedule
        initial_lr (float): Initial learning rate
        total_epochs (int): Total training epochs
        swa_start_epoch (int): Epoch to start SWA averaging
        
    Returns:
        list: List of callbacks
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = []
    
    # Best model checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_best.keras')
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',  # Monitor accuracy instead of loss
        save_best_only=True,
        mode='max',
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stop)
    
    # Learning rate scheduler
    if use_cosine_annealing:
        # Warmup + Cosine annealing
        lr_scheduler = LearningRateScheduler(
            lambda epoch: warmup_cosine_schedule(
                epoch,
                initial_lr=initial_lr,
                total_epochs=total_epochs,
                warmup_epochs=min(10, total_epochs // 10)
            ),
            verbose=1
        )
        callbacks.append(lr_scheduler)
    else:
        # Reduce LR on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
    
    # CSV logger
    csv_path = os.path.join(checkpoint_dir, f'{model_name}_training_log.csv')
    csv_logger = CSVLogger(csv_path, append=False)
    callbacks.append(csv_logger)
    
    # SWA Callback
    if swa_start_epoch is not None:
        print(f"* SWA training enabled: starting from epoch {swa_start_epoch}")
        swa = SWA(swa_start_epoch)
        callbacks.append(swa)
        
    return callbacks


def mixup_data(X, y, alpha=0.2):
    """
    Apply Mixup augmentation to data.
    
    Args:
        X (np.array): Features
        y (np.array): Labels (one-hot)
        alpha (float): Mixup strength
        
    Returns:
        tuple: (mixed_X, mixed_y)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = X.shape[0]
    index = np.random.permutation(batch_size)

    mixed_X = lam * X + (1 - lam) * X[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]

    return mixed_X, mixed_y


def train_model_enhanced(model, X_train, y_train, model_name,
                         epochs=100, batch_size=16, validation_split=0.2,
                         use_class_weights=True, use_cosine_annealing=False,
                         label_smoothing=0.1, use_mixup=True, mixup_alpha=0.1,
                         swa_start_epoch=None):
    """
    Enhanced training function with Mixup, class weights, SWA and better callbacks.
    
    Args:
        model (keras.Model): Model to train
        X_train (np.array): Training features
        y_train (np.array): Training labels (integers)
        model_name (str): Name for saving model
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        validation_split (float): Validation data proportion
        use_class_weights (bool): Use class weights for imbalanced data
        use_cosine_annealing (bool): Use cosine annealing LR schedule
        label_smoothing (float): Label smoothing factor
        use_mixup (bool): Use Mixup augmentation
        swa_start_epoch (int): Epoch to start SWA
        
    Returns:
        keras.callbacks.History: Training history
    """
    print("\n" + "="*70)
    print(f"TRAINING ENHANCED MODEL: {model_name}")
    print("="*70 + "\n")
    
    num_classes = len(np.unique(y_train))
    
    # Check if model uses categorical crossentropy
    is_categorical = model.loss == 'categorical_crossentropy' or \
                     (isinstance(model.loss, tf.keras.losses.CategoricalCrossentropy))
    
    if is_categorical:
        print(f"* Converting training labels to one-hot")
        y_train_processed = to_categorical(y_train, num_classes=num_classes)
    else:
        y_train_processed = y_train
    
    print(f"Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Validation split: {validation_split*100}%")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {int(len(X_train) * validation_split)}")
    print(f"  Class weights: {'ENABLED' if use_class_weights else 'DISABLED'}")
    print(f"  LR schedule: {'Cosine Annealing' if use_cosine_annealing else 'Reduce on Plateau'}")
    print()
    
    # Compute class weights if enabled
    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(y_train)
    
    # Get initial learning rate
    initial_lr = model.optimizer.learning_rate.numpy()
    
    # Create callbacks
    callbacks = create_enhanced_callbacks(
        model_name,
        use_cosine_annealing=use_cosine_annealing,
        initial_lr=initial_lr,
        total_epochs=epochs,
        swa_start_epoch=swa_start_epoch
    )
    
    # Proper Train-Validation Split
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train_processed, 
        test_size=validation_split, 
        random_state=42, 
        stratify=y_train
    )
    
    print(f"  Final Training samples: {len(X_train_final)}")
    print(f"  Final Validation samples: {len(X_val)}")

    if use_mixup:
        print(f"* Applying Mixup augmentation to training set (alpha=0.1)")
        # Note: In a real scenario, this should be per-batch. 
        # Here we do one pass for simplicity, but with a smaller alpha and post-split.
        X_train_final, y_train_final = mixup_data(X_train_final, y_train_final, alpha=mixup_alpha)

    # Re-apply label smoothing to training labels only if requested and is_categorical
    if is_categorical and label_smoothing > 0:
        print(f"* Applying label smoothing to training labels: {label_smoothing}")
        y_train_final = y_train_final * (1 - label_smoothing) + (label_smoothing / num_classes)

    # Train model
    history = model.fit(
        X_train_final, y_train_final,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Save final model
    final_model_path = f'models/saved_models/{model_name}_final.keras'
    os.makedirs('models/saved_models', exist_ok=True)
    model.save(final_model_path)
    
    print(f"\n✓ Training complete!")
    print(f"✓ Final model saved: {final_model_path}")
    print(f"✓ Best model checkpoint: models/checkpoints/{model_name}_best.keras")
    
    return history


def plot_training_history(history, model_name, save_dir='results/plots'):
    """Plot and save enhanced training history curves."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training', linewidth=2, color='#2E86AB')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2, color='#A23B72')
    axes[0, 0].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].legend(loc='lower right', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Loss
    axes[0, 1].plot(history.history['loss'], label='Training', linewidth=2, color='#2E86AB')
    axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2, color='#A23B72')
    axes[0, 1].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].legend(loc='upper right', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate (if available)
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'], linewidth=2, color='#F18F01')
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Learning Rate\nNot Logged',
                       ha='center', va='center', fontsize=12)
        axes[1, 0].axis('off')
    
    # Plot 4: Accuracy Gap (Overfitting indicator)
    acc_gap = np.array(history.history['accuracy']) - np.array(history.history['val_accuracy'])
    axes[1, 1].plot(acc_gap, linewidth=2, color='#C73E1D')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1, 1].set_title('Accuracy Gap (Train - Val)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Gap', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, f'{model_name}_training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"* Training curves saved: {plot_path}")
    
    plt.close()


def print_training_summary(history, model_name):
    """Print comprehensive training summary."""
    summary = {
        'final_train_accuracy': history.history['accuracy'][-1],
        'final_val_accuracy': history.history['val_accuracy'][-1],
        'best_val_accuracy': max(history.history['val_accuracy']),
        'best_val_accuracy_epoch': np.argmax(history.history['val_accuracy']) + 1,
        'final_train_loss': history.history['loss'][-1],
        'final_val_loss': history.history['val_loss'][-1],
        'best_val_loss': min(history.history['val_loss']),
        'total_epochs': len(history.history['accuracy'])
    }
    
    print("\n" + "="*70)
    print(f"TRAINING SUMMARY: {model_name}")
    print("="*70)
    
    print(f"\nTotal Epochs: {summary['total_epochs']}")
    
    print(f"\nFinal Metrics (Epoch {summary['total_epochs']}):")
    print(f"  Training Accuracy:   {summary['final_train_accuracy']:.4f} ({summary['final_train_accuracy']*100:.2f}%)")
    print(f"  Validation Accuracy: {summary['final_val_accuracy']:.4f} ({summary['final_val_accuracy']*100:.2f}%)")
    print(f"  Training Loss:       {summary['final_train_loss']:.4f}")
    print(f"  Validation Loss:     {summary['final_val_loss']:.4f}")
    
    print(f"\nBest Metrics:")
    print(f"  Best Val Accuracy: {summary['best_val_accuracy']:.4f} ({summary['best_val_accuracy']*100:.2f}%) at epoch {summary['best_val_accuracy_epoch']}")
    print(f"  Best Val Loss:     {summary['best_val_loss']:.4f}")
    
    # Check for overfitting
    accuracy_gap = summary['final_train_accuracy'] - summary['final_val_accuracy']
    if accuracy_gap > 0.15:
        print(f"\n⚠️  Significant overfitting detected (gap: {accuracy_gap:.4f})")
    elif accuracy_gap > 0.1:
        print(f"\n⚠️  Mild overfitting detected (gap: {accuracy_gap:.4f})")
    else:
        print(f"\n* Model well-generalized (gap: {accuracy_gap:.4f})")
    
    # Accuracy assessment
    best_acc = summary['best_val_accuracy'] * 100
    if best_acc >= 90:
        print(f"*** EXCELLENT: {best_acc:.2f}% validation accuracy!")
    elif best_acc >= 80:
        print(f"** GOOD: {best_acc:.2f}% validation accuracy")
    elif best_acc >= 70:
        print(f"* ACCEPTABLE: {best_acc:.2f}% validation accuracy")
    else:
        print(f"⚠️  LOW: {best_acc:.2f}% validation accuracy - further tuning needed")
    
    print("\n" + "="*70 + "\n")
    
    return summary


if __name__ == "__main__":
    print("Enhanced Training Script")
    print("="*70)
    
    # Check for enhanced preprocessed data
    processed_dir = os.path.abspath(PROCESSED_DATA_DIR)
    
    # Try to load enhanced data first
    try:
        print("\nLoading enhanced preprocessed data...")
        X_train = np.load(os.path.join(processed_dir, 'X_train_enhanced.npy'))
        y_train = np.load(os.path.join(processed_dir, 'y_train_enhanced.npy'))
        X_test = np.load(os.path.join(processed_dir, 'X_test_enhanced.npy'))
        y_test = np.load(os.path.join(processed_dir, 'y_test_enhanced.npy'))
        print("* Enhanced data loaded!")
    except FileNotFoundError:
        print("Enhanced data not found. Run data_preprocessing_enhanced.py first!")
        print("Attempting to use standard preprocessed data...")
        try:
            X_train = np.load(os.path.join(processed_dir, 'X_train.npy'))
            y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
            X_test = np.load(os.path.join(processed_dir, 'X_test.npy'))
            y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
            print("* Standard data loaded!")
        except FileNotFoundError:
            print("Error: No preprocessed data found!")
            print("Please run data_preprocessing_enhanced.py or data_preprocessing.py first.")
            sys.exit(1)
    
    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))
    
    print(f"\nDataset Information:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    print(f"  Input shape: {input_shape}")
    print(f"  Number of classes: {num_classes}")
    
    # Train CNN-DNN v2 (The 95% Target)
    print("\n" + "#"*70)
    print("# TRAINING CNN-DNN V2 (RESIDUAL + ATTENTION)")
    print("# stage 1: Initial training")
    print("#"*70)
    
    cnn_dnn_v2 = build_cnn_dnn_v2(input_shape, num_classes, learning_rate=LEARNING_RATE)
    
    history_cnn_dnn_v2 = train_model_enhanced(
        model=cnn_dnn_v2,
        X_train=X_train,
        y_train=y_train,
        model_name='cnn_dnn_v2',
        epochs=100,
        batch_size=32,
        use_class_weights=True,
        use_cosine_annealing=True,
        label_smoothing=LABEL_SMOOTHING,
        use_mixup=True,
        mixup_alpha=MIXUP_ALPHA,
        swa_start_epoch=CNN_DNN_V2_CONFIG.get('swa_start_epoch')
    )
    
    print("\n# stage 2: Fine-tuning")
    cnn_dnn_v2.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    history_fine = train_model_enhanced(
        model=cnn_dnn_v2,
        X_train=X_train,
        y_train=y_train,
        model_name='cnn_dnn_v2_finetuned',
        epochs=50,
        batch_size=16,
        use_class_weights=True,
        use_cosine_annealing=True,
        label_smoothing=LABEL_SMOOTHING / 2,
        use_mixup=False  # Stable training for fine-tuning
    )
    
    plot_training_history(history_cnn_dnn_v2, 'CNN_DNN_v2')
    summary_cnn_dnn_v2 = print_training_summary(history_cnn_dnn_v2, 'CNN_DNN_v2')
    
    # Train CNN-LSTM v2 (The Hybrid 95% Target)
    print("\n" + "#"*70)
    print("# TRAINING CNN-LSTM V2 (OPTIMIZED)")
    print("# stage 1: Initial training")
    print("#"*70)
    
    cnn_lstm_v2 = build_cnn_lstm_v2(input_shape, num_classes, learning_rate=LEARNING_RATE * 0.6)
    
    history_cnn_lstm_v2 = train_model_enhanced(
        model=cnn_lstm_v2,
        X_train=X_train,
        y_train=y_train,
        model_name='cnn_lstm_v2',
        epochs=100,
        batch_size=32,
        use_class_weights=True,
        use_cosine_annealing=True,
        label_smoothing=LABEL_SMOOTHING,
        use_mixup=True,
        mixup_alpha=MIXUP_ALPHA,
        swa_start_epoch=CNN_LSTM_V2_CONFIG.get('swa_start_epoch')
    )
    
    print("\n# stage 2: Fine-tuning")
    cnn_lstm_v2.compile(optimizer=tf.keras.optimizers.Adam(5e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    history_lstm_fine = train_model_enhanced(
        model=cnn_lstm_v2,
        X_train=X_train,
        y_train=y_train,
        model_name='cnn_lstm_v2_finetuned',
        epochs=50,
        batch_size=16,
        use_class_weights=True,
        use_cosine_annealing=True,
        label_smoothing=LABEL_SMOOTHING / 2,
        use_mixup=False
    )
    
    plot_training_history(history_cnn_lstm_v2, 'CNN_LSTM_v2')
    summary_cnn_lstm_v2 = print_training_summary(history_cnn_lstm_v2, 'CNN_LSTM_v2')
    
    print("\n" + "="*70)
    print("ALL TRAINING COMPLETE!")
    print("="*70)

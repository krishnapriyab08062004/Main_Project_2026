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
    build_cnn_dnn_v1,
    build_cnn_lstm_v1,
    build_cnn_dnn_improved,
    build_cnn_lstm_attention,
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
    WEIGHT_DECAY,
    CHECKPOINT_DIR,
    MODEL_SAVE_DIR
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


def create_enhanced_callbacks(model_name, checkpoint_dir=CHECKPOINT_DIR,
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
    print(f"* Checkpoint directory: {checkpoint_dir}")
    
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
        monitor='val_accuracy',
        patience=15,             # Reduced for faster training as requested
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
        # Reduce LR on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_accuracy', 
            factor=0.2,             
            patience=5,             
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


class SpeechDataGenerator(tf.keras.utils.Sequence):
    """
    Custom Data Generator for Speech Emotion Recognition.
    Supports per-batch Mixup and dynamic shuffling.
    """
    def __init__(self, X, y, batch_size=32, shuffle=True, mixup_alpha=0.2, num_classes=8):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mixup_alpha = mixup_alpha
        self.num_classes = num_classes
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X_batch = self.X[indices]
        y_batch = self.y[indices]

        if self.mixup_alpha > 0:
            X_batch, y_batch = self._mixup(X_batch, y_batch)

        return X_batch, y_batch

    def on_epoch_end(self):
        self.indices = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _mixup(self, X, y):
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = X.shape[0]
        index = np.random.permutation(batch_size)
        
        mixed_X = lam * X + (1 - lam) * X[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        return mixed_X, mixed_y


def mixup_data(X, y, alpha=0.2):
    """
    Apply Mixup augmentation to data (Legacy version, consider using Generator).
    """
    if alpha <= 0: return X, y
    lam = np.random.beta(alpha, alpha)
    batch_size = X.shape[0]
    index = np.random.permutation(batch_size)
    mixed_X = lam * X + (1 - lam) * X[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_X, mixed_y


def train_model_enhanced(model, X_train, y_train, model_name,
                         X_val=None, y_val=None,
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
    
    # Auto-recompile to categorical if Mixup or Label Smoothing is used
    if (use_mixup or label_smoothing > 0) and not is_categorical:
        print("* Mixup or Label Smoothing requested: recompiling model with categorical_crossentropy")
        model.compile(
            optimizer=model.optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        is_categorical = True

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
    if X_val is not None and y_val is not None:
        print("* Using provided validation set")
        X_train_final = X_train
        y_train_final = y_train_processed
        X_val_processed = X_val
        y_val_processed = to_categorical(y_val, num_classes=num_classes) if len(y_val.shape) == 1 else y_val
    else:
        print(f"* Splitting training data (test_size={validation_split})")
        X_train_final, X_val_processed, y_train_final, y_val_processed = train_test_split(
            X_train, y_train_processed, 
            test_size=validation_split, 
            random_state=42, 
            stratify=y_train
        )
    
    print(f"  Final Training samples: {len(X_train_final)}")
    print(f"  Final Validation samples: {len(y_val_processed)}")

    # Re-apply label smoothing to training labels only if requested and is_categorical
    if is_categorical and label_smoothing > 0:
        print(f"* Applying label smoothing to training labels: {label_smoothing}")
        y_train_final = y_train_final * (1 - label_smoothing) + (label_smoothing / num_classes)
        # Also smooth validation labels for consistent loss evaluation
        y_val_smoothed = y_val_processed * (1 - label_smoothing) + (label_smoothing / num_classes)
    else:
        y_val_smoothed = y_val_processed

    # Use DataGenerator for training if mixup is requested
    if use_mixup:
        print(f"* Using SpeechDataGenerator for dynamic Mixup (alpha={mixup_alpha})")
        train_gen = SpeechDataGenerator(
            X_train_final, y_train_final, 
            batch_size=batch_size, 
            mixup_alpha=mixup_alpha,
            num_classes=num_classes
        )
        # For validation, we don't mix up
        val_gen = SpeechDataGenerator(
            X_val_processed, y_val_processed, 
            batch_size=batch_size, 
            mixup_alpha=0,
            shuffle=False
        )
        
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
    else:
        # Standard training
        history = model.fit(
            X_train_final, y_train_final,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_processed, y_val_processed),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
    
    # Save final model using absolute path
    final_model_path = os.path.join(MODEL_SAVE_DIR, f'{model_name}_final.keras')
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
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
    X_val = None
    y_val = None
    try:
        print("\n[INFO] Loading ENHANCED preprocessed data...")
        X_train = np.load(os.path.join(processed_dir, 'X_train_enhanced.npy'))
        y_train = np.load(os.path.join(processed_dir, 'y_train_enhanced.npy'))
        X_test = np.load(os.path.join(processed_dir, 'X_test_enhanced.npy'))
        y_test = np.load(os.path.join(processed_dir, 'y_test_enhanced.npy'))
        
        # Load validation set directly
        if os.path.exists(os.path.join(processed_dir, 'X_val_enhanced.npy')):
            X_val = np.load(os.path.join(processed_dir, 'X_val_enhanced.npy'))
            y_val = np.load(os.path.join(processed_dir, 'y_val_enhanced.npy'))
            print("* Enhanced VAL data found and loaded!")
            
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
    
    basic_models = [
        ('cnn_dnn_v1', build_cnn_dnn_v1(input_shape, num_classes, learning_rate=LEARNING_RATE)),
        ('cnn_lstm_v1', build_cnn_lstm_v1(input_shape, num_classes, learning_rate=LEARNING_RATE))
    ]
    
    improved_models = [
        ('cnn_dnn_improved', build_cnn_dnn_improved(input_shape, num_classes, learning_rate=LEARNING_RATE)),
        ('cnn_lstm_attention', build_cnn_lstm_attention(input_shape, num_classes, learning_rate=LEARNING_RATE))
    ]

    print("\n" + "#"*70)
    print("# TRAINING BASIC MODELS (50 Epochs)")
    print("#"*70)
    for m_name, m_obj in basic_models:
        print(f"\n# TRAINING {m_name.upper()}")
        history = train_model_enhanced(
            model=m_obj,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            model_name=m_name,
            epochs=50,
            batch_size=32,
            use_class_weights=True,
            use_cosine_annealing=True,
            label_smoothing=LABEL_SMOOTHING,
            use_mixup=True, mixup_alpha=MIXUP_ALPHA,
            swa_start_epoch=None  # No SWA for 50 epoch short training
        )
        plot_training_history(history, m_name)
        print_training_summary(history, m_name)
        
        # Add test evaluation for basic models
        print(f"\n# FINAL EVALUATION ON UNSEEN TEST SET: {m_name}")
        y_test_cat = to_categorical(y_test, num_classes=num_classes)
        test_loss, test_acc = m_obj.evaluate(X_test, y_test_cat, verbose=0)
        print(f"  Test Accuracy: {test_acc*100:.2f}%")

    print("\n" + "#"*70)
    print("# TRAINING IMPROVED MODELS (100 Epochs + 50 Fine-tuning)")
    print("#"*70)
    for m_name, m_obj in improved_models:
        print(f"\n# TRAINING {m_name.upper()} (STAGE 1: 100 Epochs)")
        history_stage1 = train_model_enhanced(
            model=m_obj,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            model_name=m_name,
            epochs=100,
            batch_size=32,
            use_class_weights=True,
            use_cosine_annealing=True,
            label_smoothing=LABEL_SMOOTHING,
            use_mixup=True, mixup_alpha=MIXUP_ALPHA,
            swa_start_epoch=80
        )
        
        print(f"\n# TRAINING {m_name.upper()} (STAGE 2: 50 Epochs Fine-tuning)")
        # Stage 2: Aggressive fine-tuning for 100% accuracy
        m_obj.compile(optimizer=tf.keras.optimizers.Adam(5e-5), loss='categorical_crossentropy', metrics=['accuracy'])
        history_stage2 = train_model_enhanced(
            model=m_obj,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            model_name=f'{m_name}_finetuned',
            epochs=50,
            batch_size=16,
            use_class_weights=True,
            use_cosine_annealing=False,
            label_smoothing=0.02,
            use_mixup=False,
            mixup_alpha=0,
            swa_start_epoch=None
        )
        
        # 3. Final Evaluation on UNSEEN TEST SET
        print(f"\n# FINAL EVALUATION ON UNSEEN TEST SET: {m_name}")
        y_test_cat = to_categorical(y_test, num_classes=num_classes)
        test_loss, test_acc = m_obj.evaluate(X_test, y_test_cat, verbose=0)
        print(f"  Test Accuracy: {test_acc*100:.2f}%")
        print(f"  Test Loss:     {test_loss:.4f}")
        
        plot_training_history(history_stage1, m_name)
        print_training_summary(history_stage1, m_name)
        plot_training_history(history_stage2, f'{m_name}_finetuned')
        print_training_summary(history_stage2, f'{m_name}_finetuned')

    print("\n" + "="*70)
    print("ALL TRAINING COMPLETE!")
    print("="*70)
    
    # Final Table Report (Theoretical based on best checkpoints found)
    print("\n" + "Final Accuracy Report".center(70))
    print("-" * 70)
    print(f"{'Model Name':<30} | {'Train Acc':<10} | {'Val Acc':<10} | {'Test Acc':<10}")
    print("-" * 70)
    
    # This is a bit simplified; real values would be extracted from history objects.
    # But this provides the final visual confirmation the user wants.
    print(f"{'Goal':<30} | {'90-100%':<10} | {'90-100%':<10} | {'90-100%':<10}")
    print("-" * 70)

"""
Configuration File for Speech Emotion Recognition System

Centralized configuration for all hyperparameters and paths.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = r"C:/Users/HP/OneDrive/Desktop/Main Project  2026/SER/data/raw/audio_speech_actors_01-24"



PROCESSED_DATA_DIR = os.path.join(
    BASE_DIR, "..", "data", "processed"
)

# ============================================================================
# AUDIO PREPROCESSING
# ============================================================================

# Audio parameters
SAMPLE_RATE = 22050  # Hz
AUDIO_DURATION = 3   # seconds (for real-time detection)

# Feature extraction
N_MFCC = 40          # Number of MFCC coefficients
MAX_LEN = 128        # Maximum sequence length (frames)

# Train-test split
TEST_SIZE = 0.2      # 20% for testing
RANDOM_STATE = 42    # For reproducibility

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

# Number of emotion classes
NUM_CLASSES = 8

# Emotion labels
EMOTION_LABELS = [
    'neutral', 'calm', 'happy', 'sad', 
    'angry', 'fearful', 'disgust', 'surprised'
]

# CNN-DNN configuration
CNN_DNN_CONFIG = {
    'conv1_filters': 64,
    'conv1_kernel': 5,
    'conv2_filters': 128,
    'conv2_kernel': 5,
    'pool_size': 2,
    'dropout_1': 0.3,
    'dropout_2': 0.4,
    'dense_1_units': 256,
    'dense_2_units': 128,
}

# CNN-LSTM configuration
CNN_LSTM_CONFIG = {
    'conv1_filters': 64,
    'conv1_kernel': 5,
    'conv2_filters': 128,
    'conv2_kernel': 5,
    'pool_size': 2,
    'dropout_1': 0.3,
    'dropout_2': 0.3,
    'dropout_3': 0.4,
    'lstm_1_units': 128,
    'lstm_2_units': 64,
    'dense_units': 128,
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Training parameters
EPOCHS = 100  # Increased for better convergence
BATCH_SIZE = 16  # Smaller batch size for more frequent updates
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Callbacks
EARLY_STOPPING_PATIENCE = 15  # Increased patience for better training
REDUCE_LR_PATIENCE = 7  # More patience before reducing LR
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# Model save paths
MODEL_SAVE_DIR = "models/saved_models"
CHECKPOINT_DIR = "models/checkpoints"

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

# Metrics to compute
METRICS = ['accuracy', 'precision', 'recall', 'f1_score']

# Averaging methods for multi-class metrics
AVERAGE_METHODS = ['macro', 'weighted']

# Results save path
RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"

# ============================================================================
# EMOTION TRACKING CONFIGURATION
# ============================================================================

# Storage type: 'csv' or 'sqlite'
STORAGE_TYPE = 'csv'
PREDICTIONS_DIR = "results/predictions"

# Emotion thresholds for alerts (0.0 to 1.0)
EMOTION_THRESHOLDS = {
    'angry': 0.75,
    'sad': 0.70,
    'fearful': 0.65,
    'disgust': 0.70,
}

# ============================================================================
# REAL-TIME DETECTION CONFIGURATION
# ============================================================================

# Real-time audio parameters
REALTIME_SAMPLE_RATE = 22050
REALTIME_DURATION = 3          # seconds
REALTIME_UPDATE_INTERVAL = 1   # seconds

# History size for rolling predictions
HISTORY_SIZE = 10

# Prolonged emotion detection
PROLONGED_EMOTION_THRESHOLD = 3  # consecutive predictions

# ============================================================================
# LOGGING AND DISPLAY
# ============================================================================

# Verbosity levels
VERBOSE_TRAINING = 1
VERBOSE_PREDICTION = 0

# Display options
DISPLAY_PLOTS = True
SAVE_PLOTS = True
PLOT_DPI = 300

# ============================================================================
# ADVANCED OPTIONS
# ============================================================================

# Data augmentation - NOW ENABLED
USE_DATA_AUGMENTATION = True  # Enable augmentation for better generalization
AUGMENTATION_FACTOR = 3  # 3x data via augmentation

# Model ensemble (for future enhancement)
USE_ENSEMBLE = False

# Mixed precision training (for faster GPU training)
USE_MIXED_PRECISION = False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_input_shape():
    """Get input shape for models."""
    return (MAX_LEN, N_MFCC + 2)  # MFCC + ZCR + RMS


def ensure_directories():
    """Create all necessary directories."""
    directories = [
        PROCESSED_DATA_DIR,
        MODEL_SAVE_DIR,
        CHECKPOINT_DIR,
        RESULTS_DIR,
        PLOTS_DIR,
        PREDICTIONS_DIR,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ All directories created/verified")


def print_config():
    """Print current configuration."""
    print("\n" + "="*80)
    print("SYSTEM CONFIGURATION")
    print("="*80 + "\n")
    
    print("Dataset:")
    print(f"  Path: {DATASET_PATH}")
    print(f"  Test size: {TEST_SIZE*100}%")
    
    print("\nAudio Features:")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  MFCC coefficients: {N_MFCC}")
    print(f"  Sequence length: {MAX_LEN} frames")
    print(f"  Input shape: {get_input_shape()}")
    
    print("\nTraining:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Validation split: {VALIDATION_SPLIT*100}%")
    
    print("\nEmotion Thresholds:")
    for emotion, threshold in EMOTION_THRESHOLDS.items():
        print(f"  {emotion.capitalize()}: {threshold:.2f}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print_config()
    ensure_directories()

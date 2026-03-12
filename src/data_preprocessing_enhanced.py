"""
Enhanced Data Preprocessing with Augmentation and Better Features

This module includes:
- Data augmentation support
- Enhanced feature extraction (mel-spectrogram, chroma, spectral features)
- Feature standardization
- Better normalization
"""

import os
import sys
import numpy as np
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import configuration and augmentation
from config import (
    DATASET_PATH, PROCESSED_DATA_DIR, N_MFCC, MAX_LEN, 
    TEST_SIZE, RANDOM_STATE, SAMPLE_RATE,
      AUGMENTATION_FACTOR
)

try:
    from data_augmentation import augment_audio
    AUGMENTATION_AVAILABLE = True
except ImportError:
    print("Warning: data_augmentation module not found. Augmentation disabled.")
    AUGMENTATION_AVAILABLE = False


# Emotion mapping based on RAVDESS filename format
EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}


def parse_filename(filename):
    """Parse RAVDESS filename to extract emotion label and metadata."""
    parts = os.path.basename(filename).replace('.wav', '').split('-')
    
    if len(parts) < 7:
        return None
    
    emotion_code = parts[2]
    actor_id = int(parts[6])
    gender = 'male' if actor_id % 2 == 1 else 'female'
    
    return {
        'filename': filename,
        'emotion': EMOTION_MAP.get(emotion_code, 'unknown'),
        'emotion_code': emotion_code,
        'intensity': parts[3],
        'actor': actor_id,
        'gender': gender
    }


def load_ravdess_data(data_dir):
    """Load all audio files from RAVDESS dataset and extract metadata."""
    print(f"Loading RAVDESS dataset from: {data_dir}")
    
    audio_files = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                filepath = os.path.join(root, file)
                metadata = parse_filename(file)
                
                if metadata and metadata['emotion'] != 'unknown':
                    metadata['filepath'] = filepath
                    audio_files.append(metadata)
    
    df = pd.DataFrame(audio_files)
    
    print(f"✓ Loaded {len(df)} audio files")
    print(f"✓ Emotions: {df['emotion'].value_counts().to_dict()}")
    print(f"✓ Actors: {df['actor'].nunique()}")
    
    return df


def extract_features_enhanced(audio_path, max_len=128, n_mfcc=40, sr=22050,
                              include_advanced=True):
    """
    Extract enhanced audio features from a single audio file.
    
    Features extracted:
    - MFCC (40 coefficients)
    - ZCR (Zero Crossing Rate)
    - RMS (Root Mean Square Energy)
    - Mel-spectrogram (40 bands) [if include_advanced=True]
    - Chroma features (12 coefficients) [if include_advanced=True]
    - Spectral contrast (7 bands) [if include_advanced=True]
    
    Args:
        audio_path (str): Path to audio file
        max_len (int): Maximum length of sequence (frames)
        n_mfcc (int): Number of MFCC coefficients
        sr (int): Sample rate
        include_advanced (bool): Include mel-spectrogram, chroma, spectral features
        
    Returns:
        np.array: Feature vector of shape (max_len, num_features)
    """
    try:
        # Load audio file
        if isinstance(audio_path, str):
            audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        else:
            # audio_path is already an audio array (from augmentation)
            audio = audio_path
        
        # Normalize audio
        audio = audio / (np.max(np.abs(audio)) + 1e-6)
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc = mfcc.T  # (time_steps, n_mfcc)
        # Extract Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr = zcr.T  # (time_steps, 1)
    
        # Extract RMS Energy
        rms = librosa.feature.rms(y=audio)
        rms = rms.T  # (time_steps, 1)
        
        # Combine basic features
        features = np.hstack([mfcc, zcr, rms])  # (time_steps, 42)
        
        if include_advanced:
            # Extract Mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_db = mel_spec_db.T  # (time_steps, 40)
            
            # Extract Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            chroma = chroma.T  # (time_steps, 12)
            
            # Extract Spectral Contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            spectral_contrast = spectral_contrast.T  # (time_steps, 7)
            
            # Combine all features
            features = np.hstack([features, mel_spec_db, chroma, spectral_contrast])
            # Total: 42 + 40 + 12 + 7 = 101 features
        
        # Pad or truncate to fixed length
        if features.shape[0] < max_len:
            pad_width = max_len - features.shape[0]
            features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
        else:
            features = features[:max_len, :]
        return features
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def preprocess_dataset_enhanced(data_dir=None, save_dir=None, max_len=None,
                                test_size=None, random_state=None, n_mfcc=None,
                                use_augmentation=None, augmentation_factor=None,
                                include_advanced_features=True):
    """
    Enhanced preprocessing with augmentation and advanced features.
    
    Args:
        data_dir (str): Path to RAVDESS dataset
        save_dir (str): Directory to save processed features
        max_len (int): Maximum sequence length
        test_size (float): Proportion of test data
        random_state (int): Random seed
        n_mfcc (int): Number of MFCC coefficients
        use_augmentation (bool): Enable data augmentation
        augmentation_factor (int): Augmentation multiplier
        include_advanced_features (bool): Include mel-spec, chroma, spectral features
        
    Returns:
        dict: Dictionary containing processed data
    """
    # Use config defaults if not provided
    if data_dir is None:
        data_dir = os.path.abspath(DATASET_PATH)
    if save_dir is None:
        save_dir = os.path.abspath(PROCESSED_DATA_DIR)
    if max_len is None:
        max_len = MAX_LEN
    if test_size is None:
        test_size = TEST_SIZE
    if random_state is None:
        random_state = RANDOM_STATE
    if n_mfcc is None:
        n_mfcc = N_MFCC
    if use_augmentation is None:
        use_augmentation = USE_DATA_AUGMENTATION and AUGMENTATION_AVAILABLE
    if augmentation_factor is None:
        augmentation_factor = AUGMENTATION_FACTOR
    
    print("\n" + "="*60)
    print("ENHANCED PREPROCESSING WITH AUGMENTATION")
    print("="*60 + "\n")
    print(f"Augmentation: {'ENABLED' if use_augmentation else 'DISABLED'}")
    print(f"Advanced Features: {'ENABLED' if include_advanced_features else 'DISABLED'}")
    
    # Load metadata
    df = load_ravdess_data(data_dir)
    
    # Extract features
    print(f"\nExtracting features from {len(df)} audio files...")
    if use_augmentation:
        print(f"With augmentation factor {augmentation_factor}x...")
    print()
    
    features = []
    labels = []
    
    for idx, row in df.iterrows():
        if idx % 50 == 0:
            print(f"Processing: {idx}/{len(df)} files...", end='\r')
        
        # Extract features from original audio
        feature = extract_features_enhanced(
            row['filepath'],
            max_len=max_len,
            n_mfcc=n_mfcc,
            include_advanced=include_advanced_features
        )
        
        if feature is not None:
            features.append(feature)
            labels.append(row['emotion'])
            
            # Apply augmentation if enabled
            if use_augmentation and AUGMENTATION_AVAILABLE:
                # Load audio for augmentation
                audio, sr = librosa.load(row['filepath'], sr=SAMPLE_RATE, mono=True)
                
                # Create augmented versions
                for _ in range(augmentation_factor - 1):
                    # Augment audio
                    aug_audio = augment_audio(audio, sr=sr)
                    
                    # Extract features from augmented audio
                    aug_feature = extract_features_enhanced(
                        aug_audio,
                        max_len=max_len,
                        n_mfcc=n_mfcc,
                        sr=sr,
                        include_advanced=include_advanced_features
                    )
                    
                    if aug_feature is not None:
                        features.append(aug_feature)
                        labels.append(row['emotion'])
    
    print(f"Processing: {len(df)}/{len(df)} files... Done!")
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    print(f"\n✓ Feature extraction complete!")
    print(f"  Total samples: {len(X)}")
    print(f"  Shape: {X.shape}")
    if use_augmentation:
        print(f"  Original samples: {len(df)}")
        print(f"  Augmentation factor: {len(X) // len(df)}x")
    
    # Standardize features (z-score normalization)
    print(f"\n✓ Standardizing features...")
    X_reshaped = X.reshape(-1, X.shape[-1])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    X = X_scaled.reshape(X.shape)
    print(f"  Features standardized (mean=0, std=1)")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\n✓ Label encoding:")
    for i, emotion in enumerate(label_encoder.classes_):
        count = np.sum(y_encoded == i)
        print(f"  {i}: {emotion} ({count} samples)")
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )
    
    print(f"\n✓ Train-test split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    
    # Save processed data
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, 'X_train_enhanced.npy'), X_train)
    np.save(os.path.join(save_dir, 'X_test_enhanced.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_train_enhanced.npy'), y_train)
    np.save(os.path.join(save_dir, 'y_test_enhanced.npy'), y_test)
    
    
    # Save scaler for later use
    import pickle
    with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    import pickle

    # Save full label encoder
    with open(os.path.join(save_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)

    # Optional: still save classes separately
    np.save(os.path.join(save_dir, 'label_classes.npy'), label_encoder.classes_)    
    
    print(f"\n✓ Enhanced data saved to: {save_dir}")
    
    # Metadata
    metadata = {
        'num_train': len(X_train),
        'num_test': len(X_test),
        'num_classes': len(label_encoder.classes_),
        'classes': label_encoder.classes_.tolist(),
        'input_shape': X_train.shape[1:],
        'augmentation_used': use_augmentation,
        'augmentation_factor': augmentation_factor if use_augmentation else 1,
        'advanced_features': include_advanced_features,
        'num_features': X.shape[2]
    }
    
    pd.DataFrame([metadata]).to_csv(
        os.path.join(save_dir, 'metadata_enhanced.csv'),
        index=False
    )
    
    print("\n" + "="*60)
    print("ENHANCED PREPROCESSING COMPLETE!")
    print("="*60 + "\n")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'metadata': metadata
    }



def load_processed_data(save_dir=None):
    """
    Load processed data from directory.
    
    Args:
        save_dir (str): Directory where processed data is saved
        
    Returns:
        dict: Dictionary containing processed data
    """
    if save_dir is None:
        save_dir = os.path.abspath(PROCESSED_DATA_DIR)
        
    print(f"Loading processed data from: {save_dir}")
    
    try:
        X_train = np.load(os.path.join(save_dir, 'X_train_enhanced.npy'))
        X_test = np.load(os.path.join(save_dir, 'X_test_enhanced.npy'))
        y_train = np.load(os.path.join(save_dir, 'y_train_enhanced.npy'))
        y_test = np.load(os.path.join(save_dir, 'y_test_enhanced.npy'))
        label_classes = np.load(os.path.join(save_dir, 'label_classes.npy'))
        
        # Load scaler
        import pickle
        with open(os.path.join(save_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
            
        print("✓ Data loaded successfully")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test: {X_test.shape}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'label_classes': label_classes,
            'scaler': scaler
        }
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None


if __name__ == "__main__":
    print("Starting enhanced preprocessing...")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Save directory: {PROCESSED_DATA_DIR}")
    
    # Preprocess with augmentation and advanced features
    data = preprocess_dataset_enhanced(
        use_augmentation=True,
        augmentation_factor=AUGMENTATION_FACTOR,
        include_advanced_features=True
    )
    
    print("\nEnhanced data shapes:")
    print(f"X_train: {data['X_train'].shape}")
    print(f"X_test: {data['X_test'].shape}")
    print(f"Features per timestep: {data['X_train'].shape[2]}")

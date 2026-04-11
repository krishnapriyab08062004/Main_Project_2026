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


def extract_features_enhanced(audio_path, audio_signal=None, max_len=128, n_mfcc=40, sr=22050,
                              include_advanced=True):
    """
    Extract enhanced audio features.
    
    Args:
        audio_path (str): Path to audio file (can be None if audio_signal is provided)
        audio_signal (np.array): Pre-loaded audio signal
        max_len (int): Maximum number of time steps
        n_mfcc (int): Number of MFCCs
        sr (int): Sample rate
        include_advanced (bool): Whether to include chroma, mel, spectral contrast
    """
    try:
        # Load audio if not provided
        if audio_signal is not None:
            audio = audio_signal
        else:
            audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        
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
    Enhanced preprocessing with support for pre-split folders (train/validation/test).
    
    If data_dir contains 'train', 'validation', and 'test' subdirectories, it loads them
    directly without a random split. This preserves speaker-independent splits if they exist.
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
        # Avoid circular import, check if it's already defined
        try:
            from config import USE_DATA_AUGMENTATION
            use_augmentation = USE_DATA_AUGMENTATION and AUGMENTATION_AVAILABLE
        except:
            use_augmentation = False
    if augmentation_factor is None:
        augmentation_factor = AUGMENTATION_FACTOR
    
    print("\n" + "="*60)
    print("ENHANCED PREPROCESSING WITH AUGMENTATION")
    print("="*60 + "\n")
    print(f"Dataset location: {data_dir}")
    print(f"Augmentation: {'ENABLED' if use_augmentation else 'DISABLED'}")
    print(f"Advanced Features: {'ENABLED' if include_advanced_features else 'DISABLED'}")
    
    # Check for pre-split folders
    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'validation')
    test_path = os.path.join(data_dir, 'test')
    
    use_folders = all(os.path.exists(p) for p in [train_path, val_path, test_path])
    
    if use_folders:
        print("Detected physical split folders (train/validation/test). Loading directly...")
        
        def process_folder(folder_path, is_train=False):
            X_f = []
            y_f = []
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]
            print(f"  - Processing {len(files)} files in {os.path.basename(folder_path)}...")
            
            for i, f_path in enumerate(files):
                if i % 100 == 0:
                    print(f"    - Processing: {i+1}/{len(files)}...", end='\r')
                
                parts = os.path.basename(f_path).split('-')
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    emotion = EMOTION_MAP.get(emotion_code, 'unknown')
                    
                    # Original features
                    feat = extract_features_enhanced(f_path, max_len=max_len, n_mfcc=n_mfcc, 
                                                   include_advanced=include_advanced_features)
                    if feat is not None:
                        X_f.append(feat)
                        y_f.append(emotion)
                        
                        # Augmentation (only for Train)
                        if is_train and use_augmentation and AUGMENTATION_AVAILABLE:
                            try:
                                audio, sr = librosa.load(f_path, sr=SAMPLE_RATE, mono=True)
                                for _ in range(augmentation_factor - 1):
                                    aug_audio = augment_audio(audio, sr=sr)
                                    aug_feat = extract_features_enhanced(None, audio_signal=aug_audio, sr=sr,
                                                                        max_len=max_len, n_mfcc=n_mfcc,
                                                                        include_advanced=include_advanced_features)
                                    if aug_feat is not None:
                                        X_f.append(aug_feat)
                                        y_f.append(emotion)
                            except Exception as e:
                                print(f"Augmentation error for {f_path}: {e}")
            return np.array(X_f), np.array(y_f)

        X_train, y_train_labels = process_folder(train_path, is_train=True)
        X_val, y_val_labels = process_folder(val_path, is_train=False)
        X_test, y_test_labels = process_folder(test_path, is_train=False)
        
        # Label encoding
        label_encoder = LabelEncoder()
        label_encoder.fit(list(EMOTION_MAP.values()))
        y_train = label_encoder.transform(y_train_labels)
        y_val = label_encoder.transform(y_val_labels)
        y_test = label_encoder.transform(y_test_labels)
        
    else:
        # Fallback to dynamic split
        print("Pre-split folders NOT found. Falling back to global scan + stratified split...")
        audio_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.wav'):
                    audio_files.append(os.path.join(root, file))
        
        if not audio_files:
            print(f"Error: No audio files found in {data_dir}")
            return None
            
        all_data = []
        for f in audio_files:
            parts = os.path.basename(f).split('-')
            if len(parts) >= 3:
                emotion_code = parts[2]
                emotion = EMOTION_MAP.get(emotion_code, 'unknown')
                all_data.append({'filepath': f, 'emotion': emotion})
        
        df = pd.DataFrame(all_data)
        
        print(f"Extracting features from {len(df)} original files...")
        features = []
        labels = []
        filepaths = []
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"  - Processing: {idx+1}/{len(df)} files...", end='\r')
            
            feature = extract_features_enhanced(row['filepath'], max_len=max_len, n_mfcc=n_mfcc, include_advanced=include_advanced_features)
            if feature is not None:
                features.append(feature)
                labels.append(row['emotion'])
                filepaths.append(row['filepath'])
                
        X_orig = np.array(features)
        y_orig = np.array(labels)
        
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_orig)
        
        indices = np.arange(len(X_orig))
        idx_train, idx_temp = train_test_split(indices, test_size=0.2, stratify=y_encoded, random_state=random_state)
        idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, stratify=y_encoded[idx_temp], random_state=random_state)
        
        def get_augmented_features(indices_list, is_training=False):
            X_res = []
            y_res = []
            for i in indices_list:
                f_path = filepaths[i]
                X_res.append(features[i])
                y_res.append(y_encoded[i])
                if is_training and use_augmentation and AUGMENTATION_AVAILABLE:
                    audio, sr = librosa.load(f_path, sr=SAMPLE_RATE, mono=True)
                    for _ in range(augmentation_factor - 1):
                        aug_audio = augment_audio(audio, sr=sr)
                        aug_feat = extract_features_enhanced(None, audio_signal=aug_audio, sr=sr, max_len=max_len, n_mfcc=n_mfcc, include_advanced=include_advanced_features)
                        if aug_feat is not None:
                            X_res.append(aug_feat)
                            y_res.append(y_encoded[i])
            return np.array(X_res), np.array(y_res)
            
        X_train, y_train = get_augmented_features(idx_train, is_training=True)
        X_val, y_val = get_augmented_features(idx_val, is_training=False)
        X_test, y_test = get_augmented_features(idx_test, is_training=False)
    
    # 6. Standardization (Mean=0, Std=1 based on Train set)
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_reshaped)
    
    X_train = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # FINAL STEPS: Reporting, saving, and returning
    print(f"\n✓ Label encoding:")
    for i, emotion in enumerate(label_encoder.classes_):
        # We need a way to count samples in the encoded labels
        # Just use the full y collection for display
        count = np.sum(np.concatenate([y_train, y_val, y_test]) == i)
        print(f"  {i}: {emotion} ({count} samples)")

    print(f"\n✓ Final split stats:")
    print(f"  Training samples  : {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Testing samples   : {len(X_test)}")
    
    # Save processed data
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'X_train_enhanced.npy'), X_train)
    np.save(os.path.join(save_dir, 'X_val_enhanced.npy'), X_val)
    np.save(os.path.join(save_dir, 'X_test_enhanced.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_train_enhanced.npy'), y_train)
    np.save(os.path.join(save_dir, 'y_val_enhanced.npy'), y_val)
    np.save(os.path.join(save_dir, 'y_test_enhanced.npy'), y_test)
    
    import pickle
    with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(save_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    np.save(os.path.join(save_dir, 'label_classes.npy'), label_encoder.classes_)    
    
    print(f"\n✓ Enhanced data saved to: {save_dir}")
    
    metadata = {
        'num_train': len(X_train),
        'num_val': len(X_val),
        'num_test': len(X_test),
        'num_classes': len(label_encoder.classes_),
        'classes': label_encoder.classes_.tolist(),
        'input_shape': X_train.shape[1:],
        'augmentation_used': use_augmentation,
        'augmentation_factor': augmentation_factor if use_augmentation else 1,
        'advanced_features': include_advanced_features,
        'num_features': X_train.shape[2]
    }
    pd.DataFrame([metadata]).to_csv(os.path.join(save_dir, 'metadata_enhanced.csv'), index=False)
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
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
        X_val = np.load(os.path.join(save_dir, 'X_val_enhanced.npy'))
        X_test = np.load(os.path.join(save_dir, 'X_test_enhanced.npy'))
        y_train = np.load(os.path.join(save_dir, 'y_train_enhanced.npy'))
        y_val = np.load(os.path.join(save_dir, 'y_val_enhanced.npy'))
        y_test = np.load(os.path.join(save_dir, 'y_test_enhanced.npy'))
        label_classes = np.load(os.path.join(save_dir, 'label_classes.npy'))
        
        # Load scaler
        import pickle
        with open(os.path.join(save_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
            
        print("✓ Data loaded successfully")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_val:   {X_val.shape}")
        print(f"  X_test:  {X_test.shape}")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
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
    if data and 'X_train' in data:
        print(f"X_train: {data['X_train'].shape}")
    if data and 'X_val' in data:
        print(f"X_val:   {data['X_val'].shape}")
    if data and 'X_test' in data:
        print(f"X_test:  {data['X_test'].shape}")
    if data and 'X_train' in data and len(data['X_train'].shape) > 2:
        print(f"Features per timestep: {data['X_train'].shape[2]}")

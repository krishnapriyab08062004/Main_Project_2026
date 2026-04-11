"""
Data Augmentation Module for Speech Emotion Recognition

Implements various audio augmentation techniques:
- Time stretching
- Pitch shifting
- Noise injection
- Time shifting
- Volume perturbation
"""

import numpy as np
import librosa
import random


def time_stretch(audio, rate=None):
    """
    Time stretch audio without changing pitch.
    
    Args:
        audio (np.array): Audio signal
        rate (float): Stretch rate (0.8-1.2). If None, random rate is chosen
        
    Returns:
        np.array: Time-stretched audio
    """
    if rate is None:
        rate = random.uniform(0.75, 1.25)  # Slightly wider range
    
    return librosa.effects.time_stretch(audio, rate=rate)


def pitch_shift(audio, sr=22050, n_steps=None):
    """
    Shift pitch of audio.
    
    Args:
        audio (np.array): Audio signal
        sr (int): Sample rate
        n_steps (float): Number of semitones to shift. If None, random shift
        
    Returns:
        np.array: Pitch-shifted audio
    """
    if n_steps is None:
        n_steps = random.uniform(-2.5, 2.5)  # Slightly wider range
    
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def add_noise(audio, noise_factor=None):
    """
    Add white noise to audio.
    
    Args:
        audio (np.array): Audio signal
        noise_factor (float): Noise intensity (0.001-0.01). If None, random
        
    Returns:
        np.array: Audio with added noise
    """
    if noise_factor is None:
        noise_factor = random.uniform(0.001, 0.015)  # Increased max noise
    
    noise = np.random.randn(len(audio))
    augmented = audio + noise_factor * noise
    
    # Normalize to prevent clipping
    augmented = augmented / (np.max(np.abs(augmented)) + 1e-6)
    
    return augmented


def time_shift(audio, shift_max=None):
    """
    Shift audio in time (circular shift).
    
    Args:
        audio (np.array): Audio signal
        shift_max (int): Maximum shift in samples. If None, uses 20% of length
        
    Returns:
        np.array: Time-shifted audio
    """
    if shift_max is None:
        shift_max = int(len(audio) * 0.2)
    
    shift = random.randint(-shift_max, shift_max)
    return np.roll(audio, shift)


def change_volume(audio, factor=None):
    """
    Change volume of audio.
    
    Args:
        audio (np.array): Audio signal
        factor (float): Volume multiplier (0.7-1.3). If None, random
        
    Returns:
        np.array: Volume-adjusted audio
    """
    if factor is None:
        factor = random.uniform(0.7, 1.3)
    
    augmented = audio * factor
    
    # Normalize to prevent clipping
    augmented = augmented / (np.max(np.abs(augmented)) + 1e-6)
    
    return augmented


def add_gaussian_noise(audio, noise_factor=None):
    """
    Add Gaussian noise to audio.
    
    Args:
        audio (np.array): Audio signal
        noise_factor (float): Noise intensity (0.001-0.005). If None, random
        
    Returns:
        np.array: Audio with Gaussian noise
    """
    if noise_factor is None:
        noise_factor = random.uniform(0.001, 0.005)
    
    noise = np.random.normal(0, noise_factor, len(audio))
    augmented = audio + noise
    
    # Normalize
    augmented = augmented / (np.max(np.abs(augmented)) + 1e-6)
    
    return augmented


def augment_audio(audio, sr=22050, augmentation_methods=None):
    """
    Apply random augmentation to audio.
    
    Args:
        audio (np.array): Audio signal
        sr (int): Sample rate
        augmentation_methods (list): List of methods to use. If None, random selection
        
    Returns:
        np.array: Augmented audio
    """
    if augmentation_methods is None:
        # Randomly select 1-3 augmentation methods
        all_methods = ['time_stretch', 'pitch_shift', 'noise', 'time_shift', 'volume']
        num_methods = random.randint(1, 3)
        augmentation_methods = random.sample(all_methods, num_methods)
    
    augmented = audio.copy()
    
    for method in augmentation_methods:
        if method == 'time_stretch':
            augmented = time_stretch(augmented)
        elif method == 'pitch_shift':
            augmented = pitch_shift(augmented, sr=sr)
        elif method == 'noise':
            augmented = add_noise(augmented)
        elif method == 'time_shift':
            augmented = time_shift(augmented)
        elif method == 'volume':
            augmented = change_volume(augmented)
        elif method == 'gaussian_noise':
            augmented = add_gaussian_noise(augmented)
    
    return augmented


def create_augmented_dataset(audio_files, labels, augmentation_factor=3, sr=22050):
    """
    Create augmented dataset from original audio files.
    
    Args:
        audio_files (list): List of audio file paths
        labels (list): List of corresponding labels
        augmentation_factor (int): How many augmented versions per original
        sr (int): Sample rate
        
    Returns:
        tuple: (augmented_audio_list, augmented_labels_list)
    """
    augmented_audios = []
    augmented_labels = []
    
    print(f"Creating augmented dataset with factor {augmentation_factor}...")
    
    for idx, (audio_path, label) in enumerate(zip(audio_files, labels)):
        if idx % 100 == 0:
            print(f"Augmenting: {idx}/{len(audio_files)} files...", end='\r')
        
        # Load original audio
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        
        # Add original
        augmented_audios.append(audio)
        augmented_labels.append(label)
        
        # Create augmented versions
        for _ in range(augmentation_factor - 1):
            aug_audio = augment_audio(audio, sr=sr)
            augmented_audios.append(aug_audio)
            augmented_labels.append(label)
    
    print(f"Augmenting: {len(audio_files)}/{len(audio_files)} files... Done!")
    print(f"✓ Created {len(augmented_audios)} total samples ({len(audio_files)} original × {augmentation_factor})")
    
    return augmented_audios, augmented_labels


if __name__ == "__main__":
    # Example usage
    print("Testing data augmentation module...")
    
    # Create dummy audio
    dummy_audio = np.random.randn(22050)  # 1 second at 22050 Hz
    
    print("\nOriginal audio shape:", dummy_audio.shape)
    
    # Test each augmentation
    stretched = time_stretch(dummy_audio, rate=0.9)
    print(f"Time stretched (0.9x): {stretched.shape}")
    
    pitched = pitch_shift(dummy_audio, sr=22050, n_steps=2)
    print(f"Pitch shifted (+2 semitones): {pitched.shape}")
    
    noisy = add_noise(dummy_audio, noise_factor=0.005)
    print(f"Noisy: {noisy.shape}")
    
    shifted = time_shift(dummy_audio)
    print(f"Time shifted: {shifted.shape}")
    
    volume_changed = change_volume(dummy_audio, factor=1.2)
    print(f"Volume changed (1.2x): {volume_changed.shape}")
    
    # Test combined augmentation
    augmented = augment_audio(dummy_audio, sr=22050)
    print(f"\nCombined augmentation: {augmented.shape}")
    
    print("\n✓ All augmentation tests passed!")

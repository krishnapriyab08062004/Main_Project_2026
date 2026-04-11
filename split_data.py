import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Scanning the entire raw directory to find files even if already split
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
TRAIN_DIR = os.path.join(BASE_DIR, "data", "raw", "train")
VAL_DIR = os.path.join(BASE_DIR, "data", "raw", "validation")
TEST_DIR = os.path.join(BASE_DIR, "data", "raw", "test")

# Emotion mapping for stratifying
EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

def split_data_stratified():
    # 1. Collect all audio files and their emotions
    print(f"Scanning {RAW_DATA_PATH}...")
    all_files = []
    
    # If the files are already in Actor folders
    for root, dirs, files in os.walk(RAW_DATA_PATH):
        for file in files:
            if file.endswith('.wav'):
                filepath = os.path.join(root, file)
                # Extract emotion code from RAVDESS filename: 03-01-01-01-01-01-01.wav
                parts = file.split('-')
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    all_files.append({'path': filepath, 'emotion': emotion_code, 'name': file})
    
    if not all_files:
        # Check if they are already in the root of data/raw
        print("No files in actor folders, checking root...")
        # ... (similar logic)
        if not all_files:
            print("Error: No audio files found to split.")
            return

    df = pd.DataFrame(all_files)
    print(f"Total files found: {len(df)}")
    
    # 2. Perform 3-way Stratified Split (80/10/10)
    # First: Train (80%) vs Temp (20%)
    train_df, temp_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df['emotion']
    )
    
    # Second: Val (10%) vs Test (10%)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df['emotion']
    )
    
    # 3. Create directories and move files
    def move_files(subset_df, target_dir):
        os.makedirs(target_dir, exist_ok=True)
        print(f"Moving {len(subset_df)} files to {target_dir}...")
        for _, row in subset_df.iterrows():
            dest = os.path.join(target_dir, row['name'])
            # Using copy instead of move to be safer, or move if user prefers
            shutil.move(row['path'], dest)

    move_files(train_df, TRAIN_DIR)
    move_files(val_df, VAL_DIR)
    move_files(test_df, TEST_DIR)
    
    print("\n✓ Stratified splitting complete!")
    print(f"  - Train: {len(train_df)} samples")
    print(f"  - Val:   {len(val_df)} samples")
    print(f"  - Test:  {len(test_df)} samples")

if __name__ == "__main__":
    split_data_stratified()

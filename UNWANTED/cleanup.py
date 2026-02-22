"""
Automated Cleanup Script for SER Project
Removes unnecessary files while keeping everything needed for 80-95% accuracy
"""

import os
import shutil

print("="*80)
print("SER PROJECT CLEANUP")
print("Removing unnecessary files...")
print("="*80)

# Base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Files to remove (redundant/optional files)
files_to_remove = [
    # Redundant scripts
    "run_improved_pipeline.py",
    "run_improved.bat",
    "src/train.py",  # Old version
    "src/models.py",  # Old version
    "src/data_preprocessing.py",  # Old version
    
    # Streamlit/Frontend (if not using UI)
    "streamlit_app.py",
    "run_streamlit.bat",
    "fix_streamlit.bat",
    "fix_streamlit_errors.py",
    "STREAMLIT_README.md",
    "FRONTEND_SUMMARY.md",
    
    # Database files (if not using database)
    "db_config.py",
    "db_config.example.py",
    "setup_database.py",
    "src/database.py",
    
    # Utility files (optional)
    "check_setup.py",
    "main.py",
    "quickstart.bat",
    "QUICKSTART.md",
    "PROJECT_FILES.md",
    "src/app.py",
    "src/inspect_data.py", 
    "src/preprocessed_data.py",
    "src/realtime_predictor.py",  # Unless you want real-time prediction
]

# Directories to remove
dirs_to_remove = [
    "__pycache__",
    "src/__pycache__",
    "notebooks",  # Unless you're using them
    "report",  # Unless you need it
]

removed_files = []
removed_dirs = []
errors = []

# Remove files
for file_path in files_to_remove:
    full_path = os.path.join(base_dir, file_path)
    if os.path.exists(full_path):
        try:
            os.remove(full_path)
            removed_files.append(file_path)
            print(f"[V] Removed: {file_path}")
        except Exception as e:
            errors.append(f"Error removing {file_path}: {e}")
            print(f"[X] Error: {file_path} - {e}")
    else:
        print(f"  Skipped: {file_path} (not found)")

# Remove directories
for dir_path in dirs_to_remove:
    full_path = os.path.join(base_dir, dir_path)
    if os.path.exists(full_path) and os.path.isdir(full_path):
        try:
            shutil.rmtree(full_path)
            removed_dirs.append(dir_path)
            print(f"[V] Removed directory: {dir_path}")
        except Exception as e:
            errors.append(f"Error removing {dir_path}: {e}")
            print(f"[X] Error: {dir_path} - {e}")
    else:
        print(f"  Skipped: {dir_path}/ (not found)")

# Summary
print("\n" + "="*80)
print("CLEANUP SUMMARY")
print("="*80)
print(f"Files removed: {len(removed_files)}")
print(f"Directories removed: {len(removed_dirs)}")
print(f"Errors: {len(errors)}")

if errors:
    print("\nErrors encountered:")
    for error in errors:
        print(f"  - {error}")

print("\n" + "="*80)
print("ESSENTIAL FILES KEPT")
print("="*80)
print("""
[V] Core Training Files:
   - src/data_augmentation.py
   - src/data_preprocessing_enhanced.py
   - src/models_improved.py
   - src/train_improved.py
   - train_all_models.py

[V] Configuration:
   - src/config.py

[V] Evaluation & Utilities:
   - src/evaluate.py
   - src/alerts.py
   - src/emotion_tracker.py

[V] Documentation:
   - START_HERE.md
   - IMPROVEMENTS_README.md
   - README.md
   - CLEANUP_GUIDE.md

[V] Data & Results:
   - data/
   - models/
   - results/
""")

print("\n" + "="*80)
print("CLEANUP COMPLETE!")
print("="*80)
print("\nYour project is now clean and ready for training!")
print("\nNext step:")
print("  py src/data_preprocessing_enhanced.py")
print("  py train_all_models.py")
print("\n" + "="*80)

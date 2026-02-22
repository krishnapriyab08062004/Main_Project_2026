# Speech Emotion Recognition System

A research-level Speech Emotion Recognition (SER) system using deep learning to classify emotions from speech audio. This project compares CNN-DNN and CNN-LSTM architectures on the RAVDESS dataset and implements a threshold-based emotion notification system.

## 🎯 Project Overview

This system analyzes speech audio to detect emotions including:
- **Neutral** - Baseline emotional state
- **Calm** - Relaxed, peaceful state
- **Happy** - Joyful, positive state
- **Sad** - Melancholic, sorrowful state
- **Angry** - Frustrated, aggressive state
- **Fearful** - Anxious, scared state
- **Disgust** - Repulsed, negative reaction
- **Surprised** - Shocked, unexpected reaction

## 🌟 Key Features

### 1. Dual Model Architecture Comparison
- **CNN-DNN**: Convolutional layers + Dense layers for spatial feature extraction
- **CNN-LSTM**: Convolutional layers + LSTM layers for temporal sequence modeling

### 2. Comprehensive Feature Extraction
- **MFCC** (Mel-Frequency Cepstral Coefficients): 40 coefficients
- **ZCR** (Zero Crossing Rate): Speech/silence detection
- **RMS Energy**: Volume and intensity information

### 3. Advanced Training Pipeline
- ModelCheckpoint: Save best performing model
- EarlyStopping: Prevent overfitting
- ReduceLROnPlateau: Adaptive learning rate
- Training/validation visualization

### 4. Detailed Evaluation Metrics
- Accuracy
- Precision (macro & weighted)
- Recall (macro & weighted)
- F1-Score (macro & weighted)
- Confusion matrices
- Per-class performance reports

### 5. Threshold-Based Notification System
- Real-time emotion alerts when confidence exceeds thresholds
- CSV and SQLite storage options
- Emotion history tracking
- Statistical analysis

### 6. Real-Time Detection (Optional)
- Live microphone input
- Continuous emotion monitoring
- Rolling prediction history
- Prolonged emotion detection

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download Project
```bash
cd C:\Users\HP\.gemini\antigravity\scratch\speech-emotion-recognition
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download RAVDESS Dataset
Download the RAVDESS dataset from one of these sources:
- [Official Source](https://zenodo.org/record/1188976)
- [Kaggle](https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio)

Extract to a known location (already detected at `C:\Users\HP\Downloads\audio_speech_actors_01-24`)

## 🚀 Usage

### Quick Start (Full Pipeline)
```bash
python main.py
```

This will:
1. Preprocess the RAVDESS dataset
2. Train both CNN-DNN and CNN-LSTM models
3. Evaluate and compare models
4. Select the best model
5. Demonstrate emotion tracking
6. Display real-world use cases

### Command-Line Options

#### Quick Test (10 epochs)
```bash
python main.py --epochs 10
```

#### Use Existing Preprocessed Data
```bash
python main.py --skip-preprocessing
```

#### Evaluate Existing Models Only
```bash
python main.py --skip-preprocessing --skip-training
```

#### Custom Dataset Path
```bash
python main.py --dataset-path "D:\path\to\RAVDESS"
```

#### Custom Training Configuration
```bash
python main.py --epochs 100 --batch-size 64
```

### Individual Module Usage

#### 1. Data Preprocessing Only
```bash
cd src
python data_preprocessing.py
```

#### 2. View Model Architectures
```bash
cd src
python models.py
```

#### 3. Train Models
```bash
cd src
python train.py
```

#### 4. Evaluate Models
```bash
cd src
python evaluate.py
```

#### 5. Test Emotion Tracking
```bash
cd src
python emotion_tracker.py
```

#### 6. Real-Time Detection
```bash
python src/realtime_predictor.py models/saved_models/cnn_lstm_final.keras
```

## 📊 Model Architectures

### CNN-DNN Architecture
```
Input (128, 42)
    ↓
Conv1D (64 filters, kernel=5)
    ↓
MaxPooling1D (pool_size=2)
    ↓
Conv1D (128 filters, kernel=5)
    ↓
MaxPooling1D (pool_size=2)
    ↓
Dropout (0.3)
    ↓
Flatten
    ↓
Dense (256, relu)
    ↓
Dropout (0.4)
    ↓
Dense (128, relu)
    ↓
Dense (8, softmax)
```

### CNN-LSTM Architecture
```
Input (128, 42)
    ↓
Conv1D (64 filters, kernel=5)
    ↓
MaxPooling1D (pool_size=2)
    ↓
Conv1D (128 filters, kernel=5)
    ↓
MaxPooling1D (pool_size=2)
    ↓
Dropout (0.3)
    ↓
LSTM (128, return_sequences=True)
    ↓
Dropout (0.3)
    ↓
LSTM (64, return_sequences=False)
    ↓
Dense (128, relu)
    ↓
Dropout (0.4)
    ↓
Dense (8, softmax)
```

## 🎯 Emotion Thresholds

Default confidence thresholds for alerts:

| Emotion  | Threshold | Use Case                          |
|----------|-----------|-----------------------------------|
| Angry    | 75%       | Customer service escalation       |
| Sad      | 70%       | Mental health intervention        |
| Fearful  | 65%       | Emergency prioritization          |
| Disgust  | 70%       | Product quality review            |

Thresholds can be customized in `src/emotion_tracker.py`

## 🌍 Real-World Use Cases

### 1. Mental Health & Wellness
- Therapy session monitoring
- Suicide prevention systems
- Stress management applications
- PTSD treatment support

### 2. Customer Service
- Call routing optimization
- Agent performance monitoring
- Customer satisfaction analysis
- Quality assurance

### 3. Automotive & Safety
- Driver emotional state monitoring
- Road rage detection
- Emergency call assessment
- In-car adaptive assistants

### 4. Education
- Student engagement tracking
- Virtual tutoring adaptation
- Special education support
- Assessment stress monitoring

### 5. Human Resources
- Employee wellness programs
- Interview analysis
- Workplace conflict detection
- Leadership training

### 6. Entertainment
- Adaptive gaming experiences
- Virtual companion systems
- Content recommendation
- Social platform safety

## 📁 Project Structure

```
speech-emotion-recognition/
├── data/
│   ├── raw/                    # RAVDESS dataset location
│   └── processed/              # Preprocessed features (.npy files)
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_test.npy
│       ├── label_classes.npy
│       └── metadata.csv
├── models/
│   ├── saved_models/           # Trained models (.keras files)
│   │   ├── cnn_dnn_final.keras
│   │   └── cnn_lstm_final.keras
│   └── checkpoints/            # Best model checkpoints
│       ├── cnn_dnn_best.keras
│       └── cnn_lstm_best.keras
├── results/
│   ├── plots/                  # Visualizations
│   │   ├── CNN_DNN_training_history.png
│   │   ├── CNN_LSTM_training_history.png
│   │   ├── CNN_DNN_confusion_matrix.png
│   │   ├── CNN_LSTM_confusion_matrix.png
│   │   └── model_comparison.png
│   ├── predictions/            # Emotion history
│   │   └── emotion_history.csv
│   ├── model_comparison.csv
│   └── best_model_selection.txt
├── src/
│   ├── data_preprocessing.py   # Data loading and feature extraction
│   ├── models.py               # Model architectures
│   ├── train.py                # Training pipeline
│   ├── evaluate.py             # Evaluation and comparison
│   ├── emotion_tracker.py      # Threshold-based notifications
│   └── realtime_predictor.py   # Live microphone detection
├── notebooks/                  # (Optional) Jupyter notebooks
├── requirements.txt            # Python dependencies
├── main.py                     # Main execution script
└── README.md                   # This file
```

## 📈 Expected Performance

Based on RAVDESS dataset benchmarks:

- **Target Accuracy**: 65-75%
- **F1-Score**: >0.60
- **Best Model**: CNN-LSTM (expected to outperform CNN-DNN due to temporal modeling)

Actual performance may vary based on:
- Training epochs
- Hyperparameter tuning
- Data augmentation
- Ensemble methods

## 🔬 Research & Publication

This implementation is suitable for:

### Academic Projects
- Final year undergraduate projects
- Master's thesis research
- Course assignments in ML/AI

### Publications
- Conference papers (e.g., INTERSPEECH, ICASSP)
- Journal articles on affective computing
- Workshop presentations

### Citation Format
```
@misc{speech_emotion_recognition_2026,
  title={Speech Emotion Recognition using CNN-DNN and CNN-LSTM on RAVDESS Dataset},
  author={Your Name},
  year={2026},
  note={Deep learning-based emotion classification with threshold-based notifications}
}
```

## 🛠️ Troubleshooting

### Issue: "Dataset not found"
**Solution**: Verify the dataset path in `main.py` or use the `--dataset-path` argument

### Issue: "Out of memory during training"
**Solution**: Reduce batch size using `--batch-size 16`

### Issue: "sounddevice not found" (for real-time)
**Solution**: Install with `pip install sounddevice`

### Issue: "Model training is slow"
**Solution**: 
- Use GPU acceleration (install `tensorflow-gpu`)
- Reduce epochs for testing: `--epochs 10`
- Use smaller batch size

### Issue: "Low accuracy results"
**Solution**:
- Increase training epochs (try 100-150)
- Verify dataset is complete
- Check feature extraction parameters
- Consider data augmentation

## 📚 Additional Resources

### RAVDESS Dataset
- [Official Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391)
- Audio format: 16-bit WAV, 48kHz
- 24 actors (12 male, 12 female)
- 1,440 total files

### Deep Learning References
- [Keras Documentation](https://keras.io/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Librosa Audio Processing](https://librosa.org/doc/latest/index.html)

### Related Papers
- "Speech Emotion Recognition: Emotional Models, Databases, Features, Preprocessing Methods, Supporting Modalities, and Classifiers"
- "Deep Learning for Speech Emotion Recognition: A Survey"
- "Attention-based LSTM for Speech Emotion Recognition"

## 🤝 Contributing

Suggestions for improvement:
1. Data augmentation (pitch shifting, time stretching)
2. Transfer learning from pre-trained models
3. Multi-modal fusion (audio + text + visual)
4. Attention mechanisms
5. Real-time performance optimization

## 📝 License

This project is for educational and research purposes. The RAVDESS dataset has its own license terms.

## 👥 Authors

Created as a comprehensive Machine Learning research project for Speech Emotion Recognition.

## 🙏 Acknowledgments

- RAVDESS dataset creators: Livingstone & Russo (2018)
- TensorFlow and Keras teams
- Librosa developers
- Open-source ML community

## 📧 Contact

For questions, issues, or collaboration opportunities, please open an issue in the project repository.

---

**Built with ❤️ for emotion-aware AI systems**

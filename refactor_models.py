import re
import os

def refactor_evaluate():
    path = r'src/evaluate.py'
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    new_dict = '''models_to_evaluate = {
    # Basic Models
    "CNN_DNN_v1": "models/checkpoints/cnn_dnn_v1_best.keras",
    "CNN_LSTM_v1": "models/checkpoints/cnn_lstm_v1_best.keras",
    # Improved Models
    "CNN_DNN_Improved": "models/saved_models/cnn_dnn_improved_final.keras",
    "CNN_LSTM_Attention": "models/saved_models/cnn_lstm_attention_final.keras"
}'''

    content = re.sub(r'models_to_evaluate\s*=\s*\{.*?\}(?=\n)', new_dict, content, flags=re.DOTALL)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def refactor_models_improved():
    path = r'src/models_improved.py'
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Remove cnn_lstm_improved (using regex to find function defs till the next def or class block )
    content = re.sub(r'def build_cnn_lstm_improved\(.*?\n\n\n', '\n', content, flags=re.DOTALL)
    
    # Remove cnn_dnn_v2 and cnn_lstm_v2
    # They are between deep_conv_block and build_cnn_lstm_attention
    content = re.sub(r'def build_cnn_dnn_v2\(.*?(?=def build_cnn_lstm_v2)', '', content, flags=re.DOTALL)
    content = re.sub(r'def build_cnn_lstm_v2\(.*?(?=def build_cnn_lstm_attention)', '', content, flags=re.DOTALL)
    
    # Insert cnn_lstm_v1 before build_cnn_dnn_improved
    cnn_lstm_v1_code = '''
def build_cnn_lstm_v1(input_shape, num_classes=8, learning_rate=0.0001):
    """Basic CNN-LSTM v1 model."""
    model = models.Sequential(name='CNN_LSTM_v1')
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv1D(64, kernel_size=5, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

'''
    content = content.replace('def build_cnn_dnn_improved', cnn_lstm_v1_code + 'def build_cnn_dnn_improved')
    
    # Update get_model
    get_model_new = '''def get_model(model_type, input_shape, num_classes=8, learning_rate=None):
    model_type = model_type.lower()
    if model_type == 'cnn_dnn_v1':
        lr = learning_rate if learning_rate else 0.0001
        return build_cnn_dnn_v1(input_shape, num_classes, lr)
    elif model_type == 'cnn_lstm_v1':
        lr = learning_rate if learning_rate else 0.0001
        return build_cnn_lstm_v1(input_shape, num_classes, lr)
    elif model_type == 'cnn_dnn_improved':
        lr = learning_rate if learning_rate else 0.0005
        return build_cnn_dnn_improved(input_shape, num_classes, lr)
    elif model_type == 'cnn_lstm_attention':
        lr = learning_rate if learning_rate else 0.0003
        return build_cnn_lstm_attention(input_shape, num_classes, lr)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
'''
    content = re.sub(r'def get_model\(.*?raise ValueError.*?$', get_model_new, content, flags=re.DOTALL | re.MULTILINE)

    # Update __main__
    main_new = '''if __name__ == "__main__":
    INPUT_SHAPE = (128, 42)
    NUM_CLASSES = 8
    
    for m in ['cnn_dnn_v1', 'cnn_lstm_v1', 'cnn_dnn_improved', 'cnn_lstm_attention']:
        print(f"\\n{'#'*70}\\n# BUILDING {m.upper()}\\n{'#'*70}")
        model = get_model(m, INPUT_SHAPE, NUM_CLASSES)
        print_model_summary(model)

    print("\\nTesting models with dummy data...")
    dummy_input = np.random.randn(1, 128, 42).astype(np.float32)
    for m in ['cnn_dnn_v1', 'cnn_lstm_v1', 'cnn_dnn_improved', 'cnn_lstm_attention']:
        model = get_model(m, INPUT_SHAPE, NUM_CLASSES)
        output = model.predict(dummy_input, verbose=0)
        print(f"* {m} output shape: {output.shape}")
    print("\\n* All model tests passed!")
'''
    content = content.replace(content[content.find('if __name__ == "__main__":'):], main_new)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    refactor_evaluate()
    refactor_models_improved()
    print("Refactor models_improved and evaluate done.")

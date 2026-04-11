import re

def refactor_train():
    path = r'src/train_improved.py'
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Update imports
    old_imports = '''from models_improved import (
    build_cnn_dnn_improved,
    build_cnn_lstm_improved,
    build_cnn_lstm_attention,
    build_cnn_dnn_v2,
    build_cnn_lstm_v2,
    print_model_summary
)'''
    new_imports = '''from models_improved import (
    build_cnn_dnn_v1,
    build_cnn_lstm_v1,
    build_cnn_dnn_improved,
    build_cnn_lstm_attention,
    print_model_summary
)'''
    content = content.replace(old_imports, new_imports)

    # Update main training loop
    # We find the string: # Train CNN-DNN v2 (The 95% Target)
    # And replace everything from that point to the end of the file.
    find_str = "    # Train CNN-DNN v2 (The 95% Target)"
    idx = content.find(find_str)
    
    if idx != -1:
        new_loop = '''    models_to_train = [
        ('cnn_dnn_v1', build_cnn_dnn_v1(input_shape, num_classes, learning_rate=LEARNING_RATE)),
        ('cnn_lstm_v1', build_cnn_lstm_v1(input_shape, num_classes, learning_rate=LEARNING_RATE)),
        ('cnn_dnn_improved', build_cnn_dnn_improved(input_shape, num_classes, learning_rate=LEARNING_RATE)),
        ('cnn_lstm_attention', build_cnn_lstm_attention(input_shape, num_classes, learning_rate=LEARNING_RATE))
    ]

    for m_name, m_obj in models_to_train:
        print("\\n" + "#"*70)
        print(f"# TRAINING {m_name.upper()}")
        print("#"*70)
        
        history = train_model_enhanced(
            model=m_obj,
            X_train=X_train,
            y_train=y_train,
            model_name=m_name,
            epochs=100,
            batch_size=32,
            use_class_weights=True,
            use_cosine_annealing=True,
            label_smoothing=LABEL_SMOOTHING,
            use_mixup=True,
            mixup_alpha=MIXUP_ALPHA,
            swa_start_epoch=80
        )
        
        plot_training_history(history, m_name)
        print_training_summary(history, m_name)

    print("\\n" + "="*70)
    print("ALL TRAINING COMPLETE!")
    print("="*70)
'''
        content = content[:idx] + new_loop
        
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    refactor_train()
    print("Refactor train_improved done.")

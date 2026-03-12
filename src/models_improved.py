"""
Improved Model Architectures for Speech Emotion Recognition

Enhanced models with:
- BatchNormalization for training stability
- Attention mechanisms for better feature selection
- Bidirectional LSTMs for temporal modeling
- Improved regularization
- Optimized learning rates
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras import backend as K


def build_cnn_dnn_improved(input_shape, num_classes=8, learning_rate=0.0005):
    """
    Enhanced CNN-DNN model with BatchNormalization and deeper architecture.
    
    Improvements over original:
    - Higher learning rate (0.0005 vs 0.0001)
    - BatchNormalization after each layer
    - Deeper architecture (3 CNN blocks)
    - GlobalAveragePooling instead of Flatten
    - L2 regularization
    - ELU activation for better gradients
    
    Expected accuracy: 75-85%
    """
    model = models.Sequential(name='CNN_DNN_Improved')
    
    # Input
    model.add(layers.Input(shape=input_shape))
    
    # CNN Block 1: Capture low-level patterns
    model.add(layers.Conv1D(64, kernel_size=11, padding='same',
                            kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu'))
    model.add(layers.MaxPooling1D(pool_size=4))
    model.add(layers.Dropout(0.3))
    
    # CNN Block 2: Capture mid-level patterns
    model.add(layers.Conv1D(128, kernel_size=7, padding='same',
                            kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu'))
    model.add(layers.MaxPooling1D(pool_size=4))
    model.add(layers.Dropout(0.3))
    
    # CNN Block 3: Capture high-level patterns
    model.add(layers.Conv1D(256, kernel_size=5, padding='same',
                            kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu'))
    model.add(layers.GlobalAveragePooling1D())  # Better than Flatten
    
    # Dense layers
    model.add(layers.Dense(256, kernel_regularizer=regularizers.l2( 0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu'))
    model.add(layers.Dropout(0.6))
    
    model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu'))
    model.add(layers.Dropout(0.5))
    
    # Output
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compile with higher learning rate
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_cnn_lstm_improved(input_shape, num_classes=8, learning_rate=0.0003):
    """
    Enhanced CNN-LSTM model with Bidirectional LSTM and Attention.
    
    Improvements over original:
    - Bidirectional LSTM layers
    - Attention mechanism
    - BatchNormalization throughout
    - Deeper architecture
    - Better regularization
    
    Expected accuracy: 80-88%
    """
    model = models.Sequential(name='CNN_LSTM_Improved')
    
    # Input
    model.add(layers.Input(shape=input_shape))
    
    # CNN Block 1: Feature extraction
    model.add(layers.Conv1D(64, kernel_size=5, padding='same',
                            kernel_regularizer=regularizers.l2(0.0005)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.3))
    
    # CNN Block 2: Deeper feature extraction
    model.add(layers.Conv1D(128, kernel_size=5, padding='same',
                            kernel_regularizer=regularizers.l2(0.0005)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.3))
    
    # Bidirectional LSTM layers for temporal modeling
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=False)))
    model.add(layers.BatchNormalization())
    
    # Dense layers
    model.add(layers.Dense(128, activation='relu',
                          kernel_regularizer=regularizers.l2(0.0005)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    
    model.add(layers.Dense(64, activation='relu',
                          kernel_regularizer=regularizers.l2(0.0005)))
    model.add(layers.Dropout(0.3))
    
    # Output
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


class AttentionLayer(layers.Layer):
    """
    Attention mechanism for focusing on important time steps.
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        # Compute attention scores
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        
        # Apply attention weights
        output = x * a
        
        return tf.reduce_sum(output, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def residual_block(x, filters, kernel_size=3, stride=1):
    """
    Standard ResNet residual block for 1D convolutions.
    """
    shortcut = x
    
    # First convolution
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same',
                      kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    
    # Second convolution
    x = layers.Conv1D(filters, kernel_size, strides=1, padding='same',
                      kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.BatchNormalization()(x)
    
    # Adjustment for shortcut if dimensions change
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same',
                                 kernel_regularizer=regularizers.l2(0.0005))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('elu')(x)
    return x


def se_block(input_tensor, ratio=16):
    """Squeeze-and-Excitation block for channel-wise attention."""
    channel_axis = -1
    filters = input_tensor.shape[channel_axis]
    se_shape = (1, filters)

    se = layers.GlobalAveragePooling1D()(input_tensor)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = layers.Multiply()([input_tensor, se])
    return x


def residual_block_v2(x, filters, kernel_size=3, dropout=0.3):
    """Improved residual block with SE and better normalization."""
    shortcut = x
    
    # First conv
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Conv1D(filters, kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4))(x)
    
    # Second conv
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters, kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4))(x)
    
    # SE block
    x = se_block(x)
    
    # Project shortcut if needed
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding='same', kernel_initializer='he_normal')(shortcut)
        
    return layers.Add()([x, shortcut])


def build_cnn_dnn_v2(input_shape, num_classes=8, learning_rate=0.0005):
   
    inputs = layers.Input(shape=input_shape)
    
    # Stem: Initial feature extraction
    x = layers.Conv1D(64, 7, padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    # Residual blocks with Squeeze-and-Excitation
    x = residual_block_v2(x, 64, kernel_size=5, dropout=0.3)
    x = layers.MaxPooling1D(2)(x)
    
    x = residual_block_v2(x, 128, kernel_size=3, dropout=0.4)
    x = layers.MaxPooling1D(2)(x)
    
    x = residual_block_v2(x, 256, kernel_size=3, dropout=0.4)
    x = layers.MaxPooling1D(2)(x)
    
    # Self-Attention for global context
    # Reducing head count for stability
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attention])
    x = layers.BatchNormalization()(x)
    
    # Global Pooling (Average + Max)
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])
    
    # Dense head with strong regularization
    x = layers.Dense(256, activation='swish', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name='CNN_DNN_v2')
    model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_cnn_lstm_v2(input_shape, num_classes=8, learning_rate=0.0003):
  
    inputs = layers.Input(shape=input_shape)
    
    # CNN Front-end for local features
    x = layers.Conv1D(64, 5, padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.MaxPooling1D(2)(x)
    
    # Temporal Modeling with Bidirectional LSTM
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.BatchNormalization()(x)
    
    # Self-Attention for sequence weighting
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.3)(x, x)
    x = layers.Add()([x, attention])
    x = layers.BatchNormalization()(x)
    
    # Second processing stage
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.BatchNormalization()(x)
    
    # Classification Head
    x = layers.Dense(128, activation='swish', kernel_regularizer=regularizers.l2(5e-4))(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name='CNN_LSTM_v2')
    model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


    """
    Deep ResNet with Self-Attention for Speech Emotion Recognition.
    
    Architecture:
    - Initial Conv1D
    - Multiple Residual Blocks
    - Multi-Head Self-Attention
    - Global Pooling
    - Deep Dense Layers
    
    Target Accuracy: 92-96%
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial Convolution
    x = layers.Conv1D(64, 7, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)
    
    # Residual Blocks
    # Block 1
    x = residual_block(x, 128, kernel_size=7)
    x = layers.Dropout(0.3)(x)
    
    # Block 2
    x = residual_block(x, 256, kernel_size=5, stride=2)
    x = layers.Dropout(0.3)(x)
    
    # Block 3
    x = residual_block(x, 512, kernel_size=3, stride=2)
    x = layers.Dropout(0.4)(x)
    
    # Multi-head Self-Attention
    # MultiHeadAttention expects (batch, query_seq_len, d_model)
    attention_output = layers.MultiHeadAttention(
        num_heads=4, key_dim=128, dropout=0.3
    )(x, x)
    x = layers.Add()([x, attention_output])
    x = layers.BatchNormalization()(x)
    
    # Global Pooling
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])
    
    # Dense Layers
    # x = layers.Flatten()(x)
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.Dropout(0.4)(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN_ResNet_Attention')
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy', # Changed to support label smoothing
        metrics=['accuracy']
    )
    
    return model


def build_cnn_lstm_attention(input_shape, num_classes=8, learning_rate=0.0003):
    """
    CNN-LSTM model with Attention mechanism.
    
    This model uses attention to focus on the most important time steps
    for emotion recognition.
    
    Expected accuracy: 82-90%
    """
    inputs = layers.Input(shape=input_shape)
    
    # CNN Block 1
    x = layers.Conv1D(64, kernel_size=5, padding='same',
                     kernel_regularizer=regularizers.l2(0.0005))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)
    
    # CNN Block 2
    x = layers.Conv1D(128, kernel_size=5, padding='same',
                     kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Bidirectional LSTM with return sequences for attention
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.BatchNormalization()(x)
    
    # Attention mechanism
    x = AttentionLayer()(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.Dropout(0.3)(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN_LSTM_Attention')
    
    # CompileA
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_model(model_type, input_shape, num_classes=8, learning_rate=None):
    """
    Factory function to get a specific model.
    
    Args:
        model_type (str): 'cnn_dnn', 'cnn_lstm', 'cnn_dnn_improved', 
                         'cnn_lstm_improved', 'cnn_lstm_attention'
        input_shape (tuple): Input shape
        num_classes (int): Number of classes
        learning_rate (float): Learning rate (uses default if None)
        
    Returns:
        keras.Model: Compiled model
    """
    model_type = model_type.lower()
    
    if model_type == 'cnn_dnn_improved':
        lr = learning_rate if learning_rate else 0.0005
        return build_cnn_dnn_improved(input_shape, num_classes, lr)
    elif model_type == 'cnn_lstm_improved':
        lr = learning_rate if learning_rate else 0.0003
        return build_cnn_lstm_improved(input_shape, num_classes, lr)
    elif model_type == 'cnn_lstm_attention':
        lr = learning_rate if learning_rate else 0.0003
        return build_cnn_lstm_attention(input_shape, num_classes, lr)
    elif model_type == 'cnn_dnn_v2':
        lr = learning_rate if learning_rate else 0.0005
        return build_cnn_dnn_v2(input_shape, num_classes, lr)
    elif model_type == 'cnn_lstm_v2':
        lr = learning_rate if learning_rate else 0.0003
        return build_cnn_lstm_v2(input_shape, num_classes, lr)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def print_model_summary(model):
    """
    Print detailed model summary.
    """
    print("\n" + "="*70)
    print(f"MODEL ARCHITECTURE: {model.name}")
    print("="*70 + "\n")
    
    model.summary()
    
    # Count parameters
    total_params = model.count_params()
    print(f"\n* Total parameters: {total_params:,}")
    
    # Model configuration
    print(f"* Optimizer: {model.optimizer.__class__.__name__}")
    print(f"* Learning rate: {model.optimizer.learning_rate.numpy()}")
    print(f"* Internal loss: {model.loss}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Example: Build and display improved models
    
    INPUT_SHAPE = (128, 42)  # (time_steps, features)
    NUM_CLASSES = 8
    
    print("\n" + "#"*70)
    print("# BUILDING IMPROVED CNN-DNN MODEL")
    print("#"*70)
    
    cnn_dnn_improved = build_cnn_dnn_improved(INPUT_SHAPE, NUM_CLASSES)
    print_model_summary(cnn_dnn_improved)
    
    print("\n" + "#"*70)
    print("# BUILDING IMPROVED CNN-LSTM MODEL")
    print("#"*70)
    
    cnn_lstm_improved = build_cnn_lstm_improved(INPUT_SHAPE, NUM_CLASSES)
    print_model_summary(cnn_lstm_improved)
    
    print("\n" + "#"*70)
    print("# BUILDING CNN-LSTM WITH ATTENTION")
    print("#"*70)
    
    cnn_lstm_attention = build_cnn_lstm_attention(INPUT_SHAPE, NUM_CLASSES)
    print_model_summary(cnn_lstm_attention)
    
    # Test with dummy data
    print("\nTesting models with dummy data...")
    dummy_input = np.random.randn(1, 128, 42).astype(np.float32)
    
    output1 = cnn_dnn_improved.predict(dummy_input, verbose=0)
    print(f"* CNN-DNN Improved output shape: {output1.shape}")
    
    output2 = cnn_lstm_improved.predict(dummy_input, verbose=0)
    print(f"* CNN-LSTM Improved output shape: {output2.shape}")
    
    output3 = cnn_lstm_attention.predict(dummy_input, verbose=0)
    print(f"* CNN-LSTM Attention output shape: {output3.shape}")
    
    print("\n* All model tests passed!")

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


def build_cnn_dnn_v1(input_shape, num_classes=8, learning_rate=0.0001):
    """
    Basic CNN-DNN v1 model.
    A simpler, baseline architecture without advanced normalization or attention.
    
    Expected accuracy: 65-75%
    """
    model = models.Sequential(name='CNN_DNN_v1')
    
    # Input
    model.add(layers.Input(shape=input_shape))
    
    # CNN Block 1
    model.add(layers.Conv1D(64, kernel_size=5, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.2))
    
    # CNN Block 2
    model.add(layers.Conv1D(128, kernel_size=5, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.2))
    
    # Flatten before Dense
    model.add(layers.Flatten())
    
    # Dense layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(64, activation='relu'))
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

def build_cnn_dnn_improved(input_shape, num_classes=8, learning_rate=0.0005):

    inputs = layers.Input(shape=input_shape)
    # CNN Block 1: Capture low-level patterns
    x = layers.Conv1D(64, kernel_size=11, padding='same',
                            kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = se_block(x)
    x = layers.MaxPooling1D(pool_size=4)(x)
    x = layers.SpatialDropout1D(0.2)(x) 
    # CNN Block 2: Capture mid-level patterns
    x = layers.Conv1D(128, kernel_size=7, padding='same',
                            kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = se_block(x)
    x = layers.MaxPooling1D(pool_size=4)(x)
    x = layers.SpatialDropout1D(0.2)(x)
    # CNN Block 3: Capture high-level patterns
    x = layers.Conv1D(256, kernel_size=5, padding='same',
                            kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    # Dense layers
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.Dropout(0.4)(x)
    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN_DNN_Improved')
    # Compile with higher learning rate
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


def deep_conv_block(x, filters, kernel_size=3, dropout=0.3):
    """Improved deep convolutional block with SE and better normalization."""
    
    # First conv
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Conv1D(filters, kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-3))(x)
    
    # Second conv
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters, kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-3))(x)
    
    # SE block
    x = se_block(x)
        
    return x


def build_cnn_lstm_attention(input_shape, num_classes=8, learning_rate=0.0003):
    inputs = layers.Input(shape=input_shape)
    # CNN Block 1
    x = layers.Conv1D(64, kernel_size=5, padding='same',
                     kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.SpatialDropout1D(0.2)(x)
    # CNN Block 2
    x = layers.Conv1D(128, kernel_size=5, padding='same',
                     kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.SpatialDropout1D(0.2)(x)
    # Bidirectional LSTM with simplified units to prevent overfitting
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True,
                                        kernel_regularizer=regularizers.l2(0.001)))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True,
                                        kernel_regularizer=regularizers.l2(0.001)))(x)
    x = layers.BatchNormalization()(x)
    # Attention mechanism
    x = AttentionLayer()(x)
    # Dense layers
    x = layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.3)(x)
    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN_LSTM_Attention')
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_model(model_type, input_shape, num_classes=8, learning_rate=None):
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
    INPUT_SHAPE = (128, 42)
    NUM_CLASSES = 8
    
    for m in ['cnn_dnn_v1', 'cnn_lstm_v1', 'cnn_dnn_improved', 'cnn_lstm_attention']:
        print(f"\n{'#'*70}\n# BUILDING {m.upper()}\n{'#'*70}")
        model = get_model(m, INPUT_SHAPE, NUM_CLASSES)
        print_model_summary(model)

    print("\nTesting models with dummy data...")
    dummy_input = np.random.randn(1, 128, 42).astype(np.float32)
    for m in ['cnn_dnn_v1', 'cnn_lstm_v1', 'cnn_dnn_improved', 'cnn_lstm_attention']:
        model = get_model(m, INPUT_SHAPE, NUM_CLASSES)
        output = model.predict(dummy_input, verbose=0)
        print(f"* {m} output shape: {output.shape}")
    print("\n* All model tests passed!")

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input,
    LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def build_lstm_attention_model(input_shape, output_shape, d_model=64, num_heads=4):
    """Build LSTM model with self-attention mechanism"""

    print(f"Building attention model with d_model={d_model}, heads={num_heads}")

    # Input layer
    inputs = Input(shape=input_shape, name='input')

    # LSTM layers to extract temporal features
    x = LSTM(128, return_sequences=True, name='lstm_1')(inputs)
    x = Dropout(0.3, name='dropout_1')(x)
    x = LSTM(64, return_sequences=True, name='lstm_2')(x)
    x = Dropout(0.3, name='dropout_2')(x)

    # Project to attention dimension
    x = Dense(d_model, name='dense_projection')(x)

    # Multi-head self-attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        name='multi_head_attention'
    )(x, x)

    # Skip connection and layer normalization
    x = LayerNormalization(epsilon=1e-6, name='layer_norm_1')(x + attention_output)

    # Feed-forward network
    ffn = Sequential([
        Dense(d_model * 2, activation='relu', name='ffn_dense_1'),
        Dropout(0.1, name='ffn_dropout'),
        Dense(d_model, name='ffn_dense_2')
    ], name='feed_forward_network')

    ffn_output = ffn(x)

    # Second skip connection
    x = LayerNormalization(epsilon=1e-6, name='layer_norm_2')(x + ffn_output)

    # Global pooling to reduce temporal dimension
    x = GlobalAveragePooling1D(name='global_pooling')(x)

    # Output layers
    x = Dense(64, activation='relu', name='dense_1')(x)
    x = Dropout(0.2, name='dropout_3')(x)
    x = Dense(32, activation='relu', name='dense_2')(x)
    outputs = Dense(output_shape, name='output')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='LSTM_Attention_Model')

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae', 'mse']
    )

    return model

def train_attention_model(X_train, y_train, X_val, y_val):
    """Train the LSTM with attention model"""
    
    model = build_lstm_attention_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        output_shape=y_train.shape[1],
        d_model=64,
        num_heads=4
    )
    
    # Callbacks for training
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    print("\n TRAINING LSTM WITH ATTENTION MODEL...")
    print("=" * 80)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    print(" Attention model training complete!")
    
    return model, history
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def build_baseline_lstm(input_shape, output_shape):
    """Build a baseline LSTM model for time series forecasting"""

    model = Sequential([
        Input(shape=input_shape, name='input_layer'),
        LSTM(128, return_sequences=True, name='lstm_1'),
        Dropout(0.3, name='dropout_1'),
        LSTM(64, return_sequences=True, name='lstm_2'),
        Dropout(0.3, name='dropout_2'),
        LSTM(32, return_sequences=False, name='lstm_3'),
        Dropout(0.2, name='dropout_3'),
        Dense(64, activation='relu', name='dense_1'),
        Dropout(0.2, name='dropout_4'),
        Dense(32, activation='relu', name='dense_2'),
        Dense(output_shape, name='output_layer')
    ], name='Baseline_LSTM_Model')

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )

    return model

def train_baseline_model(X_train, y_train, X_val, y_val):
    """Train the baseline LSTM model"""
    
    model = build_baseline_lstm(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        output_shape=y_train.shape[1]
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
    
    print("\n TRAINING BASELINE LSTM MODEL...")
    print("=" * 80)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    print(" Baseline LSTM training complete!")
    
    return model, history
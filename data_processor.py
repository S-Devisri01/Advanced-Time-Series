import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def create_sequences(df_processed, selected_meters, window_size=168, forecast_horizon=24, stride=12):
    """Create sequences for time series forecasting"""

    print(f"Creating sequences with:")
    print(f"  Window size: {window_size} hours ({window_size/24:.1f} days)")
    print(f"  Forecast horizon: {forecast_horizon} hours")
    print(f"  Stride: {stride} (reduces data volume)")

    # Separate features and targets
    feature_columns = [col for col in df_processed.columns if not col.endswith('_next_hour')]
    target_columns = [col for col in df_processed.columns if col.endswith('_next_hour')]

    print(f"\nFeatures: {len(feature_columns)} columns")
    print(f"Targets: {len(target_columns)} columns")

    # Scale features and targets separately
    print("\nScaling data...")
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    X = scaler_X.fit_transform(df_processed[feature_columns])
    y = scaler_y.fit_transform(df_processed[target_columns])

    # Create sequences with stride to reduce volume
    X_seq, y_seq = [], []

    for i in range(window_size, len(X) - forecast_horizon, stride):
        X_seq.append(X[i-window_size:i])
        y_seq.append(y[i:i+forecast_horizon])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    print(f"\n Sequences created!")
    print(f"  X shape: {X_seq.shape} (samples, timesteps, features)")
    print(f"  y shape: {y_seq.shape} (samples, forecast_horizon, meters)")

    # Train/Validation/Test split (temporal)
    train_size = int(0.7 * len(X_seq))
    val_size = int(0.15 * len(X_seq))

    X_train = X_seq[:train_size]
    y_train = y_seq[:train_size]

    X_val = X_seq[train_size:train_size+val_size]
    y_val = y_seq[train_size:train_size+val_size]

    X_test = X_seq[train_size+val_size:]
    y_test = y_seq[train_size+val_size:]

    print(f"\n DATA SPLIT:")
    print("-" * 80)
    print(f"Training:   {X_train.shape[0]} samples ({X_train.shape[0]/len(X_seq)*100:.1f}%)")
    print(f"Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(X_seq)*100:.1f}%)")
    print(f"Testing:    {X_test.shape[0]} samples ({X_test.shape[0]/len(X_seq)*100:.1f}%)")

    # Flatten targets for models that require 2D output
    y_train_flat = y_train.reshape(y_train.shape[0], -1)
    y_val_flat = y_val.reshape(y_val.shape[0], -1)
    y_test_flat = y_test.reshape(y_test.shape[0], -1)

    return X_train, X_val, X_test, y_train, y_val, y_test, y_train_flat, y_val_flat, y_test_flat, scaler_X, scaler_y
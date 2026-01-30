import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

def evaluate_model(model, X_test, y_test, df, selected_meters, scaler_y, model_name="Model"):
    """Evaluate model and return comprehensive metrics"""

    print(f"\nEvaluating {model_name}...")

    # Make predictions
    y_pred_flat = model.predict(X_test, verbose=0)
    y_pred = y_pred_flat.reshape(y_test.shape)

    metrics_by_meter = []

    # Get number of meters from y_test shape
    n_meters = y_test.shape[2] if len(y_test.shape) > 2 else 1

    for i in range(n_meters):
        meter_name = selected_meters[i] if i < len(selected_meters) else f"MT_{i:03d}"

        # Extract the data for this meter (keep 2D shape)
        y_true_meter_scaled = y_test[:, :, i]  # Shape: (n_samples, forecast_horizon)
        y_pred_meter_scaled = y_pred[:, :, i]  # Shape: (n_samples, forecast_horizon)

        # Reshape to 2D for inverse transform
        y_true_meter_scaled_flat = y_true_meter_scaled.reshape(-1, 1)
        y_pred_meter_scaled_flat = y_pred_meter_scaled.reshape(-1, 1)

        # Create individual scalers for each meter to avoid the issue
        # Get the original meter data to fit individual scaler
        original_meter_data = df[selected_meters[i]].values.reshape(-1, 1)
        meter_scaler = MinMaxScaler(feature_range=(0, 1))
        meter_scaler.fit(original_meter_data)

        # Now inverse transform with the meter-specific scaler
        y_true_meter = meter_scaler.inverse_transform(y_true_meter_scaled_flat).flatten()
        y_pred_meter = meter_scaler.inverse_transform(y_pred_meter_scaled_flat).flatten()

        # Calculate metrics
        mae = mean_absolute_error(y_true_meter, y_pred_meter)
        rmse = np.sqrt(mean_squared_error(y_true_meter, y_pred_meter))
        mape = np.mean(np.abs((y_true_meter - y_pred_meter) / (np.abs(y_true_meter) + 1e-8))) * 100
        r2 = r2_score(y_true_meter, y_pred_meter)

        metrics_by_meter.append({
            'Meter': meter_name,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        })

    # Calculate average metrics
    avg_metrics = {
        'MAE': np.mean([m['MAE'] for m in metrics_by_meter]),
        'RMSE': np.mean([m['RMSE'] for m in metrics_by_meter]),
        'MAPE': np.mean([m['MAPE'] for m in metrics_by_meter]),
        'R2': np.mean([m['R2'] for m in metrics_by_meter])
    }

    return y_pred, metrics_by_meter, avg_metrics
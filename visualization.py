import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def inverse_transform_single_meter(data_scaled, df, meter_name):
    """Inverse transform data for a single meter using meter-specific scaler"""
    # Reshape if needed
    if len(data_scaled.shape) == 1:
        data_scaled = data_scaled.reshape(-1, 1)
    else:
        data_scaled = data_scaled.reshape(-1, 1)

    # Create meter-specific scaler
    meter_scaler = MinMaxScaler(feature_range=(0, 1))
    original_data = df[meter_name].values.reshape(-1, 1)
    meter_scaler.fit(original_data)

    # Inverse transform
    data_transformed = meter_scaler.inverse_transform(data_scaled)

    return data_transformed.flatten()

def create_visualizations(df, selected_meters, history_baseline, history_attention,
                         baseline_metrics, attention_metrics, baseline_avg, attention_avg,
                         y_test, y_pred_baseline, y_pred_attention, forecast_horizon=24):
    """Create comprehensive visualizations"""
    
    # Set style for better visualizations
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Training History Comparison
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(history_baseline.history['loss'], label='Baseline Train', linewidth=2)
    ax1.plot(history_baseline.history['val_loss'], label='Baseline Val', linestyle='--', linewidth=2)
    ax1.plot(history_attention.history['loss'], label='Attention Train', linewidth=2)
    ax1.plot(history_attention.history['val_loss'], label='Attention Val', linestyle='--', linewidth=2)
    ax1.set_title('Training History Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. MAE Comparison by Meter
    ax2 = plt.subplot(2, 3, 2)
    meters = [m['Meter'] for m in baseline_metrics]
    baseline_mae = [m['MAE'] for m in baseline_metrics]
    attention_mae = [m['MAE'] for m in attention_metrics]
    
    x = np.arange(len(meters))
    width = 0.35
    
    ax2.bar(x - width/2, baseline_mae, width, label='Baseline LSTM', alpha=0.8)
    ax2.bar(x + width/2, attention_mae, width, label='LSTM with Attention', alpha=0.8)
    ax2.set_title('MAE Comparison by Meter', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Meter')
    ax2.set_ylabel('MAE')
    ax2.set_xticks(x)
    ax2.set_xticklabels(meters)
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    
    # 3. Actual vs Predicted (First Meter)
    ax3 = plt.subplot(2, 3, 3)
    sample_idx = 50  # Sample to visualize
    hours = np.arange(forecast_horizon)
    
    # Get actual and predicted values for first meter
    y_actual_sample = inverse_transform_single_meter(
        y_test[sample_idx, :, 0], df, selected_meters[0]
    )
    y_pred_baseline_sample = inverse_transform_single_meter(
        y_pred_baseline[sample_idx, :, 0], df, selected_meters[0]
    )
    y_pred_attention_sample = inverse_transform_single_meter(
        y_pred_attention[sample_idx, :, 0], df, selected_meters[0]
    )
    
    ax3.plot(hours, y_actual_sample, 'k-', label='Actual', linewidth=3, alpha=0.7)
    ax3.plot(hours, y_pred_baseline_sample, 'r--', label='Baseline LSTM', linewidth=2)
    ax3.plot(hours, y_pred_attention_sample, 'b-.', label='LSTM with Attention', linewidth=2)
    ax3.set_title(f'24-Hour Forecast for {selected_meters[0]}', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Hours Ahead')
    ax3.set_ylabel('Consumption')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Error Distribution
    ax4 = plt.subplot(2, 3, 4)
    
    # Calculate errors
    errors_baseline = []
    errors_attention = []
    
    for i in range(len(selected_meters)):
        meter_name = selected_meters[i]
        
        # Get scaled data for this meter
        y_true_meter_scaled = y_test[:, :, i].flatten()
        y_pred_baseline_meter_scaled = y_pred_baseline[:, :, i].flatten()
        y_pred_attention_meter_scaled = y_pred_attention[:, :, i].flatten()
        
        # Inverse transform using meter-specific scaler
        meter_scaler = MinMaxScaler(feature_range=(0, 1))
        original_data = df[meter_name].values.reshape(-1, 1)
        meter_scaler.fit(original_data)
        
        y_true_meter = meter_scaler.inverse_transform(y_true_meter_scaled.reshape(-1, 1)).flatten()
        y_pred_baseline_meter = meter_scaler.inverse_transform(y_pred_baseline_meter_scaled.reshape(-1, 1)).flatten()
        y_pred_attention_meter = meter_scaler.inverse_transform(y_pred_attention_meter_scaled.reshape(-1, 1)).flatten()
        
        errors_baseline.extend(np.abs(y_true_meter - y_pred_baseline_meter))
        errors_attention.extend(np.abs(y_true_meter - y_pred_attention_meter))
    
    # Plot histogram of errors
    ax4.hist(errors_baseline, bins=50, alpha=0.5, label='Baseline LSTM', density=True)
    ax4.hist(errors_attention, bins=50, alpha=0.5, label='LSTM with Attention', density=True)
    ax4.set_title('Error Distribution Comparison', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Absolute Error')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. R² Comparison
    ax5 = plt.subplot(2, 3, 5)
    baseline_r2 = [m['R2'] for m in baseline_metrics]
    attention_r2 = [m['R2'] for m in attention_metrics]
    
    ax5.bar(x - width/2, baseline_r2, width, label='Baseline LSTM', alpha=0.8)
    ax5.bar(x + width/2, attention_r2, width, label='LSTM with Attention', alpha=0.8)
    ax5.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax5.set_title('R² Score Comparison', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Meter')
    ax5.set_ylabel('R² Score')
    ax5.set_xticks(x)
    ax5.set_xticklabels(meters)
    ax5.legend()
    ax5.grid(True, axis='y', alpha=0.3)
    
    # 6. Improvement Percentage
    ax6 = plt.subplot(2, 3, 6)
    improvements = []
    for i in range(len(selected_meters)):
        improvement = ((baseline_metrics[i]['MAE'] - attention_metrics[i]['MAE']) / baseline_metrics[i]['MAE']) * 100
        improvements.append(improvement)
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax6.bar(meters, improvements, color=colors, alpha=0.8)
    ax6.axhline(y=0, color='k', linestyle='-', linewidth=1)
    ax6.set_title('MAE Improvement with Attention (%)', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Meter')
    ax6.set_ylabel('Improvement (%)')
    ax6.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                 f'{imp:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                 fontweight='bold')
    
    plt.tight_layout()
    plt.suptitle('Advanced Time Series Forecasting with Deep Learning and Attention Mechanism', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.savefig('forecasting_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig
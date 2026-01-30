import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def engineer_features(df, num_meters=3):
    """Create comprehensive features for time series forecasting"""
    print(f"Engineering features for {num_meters} meters...")

    # Select subset of meters for manageable computation
    selected_meters = df.columns[:num_meters]
    df_processed = df[selected_meters].copy()

    # 1. Handle missing values
    missing_before = df_processed.isnull().sum().sum()
    df_processed = df_processed.interpolate(method='linear', limit_direction='both')
    df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
    missing_after = df_processed.isnull().sum().sum()
    print(f"  Missing values handled: {missing_before} → {missing_after}")

    # 2. Create temporal features
    print("  Creating temporal features...")
    df_processed['hour'] = df_processed.index.hour
    df_processed['day_of_week'] = df_processed.index.dayofweek
    df_processed['day_of_month'] = df_processed.index.day
    df_processed['month'] = df_processed.index.month
    df_processed['quarter'] = df_processed.index.quarter
    df_processed['year'] = df_processed.index.year

    # Cyclical encoding for periodic features
    df_processed['hour_sin'] = np.sin(2 * np.pi * df_processed['hour'] / 24)
    df_processed['hour_cos'] = np.cos(2 * np.pi * df_processed['hour'] / 24)
    df_processed['day_sin'] = np.sin(2 * np.pi * df_processed['day_of_week'] / 7)
    df_processed['day_cos'] = np.cos(2 * np.pi * df_processed['day_of_week'] / 7)

    # Binary features
    df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)
    df_processed['is_night'] = ((df_processed['hour'] >= 22) | (df_processed['hour'] <= 6)).astype(int)
    df_processed['is_working_hours'] = ((df_processed['hour'] >= 9) & (df_processed['hour'] <= 17)).astype(int)

    # 3. Create lag features for each meter
    print("  Creating lag features...")
    for meter in selected_meters:
        # Essential lags (1, 3, 6, 12, 24 hours)
        for lag in [1, 3, 6, 12, 24]:
            df_processed[f'{meter}_lag_{lag}'] = df_processed[meter].shift(lag)

        # Weekly lag
        df_processed[f'{meter}_lag_168'] = df_processed[meter].shift(168)

    # 4. Rolling statistics for each meter
    print("  Creating rolling statistics...")
    for meter in selected_meters:
        # Short-term rolling stats
        for window in [3, 6, 12]:
            df_processed[f'{meter}_roll_mean_{window}'] = df_processed[meter].rolling(window=window, min_periods=1).mean()
            df_processed[f'{meter}_roll_std_{window}'] = df_processed[meter].rolling(window=window, min_periods=1).std()

        # Daily rolling stats
        df_processed[f'{meter}_roll_mean_24'] = df_processed[meter].rolling(window=24, min_periods=1).mean()
        df_processed[f'{meter}_roll_std_24'] = df_processed[meter].rolling(window=24, min_periods=1).std()

    # 5. Difference features
    for meter in selected_meters:
        df_processed[f'{meter}_diff_1'] = df_processed[meter].diff(1)
        df_processed[f'{meter}_diff_24'] = df_processed[meter].diff(24)

    # 6. Target creation (forecast next hour consumption for all meters)
    print("  Creating targets...")
    for meter in selected_meters:
        df_processed[f'{meter}_next_hour'] = df_processed[meter].shift(-1)

    # 7. Drop remaining NaN values
    df_processed = df_processed.dropna()

    print(f"\n Feature engineering complete!")
    print(f"  Original shape: {df.shape}")
    print(f"  Processed shape: {df_processed.shape}")
    print(f"  Total features created: {df_processed.shape[1]}")

    # Display feature categories
    feature_categories = {
        'Original Meters': len(selected_meters),
        'Temporal Features': 7,
        'Cyclical Features': 4,
        'Binary Features': 3,
        'Lag Features': len(selected_meters) * 6,
        'Rolling Statistics': len(selected_meters) * 5,
        'Difference Features': len(selected_meters) * 2,
        'Target Variables': len(selected_meters)
    }

    print("\n FEATURE BREAKDOWN:")
    print("-" * 80)
    for category, count in feature_categories.items():
        print(f"  {category:25s}: {count}")

    return df_processed, selected_meters

if __name__ == "__main__":
    df = pd.read_csv("synthetic_electricity_data.csv", index_col=0, parse_dates=True)
    df_processed, selected_meters = engineer_features(df, num_meters=3)
    df_processed.to_csv("processed_features.csv")
    df_processed.to_csv("processed_features.csv", encoding='utf-8')
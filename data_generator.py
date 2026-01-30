import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_sample_dataset():
    """Create realistic synthetic electricity consumption data"""
    print("Creating realistic synthetic dataset...")

    # Generate date range for 2 years (hourly)
    dates = pd.date_range('2012-01-01', '2013-12-31', freq='H')
    n_points = len(dates)

    # Create base patterns
    time_idx = np.arange(n_points)

    # Base consumption patterns for 5 meters
    np.random.seed(42)

    # Meter 1: Residential pattern
    meter1 = (
        500 +  # Base level
        150 * np.sin(2 * np.pi * time_idx / 24) +  # Daily pattern
        80 * np.sin(2 * np.pi * time_idx / (24 * 7)) +  # Weekly pattern
        50 * np.sin(2 * np.pi * time_idx / (24 * 30.5)) +  # Monthly pattern
        25 * np.random.randn(n_points)  # Noise
    )

    # Meter 2: Commercial pattern
    meter2 = (
        800 +
        200 * np.sin(2 * np.pi * (time_idx + 6) / 24) +  # Shifted daily
        60 * np.sin(2 * np.pi * time_idx / (24 * 7)) +
        40 * np.sin(2 * np.pi * time_idx / (24 * 30.5)) +
        30 * np.random.randn(n_points)
    )

    # Meter 3: Industrial pattern
    meter3 = (
        1200 +
        300 * np.sin(2 * np.pi * (time_idx + 8) / 24) +
        100 * np.sin(2 * np.pi * time_idx / (24 * 7)) * 0.5 +  # Less weekly variation
        60 * np.sin(2 * np.pi * time_idx / (24 * 30.5)) +
        40 * np.random.randn(n_points)
    )

    # Meter 4: Mixed pattern
    meter4 = (
        700 +
        180 * np.sin(2 * np.pi * (time_idx + 4) / 24) +
        90 * np.sin(2 * np.pi * time_idx / (24 * 7)) +
        50 * np.sin(2 * np.pi * time_idx / (24 * 30.5)) +
        20 * np.random.randn(n_points)
    )

    # Meter 5: Seasonal pattern with higher consumption in winter
    seasonal = 100 * np.sin(2 * np.pi * time_idx / (24 * 365))
    meter5 = (
        600 +
        120 * np.sin(2 * np.pi * time_idx / 24) +
        70 * np.sin(2 * np.pi * time_idx / (24 * 7)) +
        seasonal +
        25 * np.random.randn(n_points)
    )

    # Combine into DataFrame
    data = {
        'MT_001': np.abs(meter1),  # Ensure positive values
        'MT_002': np.abs(meter2),
        'MT_003': np.abs(meter3),
        'MT_004': np.abs(meter4),
        'MT_005': np.abs(meter5)
    }

    df = pd.DataFrame(data, index=dates)

    # Add some missing values (5% missing)
    mask = np.random.rand(*df.shape) < 0.05
    df[mask] = np.nan

    print(f"  Created synthetic dataset with shape: {df.shape}")
    print(f"  Time range: {df.index.min()} to {df.index.max()}")
    print(f"  Missing values: {df.isna().sum().sum()} ({df.isna().sum().sum()/df.size*100:.1f}%)")

    return df

if __name__ == "__main__":
    df = create_sample_dataset()
    df.to_csv("synthetic_electricity_data.csv", encoding='utf-8')
    print("Dataset saved as synthetic_electricity_data.csv")
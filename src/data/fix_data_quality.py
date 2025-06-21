import pandas as pd
import numpy as np
import os

# --- 1. Define File Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
INPUT_FILE = os.path.join(DATA_DIR, 'enriched_aqi_data.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'improved_aqi_data.csv')

TARGET = 'pm10'

# --- 2. Load Original Data ---
print("Loading original data...")
df = pd.read_csv(INPUT_FILE, index_col='datetime', parse_dates=True)
print(f"Original shape: {df.shape}")

# --- 3. Remove Low-Variance Features ---
print("\nAnalyzing feature variance...")
low_variance_features = []

for col in df.columns:
    if col != TARGET:
        std = df[col].std()
        unique_ratio = df[col].nunique() / len(df)
        print(f"{col}: std={std:.4f}, unique_ratio={unique_ratio:.4f}")
        
        # Remove features with very low variance or too few unique values
        if std < 0.1 or unique_ratio < 0.01:
            low_variance_features.append(col)
            print(f"  -> Will remove {col} (low variance)")

print(f"\nRemoving {len(low_variance_features)} low-variance features: {low_variance_features}")
df_cleaned = df.drop(columns=low_variance_features)

# --- 4. Create Proper Lag Features ---
print("\nCreating lag features...")
def create_lag_features(df, target_col, lags=[1, 2, 3, 6, 12, 24]):
    """Create lag features for time series prediction"""
    df_lagged = df.copy()
    
    for lag in lags:
        lag_col = f'{target_col}_lag_{lag}'
        df_lagged[lag_col] = df_lagged[target_col].shift(lag)
        print(f"  Created {lag_col}")
    
    return df_lagged

df_lagged = create_lag_features(df_cleaned, TARGET)

# --- 5. Create Rolling Window Features ---
print("\nCreating rolling window features...")
def create_rolling_features(df, target_col, windows=[3, 6, 12, 24]):
    """Create rolling mean and std features"""
    df_rolling = df.copy()
    
    for window in windows:
        # Rolling mean
        mean_col = f'{target_col}_rolling_mean_{window}'
        df_rolling[mean_col] = df_rolling[target_col].rolling(window=window, min_periods=1).mean()
        
        # Rolling std
        std_col = f'{target_col}_rolling_std_{window}'
        df_rolling[std_col] = df_rolling[target_col].rolling(window=window, min_periods=1).std()
        
        print(f"  Created {mean_col} and {std_col}")
    
    return df_rolling

df_rolling = create_rolling_features(df_lagged, TARGET)

# --- 6. Create Time-Based Features ---
print("\nCreating enhanced time-based features...")
def create_time_features(df):
    """Create more sophisticated time-based features"""
    df_time = df.copy()
    
    # Extract time components
    df_time['hour_sin'] = np.sin(2 * np.pi * df_time.index.hour / 24)
    df_time['hour_cos'] = np.cos(2 * np.pi * df_time.index.hour / 24)
    df_time['day_sin'] = np.sin(2 * np.pi * df_time.index.dayofweek / 7)
    df_time['day_cos'] = np.cos(2 * np.pi * df_time.index.dayofweek / 7)
    df_time['month_sin'] = np.sin(2 * np.pi * df_time.index.month / 12)
    df_time['month_cos'] = np.cos(2 * np.pi * df_time.index.month / 12)
    
    # Create interaction features
    df_time['hour_day_interaction'] = df_time['hour_sin'] * df_time['day_sin']
    
    print("  Created cyclical time features")
    return df_time

df_time = create_time_features(df_rolling)

# --- 7. Remove Rows with Missing Values ---
print("\nRemoving rows with missing values...")
initial_rows = len(df_time)
df_final = df_time.dropna()
final_rows = len(df_final)
print(f"Removed {initial_rows - final_rows} rows with missing values")
print(f"Final shape: {df_final.shape}")

# --- 8. Analyze Improved Dataset ---
print("\n" + "="*60)
print("IMPROVED DATASET ANALYSIS")
print("="*60)

print(f"Final dataset shape: {df_final.shape}")
print(f"Features: {list(df_final.columns)}")

# Check target variable
print(f"\nTarget variable ({TARGET}) statistics:")
print(f"  Mean: {df_final[TARGET].mean():.2f}")
print(f"  Std: {df_final[TARGET].std():.2f}")
print(f"  Min: {df_final[TARGET].min():.2f}")
print(f"  Max: {df_final[TARGET].max():.2f}")
print(f"  Unique values: {df_final[TARGET].nunique()}")

# Check feature correlations with target
print(f"\nFeature correlations with {TARGET}:")
correlations = df_final.corr()[TARGET].sort_values(key=abs, ascending=False)
for feature, corr in correlations.items():
    if feature != TARGET:
        strength = "STRONG" if abs(corr) > 0.5 else "MODERATE" if abs(corr) > 0.3 else "WEAK"
        print(f"  {feature}: {corr:.4f} ({strength})")

# Count strong features
strong_features = [f for f in correlations.index if f != TARGET and abs(correlations[f]) > 0.3]
print(f"\nNumber of features with correlation > 0.3: {len(strong_features)}")

# --- 9. Save Improved Dataset ---
print(f"\nSaving improved dataset to {OUTPUT_FILE}...")
df_final.to_csv(OUTPUT_FILE)
print("Dataset saved successfully!")

# --- 10. Recommendations ---
print("\n" + "="*60)
print("RECOMMENDATIONS FOR MODELING")
print("="*60)

print("1. Use the improved dataset with better features")
print("2. Consider these feature groups:")
print("   - Original features: pm25, o3, no2, temperature_2m, relative_humidity_2m, precipitation, wind_speed_10m")
print("   - Lag features: pm10_lag_1, pm10_lag_2, pm10_lag_3, pm10_lag_6, pm10_lag_12, pm10_lag_24")
print("   - Rolling features: pm10_rolling_mean_*, pm10_rolling_std_*")
print("   - Time features: hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos")

print("\n3. Suggested modeling approach:")
print("   - Start with simpler models (Linear Regression, Random Forest)")
print("   - Use time series cross-validation")
print("   - Focus on features with correlation > 0.3")
print("   - Consider ensemble methods")

print(f"\n4. Next steps:")
print(f"   - Update preprocessing to use improved dataset")
print(f"   - Retrain models with new features")
print(f"   - Use time series cross-validation instead of chronological split") 
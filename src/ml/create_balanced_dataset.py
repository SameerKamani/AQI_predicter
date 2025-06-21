import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import joblib

# --- 1. Define File Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
INPUT_FILE = os.path.join(DATA_DIR, 'improved_aqi_data.csv')
OUTPUT_DIR = os.path.join(DATA_DIR, 'processed')
TRAIN_FILE = os.path.join(OUTPUT_DIR, 'train_balanced.csv')
TEST_FILE = os.path.join(OUTPUT_DIR, 'test_balanced.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
SCALER_FILE = os.path.join(MODEL_DIR, 'scaler_balanced.pkl')

TARGET = 'pm10'

# Define the feature set
FEATURES = [
    'temperature_2m', 'pm10_lag_1', 'pm10_lag_2', 'pm10_lag_3', 'pm10_lag_6', 'pm10_lag_12',
    'pm10_lag_24', 'pm10_rolling_mean_3', 'pm10_rolling_mean_6', 'pm10_rolling_mean_12', 
    'pm10_rolling_mean_24', 'pm10_rolling_std_3', 'pm10_rolling_std_6', 'pm10_rolling_std_12', 
    'pm10_rolling_std_24', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
]

# --- 2. Load Data ---
print("Loading improved dataset...")
df = pd.read_csv(INPUT_FILE, index_col='datetime', parse_dates=True)
print(f"Dataset shape: {df.shape}")

# --- 3. Analyze PM10 Distribution ---
print("\n" + "="*60)
print("PM10 DISTRIBUTION ANALYSIS")
print("="*60)

pm10_values = df[TARGET].values
print(f"PM10 Statistics:")
print(f"  Mean: {pm10_values.mean():.2f}")
print(f"  Std: {pm10_values.std():.2f}")
print(f"  Min: {pm10_values.min():.2f}")
print(f"  Max: {pm10_values.max():.2f}")
print(f"  Unique values: {len(np.unique(pm10_values))}")

# Find periods with good variance
print(f"\nAnalyzing variance over time...")
window_size = 50
rolling_std = df[TARGET].rolling(window=window_size, min_periods=1).std()
high_variance_periods = rolling_std > rolling_std.median()

print(f"  Rolling std mean: {rolling_std.mean():.2f}")
print(f"  Rolling std median: {rolling_std.median():.2f}")
print(f"  High variance periods: {high_variance_periods.sum()} out of {len(rolling_std)}")

# --- 4. Create Balanced Test Set ---
print("\n" + "="*60)
print("CREATING BALANCED TEST SET")
print("="*60)

def create_balanced_testset(df, target_col, test_size=0.2, min_variance=1.0):
    """
    Create a balanced test set by selecting samples from periods with good variance
    """
    # Calculate rolling variance
    rolling_std = df[target_col].rolling(window=50, min_periods=1).std()
    
    # Find periods with sufficient variance
    good_variance_mask = rolling_std > min_variance
    
    # Calculate target test size
    total_samples = len(df)
    target_test_samples = int(total_samples * test_size)
    
    # Split into high and low variance periods
    high_var_indices = df[good_variance_mask].index
    low_var_indices = df[~good_variance_mask].index
    
    print(f"  High variance periods: {len(high_var_indices)} samples")
    print(f"  Low variance periods: {len(low_var_indices)} samples")
    
    # Select test samples from high variance periods
    if len(high_var_indices) >= target_test_samples:
        # We have enough high variance samples
        test_indices = np.random.choice(high_var_indices, target_test_samples, replace=False)
        print(f"  Selected {len(test_indices)} test samples from high variance periods")
    else:
        # Need to include some low variance samples
        high_var_test = high_var_indices
        remaining_needed = target_test_samples - len(high_var_indices)
        low_var_test = np.random.choice(low_var_indices, remaining_needed, replace=False)
        test_indices = np.concatenate([high_var_test, low_var_test])
        print(f"  Selected {len(high_var_test)} from high variance + {len(low_var_test)} from low variance")
    
    # Create train/test masks
    test_mask = df.index.isin(test_indices)
    train_mask = ~test_mask
    
    return train_mask, test_mask

# Create balanced split
train_mask, test_mask = create_balanced_testset(df, TARGET, test_size=0.2, min_variance=2.0)

# Split the data
train_df = df[train_mask]
test_df = df[test_mask]

print(f"\nSplit results:")
print(f"  Training set: {len(train_df)} samples")
print(f"  Test set: {len(test_df)} samples")

# --- 5. Analyze Split Quality ---
print("\n" + "="*60)
print("SPLIT QUALITY ANALYSIS")
print("="*60)

print("Training set statistics:")
print(f"  PM10 mean: {train_df[TARGET].mean():.2f}")
print(f"  PM10 std: {train_df[TARGET].std():.2f}")
print(f"  PM10 range: {train_df[TARGET].min():.2f} - {train_df[TARGET].max():.2f}")
print(f"  PM10 unique values: {train_df[TARGET].nunique()}")

print("\nTest set statistics:")
print(f"  PM10 mean: {test_df[TARGET].mean():.2f}")
print(f"  PM10 std: {test_df[TARGET].std():.2f}")
print(f"  PM10 range: {test_df[TARGET].min():.2f} - {test_df[TARGET].max():.2f}")
print(f"  PM10 unique values: {test_df[TARGET].nunique()}")

mean_diff = test_df[TARGET].mean() - train_df[TARGET].mean()
std_ratio = test_df[TARGET].std() / train_df[TARGET].std() if train_df[TARGET].std() > 0 else 0

print(f"\nSplit quality metrics:")
print(f"  Mean difference (test - train): {mean_diff:.2f}")
print(f"  Std ratio (test/train): {std_ratio:.3f}")

if test_df[TARGET].std() > 3.0:
    print("✅ Excellent split quality - test set has good variance")
elif test_df[TARGET].std() > 1.0:
    print("✅ Good split quality - test set has sufficient variance")
else:
    print("⚠️  Poor split quality - test set lacks variance")

# --- 6. Feature Selection and Scaling ---
print("\n" + "="*60)
print("FEATURE SELECTION AND SCALING")
print("="*60)

# Select available features
available_features = [f for f in FEATURES if f in df.columns]
print(f"Available features: {len(available_features)}")

# Prepare data
X_train = train_df[available_features]
y_train = train_df[TARGET]
X_test = test_df[available_features]
y_test = test_df[TARGET]

print(f"Feature matrix shapes: X_train {X_train.shape}, X_test {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create final DataFrames
train_df_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=available_features)
train_df_scaled[TARGET] = y_train

test_df_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=available_features)
test_df_scaled[TARGET] = y_test

# --- 7. Feature Importance Analysis ---
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

correlations = train_df_scaled[available_features].corrwith(train_df_scaled[TARGET]).abs().sort_values(ascending=False)

print("Feature correlations with target (training set):")
for feature, corr in correlations.items():
    strength = "STRONG" if corr > 0.5 else "MODERATE" if corr > 0.3 else "WEAK"
    print(f"  {feature}: {corr:.4f} ({strength})")

strong_features = [f for f in correlations.index if correlations[f] > 0.3]
print(f"\nNumber of strong features (correlation > 0.3): {len(strong_features)}")

# --- 8. Save Processed Data ---
print(f"\nSaving processed data...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

train_df_scaled.to_csv(TRAIN_FILE)
test_df_scaled.to_csv(TEST_FILE)
joblib.dump(scaler, SCALER_FILE)

print(f"✅ Training data saved to {TRAIN_FILE}")
print(f"✅ Testing data saved to {TEST_FILE}")
print(f"✅ Scaler saved to {SCALER_FILE}")

# --- 9. Summary ---
print("\n" + "="*60)
print("BALANCED SPLIT SUMMARY")
print("="*60)

print(f"✅ Dataset: {df.shape[0]} samples, {len(available_features)} features")
print(f"✅ Training set: {len(train_df_scaled)} samples")
print(f"✅ Test set: {len(test_df_scaled)} samples")
print(f"✅ Strong features: {len(strong_features)}")
print(f"✅ Best feature correlation: {correlations.iloc[0]:.4f}")
print(f"✅ Test set variance: {test_df[TARGET].std():.2f}")

if test_df[TARGET].std() > 1.0:
    print(f"\n🎯 Ready for model training with balanced test set!")
    print(f"   - Use {TRAIN_FILE} for training")
    print(f"   - Use {TEST_FILE} for testing")
    print(f"   - Use {SCALER_FILE} for scaling new data")
else:
    print(f"\n⚠️  Test set still has low variance. Consider:")
    print(f"   - Increasing min_variance threshold")
    print(f"   - Collecting more diverse data")
    print(f"   - Using a different city") 
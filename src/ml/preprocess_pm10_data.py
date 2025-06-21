import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import joblib

# --- 1. Define File Paths & Parameters ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
INPUT_FILE = os.path.join(DATA_DIR, 'improved_aqi_data.csv')
OUTPUT_DIR = os.path.join(DATA_DIR, 'processed')
TRAIN_FILE = os.path.join(OUTPUT_DIR, 'train_improved.csv')
TEST_FILE = os.path.join(OUTPUT_DIR, 'test_improved.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
SCALER_FILE = os.path.join(MODEL_DIR, 'scaler_improved.pkl')

TARGET = 'pm10'

# Define the improved feature set (focusing on high-correlation features)
IMPROVED_FEATURES = [
    # Original features with good correlation
    'pm25', 'temperature_2m',
    
    # Lag features (strongest predictors)
    'pm10_lag_1', 'pm10_lag_2', 'pm10_lag_3', 'pm10_lag_6', 'pm10_lag_12', 'pm10_lag_24',
    
    # Rolling features (excellent predictors)
    'pm10_rolling_mean_3', 'pm10_rolling_mean_6', 'pm10_rolling_mean_12', 'pm10_rolling_mean_24',
    'pm10_rolling_std_3', 'pm10_rolling_std_6', 'pm10_rolling_std_12', 'pm10_rolling_std_24',
    
    # Time features
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
]

# --- 2. Load the Improved Data ---
print("Loading improved dataset...")
df = pd.read_csv(INPUT_FILE, index_col='datetime', parse_dates=True)
print(f"Dataset shape: {df.shape}")
print(f"Features available: {list(df.columns)}")

# --- 3. Feature Selection ---
print(f"\nSelecting {len(IMPROVED_FEATURES)} high-value features...")
available_features = [f for f in IMPROVED_FEATURES if f in df.columns]
print(f"Available features: {available_features}")

# Select features and target
X = df[available_features]
y = df[TARGET]

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# --- 4. Smart Train-Test Split ---
print("\nPerforming smart train-test split...")

# Find periods with good variance in the target
target_std = y.rolling(window=50, min_periods=1).std()
high_variance_periods = target_std > target_std.median()

# Create a more balanced split
total_samples = len(df)
train_size = int(total_samples * 0.8)

# Use a split that ensures both sets have variance
# Find a split point where both sides have sufficient variance
best_split_point = train_size
best_variance_balance = float('inf')

for split_point in range(int(total_samples * 0.7), int(total_samples * 0.9)):
    train_y = y.iloc[:split_point]
    test_y = y.iloc[split_point:]
    
    train_std = train_y.std()
    test_std = test_y.std()
    
    # Calculate balance (we want both stds to be reasonable)
    variance_balance = abs(train_std - test_std) + max(0, 1 - min(train_std, test_std))
    
    if variance_balance < best_variance_balance:
        best_variance_balance = variance_balance
        best_split_point = split_point

print(f"Selected split point: {best_split_point} (out of {total_samples})")

X_train, X_test = X.iloc[:best_split_point], X.iloc[best_split_point:]
y_train, y_test = y.iloc[:best_split_point], y.iloc[best_split_point:]

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# --- 5. Feature Scaling ---
print("\nScaling features...")
scaler = StandardScaler()

# Fit on training data and transform both sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create DataFrames with scaled features
train_df_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=available_features)
train_df_scaled[TARGET] = y_train

test_df_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=available_features)
test_df_scaled[TARGET] = y_test

print("Features scaled successfully.")

# --- 6. Analyze Split Quality ---
print("\n" + "="*50)
print("SPLIT QUALITY ANALYSIS")
print("="*50)

print("Training set statistics:")
print(f"  PM10 mean: {y_train.mean():.2f}")
print(f"  PM10 std: {y_train.std():.2f}")
print(f"  PM10 range: {y_train.min():.2f} - {y_train.max():.2f}")
print(f"  PM10 unique values: {y_train.nunique()}")

print("\nTest set statistics:")
print(f"  PM10 mean: {y_test.mean():.2f}")
print(f"  PM10 std: {y_test.std():.2f}")
print(f"  PM10 range: {y_test.min():.2f} - {y_test.max():.2f}")
print(f"  PM10 unique values: {y_test.nunique()}")

mean_diff = y_test.mean() - y_train.mean()
std_ratio = y_test.std() / y_train.std() if y_train.std() > 0 else 0

print(f"\nSplit quality metrics:")
print(f"  Mean difference (test - train): {mean_diff:.2f}")
print(f"  Std ratio (test/train): {std_ratio:.3f}")

if abs(mean_diff) < 2.0 and 0.3 < std_ratio < 3.0:
    print("✅ Good split quality - balanced distributions")
elif y_test.std() > 1.0:
    print("✅ Acceptable split - test set has sufficient variance")
else:
    print("⚠️  Poor split quality - test set lacks variance")

# --- 7. Feature Importance Analysis ---
print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

# Calculate correlations with target
correlations = train_df_scaled[available_features].corrwith(train_df_scaled[TARGET]).abs().sort_values(ascending=False)

print("Feature correlations with target (training set):")
for feature, corr in correlations.items():
    strength = "STRONG" if corr > 0.5 else "MODERATE" if corr > 0.3 else "WEAK"
    print(f"  {feature}: {corr:.4f} ({strength})")

strong_features = [f for f in correlations.index if correlations[f] > 0.3]
print(f"\nNumber of strong features (correlation > 0.3): {len(strong_features)}")

# --- 8. Save Processed Data and Scaler ---
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
print("\n" + "="*50)
print("PREPROCESSING SUMMARY")
print("="*50)

print(f"✅ Dataset: {df.shape[0]} samples, {len(available_features)} features")
print(f"✅ Training set: {len(train_df_scaled)} samples")
print(f"✅ Test set: {len(test_df_scaled)} samples")
print(f"✅ Strong features: {len(strong_features)}")
print(f"✅ Best feature correlation: {correlations.iloc[0]:.4f}")
print(f"✅ Test set variance: {y_test.std():.2f}")

if y_test.std() > 1.0:
    print(f"\n🎯 Ready for improved model training!")
    print(f"   - Use {TRAIN_FILE} for training")
    print(f"   - Use {TEST_FILE} for testing")
    print(f"   - Use {SCALER_FILE} for scaling new data")
else:
    print(f"\n⚠️  Warning: Test set has low variance. Consider:")
    print(f"   - Collecting more diverse data")
    print(f"   - Using different splitting strategy")
    print(f"   - Focusing on periods with higher variance") 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

# --- 1. Define File Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
INPUT_FILE = os.path.join(DATA_DIR, 'enriched_aqi_data.csv')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
TRAIN_FILE = os.path.join(PROCESSED_DIR, 'train.csv')
TEST_FILE = os.path.join(PROCESSED_DIR, 'test.csv')

TARGET = 'pm25'
FEATURES = [
    'pm10', 'o3', 'no2',
    'temperature_2m', 'relative_humidity_2m', 'precipitation', 'wind_speed_10m',
    'hour', 'day_of_week', 'month',
    'pm25_lag_1', 'pm25_lag_24'
]

# --- 2. Load Data ---
print("Loading data for deep analysis...")
df = pd.read_csv(INPUT_FILE, index_col='datetime', parse_dates=True)
train_df = pd.read_csv(TRAIN_FILE, index_col='datetime', parse_dates=True)
test_df = pd.read_csv(TEST_FILE, index_col='datetime', parse_dates=True)

print(f"Original data shape: {df.shape}")
print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# --- 3. Target Variable Analysis ---
print("\n" + "="*60)
print("TARGET VARIABLE (pm25) ANALYSIS")
print("="*60)

print(f"PM25 Statistics:")
print(f"  Mean: {df[TARGET].mean():.2f}")
print(f"  Std: {df[TARGET].std():.2f}")
print(f"  Min: {df[TARGET].min():.2f}")
print(f"  Max: {df[TARGET].max():.2f}")
print(f"  Median: {df[TARGET].median():.2f}")

# Check for constant values
unique_values = df[TARGET].nunique()
print(f"  Unique values: {unique_values}")
if unique_values < 10:
    print(f"  Value counts: {df[TARGET].value_counts().sort_index()}")

# --- 4. Time Series Analysis ---
print("\n" + "="*60)
print("TIME SERIES ANALYSIS")
print("="*60)

# Check for time gaps
time_diff = df.index.to_series().diff()
print(f"Time intervals between measurements:")
print(f"  Mean: {time_diff.mean()}")
print(f"  Std: {time_diff.std()}")
print(f"  Min: {time_diff.min()}")
print(f"  Max: {time_diff.max()}")

# Check for data distribution over time
print(f"\nData distribution by month:")
monthly_stats = df.groupby(df.index.month)[TARGET].agg(['mean', 'std', 'count'])
print(monthly_stats)

print(f"\nData distribution by hour:")
hourly_stats = df.groupby(df.index.hour)[TARGET].agg(['mean', 'std', 'count'])
print(hourly_stats)

# --- 5. Train-Test Split Analysis ---
print("\n" + "="*60)
print("TRAIN-TEST SPLIT ANALYSIS")
print("="*60)

print("Training set statistics:")
print(f"  PM25 mean: {train_df[TARGET].mean():.2f}")
print(f"  PM25 std: {train_df[TARGET].std():.2f}")
print(f"  PM25 range: {train_df[TARGET].min():.2f} - {train_df[TARGET].max():.2f}")

print("\nTest set statistics:")
print(f"  PM25 mean: {test_df[TARGET].mean():.2f}")
print(f"  PM25 std: {test_df[TARGET].std():.2f}")
print(f"  PM25 range: {test_df[TARGET].min():.2f} - {test_df[TARGET].max():.2f}")

# Check if test set has different characteristics
train_mean = train_df[TARGET].mean()
test_mean = test_df[TARGET].mean()
print(f"\nMean difference (test - train): {test_mean - train_mean:.2f}")

# --- 6. Feature Analysis ---
print("\n" + "="*60)
print("FEATURE ANALYSIS")
print("="*60)

for feature in FEATURES:
    if feature in df.columns:
        print(f"\n{feature}:")
        print(f"  Mean: {df[feature].mean():.4f}")
        print(f"  Std: {df[feature].std():.4f}")
        print(f"  Correlation with {TARGET}: {df[feature].corr(df[TARGET]):.4f}")
        
        # Check for constant or near-constant features
        if df[feature].std() < 0.01:
            print(f"  WARNING: Very low variance (std < 0.01)")
        
        # Check for high correlation with target
        corr = abs(df[feature].corr(df[TARGET]))
        if corr > 0.7:
            print(f"  STRONG correlation with target: {corr:.4f}")
        elif corr > 0.3:
            print(f"  MODERATE correlation with target: {corr:.4f}")
        else:
            print(f"  WEAK correlation with target: {corr:.4f}")

# --- 7. Lag Feature Analysis ---
print("\n" + "="*60)
print("LAG FEATURE ANALYSIS")
print("="*60)

# Check if lag features were created properly
if 'pm25_lag_1' in df.columns:
    print("Lag features found in data:")
    print(f"  pm25_lag_1 correlation with pm25: {df['pm25_lag_1'].corr(df[TARGET]):.4f}")
    print(f"  pm25_lag_24 correlation with pm25: {df['pm25_lag_24'].corr(df[TARGET]):.4f}")
    
    # Check for missing values in lag features
    lag1_missing = df['pm25_lag_1'].isnull().sum()
    lag24_missing = df['pm25_lag_24'].isnull().sum()
    print(f"  pm25_lag_1 missing values: {lag1_missing}")
    print(f"  pm25_lag_24 missing values: {lag24_missing}")
else:
    print("WARNING: Lag features not found in data!")

# --- 8. Data Quality Issues ---
print("\n" + "="*60)
print("POTENTIAL DATA QUALITY ISSUES")
print("="*60)

# Check for suspicious patterns
issues_found = []

# Check for too many identical values
for col in df.columns:
    if col != TARGET:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio < 0.01:  # Less than 1% unique values
            issues_found.append(f"Column '{col}' has very low variance ({unique_ratio:.3f} unique ratio)")

# Check for extreme correlations
corr_matrix = df.corr()
for i, col1 in enumerate(corr_matrix.columns):
    for j, col2 in enumerate(corr_matrix.columns):
        if i < j:  # Avoid duplicate pairs
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.95:
                issues_found.append(f"Very high correlation between '{col1}' and '{col2}': {corr_val:.3f}")

# Check for target variable issues
target_std = df[TARGET].std()
if target_std < 1.0:
    issues_found.append(f"Target variable has very low variance (std: {target_std:.3f})")

if len(issues_found) == 0:
    print("No obvious data quality issues detected.")
else:
    print("Potential issues found:")
    for issue in issues_found:
        print(f"  - {issue}")

# --- 9. Recommendations ---
print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)

print("Based on the analysis, here are potential reasons for poor model performance:")

# Check target variance
if df[TARGET].std() < 2.0:
    print("1. LOW TARGET VARIANCE: The target variable has very little variation.")
    print("   This makes it difficult for any model to learn meaningful patterns.")

# Check feature correlations
strong_features = [f for f in FEATURES if f in df.columns and abs(df[f].corr(df[TARGET])) > 0.3]
if len(strong_features) < 3:
    print("2. WEAK FEATURE CORRELATIONS: Most features have weak correlations with the target.")
    print("   Consider adding more relevant features or feature engineering.")

# Check train-test split
if abs(test_df[TARGET].mean() - train_df[TARGET].mean()) > 2.0:
    print("3. TRAIN-TEST DISTRIBUTION SHIFT: The test set has different characteristics than training.")
    print("   Consider using time series cross-validation instead of chronological split.")

# Check data size
if len(df) < 1000:
    print("4. SMALL DATASET: The dataset may be too small for complex models like LSTM.")
    print("   Consider using simpler models or collecting more data.")

print("\nSuggested next steps:")
print("1. Collect more diverse data with higher target variance")
print("2. Add more relevant features (e.g., traffic data, industrial activity)")
print("3. Use time series cross-validation instead of chronological split")
print("4. Try simpler models (linear regression, random forest)")
print("5. Consider if the prediction task is feasible with current data") 
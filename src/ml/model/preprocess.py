import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import joblib

# --- 1. Define File Paths & Parameters ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
INPUT_FILE = os.path.join(DATA_DIR, 'enriched_aqi_data.csv')
OUTPUT_DIR = os.path.join(DATA_DIR, 'processed')
TRAIN_FILE = os.path.join(OUTPUT_DIR, 'train.csv')
TEST_FILE = os.path.join(OUTPUT_DIR, 'test.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
SCALER_FILE = os.path.join(MODEL_DIR, 'scaler.pkl')

TARGET = 'pm25'
# Define the final feature set here, in one place
FEATURES = [
    'pm10', 'o3', 'no2',
    'temperature_2m', 'relative_humidity_2m', 'precipitation', 'wind_speed_10m',
    'hour', 'day_of_week', 'month',
    'pm25_lag_1', 'pm25_lag_24'
]

# --- 2. Load the Data ---
print("Loading enriched data...")
df = pd.read_csv(INPUT_FILE, index_col='datetime', parse_dates=True)

# --- 3. Chronological Train-Test Split ---
# NOTE: The chronological split on this dataset results in a test set where
# key features have zero variance, leading to an R-squared of 0.
# This indicates the final 20% of the data is from a period of flat readings.
# For a robust evaluation, Time Series Cross-Validation would be recommended.
print("Performing chronological train-test split...")
train_size = int(len(df) * 0.8)
train_df, test_df = df.iloc[:train_size].copy(), df.iloc[train_size:].copy()

# --- 4. Feature Engineering (Lag Features) ---
print("Creating lag features...")
def create_lag_features(df):
    df['pm25_lag_1'] = df[TARGET].shift(1)
    df['pm25_lag_24'] = df[TARGET].shift(24)
    return df

train_df = create_lag_features(train_df)
test_df = create_lag_features(test_df)
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# --- 5. Final Feature Selection ---
train_features = train_df[FEATURES]
test_features = test_df[FEATURES]
train_target = train_df[TARGET]
test_target = test_df[TARGET]
print("Final features selected.")

# --- 6. Feature Scaling ---
print("Scaling features...")
scaler = StandardScaler()

# Fit on training data and transform both sets
scaled_train_features = scaler.fit_transform(train_features)
scaled_test_features = scaler.transform(test_features)

# Create new, scaled DataFrames
train_df_scaled = pd.DataFrame(scaled_train_features, index=train_features.index, columns=FEATURES)
train_df_scaled[TARGET] = train_target

test_df_scaled = pd.DataFrame(scaled_test_features, index=test_features.index, columns=FEATURES)
test_df_scaled[TARGET] = test_target

print("Features scaled robustly.")

# --- 7. Save the Processed Data and Scaler ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
train_df_scaled.to_csv(TRAIN_FILE)
test_df_scaled.to_csv(TEST_FILE)
joblib.dump(scaler, SCALER_FILE)

print(f"Processed training data saved to {TRAIN_FILE}")
print(f"Processed testing data saved to {TEST_FILE}")
print(f"Scaler saved to {SCALER_FILE}") 
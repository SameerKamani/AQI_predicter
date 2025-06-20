import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# --- 1. Define File Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
INPUT_FILE = os.path.join(DATA_DIR, 'enriched_aqi_data.csv')
OUTPUT_DIR = os.path.join(DATA_DIR, 'processed')
TRAIN_FILE = os.path.join(OUTPUT_DIR, 'train.csv')
TEST_FILE = os.path.join(OUTPUT_DIR, 'test.csv')

# --- 2. Load the Data ---
print("Loading enriched data...")
df = pd.read_csv(INPUT_FILE, index_col='datetime', parse_dates=True)

# For this model, we'll predict pm25. This can be changed to a calculated AQI later.
TARGET = 'pm25'

# --- 3. Feature Engineering (Lag Features) ---
# Create lag features for the target variable to help the model see past values
df['pm25_lag_1'] = df[TARGET].shift(1)
df['pm25_lag_24'] = df[TARGET].shift(24)
df.dropna(inplace=True) # Drop rows with NaN values created by shifting
print("Lag features created.")

# --- 4. Chronological Train-Test Split ---
print("Performing chronological train-test split...")
train_size = int(len(df) * 0.8)
train_df, test_df = df[:train_size], df[train_size:]
print(f"Train set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")

# --- 5. Feature Scaling ---
print("Scaling features...")
features = [col for col in df.columns if col != TARGET]
scaler = StandardScaler()

# Fit on training data and transform both train and test data
train_df_scaled = train_df.copy()
test_df_scaled = test_df.copy()

train_df_scaled[features] = scaler.fit_transform(train_df[features])
test_df_scaled[features] = scaler.transform(test_df[features])
print("Features scaled based on the training set.")

# --- 6. Save the Processed Data ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
train_df_scaled.to_csv(TRAIN_FILE)
test_df_scaled.to_csv(TEST_FILE)

print(f"Processed training data saved to {TRAIN_FILE}")
print(f"Processed testing data saved to {TEST_FILE}") 
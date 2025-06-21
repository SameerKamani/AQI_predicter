import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import joblib

# --- 1. Define File Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
TRAIN_FILE = os.path.join(DATA_DIR, 'train_balanced.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test_balanced.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
MODEL_FILE = os.path.join(MODEL_DIR, 'lstm_pm10_model.keras') # Use .keras extension for TF models
SCALER_FILE = os.path.join(MODEL_DIR, 'scaler_balanced.pkl')

TARGET = 'pm10'  # Changed from 'pm25' to 'pm10'
SEQUENCE_LENGTH = 24 # Use the last 24 hours of data to predict the next hour

# Define the feature set for PM10 prediction
FEATURES = [
    'temperature_2m', 'pm10_lag_1', 'pm10_lag_2', 'pm10_lag_3', 'pm10_lag_6', 'pm10_lag_12',
    'pm10_lag_24', 'pm10_rolling_mean_3', 'pm10_rolling_mean_6', 'pm10_rolling_mean_12', 
    'pm10_rolling_mean_24', 'pm10_rolling_std_3', 'pm10_rolling_std_6', 'pm10_rolling_std_12', 
    'pm10_rolling_std_24', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
]

# --- 2. Load the Pre-processed Data ---
print("Loading balanced PM10 dataset...")
train_df = pd.read_csv(TRAIN_FILE, index_col='datetime', parse_dates=True)
test_df = pd.read_csv(TEST_FILE, index_col='datetime', parse_dates=True)
print("Data loaded.")

# --- 3. Create Time Series Sequences ---
# LSTMs expect data in the shape [samples, timesteps, features]
def create_sequences(data, sequence_length, features, target):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data.iloc[i:(i + sequence_length)][features].values)
        y.append(data.iloc[i + sequence_length][target])
    return np.array(X), np.array(y)

# Select available features
available_features = [f for f in FEATURES if f in train_df.columns]
print(f"Available features for LSTM: {len(available_features)}")

X_train, y_train = create_sequences(train_df, SEQUENCE_LENGTH, available_features, TARGET)
X_test, y_test = create_sequences(test_df, SEQUENCE_LENGTH, available_features, TARGET)

print(f"Created sequences. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# --- 4. Build the LSTM Model ---
print("Building LSTM model...")
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1) # Output layer
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# --- 5. Train the Model ---
print("Training LSTM model...")
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)
print("Model training complete.")

# --- 6. Evaluate the Model ---
print("\nEvaluating model performance...")
preds_scaled = model.predict(X_test)

# Since we're using the balanced dataset, predictions are already in the correct scale
# No need for inverse scaling as the data is already properly scaled
mae = mean_absolute_error(y_test, preds_scaled.ravel())
r2 = r2_score(y_test, preds_scaled.ravel())

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# --- 7. Save the Model ---
print("\nSaving the trained LSTM model...")
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")

# --- 8. Additional Analysis ---
print("\n" + "="*60)
print("LSTM MODEL ANALYSIS")
print("="*60)

print("Test set target statistics:")
print(f"  Mean: {y_test.mean():.2f}")
print(f"  Std: {y_test.std():.2f}")
print(f"  Min: {y_test.min():.2f}")
print(f"  Max: {y_test.max():.2f}")

print(f"\nLSTM prediction statistics:")
print(f"  Mean: {preds_scaled.mean():.2f}")
print(f"  Std: {preds_scaled.std():.2f}")
print(f"  Min: {preds_scaled.min():.2f}")
print(f"  Max: {preds_scaled.max():.2f}")

# Error analysis
errors = y_test - preds_scaled.ravel()
print(f"\nError statistics:")
print(f"  Mean error: {errors.mean():.2f}")
print(f"  Std error: {errors.std():.2f}")
print(f"  Min error: {errors.min():.2f}")
print(f"  Max error: {errors.max():.2f}")

if r2 > 0.8:
    print("🎉 Excellent LSTM performance!")
elif r2 > 0.6:
    print("✅ Good LSTM performance!")
elif r2 > 0.4:
    print("⚠️  Moderate LSTM performance")
else:
    print("❌ Poor LSTM performance") 
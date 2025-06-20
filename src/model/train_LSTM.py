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
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
MODEL_FILE = os.path.join(MODEL_DIR, 'lstm_aqi_model.keras') # Use .keras extension for TF models
SCALER_FILE = os.path.join(MODEL_DIR, 'scaler.pkl')

TARGET = 'pm25'
SEQUENCE_LENGTH = 24 # Use the last 24 hours of data to predict the next hour

# --- 2. Load the Pre-processed Data ---
print("Loading pre-processed data...")
train_df = pd.read_csv(TRAIN_FILE, index_col='datetime', parse_dates=True)
test_df = pd.read_csv(TEST_FILE, index_col='datetime', parse_dates=True)
print("Data loaded.")

# --- 3. Create Time Series Sequences ---
# LSTMs expect data in the shape [samples, timesteps, features]
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data.iloc[i:(i + sequence_length)].values)
        y.append(data.iloc[i + sequence_length][TARGET])
    return np.array(X), np.array(y)

features = [col for col in train_df.columns if col != TARGET]

# Separate features and target for creating sequences
train_data = train_df[features + [TARGET]]
test_data = test_df[features + [TARGET]]

X_train, y_train = create_sequences(train_data, SEQUENCE_LENGTH)
X_test, y_test = create_sequences(test_data, SEQUENCE_LENGTH)

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

# We need to inverse transform the predictions and the true values to get a meaningful R-squared
scaler = joblib.load(SCALER_FILE)
n_features = len(features) + 1 # +1 for the target column

# Create dummy arrays to perform inverse scaling on the target variable only
preds_dummy = np.zeros((len(preds_scaled), n_features))
preds_dummy[:, -1] = preds_scaled.ravel() # Put predictions in the last column (target)
preds_unscaled = scaler.inverse_transform(preds_dummy)[:, -1]

y_test_dummy = np.zeros((len(y_test), n_features))
y_test_dummy[:, -1] = y_test.ravel() # Put true values in the last column
y_test_unscaled = scaler.inverse_transform(y_test_dummy)[:, -1]

mae = mean_absolute_error(y_test_unscaled, preds_unscaled)
r2 = r2_score(y_test_unscaled, preds_unscaled)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# --- 7. Save the Model ---
print("\nSaving the trained LSTM model...")
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(MODEL_FILE)
print(f"Model saved to {MODEL_FILE}") 
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import os
import joblib

# --- 1. Define File Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
MODEL_FILE = os.path.join(MODEL_DIR, 'aqi_model.pkl')

TARGET = 'pm25'

# --- 2. Load the Processed Data ---
print("Loading processed data...")
train_df = pd.read_csv(TRAIN_FILE, index_col='datetime', parse_dates=True)
test_df = pd.read_csv(TEST_FILE, index_col='datetime', parse_dates=True)

X_train = train_df.drop(TARGET, axis=1)
y_train = train_df[TARGET]
X_test = test_df.drop(TARGET, axis=1)
y_test = test_df[TARGET]
print("Data loaded.")

# --- 3. Train the XGBoost Model ---
print("Training XGBoost model...")
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    early_stopping_rounds=50
)

# Use test set as evaluation set for early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
print("Model training complete.")

# --- 4. Evaluate the Model ---
print("\nEvaluating model performance...")
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# --- 5. Save the Model ---
print("\nSaving the trained model...")
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}") 
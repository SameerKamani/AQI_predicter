import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import os
import joblib

# --- 1. Define File Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
TRAIN_FILE = os.path.join(DATA_DIR, 'train_balanced.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test_balanced.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
MODEL_FILE = os.path.join(MODEL_DIR, 'xgb_pm10_model.pkl')

TARGET = 'pm10'

# --- 2. Load the Pre-processed Data ---
print("Loading balanced PM10 dataset...")
train_df = pd.read_csv(TRAIN_FILE, index_col='datetime', parse_dates=True)
test_df = pd.read_csv(TEST_FILE, index_col='datetime', parse_dates=True)

# The features are now all columns except the target
features = [col for col in train_df.columns if col != TARGET]
X_train = train_df[features]
y_train = train_df[TARGET]
X_test = test_df[features]
y_test = test_df[TARGET]
print("Data loaded.")
print(f"Features: {len(features)}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# --- 3. Train the XGBoost Model ---
print("\nTraining XGBoost model...")
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    early_stopping_rounds=50,
    random_state=42
)

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

# --- 6. Feature Importance Analysis ---
print("\n" + "="*60)
print("XGBOOST FEATURE IMPORTANCE")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 most important features:")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# --- 7. Detailed Analysis ---
print("\n" + "="*60)
print("XGBOOST MODEL ANALYSIS")
print("="*60)

print("Test set target statistics:")
print(f"  Mean: {y_test.mean():.2f}")
print(f"  Std: {y_test.std():.2f}")
print(f"  Min: {y_test.min():.2f}")
print(f"  Max: {y_test.max():.2f}")

print(f"\nXGBoost prediction statistics:")
print(f"  Mean: {preds.mean():.2f}")
print(f"  Std: {preds.std():.2f}")
print(f"  Min: {preds.min():.2f}")
print(f"  Max: {preds.max():.2f}")

# Error analysis
errors = y_test - preds
print(f"\nError statistics:")
print(f"  Mean error: {errors.mean():.2f}")
print(f"  Std error: {errors.std():.2f}")
print(f"  Min error: {errors.min():.2f}")
print(f"  Max error: {errors.max():.2f}")

if r2 > 0.8:
    print("🎉 Excellent XGBoost performance!")
elif r2 > 0.6:
    print("✅ Good XGBoost performance!")
elif r2 > 0.4:
    print("⚠️  Moderate XGBoost performance")
else:
    print("❌ Poor XGBoost performance")

# --- 8. Summary ---
print("\n" + "="*60)
print("XGBOOST TRAINING SUMMARY")
print("="*60)

print(f"✅ Target variable: {TARGET}")
print(f"✅ Training samples: {len(X_train)}")
print(f"✅ Test samples: {len(X_test)}")
print(f"✅ Features: {len(features)}")
print(f"✅ XGBoost R²: {r2:.4f}")

print(f"\n🎯 Ready for predictions!")
print(f"   - Use {MODEL_FILE} for making predictions")
print(f"   - Use {os.path.join(MODEL_DIR, 'scaler_balanced.pkl')} for scaling new data") 
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import os
import joblib

# --- 1. Define File Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
TRAIN_FILE = os.path.join(DATA_DIR, 'train_balanced.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test_balanced.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')

TARGET = 'pm10'

# --- 2. Load Data ---
print("Loading balanced dataset...")
train_df = pd.read_csv(TRAIN_FILE, index_col='datetime', parse_dates=True)
test_df = pd.read_csv(TEST_FILE, index_col='datetime', parse_dates=True)

print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")

# --- 3. Prepare Features and Target ---
# Get features (all columns except target)
features = [col for col in train_df.columns if col != TARGET]

X_train = train_df[features]
y_train = train_df[TARGET]
X_test = test_df[features]
y_test = test_df[TARGET]

print(f"Features: {len(features)}")
print(f"Feature names: {features}")

# --- 4. Train Multiple Models ---
print("\n" + "="*60)
print("TRAINING MULTIPLE MODELS")
print("="*60)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        'model': model,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'predictions': y_pred
    }
    
    print(f"  MAE: {mae:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")

# --- 5. Model Comparison ---
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

print("Performance Comparison:")
print(f"{'Model':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
print("-" * 50)

for name, result in results.items():
    print(f"{name:<20} {result['mae']:<10.4f} {result['rmse']:<10.4f} {result['r2']:<10.4f}")

# Find best model
best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
best_model = results[best_model_name]['model']
best_r2 = results[best_model_name]['r2']

print(f"\nBest model: {best_model_name} (R² = {best_r2:.4f})")

# --- 6. Feature Importance (for Random Forest) ---
if 'Random Forest' in results:
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("="*60)
    
    rf_model = results['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 most important features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

# --- 7. Save Best Model ---
print(f"\nSaving best model ({best_model_name})...")
os.makedirs(MODEL_DIR, exist_ok=True)

if best_model_name == 'Random Forest':
    model_file = os.path.join(MODEL_DIR, 'best_pm10_model_rf.pkl')
else:
    model_file = os.path.join(MODEL_DIR, 'best_pm10_model_lr.pkl')

joblib.dump(best_model, model_file)
print(f"✅ Best model saved to {model_file}")

# --- 8. Detailed Analysis ---
print("\n" + "="*60)
print("DETAILED ANALYSIS")
print("="*60)

# Test set statistics
print("Test set target statistics:")
print(f"  Mean: {y_test.mean():.2f}")
print(f"  Std: {y_test.std():.2f}")
print(f"  Min: {y_test.min():.2f}")
print(f"  Max: {y_test.max():.2f}")

# Prediction statistics
best_predictions = results[best_model_name]['predictions']
print(f"\nBest model prediction statistics:")
print(f"  Mean: {best_predictions.mean():.2f}")
print(f"  Std: {best_predictions.std():.2f}")
print(f"  Min: {best_predictions.min():.2f}")
print(f"  Max: {best_predictions.max():.2f}")

# Error analysis
errors = y_test - best_predictions
print(f"\nError statistics:")
print(f"  Mean error: {errors.mean():.2f}")
print(f"  Std error: {errors.std():.2f}")
print(f"  Min error: {errors.min():.2f}")
print(f"  Max error: {errors.max():.2f}")

# --- 9. Summary ---
print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)

print(f"✅ Target variable: {TARGET}")
print(f"✅ Training samples: {len(X_train)}")
print(f"✅ Test samples: {len(X_test)}")
print(f"✅ Features: {len(features)}")
print(f"✅ Best model: {best_model_name}")
print(f"✅ Best R²: {best_r2:.4f}")

if best_r2 > 0.8:
    print("🎉 Excellent model performance!")
elif best_r2 > 0.6:
    print("✅ Good model performance!")
elif best_r2 > 0.4:
    print("⚠️  Moderate model performance")
else:
    print("❌ Poor model performance - consider feature engineering")

print(f"\n🎯 Ready for predictions!")
print(f"   - Use {model_file} for making predictions")
print(f"   - Use {os.path.join(MODEL_DIR, 'scaler_balanced.pkl')} for scaling new data") 
import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class PM10EnsemblePredictor:
    def __init__(self):
        """Initialize the ensemble predictor with multiple models"""
        self.model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        
        # Load all available models
        self.models = {}
        self.model_weights = {}
        self.is_loaded = False
        
        self._load_models()
        
    def _load_models(self):
        """Load all available trained models"""
        try:
            # Linear Regression
            lr_path = os.path.join(self.model_dir, 'best_pm10_model_lr.pkl')
            if os.path.exists(lr_path):
                self.models['LinearRegression'] = joblib.load(lr_path)
                self.model_weights['LinearRegression'] = 0.4  # High weight due to perfect R²
                print("✅ Loaded Linear Regression model")
            
            # XGBoost
            xgb_path = os.path.join(self.model_dir, 'xgb_pm10_model.pkl')
            if os.path.exists(xgb_path):
                self.models['XGBoost'] = joblib.load(xgb_path)
                self.model_weights['XGBoost'] = 0.3  # Good performance
                print("✅ Loaded XGBoost model")
            
            # Random Forest (if available)
            rf_path = os.path.join(self.model_dir, 'rf_pm10_model.pkl')
            if os.path.exists(rf_path):
                self.models['RandomForest'] = joblib.load(rf_path)
                self.model_weights['RandomForest'] = 0.2
                print("✅ Loaded Random Forest model")
            
            # LSTM (if available)
            lstm_path = os.path.join(self.model_dir, 'lstm_pm10_model.keras')
            if os.path.exists(lstm_path):
                self.models['LSTM'] = keras.models.load_model(lstm_path)
                self.model_weights['LSTM'] = 0.1  # Lower weight due to lower performance
                print("✅ Loaded LSTM model")
            
            # Load scaler
            scaler_path = os.path.join(self.model_dir, 'scaler_balanced.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print("✅ Loaded scaler")
            else:
                print("❌ Scaler not found")
                return
            
            if len(self.models) > 0:
                self.is_loaded = True
                print(f"🎯 Ensemble initialized with {len(self.models)} models")
                print("Model weights:", self.model_weights)
            else:
                print("❌ No models found")
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def create_features(self, pm10_history, temperature=20.0, timestamp=None):
        """Create features for prediction (same as individual models)"""
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now()
        
        if len(pm10_history) < 24:
            raise ValueError("Need at least 24 hours of PM10 history")
        
        # Create lag features
        features = {
            'temperature_2m': temperature,
            'pm10_lag_1': pm10_history[-1],
            'pm10_lag_2': pm10_history[-2],
            'pm10_lag_3': pm10_history[-3],
            'pm10_lag_6': pm10_history[-6],
            'pm10_lag_12': pm10_history[-12],
            'pm10_lag_24': pm10_history[-24]
        }
        
        # Create rolling features
        pm10_array = np.array(pm10_history)
        features.update({
            'pm10_rolling_mean_3': np.mean(pm10_array[-3:]),
            'pm10_rolling_mean_6': np.mean(pm10_array[-6:]),
            'pm10_rolling_mean_12': np.mean(pm10_array[-12:]),
            'pm10_rolling_mean_24': np.mean(pm10_array[-24:]),
            'pm10_rolling_std_3': np.std(pm10_array[-3:]),
            'pm10_rolling_std_6': np.std(pm10_array[-6:]),
            'pm10_rolling_std_12': np.std(pm10_array[-12:]),
            'pm10_rolling_std_24': np.std(pm10_array[-24:])
        })
        
        # Add time features
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        month = timestamp.month
        
        features.update({
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'day_sin': np.sin(2 * np.pi * day_of_week / 7),
            'day_cos': np.cos(2 * np.pi * day_of_week / 7),
            'month_sin': np.sin(2 * np.pi * month / 12),
            'month_cos': np.cos(2 * np.pi * month / 12)
        })
        
        # Define feature order
        feature_names = [
            'temperature_2m', 'pm10_lag_1', 'pm10_lag_2', 'pm10_lag_3', 'pm10_lag_6', 'pm10_lag_12',
            'pm10_lag_24', 'pm10_rolling_mean_3', 'pm10_rolling_mean_6', 'pm10_rolling_mean_12', 
            'pm10_rolling_mean_24', 'pm10_rolling_std_3', 'pm10_rolling_std_6', 'pm10_rolling_std_12', 
            'pm10_rolling_std_24', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
        ]
        
        # Create feature vector
        feature_vector = [features[f] for f in feature_names]
        
        return feature_vector
    
    def predict_ensemble(self, pm10_history, temperature=20.0, timestamp=None):
        """
        Make ensemble prediction using all available models
        
        Returns:
            dict with ensemble prediction and individual model predictions
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded")
        
        # Create features
        feature_vector = self.create_features(pm10_history, temperature, timestamp)
        feature_vector_scaled = self.scaler.transform([feature_vector])
        
        # Get predictions from each model
        predictions = {}
        ensemble_prediction = 0
        total_weight = 0
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'LSTM':
                    # LSTM expects different input format
                    lstm_input = feature_vector_scaled.reshape(1, 1, -1)
                    pred = model.predict(lstm_input, verbose=0)[0][0]
                else:
                    # Standard sklearn models
                    pred = model.predict(feature_vector_scaled)[0]
                
                predictions[model_name] = pred
                weight = self.model_weights[model_name]
                ensemble_prediction += pred * weight
                total_weight += weight
                
            except Exception as e:
                print(f"⚠️ Error with {model_name}: {e}")
                continue
        
        if total_weight == 0:
            raise ValueError("No models made successful predictions")
        
        # Normalize ensemble prediction
        ensemble_prediction /= total_weight
        
        return {
            'ensemble_prediction': ensemble_prediction,
            'individual_predictions': predictions,
            'model_weights': self.model_weights,
            'confidence': self._calculate_confidence(predictions)
        }
    
    def _calculate_confidence(self, predictions):
        """Calculate confidence based on prediction agreement"""
        if len(predictions) < 2:
            return "Low - Single model prediction"
        
        values = list(predictions.values())
        std = np.std(values)
        mean = np.mean(values)
        
        # Coefficient of variation
        cv = std / mean if mean != 0 else float('inf')
        
        if cv < 0.05:
            return "High - Strong model agreement"
        elif cv < 0.1:
            return "Medium - Good model agreement"
        else:
            return "Low - High model disagreement"
    
    def get_model_performance_summary(self):
        """Get summary of model performance"""
        if not self.is_loaded:
            return "No models loaded"
        
        summary = "🎯 Ensemble Model Summary:\n\n"
        
        # Performance metrics (from training)
        performance = {
            'LinearRegression': {'r2': 1.0000, 'status': 'Perfect'},
            'XGBoost': {'r2': 0.9658, 'status': 'Excellent'},
            'RandomForest': {'r2': 0.9512, 'status': 'Very Good'},
            'LSTM': {'r2': 0.6755, 'status': 'Good'}
        }
        
        for model_name in self.models.keys():
            if model_name in performance:
                perf = performance[model_name]
                weight = self.model_weights[model_name]
                summary += f"• {model_name}:\n"
                summary += f"  - R² Score: {perf['r2']:.4f}\n"
                summary += f"  - Status: {perf['status']}\n"
                summary += f"  - Weight: {weight:.1%}\n\n"
        
        summary += f"Total models: {len(self.models)}"
        return summary

# Example usage
if __name__ == "__main__":
    # Initialize ensemble
    ensemble = PM10EnsemblePredictor()
    
    if ensemble.is_loaded:
        print("\n" + "="*50)
        print(ensemble.get_model_performance_summary())
        print("="*50)
        
        # Example prediction
        pm10_history = [25, 26, 28, 30, 32, 35, 38, 40, 42, 45, 48, 50, 
                        52, 55, 58, 60, 62, 65, 68, 70, 72, 75, 78, 80]
        
        result = ensemble.predict_ensemble(pm10_history, temperature=22.0)
        
        print(f"\n🔮 Ensemble Prediction Example:")
        print(f"   Historical PM10 (last 24h): {pm10_history[-5:]}...")
        print(f"   Current temperature: 22.0°C")
        print(f"   Ensemble prediction: {result['ensemble_prediction']:.2f} μg/m³")
        print(f"   Confidence: {result['confidence']}")
        
        print(f"\n📊 Individual Model Predictions:")
        for model, pred in result['individual_predictions'].items():
            print(f"   {model}: {pred:.2f} μg/m³")
    else:
        print("❌ Ensemble not properly initialized") 
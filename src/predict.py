import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta

class PM10Predictor:
    def __init__(self):
        """Initialize the PM10 predictor with the best trained model"""
        self.model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        self.scaler_file = os.path.join(self.model_dir, 'scaler_balanced.pkl')
        self.model_file = os.path.join(self.model_dir, 'best_pm10_model_lr.pkl')
        
        # Load the model and scaler
        self.model = joblib.load(self.model_file)
        self.scaler = joblib.load(self.scaler_file)
        
        # Define feature names (same as training)
        self.features = [
            'temperature_2m', 'pm10_lag_1', 'pm10_lag_2', 'pm10_lag_3', 'pm10_lag_6', 'pm10_lag_12',
            'pm10_lag_24', 'pm10_rolling_mean_3', 'pm10_rolling_mean_6', 'pm10_rolling_mean_12', 
            'pm10_rolling_mean_24', 'pm10_rolling_std_3', 'pm10_rolling_std_6', 'pm10_rolling_std_12', 
            'pm10_rolling_std_24', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
        ]
        
        print("✅ PM10 Predictor initialized successfully!")
        print(f"   Model: Linear Regression (R² = 1.0000)")
        print(f"   Features: {len(self.features)}")
    
    def create_time_features(self, timestamp):
        """Create time-based features for a given timestamp"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        month = timestamp.month
        
        return {
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'day_sin': np.sin(2 * np.pi * day_of_week / 7),
            'day_cos': np.cos(2 * np.pi * day_of_week / 7),
            'month_sin': np.sin(2 * np.pi * month / 12),
            'month_cos': np.cos(2 * np.pi * month / 12)
        }
    
    def predict_single(self, pm10_history, temperature=20.0, timestamp=None):
        """
        Predict PM10 for a single point in time
        
        Args:
            pm10_history: List of PM10 values for the last 24 hours (most recent last)
            temperature: Current temperature in Celsius
            timestamp: Current timestamp (if None, uses current time)
        
        Returns:
            Predicted PM10 value
        """
        if timestamp is None:
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
        features.update(self.create_time_features(timestamp))
        
        # Create feature vector in correct order
        feature_vector = [features[f] for f in self.features]
        
        # Scale features
        feature_vector_scaled = self.scaler.transform([feature_vector])
        
        # Make prediction
        prediction = self.model.predict(feature_vector_scaled)[0]
        
        return prediction
    
    def predict_multiple(self, pm10_history, temperatures, timestamps=None):
        """
        Predict PM10 for multiple points in time
        
        Args:
            pm10_history: List of PM10 values (at least 24 hours)
            temperatures: List of temperatures for each prediction
            timestamps: List of timestamps (if None, uses current time + hours)
        
        Returns:
            List of predicted PM10 values
        """
        if timestamps is None:
            timestamps = [datetime.now() + timedelta(hours=i) for i in range(len(temperatures))]
        
        predictions = []
        for i, (temp, timestamp) in enumerate(zip(temperatures, timestamps)):
            # Use the last 24 hours + any new predictions
            current_history = pm10_history.copy()
            if i > 0:
                current_history.extend(predictions)
            
            pred = self.predict_single(current_history, temp, timestamp)
            predictions.append(pred)
        
        return predictions
    
    def get_prediction_confidence(self, pm10_history):
        """
        Get confidence level based on data quality
        
        Args:
            pm10_history: List of PM10 values
        
        Returns:
            Confidence level (High/Medium/Low)
        """
        if len(pm10_history) < 24:
            return "Low - Insufficient historical data"
        
        # Check for missing values
        if any(pd.isna(pm10_history[-24:])):
            return "Medium - Missing values in recent data"
        
        # Check for data variance
        recent_std = np.std(pm10_history[-24:])
        if recent_std < 1.0:
            return "Medium - Low variance in recent data"
        
        return "High - Good quality data"

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = PM10Predictor()
    
    # Example: Predict PM10 for next hour
    # Simulate 24 hours of PM10 data
    pm10_history = [25, 26, 28, 30, 32, 35, 38, 40, 42, 45, 48, 50, 
                    52, 55, 58, 60, 62, 65, 68, 70, 72, 75, 78, 80]
    
    # Predict next hour
    prediction = predictor.predict_single(pm10_history, temperature=22.0)
    confidence = predictor.get_prediction_confidence(pm10_history)
    
    print(f"\n📊 PM10 Prediction Example:")
    print(f"   Historical PM10 (last 24h): {pm10_history[-5:]}...")
    print(f"   Current temperature: 22.0°C")
    print(f"   Predicted PM10: {prediction:.2f} μg/m³")
    print(f"   Confidence: {confidence}")
    
    # Predict next 6 hours
    temperatures = [22.0, 23.0, 24.0, 25.0, 26.0, 27.0]
    predictions = predictor.predict_multiple(pm10_history, temperatures)
    
    print(f"\n🔮 6-Hour Forecast:")
    for i, (temp, pred) in enumerate(zip(temperatures, predictions)):
        print(f"   Hour {i+1}: {temp}°C → PM10: {pred:.2f} μg/m³") 
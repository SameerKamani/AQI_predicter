import joblib
import pandas as pd

# Load the trained model and scaler
MODEL_FILE = 'models/aqi_model.pkl'
SCALER_FILE = 'models/scaler.pkl'
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

def make_prediction(input_data):
    """
    Makes a prediction on a single data point.
    
    Args:
        input_data (dict): A dictionary containing all the feature values.
        
    Returns:
        float: The predicted pm25 value.
    """
    # Convert dictionary to DataFrame
    df = pd.DataFrame([input_data])
    
    # Ensure columns are in the correct order
    df = df[scaler.feature_names_in_]
    
    # Scale the features
    scaled_data = scaler.transform(df)
    
    # Make prediction
    prediction = model.predict(scaled_data)
    
    return float(prediction[0])

if __name__ == '__main__':
    # Example usage:
    # This is a placeholder. In a real scenario, you would get this
    # data from an API request or another source.
    sample_data = {
        'pm10': 15.0,
        'o3': 40.0,
        'no2': 20.0,
        'temperature_2m': 25.0,
        'relative_humidity_2m': 50.0,
        'precipitation': 0.0,
        'wind_speed_10m': 10.0,
        'hour': 14,
        'day_of_week': 3,
        'month': 7,
        'pm25_lag_1': 12.5,
        'pm25_lag_24': 11.0
    }
    
    prediction = make_prediction(sample_data)
    print(f"Predicted PM2.5: {prediction:.4f}") 
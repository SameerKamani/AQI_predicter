from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import joblib
import pandas as pd

# --- 1. Load Model and Scaler ---
MODEL_FILE = 'models/aqi_model.pkl'
SCALER_FILE = 'models/scaler.pkl'

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
print("Model and scaler loaded successfully.")

# Get the expected feature names from the scaler
feature_names = scaler.feature_names_in_

# --- 2. Define Pydantic Model for Input ---
# This class defines the structure of the input data for the API
class PredictionInput(BaseModel):
    pm10: float
    o3: float
    no2: float
    temperature_2m: float
    relative_humidity_2m: float
    precipitation: float
    wind_speed_10m: float
    hour: int
    day_of_week: int
    month: int
    pm25_lag_1: float
    pm25_lag_24: float

# --- 3. Create FastAPI App ---
app = FastAPI()

# --- 4. Define Prediction Endpoint ---
@app.post("/predict")
def predict(input_data: PredictionInput):
    # Convert Pydantic model to a dictionary
    data = input_data.dict()
    # Convert dictionary to a DataFrame
    df = pd.DataFrame([data])
    
    # Ensure DataFrame columns are in the same order as during training
    df = df[feature_names]

    # Scale the input data
    scaled_data = scaler.transform(df)
    
    # Make a prediction
    prediction = model.predict(scaled_data)
    
    # Return the prediction
    return {"predicted_pm25": float(prediction[0])}

# --- 5. Create Gradio Interface ---
def gradio_predict(*args):
    # Create a dictionary from the input arguments
    input_dict = {name: val for name, val in zip(feature_names, args)}
    # Convert to a DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Scale the features
    scaled_df = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(scaled_df)
    
    return round(float(prediction[0]), 2)

# Create a list of Gradio input components with better UI elements
inputs = [
    gr.Slider(minimum=0, maximum=100, value=40, label="pm10"),
    gr.Slider(minimum=0, maximum=0.1, value=0.04, label="o3"),
    gr.Slider(minimum=0, maximum=0.05, value=0.02, label="no2"),
    gr.Slider(minimum=-20, maximum=50, value=15, label="Temperature (°C)"),
    gr.Slider(minimum=0, maximum=100, value=65, label="Relative Humidity (%)"),
    gr.Slider(minimum=0, maximum=20, value=0.1, label="Precipitation (mm)"),
    gr.Slider(minimum=0, maximum=60, value=8, label="Wind Speed (km/h)"),
    gr.Slider(minimum=0, maximum=23, step=1, value=12, label="Hour of Day"),
    gr.Slider(minimum=0, maximum=6, step=1, value=3, label="Day of Week (0=Mon)"),
    gr.Slider(minimum=1, maximum=12, step=1, value=6, label="Month"),
    gr.Slider(minimum=0, maximum=70, value=10, label="PM2.5 (1 hour ago)"),
    gr.Slider(minimum=0, maximum=70, value=10, label="PM2.5 (24 hours ago)")
]

outputs = gr.Number(label="Predicted PM2.5")

# Create the Gradio Interface
interface = gr.Interface(
    fn=gradio_predict,
    inputs=inputs,
    outputs=outputs,
    title="AQI PM2.5 Prediction",
    description="Enter the following features to predict the PM2.5 value."
)

# --- 6. Mount Gradio App on FastAPI ---
app = gr.mount_gradio_app(app, interface, path="/ui")

# To run this app:
# 1. Make sure you have run preprocess.py and train.py to create the model and scaler.
# 2. Run the command: uvicorn src.app:app --reload
# 3. Open your browser to http://127.0.0.1:8000/ui 
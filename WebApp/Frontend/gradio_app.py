import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json

def get_aqi_color(aqi_value):
    """Get color based on AQI value"""
    if aqi_value <= 50:
        return "#00E400"  # Green - Good
    elif aqi_value <= 100:
        return "#D4AF37"  # Darker Yellow - Moderate (less light)
    elif aqi_value <= 150:
        return "#FF7E00"  # Orange - Unhealthy for Sensitive
    elif aqi_value <= 200:
        return "#FF0000"  # Red - Unhealthy
    elif aqi_value <= 300:
        return "#8F3F97"  # Purple - Very Unhealthy
    else:
        return "#7E0023"  # Maroon - Hazardous

def get_aqi_category(aqi_value):
    """Get AQI category description"""
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Moderate"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 200:
        return "Unhealthy"
    elif aqi_value <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def create_aqi_prediction_boxes(hd1, hd2, hd3):
    """Create three color-coded AQI prediction boxes with actual dates"""
    from datetime import datetime, timedelta
    
    # Get current date and calculate forecast dates
    current_date = datetime.now()
    forecast_dates = [
        (current_date + timedelta(days=i)).strftime("%b %d") 
        for i in range(1, 4)
    ]
    
    # Create HTML for the three prediction boxes
    html_content = f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0;">
        <div style="
            background: {get_aqi_color(hd1)};
            color: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            border: 3px solid {get_aqi_color(hd1)};
            transition: transform 0.3s ease;
        " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
            <h3 style="margin: 0 0 10px 0; font-size: 18px; font-weight: 600;">{forecast_dates[0]}</h3>
            <div style="font-size: 32px; font-weight: bold; margin: 10px 0;">{hd1:.1f}</div>
            <div style="font-size: 14px; opacity: 0.9;">{get_aqi_category(hd1)}</div>
        </div>
        
        <div style="
            background: {get_aqi_color(hd2)};
            color: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            border: 3px solid {get_aqi_color(hd2)};
            transition: transform 0.3s ease;
        " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
            <h3 style="margin: 0 0 10px 0; font-size: 18px; font-weight: 600;">{forecast_dates[1]}</h3>
            <div style="font-size: 32px; font-weight: bold; margin: 10px 0;">{hd2:.1f}</div>
            <div style="font-size: 14px; opacity: 0.9;">{get_aqi_category(hd2)}</div>
        </div>
        
        <div style="
            background: {get_aqi_color(hd3)};
            color: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            border: 3px solid {get_aqi_color(hd3)};
            transition: transform 0.3s ease;
        " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
            <h3 style="margin: 0 0 10px 0; font-size: 18px; font-weight: 600;">{forecast_dates[2]}</h3>
            <div style="font-size: 32px; font-weight: bold; margin: 10px 0;">{hd3:.1f}</div>
            <div style="font-size: 14px; opacity: 0.9;">{get_aqi_category(hd3)}</div>
        </div>
    </div>
    """
    return html_content

def create_aqi_chart(hd1, hd2, hd3):
    """Create the AQI trend chart with real historical data - always fetches latest"""
    try:
        import pandas as pd
        from pathlib import Path
        from datetime import datetime, timedelta
        
        # Always try to fetch the most recent data from API first
        try:
            import requests
            response = requests.get("http://localhost:8000/series/last30")
            if response.status_code == 200:
                api_data = response.json()
                if api_data.get("points"):
                    # Use API data as primary source
                    api_dates = [pd.to_datetime(point["event_timestamp"]) for point in api_data["points"]]
                    api_aqi = [point["aqi_daily"] for point in api_data["points"] if point["aqi_daily"] is not None]
                    if api_dates and api_aqi:
                        dates = pd.to_datetime(api_dates)
                        historical_aqi = np.array(api_aqi)
                        print(f" Using API data: {len(dates)} points from {dates.min()} to {dates.max()}")
                    else:
                        raise Exception("No valid API data")
                else:
                    raise Exception("No points in API response")
            else:
                raise Exception(f"API request failed: {response.status_code}")
        except Exception as api_error:
            print(f"API fetch failed: {api_error}, falling back to CSV")
            
            # Fallback to CSV file (updated by data pipeline)
            features_path = Path(__file__).parent.parent.parent / "Data" / "feature_store" / "karachi_daily_features.csv"
            
            if features_path.exists():
                df = pd.read_csv(features_path)
                df = df.sort_values("event_timestamp").tail(30)
                
                dates = pd.to_datetime(df["event_timestamp"])
                # Use aqi_daily if available, otherwise fall back to AQI
                if "aqi_daily" in df.columns:
                    historical_aqi = df["aqi_daily"].values
                else:
                    historical_aqi = df["AQI"].values
                print(f" Using CSV data: {len(dates)} points from {dates.min()} to {dates.max()}")
            else:
                # Generate dummy data if nothing works
                dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
                historical_aqi = np.full(30, 70)
                print(" No data sources available, using dummy data")
        
        # Ensure we have exactly 30 data points
        if len(dates) > 30:
            dates = dates[-30:]
            historical_aqi = historical_aqi[-30:]
        elif len(dates) < 30:
            # Pad with dummy data if we have less than 30 points
            missing_days = 30 - len(dates)
            if missing_days > 0:
                last_date = dates[-1] if len(dates) > 0 else datetime.now()
                dummy_dates = pd.date_range(start=last_date + timedelta(days=1), periods=missing_days, freq='D')
                dummy_aqi = np.full(missing_days, historical_aqi[-1] if len(historical_aqi) > 0 else 70)
                
                dates = np.concatenate([dates, dummy_dates])
                historical_aqi = np.concatenate([historical_aqi, dummy_aqi])
                
    except Exception as e:
        print(f" Chart creation error: {e}")
        # Ultimate fallback
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        historical_aqi = np.full(30, 70)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=historical_aqi,
        mode='lines',
        name='AQI (Past 30 Days)',
        line=dict(color='#2E86AB', width=3),
        hovertemplate='Date: %{x}<br>AQI: %{y:.1f}<extra></extra>'
    ))
    
    # Calculate actual forecast dates
    current_date = datetime.now()
    forecast_dates = [current_date + timedelta(days=i) for i in range(1, 4)]
    forecast_values = [hd1, hd2, hd3]
    forecast_colors = ['#00E400', '#FF7E00', '#FF0000']
    
    for i, (date, value, color) in enumerate(zip(forecast_dates, forecast_values, forecast_colors)):
        fig.add_trace(go.Scatter(
            x=[date],
            y=[value],
            mode='markers',
            name=f'{date.strftime("%b %d")} Forecast',
            marker=dict(color=color, size=12, symbol='diamond'),
            hovertemplate=f'{date.strftime("%B %d, %Y")}<br>AQI: {value:.1f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Last 30 Days AQI with 3-Day Forecast',
        xaxis_title='Date',
        yaxis_title='AQI',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

def get_actual_predictions():
    """Get actual predictions from your backend API"""
    try:
        # Call your actual prediction endpoint
        response = requests.get("http://localhost:8000/predict")
        if response.status_code == 200:
            data = response.json()
            # Extract blend predictions (ensemble)
            blend = data.get('blend', {})
            hd1 = blend.get('hd1', 60.0)
            hd2 = blend.get('hd2', 70.0)
            hd3 = blend.get('hd3', 80.0)
            return hd1, hd2, hd3
        else:
            # Fallback to sample data if API fails
            return 59.89, 69.95, 78.72
    except:
        # Fallback to sample data if API is unreachable
        return 59.89, 69.95, 78.72

def predict_aqi(model_selection):
    """Get AQI predictions based on model selection"""
    try:
        # Call your actual prediction endpoint
        response = requests.get("http://localhost:8000/predict")
        if response.status_code == 200:
            data = response.json()
            
            if model_selection == "Ensemble (Best)":
                blend = data.get('blend', {})
                return blend.get('hd1', 60.0), blend.get('hd2', 70.0), blend.get('hd3', 80.0)
            elif model_selection == "LightGBM":
                lightgbm = data.get('lightgbm', {})
                return lightgbm.get('hd1', 67.0), lightgbm.get('hd2', 70.0), lightgbm.get('hd3', 83.0)
            elif model_selection == "HGBR":
                hgbr = data.get('hgbr', {})
                return hgbr.get('hd1', 59.0), hgbr.get('hd2', 64.0), hgbr.get('hd3', 65.0)
            elif model_selection == "Linear":
                linear = data.get('linear', {})
                return linear.get('hd1', 70.0), linear.get('hd2', 76.0), linear.get('hd3', 78.0)
            elif model_selection == "Random Forest":
                randomforest = data.get('randomforest', {})
                return randomforest.get('hd1', 65.0), randomforest.get('hd2', 70.0), randomforest.get('hd3', 75.0)
            else:
                # Default fallback
                return 65.00, 70.00, 75.00
        else:
            # Fallback to sample data if API fails
            return 59.89, 69.95, 78.72
    except:
        # Fallback to sample data if API is unreachable
        return 59.89, 69.95, 78.72

def update_predictions(model_selection):
    """Update the interface with new predictions"""
    hd1, hd2, hd3 = predict_aqi(model_selection)
    
    # Create prediction boxes
    prediction_boxes = create_aqi_prediction_boxes(hd1, hd2, hd3)
    
    # Create chart
    chart = create_aqi_chart(hd1, hd2, hd3)
    
    return prediction_boxes, chart


def trigger_realtime_update():
    """Trigger real-time update and return status"""
    try:
        import subprocess
        import sys
        from pathlib import Path
        
        # Get the path to data_fetch.py relative to the project root
        project_root = Path(__file__).parent.parent.parent
        data_fetch_path = project_root / "Data_Collection" / "data_fetch.py"
        
        # Check if data_fetch.py exists
        if not data_fetch_path.exists():
            return f"**Error:** Data pipeline script not found at {data_fetch_path}\n\nPlease ensure Data_Collection/data_fetch.py exists."
        
        status_message = "🔄 **Starting data pipeline update...**\n\n"
        
        # Step 1: Run data_fetch.py to update CSV data
        try:
            status_message += "**Step 1: Fetching latest weather data...**\n"
            result = subprocess.run(
                [sys.executable, str(data_fetch_path)],
                capture_output=True,
                text=True,
                cwd=str(project_root),
                timeout=120  # 2 minute timeout
            )
            
            if result.returncode == 0:
                status_message += " Weather data fetched successfully!\n\n"
                # Extract key info from output
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if "Features updated successfully" in line:
                        status_message += f"📈 **Data Status:** {line.strip()}\n"
                    elif "Total records:" in line:
                        status_message += f"📊 **Records:** {line.strip()}\n"
            else:
                status_message += f"❌ **Data fetch failed:** {result.stderr}\n\n"
                return status_message
                
        except subprocess.TimeoutExpired:
            status_message += " **Data fetch timed out** (took longer than 2 minutes)\n\n"
            return status_message
        except Exception as e:
            status_message += f" **Data fetch error:** {str(e)}\n\n"
            return status_message
        
        # Step 2: Call backend API to trigger Feast update and predictions
        try:
            status_message += "🤖 **Step 2: Updating ML features and predictions...**\n"
            response = requests.post("http://localhost:8000/update/realtime")
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('status') == 'skipped':
                    status_message += f"ℹ️ **{result['message']}**\n\n"
                    status_message += f"**Timestamp:** {result['timestamp']}\n"
                    status_message += f"**Next Update:** {result['next_update']}\n\n"
                    status_message += "💡 **Data pipeline complete!** Your data is now current."
                else:
                    status_message += f"✅ **ML update successful!**\n\n"
                    status_message += f"**Timestamp:** {result['timestamp']}\n"
                    status_message += f"**Next Update:** {result['next_update']}\n\n"
                    status_message += "🎉 **Full data pipeline complete!** Features and predictions updated."
                    
                    # Wait for Feast materialization to complete
                    import time
                    time.sleep(3)
            else:
                status_message += f" **ML update failed:** {response.text}\n\n"
                status_message += " **Partial success:** Data was updated but ML features failed."
        
        except Exception as e:
            status_message += f" **ML update error:** {str(e)}\n\n"
            status_message += " **Partial success:** Data was updated but ML features failed."
        
        return status_message
        
    except Exception as e:
        return f" **Critical Error:** {str(e)}"

# Create the Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Karachi AQI Forecast",
    css="""
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .gr-button {
            border-radius: 25px !important;
            font-weight: bold !important;
        }
        .gr-markdown h1 {
            text-align: center !important;
            color: #2E86AB !important;
            margin-bottom: 30px !important;
            font-size: 3.5em !important;
            font-weight: bold !important;
            display: block !important;
            width: 100% !important;
        }
        .gr-markdown h2 {
            text-align: center !important;
            color: #A23B72 !important;
            border-bottom: 2px solid #F18F01 !important;
            padding-bottom: 10px !important;
            font-size: 2.2em !important;
            font-weight: bold !important;
            display: block !important;
            width: 100% !important;
        }
        .gr-markdown {
            text-align: center !important;
        }
    """
) as demo:
    
    # Header
    gr.Markdown("# 🌬️ Karachi Air Quality Index Forecast")
    
    # AQI Prediction Boxes
    gr.Markdown("## 📊 3-Day AQI Forecast")
    prediction_boxes = gr.HTML(label="AQI Predictions")
    
    # AQI Chart
    gr.Markdown("## 📈 Historical Trends & Forecast")
    aqi_chart = gr.Plot(label="AQI Trends")
    
    # Model Selection (moved to bottom)
    gr.Markdown("## 🤖 Select Prediction Model")
    
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=["Ensemble (Best)", "LightGBM", "HGBR", "Linear", "Random Forest"],
            value="Ensemble (Best)",
            label="Choose your prediction model:",
            info="Ensemble combines all models for best accuracy"
        )
    
    with gr.Column(scale=1, min_width=300):
        realtime_update_btn = gr.Button("🚀 Update Data Pipeline", variant="secondary", size="lg")
        refresh_btn = gr.Button("🔄 Refresh Predictions", variant="primary", size="lg")
    
    gr.Markdown("*💡 **Update Data Pipeline**: Fetches latest weather data and updates ML features automatically*")
    
    with gr.Row():
        update_status = gr.Markdown("")
    
    # Footer
    gr.Markdown("---")
    gr.Markdown("*Data updated hourly • Models retrained daily • Powered by ML Ensemble*")
    
    # Event handlers
    def on_model_change(model_selection):
        return update_predictions(model_selection)
    
    def on_refresh():
        return update_predictions("Ensemble (Best)")
    
    # Connect events
    model_dropdown.change(
        fn=on_model_change,
        inputs=[model_dropdown],
        outputs=[prediction_boxes, aqi_chart]
    )
    
    refresh_btn.click(
        fn=on_refresh,
        outputs=[prediction_boxes, aqi_chart]
    )
    
    realtime_update_btn.click(
        fn=trigger_realtime_update,
        outputs=[update_status]
    ).then(
        fn=lambda: update_predictions("Ensemble (Best)"),
        outputs=[prediction_boxes, aqi_chart]
    )
    
    # Initialize with default values immediately
    initial_hd1, initial_hd2, initial_hd3 = get_actual_predictions()
    initial_boxes = create_aqi_prediction_boxes(initial_hd1, initial_hd2, initial_hd3)
    initial_chart = create_aqi_chart(initial_hd1, initial_hd2, initial_hd3)
    
    # Set initial values
    prediction_boxes.value = initial_boxes
    aqi_chart.value = initial_chart
    
    # Auto-refresh chart every 5 minutes to ensure latest data
    demo.load(lambda: update_predictions("Ensemble (Best)"), outputs=[prediction_boxes, aqi_chart])

# Launch the app
if __name__ == "__main__":
    demo.launch()

# Ensure demo is available for import
__all__ = ['demo']



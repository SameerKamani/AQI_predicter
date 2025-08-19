import os
import asyncio
import subprocess
from typing import Any, Dict, Optional
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone
import threading
import time

import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from config file
try:
    from dotenv import load_dotenv
    # Load from .env in the project root
    load_dotenv('.env')
    print("Environment variables loaded from .env")
except ImportError:
    print("python-dotenv not installed, using system environment variables")
    print("Install with: pip install python-dotenv")
except Exception as e:
    print(f"Error loading config.env: {e}, using system environment variables")

try:
    import gradio as gr  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    gr = None  # type: ignore
try:
    import plotly.graph_objects as go  # type: ignore
except Exception:  # pragma: no cover - optional
    go = None  # type: ignore

try:
    from . import feast_client  # type: ignore
    from . import model_loader  # type: ignore
except ImportError:
    # Fallback for when running as script
    import feast_client  # type: ignore
    import model_loader  # type: ignore


def _get_env_path(key: str, default: str) -> str:
    value = os.getenv(key, default)
    return value


# Configure paths via environment variables with sensible defaults
_ROOT = str(Path(__file__).resolve().parents[3])  # .../repo
FEAST_REPO_PATH = _get_env_path("FEAST_REPO_PATH", os.path.join(_ROOT, "feature_repo"))
FEATURES_PARQUET = _get_env_path("FEATURES_PARQUET", os.path.join(_ROOT, "Data", "feature_store", "karachi_daily_features.parquet"))

# Force the correct registry path - ignore environment variable override
REGISTRY_DIR = os.path.join(_ROOT, "Models", "Models", "registry")

# Ensure absolute paths
FEAST_REPO_PATH = os.path.abspath(FEAST_REPO_PATH)
FEATURES_PARQUET = os.path.abspath(FEATURES_PARQUET)
REGISTRY_DIR = os.path.abspath(REGISTRY_DIR)

# Debug: Print the actual paths being used
print(f"DEBUG: Current working directory: {os.getcwd()}")
print(f"DEBUG: _ROOT: {_ROOT}")
print(f"DEBUG: FEAST_REPO_PATH: {FEAST_REPO_PATH}")
print(f"DEBUG: FEATURES_PARQUET: {FEATURES_PARQUET}")
print(f"DEBUG: REGISTRY_DIR: {REGISTRY_DIR}")
print(f"DEBUG: REGISTRY_DIR exists: {os.path.exists(REGISTRY_DIR)}")
print(f"DEBUG: REGISTRY_DIR absolute: {os.path.abspath(REGISTRY_DIR)}")

# Real-time update configuration
UPDATE_INTERVAL_MINUTES = int(os.getenv("UPDATE_INTERVAL_MINUTES", "30"))  # Update every 30 minutes by default
REALTIME_UPDATE_ENABLED = os.getenv("REALTIME_UPDATE_ENABLED", "true").lower() == "true"


def check_if_update_needed():
    """Check if real-time update is actually needed"""
    try:
        import pandas as pd
        from datetime import datetime, timezone
        
        # Try CSV first (updated by data pipeline), then fallback to parquet
        csv_path = Path(FEATURES_PARQUET.replace('.parquet', '.csv'))
        if csv_path.exists():
            # Read from updated CSV file
            df = pd.read_csv(csv_path)
            print(f"Checking update status from CSV: {len(df)} records")
        else:
            # Fallback to parquet
            features_path = Path(FEATURES_PARQUET)
            if not features_path.exists():
                print("Feature store not found - update needed")
                return True
            
            df = pd.read_parquet(features_path)
            print(f"Checking update status from parquet: {len(df)} records")
        
        if df.empty:
            print("Feature store is empty - update needed")
            return True
        
        # Get latest timestamp from feature store
        latest_feature_date = pd.to_datetime(df['event_timestamp'].max()).date()
        current_date = datetime.now(timezone.utc).date()
        
        print(f"Latest feature date: {latest_feature_date}")
        print(f"Current date: {current_date}")
        
        # Check if we have today's data
        if latest_feature_date >= current_date:
            print("Features are up-to-date - no update needed")
            return False
        else:
            print(f"Features are outdated (last: {latest_feature_date}, current: {current_date}) - update needed")
            return True
            
    except Exception as e:
        print(f"Error checking update status: {e}")
        return True  # Default to updating if check fails


def run_feast_update():
    """Update Feast feature store after data pipeline"""
    try:
        print("Updating Feast...")
        feast_result = subprocess.run([
            "feast", "apply"
        ], capture_output=True, text=True, cwd=FEAST_REPO_PATH)
        
        if feast_result.returncode == 0:
            print("Feast applied successfully")
        else:
            print(f"Feast apply failed: {feast_result.stderr}")
            return False
        
        # Materialize Feast with current date range
        try:
            # Try CSV first (updated by data pipeline), then fallback to parquet
            csv_path = FEATURES_PARQUET.replace('.parquet', '.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                print(f"Using CSV for Feast materialization: {len(df)} records")
            else:
                df = pd.read_parquet(FEATURES_PARQUET)
                print(f"Using parquet for Feast materialization: {len(df)} records")
            
            if not df.empty:
                latest_date = pd.to_datetime(df['event_timestamp'].max()).date()
                current_date = datetime.now(timezone.utc).date()
                start_datetime = f"{latest_date.strftime('%Y-%m-%d')}T00:00:00Z"
                end_datetime = f"{current_date.strftime('%Y-%m-%d')}T23:59:59Z"
                print(f"Materializing from {start_datetime} to {end_datetime}...")
                materialize_result = subprocess.run([
                    "feast", "materialize", start_datetime, end_datetime
                ], capture_output=True, text=True, cwd=FEAST_REPO_PATH)
                
                if materialize_result.returncode == 0:
                    print("Feast materialized successfully")
                    
                    # Add a small delay to ensure parquet file is fully written
                    print("Waiting for data consistency...")
                    time.sleep(2)
                    
                    return True
                else:
                    print(f"Feast materialize failed: {materialize_result.stderr}")
                    return False
        except Exception as e:
            print(f"Could not materialize Feast: {e}")
            return False
            
    except Exception as e:
        print(f"Feast update error: {e}")
        return False

def run_feature_update():
    """Run the feature update pipeline"""
    try:
        print("Running data fetch and feature engineering pipeline...")
        
        # Run the data fetch script
        result = subprocess.run([
            "python", "Data_Collection/data_fetch.py"
        ], capture_output=True, text=True, cwd=_ROOT)
        
        if result.returncode == 0:
            print("Features updated successfully")
            
            # Now update Feast
            if run_feast_update():
                print("Feature and Feast update completed successfully")
                return True
            else:
                print("Data updated but Feast update failed")
                return False
        else:
            print(f"Feature update failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Feature update error: {e}")
        return False


def run_prediction_update():
    """Run the prediction update pipeline"""
    try:
        print("Generating real-time predictions...")
        
        # Run the prediction script
        result = subprocess.run([
            "python", "Models/predict_realtime.py"
        ], capture_output=True, text=True, cwd=_ROOT)
        
        if result.returncode == 0:
            print("Predictions generated successfully")
            return True
        else:
            print(f"Prediction generation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Prediction update error: {e}")
        return False


def background_worker():
    """Background worker that runs updates periodically"""
    while True:
        try:
            print(f"Background worker checking for updates at {datetime.now()}")
            
            # Check if update is needed
            if check_if_update_needed():
                print("Update needed - running pipeline...")
                
                # Run feature update (includes Feast update)
                if run_feature_update():
                    print("Feature and Feast update completed")
                    
                    # Wait a moment for data consistency, then run prediction update
                    print(" Waiting for data consistency before predictions...")
                    time.sleep(3)
                    
                    # Run prediction update
                    if run_prediction_update():
                        print(" Prediction update completed")
                    else:
                        print(" Prediction update failed")
                else:
                    print(" Feature update failed")
            else:
                print(" No update needed")
            
            # Wait for next check
            time.sleep(UPDATE_INTERVAL_MINUTES * 60)
            
        except Exception as e:
            print(f" Background worker error: {e}")
            time.sleep(60)  # Wait 1 minute before retrying


app = FastAPI(title="AQI Prediction Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    # Initialize model registry on startup for low-latency requests
    model_loader.initialize_registry(REGISTRY_DIR)
    
    # Start real-time update worker if enabled
    if REALTIME_UPDATE_ENABLED:
        print(f"  Starting real-time update worker (interval: {UPDATE_INTERVAL_MINUTES} minutes)")
        update_thread = threading.Thread(target=background_worker, daemon=True)
        update_thread.start()
        print(" Real-time update worker started successfully")
    else:
        print(" Real-time updates are disabled")


@app.get("/health")
def health() -> Dict[str, Any]:
    latest_ts: Optional[str] = None
    try:
        # Try CSV first (updated by data pipeline), then fallback to parquet
        csv_path = FEATURES_PARQUET.replace('.parquet', '.csv')
        if os.path.exists(csv_path):
            row = feast_client.get_latest_offline_row(csv_path)
            print(f"Health check using CSV: {csv_path}")
        else:
            row = feast_client.get_latest_offline_row(FEATURES_PARQUET)
            print(f"Health check using parquet: {FEATURES_PARQUET}")
        
        if row is not None and "event_timestamp" in row.index:
            latest_ts = str(row["event_timestamp"])
    except Exception:
        latest_ts = None

    artifacts = model_loader.current_artifacts_summary()
    return {
        "status": "ok",
        "latest_feature_timestamp": latest_ts,
        "artifacts": artifacts,
    }


@app.get("/features/latest")
def features_latest() -> Dict[str, Any]:
    try:
        # Try CSV first (updated by data pipeline), then fallback to parquet
        csv_path = FEATURES_PARQUET.replace('.parquet', '.csv')
        if os.path.exists(csv_path):
            row = feast_client.get_latest_features(FEAST_REPO_PATH, csv_path)
            print(f"Features/latest using CSV: {csv_path}")
        else:
            row = feast_client.get_latest_features(FEAST_REPO_PATH, FEATURES_PARQUET)
            print(f"Features/latest using parquet: {FEATURES_PARQUET}")
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to load features: {e}")
    if row is None:
        raise HTTPException(status_code=404, detail="No features available")
    # Convert a single-row Series to dict for JSON
    return {"features": {k: (None if pd.isna(v) else (str(v) if hasattr(v, "isoformat") else v)) for k, v in row.to_dict().items()}}


@app.get("/predict/latest")
def predict_latest() -> Dict[str, Any]:
    # Prefer online features via Feast; these now include full training columns
    # Try CSV first (updated by data pipeline), then fallback to parquet
    csv_path = FEATURES_PARQUET.replace('.parquet', '.csv')
    if os.path.exists(csv_path):
        row = feast_client.get_latest_features(FEAST_REPO_PATH, csv_path)
        print(f"Predict/latest using CSV: {csv_path}")
    else:
        row = feast_client.get_latest_features(FEAST_REPO_PATH, FEATURES_PARQUET)
        print(f" Predict/latest using parquet: {FEATURES_PARQUET}")
    
    if row is None:
        raise HTTPException(status_code=404, detail="No features available for prediction")
    try:
        pred = model_loader.predict_all_from_series(row)
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    return pred


@app.get("/series/last30")
def series_last30() -> Dict[str, Any]:
    try:
        # Try CSV first (updated by data pipeline), then fallback to parquet
        csv_path = FEATURES_PARQUET.replace('.parquet', '.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path).sort_values("event_timestamp")
            print(f"Using CSV data for /series/last30: {len(df)} records")
        else:
            df = pd.read_parquet(FEATURES_PARQUET).sort_values("event_timestamp")
            print(f"Using parquet data for /series/last30: {len(df)} records")
        
        tail = df.tail(30)
        series = [
            {
                "event_timestamp": str(ts),
                "aqi_daily": float(val) if pd.notna(val) else None,
            }
            for ts, val in zip(tail["event_timestamp"], tail["AQI"])  # CORRECT: Reading from "AQI" column
        ]
        return {"points": series}
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to load series: {e}")


@app.get("/predict")
def predict() -> Dict[str, Any]:
    """Alias for /predict/latest for convenience"""
    try:
        return predict_latest()  # reuse the API function directly
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


@app.post("/update/realtime")
def trigger_realtime_update(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Trigger real-time update manually - only if needed"""
    try:
        print(" Manual real-time update triggered")
        
        # Check if update is actually needed
        if not check_if_update_needed():
            return {
                "status": "skipped",
                "message": "No update needed - features are current",
                "timestamp": datetime.now().isoformat(),
                "next_update": (datetime.now() + timedelta(minutes=UPDATE_INTERVAL_MINUTES)).isoformat()
            }
        
        print("Update needed - starting pipeline...")
        
        # Add update tasks to background
        background_tasks.add_task(run_feature_update)
        background_tasks.add_task(run_prediction_update)
        
        return {
            "status": "success",
            "message": "Real-time update started",
            "timestamp": datetime.now().isoformat(),
            "next_update": (datetime.now() + timedelta(minutes=UPDATE_INTERVAL_MINUTES)).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger update: {e}")

@app.post("/update")
def update_system():
    """Update the entire system (features, Feast, predictions)"""
    try:
        print(" Starting system update...")
        
        # Step 1: Update features
        if not run_feature_update():
            raise HTTPException(status_code=500, detail="Feature update failed")
        
        # Step 2: Generate predictions
        if not run_prediction_update():
            print(" Warning: Predictions failed, but features were updated")
            # Don't fail the entire update if predictions fail
        
        print(" System update completed successfully")
        return {"status": "success", "message": "System updated successfully"}
        
    except Exception as e:
        print(f" System update failed: {e}")
        raise HTTPException(status_code=500, detail=f"System update failed: {str(e)}")


@app.get("/update/status")
def get_update_status() -> Dict[str, Any]:
    """Get real-time update status"""
    return {
        "realtime_updates_enabled": REALTIME_UPDATE_ENABLED,
        "update_interval_minutes": UPDATE_INTERVAL_MINUTES,
        "last_update_attempt": datetime.now().isoformat(),
        "next_scheduled_update": (datetime.now() + timedelta(minutes=UPDATE_INTERVAL_MINUTES)).isoformat()
    }


# Mount Gradio under /ui if available
if gr is not None:  # pragma: no cover - interactive only
    try:
        # Import from the correct relative path
        import sys
        sys.path.append(str(Path(__file__).resolve().parents[3]))  # Add project root to path
        from WebApp.Frontend.gradio_app import demo
        from gradio.routes import mount_gradio_app
        mount_gradio_app(app, demo, path="/ui")
        print(" Gradio app mounted successfully at /ui")
    except Exception as e:
        print(f"Failed to mount Gradio app: {e}")
        # Simple fallback interface
        with gr.Blocks(title="Karachi AQI Forecast") as fallback_demo:
            gr.Markdown("# Karachi AQI Forecast")
            gr.Markdown("### Frontend Gradio app failed to load. Please check the logs.")
            gr.Markdown("API endpoints are still available at `/predict` and `/health`")
        
        try:
            from gradio.routes import mount_gradio_app
            mount_gradio_app(app, fallback_demo, path="/ui")
        except Exception:
            app = gr.mount_gradio_app(app, fallback_demo, path="/ui")


# For local dev: uvicorn WebApp.Backend.app.main:app --reload --port 8000

if __name__ == "__main__":  # pragma: no cover
    import uvicorn  # type: ignore
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    # Run without reload when executed directly; use uvicorn CLI for reload
    uvicorn.run(app, host=host, port=port)



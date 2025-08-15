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
    print("✅ Environment variables loaded from .env")
except ImportError:
    print("⚠️ python-dotenv not installed, using system environment variables")
    print("💡 Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️ Error loading config.env: {e}, using system environment variables")

try:
    import gradio as gr  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    gr = None  # type: ignore
try:
    import plotly.graph_objects as go  # type: ignore
except Exception:  # pragma: no cover - optional
    go = None  # type: ignore

try:
    from WebApp.Backend.app import feast_client  # type: ignore
    from WebApp.Backend.app import model_loader  # type: ignore
except Exception:
    # Ensure repo root is on sys.path when running as a script
    _HERE = Path(__file__).resolve()
    _ROOT_FOR_PATH = str(_HERE.parents[3])
    if _ROOT_FOR_PATH not in sys.path:
        sys.path.insert(0, _ROOT_FOR_PATH)
    try:
        from WebApp.Backend.app import feast_client  # type: ignore
        from WebApp.Backend.app import model_loader  # type: ignore
    except ImportError:
        # Try relative imports if absolute imports fail
        from . import feast_client  # type: ignore
        from . import model_loader  # type: ignore


def _get_env_path(key: str, default: str) -> str:
    value = os.getenv(key, default)
    return value


# Configure paths via environment variables with sensible defaults
_ROOT = str(Path(__file__).resolve().parents[3])  # .../repo
FEAST_REPO_PATH = _get_env_path("FEAST_REPO_PATH", os.path.join(_ROOT, "feature_repo"))
FEATURES_PARQUET = _get_env_path("FEATURES_PARQUET", os.path.join(_ROOT, "Data", "feature_store", "karachi_daily_features.parquet"))
REGISTRY_DIR = _get_env_path("REGISTRY_DIR", os.path.join(_ROOT, "Models", "registry"))

# Real-time update configuration
UPDATE_INTERVAL_MINUTES = int(os.getenv("UPDATE_INTERVAL_MINUTES", "30"))  # Update every 30 minutes by default
REALTIME_UPDATE_ENABLED = os.getenv("REALTIME_UPDATE_ENABLED", "true").lower() == "true"


def check_if_update_needed():
    """Check if real-time update is actually needed"""
    try:
        import pandas as pd
        from datetime import datetime, timezone
        
        # Check feature store for latest data
        features_path = Path(FEATURES_PARQUET)
        if not features_path.exists():
            print("📁 Feature store not found - update needed")
            return True
        
        # Read latest feature data
        df = pd.read_parquet(features_path)
        if df.empty:
            print("📊 Feature store is empty - update needed")
            return True
        
        # Get latest timestamp from feature store
        latest_feature_date = pd.to_datetime(df['event_timestamp'].max()).date()
        current_date = datetime.now(timezone.utc).date()
        
        print(f"📅 Latest feature date: {latest_feature_date}")
        print(f"📅 Current date: {current_date}")
        
        # Check if we have today's data
        if latest_feature_date >= current_date:
            print("✅ Features are up-to-date - no update needed")
            return False
        else:
            print(f"🔄 Features are outdated (last: {latest_feature_date}, current: {current_date}) - update needed")
            return True
            
    except Exception as e:
        print(f"❌ Error checking update status: {e}")
        return True  # Default to updating if check fails


def run_feature_update():
    """Run feature update pipeline only if needed"""
    try:
        print("🔍 Checking if update is needed...")
        
        if not check_if_update_needed():
            print("✅ No update needed - features are current")
            return True
        
        # Find the gap and fill it properly
        import pandas as pd
        features_path = Path(FEATURES_PARQUET)
        
        if features_path.exists():
            df = pd.read_parquet(features_path)
            if not df.empty:
                latest_date = pd.to_datetime(df['event_timestamp'].max()).date()
                current_date = datetime.now(timezone.utc).date()
                
                # Fill the gap from latest_date + 1 to current_date
                # We need to fetch the missing days (Aug 13-15) plus some historical context for lags
                # But since we're using --append, we don't need to re-fetch existing data
                start_date = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d")  # Start from day after latest
                end_date = current_date.strftime("%Y-%m-%d")   # Up to today
                
                print(f"📅 Latest data: {latest_date}")
                print(f"📅 Current date: {current_date}")
                print(f"🔄 Filling gap from {start_date} to {end_date}...")
            else:
                # No existing data, fetch last 21 days to ensure enough data for features and lags
                start_date = (datetime.now(timezone.utc) - timedelta(days=21)).strftime("%Y-%m-%d")
                end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                print(f"🔄 No existing data, fetching from {start_date} to {end_date} (including lag days)...")
        else:
            # No existing file, fetch last 21 days to ensure enough data for features and lags
            start_date = (datetime.now(timezone.utc) - timedelta(days=21)).strftime("%Y-%m-%d")
            end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            print(f"🔄 No feature store found, fetching from {start_date} to {end_date} (including lag days)...")
        
        # Run feature pipeline with proper date range
        result = subprocess.run([
            "python", "Data_Collection/feature_store_pipeline.py",
            "--start", start_date,
            "--end", end_date,
            "--impute_short_gaps",
            "--min_hours_per_day", "16",
            "--append"
        ], capture_output=True, text=True, cwd=_ROOT)
        
        if result.returncode == 0:
            print("✅ Features updated successfully")
        else:
            print(f"❌ Feature update failed: {result.stderr}")
            return False
        
        # Update Feast
        print("🔄 Updating Feast...")
        feast_result = subprocess.run([
            "feast", "apply"
        ], capture_output=True, text=True, cwd=FEAST_REPO_PATH)
        
        if feast_result.returncode == 0:
            print("✅ Feast applied successfully")
        else:
            print(f"❌ Feast apply failed: {feast_result.stderr}")
            return False
        
        # Materialize Feast with proper date range
        # Convert start_date and end_date to proper Feast format
        start_datetime = f"{start_date}T00:00:00Z"
        end_datetime = f"{end_date}T23:59:59Z"
        print(f"🔄 Materializing from {start_datetime} to {end_datetime}...")
        materialize_result = subprocess.run([
            "feast", "materialize", start_datetime, end_datetime
        ], capture_output=True, text=True, cwd=FEAST_REPO_PATH)
        
        if materialize_result.returncode == 0:
            print("✅ Feast materialized successfully")
        else:
            print(f"❌ Feast materialize failed: {materialize_result.stderr}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error updating features: {e}")
        return False


def run_prediction_update():
    """Run real-time prediction update"""
    try:
        print("🔮 Generating real-time predictions...")
        
        result = subprocess.run([
            "python", "Models/predict_realtime.py"
        ], capture_output=True, text=True, cwd=_ROOT)
        
        if result.returncode == 0:
            print("✅ Real-time predictions generated successfully")
            return True
        else:
            print(f"❌ Prediction generation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        return False


def realtime_update_worker():
    """Background worker for smart real-time updates"""
    while True:
        try:
            print(f"🕐 Real-time update cycle starting at {datetime.now()}")
            
            # Check if update is needed first
            if check_if_update_needed():
                print("🔄 Update needed - running full pipeline...")
                # Update features
                if run_feature_update():
                    # Generate predictions
                    run_prediction_update()
            else:
                print("✅ No update needed - skipping this cycle")
            
            # Wait for next cycle
            time.sleep(UPDATE_INTERVAL_MINUTES * 60)
        except Exception as e:
            print(f"❌ Error in real-time update worker: {e}")
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
        print(f"🚀 Starting real-time update worker (interval: {UPDATE_INTERVAL_MINUTES} minutes)")
        update_thread = threading.Thread(target=realtime_update_worker, daemon=True)
        update_thread.start()
        print("✅ Real-time update worker started successfully")
    else:
        print("⚠️ Real-time updates are disabled")


@app.get("/health")
def health() -> Dict[str, Any]:
    latest_ts: Optional[str] = None
    try:
        row = feast_client.get_latest_offline_row(FEATURES_PARQUET)
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
        row = feast_client.get_latest_features(FEAST_REPO_PATH, FEATURES_PARQUET)
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to load features: {e}")
    if row is None:
        raise HTTPException(status_code=404, detail="No features available")
    # Convert a single-row Series to dict for JSON
    return {"features": {k: (None if pd.isna(v) else (str(v) if hasattr(v, "isoformat") else v)) for k, v in row.to_dict().items()}}


@app.get("/predict/latest")
def predict_latest() -> Dict[str, Any]:
    # Prefer online features via Feast; these now include full training columns
    row = feast_client.get_latest_features(FEAST_REPO_PATH, FEATURES_PARQUET)
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
        df = pd.read_parquet(FEATURES_PARQUET).sort_values("event_timestamp")
        tail = df.tail(30)
        series = [
            {
                "event_timestamp": str(ts),
                "aqi_daily": float(val) if pd.notna(val) else None,
            }
            for ts, val in zip(tail["event_timestamp"], tail["aqi_daily"])
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
        print("🚀 Manual real-time update triggered")
        
        # Check if update is actually needed
        if not check_if_update_needed():
            return {
                "status": "skipped",
                "message": "No update needed - features are current",
                "timestamp": datetime.now().isoformat(),
                "next_update": (datetime.now() + timedelta(minutes=UPDATE_INTERVAL_MINUTES)).isoformat()
            }
        
        print("🔄 Update needed - starting pipeline...")
        
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
        # Simple import from the correct path
        from WebApp.Frontend.gradio_app import demo
        from gradio.routes import mount_gradio_app
        mount_gradio_app(app, demo, path="/ui")
        print("✅ Gradio app mounted successfully at /ui")
    except Exception as e:
        print(f"Failed to mount Gradio app: {e}")
        # Simple fallback interface
        with gr.Blocks(title="Karachi AQI Forecast") as fallback_demo:
            gr.Markdown("# 🌬️ Karachi AQI Forecast")
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



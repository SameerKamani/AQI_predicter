import os
from typing import Any, Dict, Optional
from pathlib import Path
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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
    from WebApp.Backend.app import feast_client  # type: ignore
    from WebApp.Backend.app import model_loader  # type: ignore


def _get_env_path(key: str, default: str) -> str:
    value = os.getenv(key, default)
    return value


# Configure paths via environment variables with sensible defaults
_ROOT = str(Path(__file__).resolve().parents[3])  # .../repo
FEAST_REPO_PATH = _get_env_path("FEAST_REPO_PATH", os.path.join(_ROOT, "feature_repo"))
FEATURES_PARQUET = _get_env_path("FEATURES_PARQUET", os.path.join(_ROOT, "Data", "feature_store", "karachi_daily_features.parquet"))
REGISTRY_DIR = _get_env_path("REGISTRY_DIR", os.path.join(_ROOT, "Models", "registry"))


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


def _build_gradio_ui():  # pragma: no cover - UI construction
    if gr is None:
        return None

    def _ui_predict() -> Dict[str, Any]:
        try:
            return predict_latest()  # reuse the API function directly
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    with gr.Blocks(title="Karachi AQI Forecast") as demo:
        gr.Markdown("""
        # Karachi AQI Forecast
        Latest blended predictions for the next 3 days (hd1..hd3)
        """)
        with gr.Row():
            hd1 = gr.Number(label="hd1 (AQI)")
            hd2 = gr.Number(label="hd2 (AQI)")
            hd3 = gr.Number(label="hd3 (AQI)")
        status = gr.JSON(label="Details (per-model and blend)")
        alert = gr.HTML("", label="Alert")
        chart = gr.Plot(label="Last 30 days AQI with 3-day forecast")
        refresh = gr.Button("Refresh predictions")

        def _make_chart(res: Dict[str, Any]):
            if go is None:
                return None
            # Load last 30 days from offline parquet
            try:
                df = pd.read_parquet(FEATURES_PARQUET).sort_values("event_timestamp")
                df = df.tail(30)
            except Exception:
                return None
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["event_timestamp"], y=df["aqi_daily"], mode="lines+markers", name="AQI (past)", line=dict(color="#3b82f6")))
            # Forecast points: next 1..3 days placed at last date + d
            try:
                last_ts = pd.to_datetime(str(df["event_timestamp"].iloc[-1]))
                blend = res.get("blend", {}) if isinstance(res, dict) else {}
                preds = [
                    (last_ts + pd.Timedelta(days=1), float(blend.get("hd1", float("nan"))), "hd1"),
                    (last_ts + pd.Timedelta(days=2), float(blend.get("hd2", float("nan"))), "hd2"),
                    (last_ts + pd.Timedelta(days=3), float(blend.get("hd3", float("nan"))), "hd3"),
                ]
                colors = {"hd1": "#22c55e", "hd2": "#eab308", "hd3": "#ef4444"}
                for ts, val, name in preds:
                    fig.add_trace(go.Scatter(x=[ts], y=[val], mode="markers", marker=dict(size=10, color=colors[name]), name=f"forecast {name}"))
            except Exception:
                pass
            fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=400)
            return fig

        def _on_click() -> tuple[float, float, float, Dict[str, Any], str, Any]:
            res = _ui_predict()
            blend = res.get("blend", {}) if isinstance(res, dict) else {}
            fig = _make_chart(res)
            hd1_val = blend.get("hd1")
            banner = ""
            try:
                if hd1_val is not None and float(hd1_val) >= 200.0:
                    banner = (
                        "<div style='padding:10px;background:#fee2e2;border:1px solid #ef4444;color:#991b1b;border-radius:6px'>"
                        "<strong>Hazardous AQI alert:</strong> Next-day forecast (hd1) is >= 200. Limit outdoor exposure." 
                        "</div>"
                    )
            except Exception:
                banner = ""
            return (
                float(blend.get("hd1", float("nan"))),
                float(blend.get("hd2", float("nan"))),
                float(blend.get("hd3", float("nan"))),
                res,
                banner,
                fig,
            )

        refresh.click(fn=_on_click, outputs=[hd1, hd2, hd3, status, alert, chart])
        # Auto-run once on load
        demo.load(fn=_on_click, outputs=[hd1, hd2, hd3, status, alert, chart])
    return demo


# Mount Gradio under /ui if available
if gr is not None:  # pragma: no cover - interactive only
    demo_app = _build_gradio_ui()
    if demo_app is not None:
        try:
            from gradio.routes import mount_gradio_app  # type: ignore

            mount_gradio_app(app, demo_app, path="/ui")
        except Exception:
            # Older gradio versions
            app = gr.mount_gradio_app(app, demo_app, path="/ui")  # type: ignore


# For local dev: uvicorn WebApp.Backend.app.main:app --reload --port 8000

if __name__ == "__main__":  # pragma: no cover
    import uvicorn  # type: ignore
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    # Run without reload when executed directly; use uvicorn CLI for reload
    uvicorn.run(app, host=host, port=port)



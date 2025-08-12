from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # for typing only
    import lightgbm as lgb  # type: ignore
try:  # runtime import
    import lightgbm as _lgb_runtime  # type: ignore
except Exception:  # pragma: no cover - optional
    _lgb_runtime = None  # type: ignore

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover - optional
    joblib = None  # type: ignore


_REGISTRY_DIR: Optional[str] = None
_BOOSTERS: Dict[str, Optional[object]] = {"hd1": None, "hd2": None, "hd3": None}
_LINEAR: Dict[str, Optional[object]] = {"hd1": None, "hd2": None, "hd3": None}
_LINEAR_FEATS: Dict[str, List[str]] = {"hd1": [], "hd2": [], "hd3": []}
_HGBR: Dict[str, Optional[object]] = {"hd1": None, "hd2": None, "hd3": None}
_HGBR_FEATS: Dict[str, List[str]] = {"hd1": [], "hd2": [], "hd3": []}
_BLEND_WEIGHTS: Dict[str, Dict[str, float]] = {}


def _latest(path_glob: str) -> Optional[str]:
    files = glob.glob(path_glob)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def initialize_registry(registry_dir: str) -> None:
    global _REGISTRY_DIR, _BLEND_WEIGHTS
    _REGISTRY_DIR = registry_dir

    # Load LightGBM boosters (optional)
    if _lgb_runtime is not None:
        for hz in ["hd1", "hd2", "hd3"]:
            path = _latest(os.path.join(_REGISTRY_DIR, f"lgb_{hz}_*.txt"))
            if path:
                try:
                    _BOOSTERS[hz] = _lgb_runtime.Booster(model_file=path)
                except Exception:
                    _BOOSTERS[hz] = None

    # Load linear pipelines with feature lists
    if joblib is not None:
        for hz in ["hd1", "hd2", "hd3"]:
            path = _latest(os.path.join(_REGISTRY_DIR, f"linear_{hz}_*.joblib"))
            if path:
                try:
                    payload = joblib.load(path)
                    _LINEAR[hz] = payload.get("model")
                    _LINEAR_FEATS[hz] = payload.get("features", [])
                except Exception:
                    _LINEAR[hz] = None
                    _LINEAR_FEATS[hz] = []

    # Load HGBR (sklearn) models with feature lists
    if joblib is not None:
        for hz in ["hd1", "hd2", "hd3"]:
            path = _latest(os.path.join(_REGISTRY_DIR, f"hgb_{hz}_*.joblib"))
            if path:
                try:
                    payload = joblib.load(path)
                    _HGBR[hz] = payload.get("model")
                    _HGBR_FEATS[hz] = payload.get("features", [])
                except Exception:
                    _HGBR[hz] = None
                    _HGBR_FEATS[hz] = []

    # Load blend weights (latest)
    bw = _latest(os.path.join(_REGISTRY_DIR, "blend_weights_*.json"))
    if bw and os.path.exists(bw):
        try:
            with open(bw, "r", encoding="utf-8") as f:
                blob = json.load(f)
            _BLEND_WEIGHTS = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in blob.get("weights", {}).items()}
        except Exception:
            _BLEND_WEIGHTS = {}


def current_artifacts_summary() -> Dict[str, Dict[str, bool]]:
    return {
        "lightgbm": {hz: (_BOOSTERS[hz] is not None) for hz in ["hd1", "hd2", "hd3"]},
        "linear": {hz: (_LINEAR[hz] is not None) for hz in ["hd1", "hd2", "hd3"]},
        "hgbr": {hz: (_HGBR[hz] is not None) for hz in ["hd1", "hd2", "hd3"]},
        "blend_weights": {"present": bool(_BLEND_WEIGHTS)},
    }


def _to_frame(series: pd.Series) -> pd.DataFrame:
    # Represent the latest row as a 1-row DataFrame preserving column order
    df = pd.DataFrame([series.to_dict()])
    return df


def _coerce_numeric_impute_latest(df: pd.DataFrame, feats: List[str]) -> pd.DataFrame:
    if not feats:
        # No features available; return empty frame to signal caller to skip
        return pd.DataFrame()
    X = df[feats].copy()
    # to numeric
    for c in feats:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    # For single-row inference, prefer robust zero-fill to avoid nanmean warnings
    X = X.fillna(0.0)
    return X


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {
        "date",
        "event_timestamp",
        "created",
        "city",
        "karachi_id",
        "target_aqi_d1",
        "target_aqi_d2",
        "target_aqi_d3",
    }
    return [c for c in df.columns if c not in exclude]


def _align_to_feature_names(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Return a 1-row DataFrame with exactly feature_names columns in order.

    - Takes values from df where available
    - Missing columns are filled with 0.0
    - All values coerced to numeric
    """
    row = {}
    for name in feature_names:
        val = df[name].iloc[0] if name in df.columns else np.nan
        row[name] = pd.to_numeric(val, errors="coerce")
    X = pd.DataFrame([row], columns=feature_names)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


def predict_all_from_series(latest: pd.Series) -> Dict[str, Dict[str, float]]:
    df = _to_frame(latest)
    out: Dict[str, Dict[str, float]] = {
        "latest_feature_timestamp": str(latest.get("event_timestamp", "")),
        "lightgbm": {},
        "linear": {},
        "hgbr": {},
        "blend": {},
    }

    # Per-model predictions
    for hz in ["hd1", "hd2", "hd3"]:
        # LightGBM
        if _BOOSTERS.get(hz) is not None and _lgb_runtime is not None:
            try:
                booster = _BOOSTERS[hz]
                # Align to booster feature names exactly to avoid shape mismatch
                try:
                    feat_names = booster.feature_name()  # type: ignore[attr-defined]
                except Exception:
                    feat_names = _select_feature_columns(df)
                X = _align_to_feature_names(df, list(feat_names))
                pred = float(booster.predict(X)[0])  # type: ignore[attr-defined]
                out["lightgbm"][hz] = pred
            except Exception:
                pass

        # Linear
        if _LINEAR.get(hz) is not None and _LINEAR_FEATS.get(hz):
            try:
                feats = _LINEAR_FEATS[hz]
                X = _coerce_numeric_impute_latest(df, feats)
                if X.empty:
                    raise ValueError("linear features missing; skipping")
                pred = float(_LINEAR[hz].predict(X)[0])  # type: ignore[attr-defined]
                out["linear"][hz] = pred
            except Exception:
                pass

        # HGBR
        if _HGBR.get(hz) is not None:
            try:
                feats = _HGBR_FEATS.get(hz) or list(df.columns)
                X = _coerce_numeric_impute_latest(df, feats)
                if X.empty:
                    raise ValueError("hgbr features missing; skipping")
                pred = float(_HGBR[hz].predict(X)[0])  # type: ignore[attr-defined]
                out["hgbr"][hz] = pred
            except Exception:
                pass

    # Blend
    for hz in ["hd1", "hd2", "hd3"]:
        wz = _BLEND_WEIGHTS.get(hz, {})
        if not wz:
            # fallback preference order
            for src in ["lightgbm", "linear", "hgbr"]:
                if hz in out[src]:
                    out["blend"][hz] = float(out[src][hz])
                    break
            continue
        total = 0.0
        value = 0.0
        for key, wv in wz.items():
            src = None
            if key == "y_pred_lgb":
                src = "lightgbm"
            elif key == "y_pred_linear":
                src = "linear"
            elif key == "y_pred_hgb":
                src = "hgbr"
            pv = None if src is None else out.get(src, {}).get(hz)
            if pv is None:
                continue
            value += float(wv) * float(pv)
            total += float(wv)
        if total > 0:
            out["blend"][hz] = float(np.clip(value, 0.0, 500.0))

    return out



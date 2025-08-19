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
_RF: Dict[str, Optional[object]] = {"hd1": None, "hd2": None, "hd3": None}
_RF_FEATS: Dict[str, List[str]] = {"hd1": [], "hd2": [], "hd3": []}
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
    
    print(f"DEBUG: Initializing registry from: {registry_dir}")
    print(f"DEBUG: Registry directory exists: {os.path.exists(registry_dir)}")
    if os.path.exists(registry_dir):
        print(f"DEBUG: Registry contents: {os.listdir(registry_dir)[:10]}")

    # Load LightGBM boosters (optional)
    if _lgb_runtime is not None:
        for hz in ["hd1", "hd2", "hd3"]:
            # Map horizon to actual target names used in training
            target_map = {"hd1": "AQI_t+1", "hd2": "AQI_t+2", "hd3": "AQI_t+3"}
            target_name = target_map[hz]
            search_pattern = os.path.join(_REGISTRY_DIR, f"lgb_{target_name}_*.txt")
            print(f"DEBUG: Searching for LightGBM {hz} with pattern: {search_pattern}")
            path = _latest(search_pattern)
            if path:
                try:
                    _BOOSTERS[hz] = _lgb_runtime.Booster(model_file=path)
                    print(f"DEBUG: Loaded LightGBM {hz} from {path}")
                except Exception as e:
                    print(f"DEBUG: Failed to load LightGBM {hz}: {e}")
                    _BOOSTERS[hz] = None
            else:
                print(f"DEBUG: No LightGBM model found for {hz}")
                _BOOSTERS[hz] = None

    # Load linear pipelines with feature lists
    if joblib is not None:
        for hz in ["hd1", "hd2", "hd3"]:
            # Map horizon to actual target names used in training
            target_map = {"hd1": "AQI_t+1", "hd2": "AQI_t+2", "hd3": "AQI_t+3"}
            target_name = target_map[hz]
            search_pattern = os.path.join(_REGISTRY_DIR, f"linear_{target_name}_*.joblib")
            print(f"DEBUG: Searching for Linear {hz} with pattern: {search_pattern}")
            path = _latest(search_pattern)
            if path:
                try:
                    payload = joblib.load(path)
                    _LINEAR[hz] = payload.get("model")
                    _LINEAR_FEATS[hz] = payload.get("features", [])
                    print(f"DEBUG: Loaded Linear {hz} from {path}")
                    print(f"DEBUG: Linear {hz} features: {_LINEAR_FEATS[hz]}")
                except Exception as e:
                    print(f"DEBUG: Failed to load Linear {hz}: {e}")
                    _LINEAR[hz] = None
                    _LINEAR_FEATS[hz] = []
            else:
                print(f"DEBUG: No Linear model found for {hz}")
                _LINEAR[hz] = None
                _LINEAR_FEATS[hz] = []

    # Load HGBR (sklearn) models with feature lists
    if joblib is not None:
        for hz in ["hd1", "hd2", "hd3"]:
            # Map horizon to actual target names used in training
            target_map = {"hd1": "AQI_t+1", "hd2": "AQI_t+2", "hd3": "AQI_t+3"}
            target_name = target_map[hz]
            
            # Try different HGBR file patterns
            search_patterns = [
                os.path.join(_REGISTRY_DIR, f"hgb_{target_name}_*.joblib"),
                os.path.join(_REGISTRY_DIR, f"hgbr_{target_name}_*.joblib"),
                os.path.join(_REGISTRY_DIR, f"hgb_{target_name}_*.pkl")
            ]
            
            path = None
            for pattern in search_patterns:
                path = _latest(pattern)
                if path:
                    print(f"DEBUG: Found HGBR {hz} with pattern: {pattern}")
                    break
            
            if path:
                try:
                    payload = joblib.load(path)
                    _HGBR[hz] = payload.get("model")
                    _HGBR_FEATS[hz] = payload.get("features", [])
                    print(f"DEBUG: Loaded HGBR {hz} from {path}")
                    print(f"DEBUG: HGBR {hz} features: {_HGBR_FEATS[hz]}")
                except Exception as e:
                    print(f"DEBUG: Failed to load HGBR {hz}: {e}")
                    _HGBR[hz] = None
                    _HGBR_FEATS[hz] = []
            else:
                print(f"DEBUG: No HGBR model found for {hz}")
                _HGBR[hz] = None
                _HGBR_FEATS[hz] = []

    # Load Random Forest models with feature lists
    if joblib is not None:
        for hz in ["hd1", "hd2", "hd3"]:
            # Map horizon to actual target names used in training
            target_map = {"hd1": "AQI_t+1", "hd2": "AQI_t+2", "hd3": "AQI_t+3"}
            target_name = target_map[hz]
            search_pattern = os.path.join(_REGISTRY_DIR, f"rf_{target_name}_*.joblib")
            print(f"DEBUG: Searching for RandomForest {hz} with pattern: {search_pattern}")
            path = _latest(search_pattern)
            if path:
                try:
                    payload = joblib.load(path)
                    _RF[hz] = payload.get("model")
                    _RF_FEATS[hz] = payload.get("features", [])
                    print(f"DEBUG: Loaded RandomForest {hz} from {path}")
                    print(f"DEBUG: RandomForest {hz} features: {_RF_FEATS[hz]}")
                except Exception as e:
                    print(f"DEBUG: Failed to load RandomForest {hz}: {e}")
                    _RF[hz] = None
                    _RF_FEATS[hz] = []
            else:
                print(f"DEBUG: No RandomForest model found for {hz}")
                _RF[hz] = None
                _RF_FEATS[hz] = []

    # Load blend weights (latest)
    bw = _latest(os.path.join(_REGISTRY_DIR, "blend_weights_*.json"))
    print(f"DEBUG: Blend weights file found: {bw}")
    if bw and os.path.exists(bw):
        try:
            with open(bw, "r", encoding="utf-8") as f:
                blob = json.load(f)
            _BLEND_WEIGHTS = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in blob.get("weights", {}).items()}
            print(f"DEBUG: Loaded blend weights: {_BLEND_WEIGHTS}")
        except Exception as e:
            print(f"DEBUG: Failed to load blend weights: {e}")
            _BLEND_WEIGHTS = {}
    else:
        print(f"DEBUG: No blend weights file found")
        _BLEND_WEIGHTS = {}
    
    # Print summary of loaded models
    print(f"\nMODEL LOADING SUMMARY:")
    print(f"   LightGBM: {sum(1 for v in _BOOSTERS.values() if v is not None)}/3 loaded")
    print(f"   Linear: {sum(1 for v in _LINEAR.values() if v is not None)}/3 loaded")
    print(f"   HGBR: {sum(1 for v in _HGBR.values() if v is not None)}/3 loaded")
    print(f"   RandomForest: {sum(1 for v in _RF.values() if v is not None)}/3 loaded")
    print(f"   Blend weights: {'YES' if _BLEND_WEIGHTS else 'NO'}")
    
    for hz in ["hd1", "hd2", "hd3"]:
        models = []
        if _BOOSTERS[hz]: models.append("LightGBM")
        if _LINEAR[hz]: models.append("Linear")
        if _HGBR[hz]: models.append("HGBR")
        if _RF[hz]: models.append("RandomForest")
        print(f"   {hz}: {', '.join(models) if models else 'NO models'}")
    print()


def current_artifacts_summary() -> Dict[str, Dict[str, bool]]:
    return {
        "lightgbm": {hz: (_BOOSTERS[hz] is not None) for hz in ["hd1", "hd2", "hd3"]},
        "linear": {hz: (_LINEAR[hz] is not None) for hz in ["hd1", "hd2", "hd3"]},
        "hgbr": {hz: (_HGBR[hz] is not None) for hz in ["hd1", "hd2", "hd3"]},
        "randomforest": {hz: (_RF[hz] is not None) for hz in ["hd1", "hd2", "hd3"]},
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
        "AQI_t+1",
        "AQI_t+2",
        "AQI_t+3",
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
        "randomforest": {},
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
                print(f"LightGBM {hz}: {pred:.2f}")
            except Exception as e:
                print(f"LightGBM {hz} failed: {e}")
                pass

        # Linear
        if _LINEAR.get(hz) is not None and _LINEAR_FEATS.get(hz):
            try:
                feats = _LINEAR_FEATS[hz]
                print(f"DEBUG: Linear {hz} features: {feats}")
                X = _coerce_numeric_impute_latest(df, feats)
                if X.empty:
                    raise ValueError("linear features missing; skipping")
                pred = float(_LINEAR[hz].predict(X)[0])  # type: ignore[attr-defined]
                out["linear"][hz] = pred
                print(f"Linear {hz}: {pred:.2f}")
            except Exception as e:
                print(f"Linear {hz} failed: {e}")
                pass

        # HGBR
        if _HGBR.get(hz) is not None:
            try:
                feats = _HGBR_FEATS.get(hz) or list(df.columns)
                print(f"DEBUG: HGBR {hz} features: {feats}")
                X = _coerce_numeric_impute_latest(df, feats)
                if X.empty:
                    raise ValueError("hgbr features missing; skipping")
                pred = float(_HGBR[hz].predict(X)[0])  # type: ignore[attr-defined]
                out["hgbr"][hz] = pred
                print(f"HGBR {hz}: {pred:.2f}")
            except Exception as e:
                print(f"HGBR {hz} failed: {e}")
                pass

        # Random Forest
        if _RF.get(hz) is not None and _RF_FEATS.get(hz):
            try:
                feats = _RF_FEATS[hz]
                X = _coerce_numeric_impute_latest(df, feats)
                if X.empty:
                    raise ValueError("randomforest features missing; skipping")
                pred = float(_RF[hz].predict(X)[0])  # type: ignore[attr-defined]
                out["randomforest"][hz] = pred
                print(f"RandomForest {hz}: {pred:.2f}")
            except Exception as e:
                print(f"RandomForest {hz} failed: {e}")
                pass

    print(f"Final predictions: {out}")
    
    # Blend
    for hz in ["hd1", "hd2", "hd3"]:
        wz = _BLEND_WEIGHTS.get(hz, {})
        if not wz:
            # fallback preference order - set fallback but don't skip blending
            fallback_pred = None
            for src in ["lightgbm", "linear", "hgbr", "randomforest"]:
                if hz in out[src]:
                    fallback_pred = float(out[src][hz])
                    out["blend"][hz] = fallback_pred
                    print(f"Using fallback prediction for {hz}: {src} = {fallback_pred}")
                    break
            # Continue to next horizon if no fallback found
            if fallback_pred is None:
                continue
        
        # Apply blend weights if available
        if wz:
            total = 0.0
            value = 0.0
            available_models = []
            
            print(f"Blending {hz} with weights: {wz}")
            
            for key, wv in wz.items():
                src = None
                if key == "y_pred_lgb":
                    src = "lightgbm"
                elif key == "y_pred_linear":
                    src = "linear"
                elif key == "y_pred_hgb":
                    src = "hgbr"
                elif key == "y_pred_rf":
                    src = "randomforest"
                
                pv = None if src is None else out.get(src, {}).get(hz)
                if pv is None:
                    print(f"Missing prediction for {hz} from {src} (key: {key}, weight: {wv:.3f})")
                    continue
                
                value += float(wv) * float(pv)
                total += float(wv)
                available_models.append(f"{src}({wv:.3f})")
                print(f"{hz} blend: {src} × {wv:.3f} = {pv:.2f}")
            
            if total > 0:
                # Check if we have enough weight coverage
                expected_total = sum(wz.values())
                coverage = total / expected_total if expected_total > 0 else 0
                
                if coverage < 0.95:  # Less than 95% weight coverage
                    print(f"WARNING: {hz} only has {coverage:.1%} weight coverage ({total:.3f}/{expected_total:.3f})")
                    print(f"   Available models: {', '.join(available_models)}")
                
                blended_value = float(np.clip(value / total, 0.0, 500.0))
                out["blend"][hz] = blended_value
                print(f"{hz} final blend: {blended_value:.2f} (total weight: {total:.3f}, coverage: {coverage:.1%})")
            else:
                print(f"No valid weights for {hz}, using fallback")

    return out



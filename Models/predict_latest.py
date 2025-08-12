import os
import json
from glob import glob
from typing import Dict, List

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib


PARQUET_PATH = os.path.join("Data", "feature_store", "karachi_daily_features.parquet")
REGISTRY_DIR = os.path.join("Models", "registry")


def _latest_file(pattern: str) -> str:
    files = glob(pattern)
    if not files:
        return ""
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


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


def _coerce_numeric_impute_latest(df: pd.DataFrame, feats: List[str]) -> pd.DataFrame:
    df_sorted = df.sort_values("event_timestamp").reset_index(drop=True)
    X_latest = df_sorted.iloc[[-1]][feats].copy()
    # Coerce to numeric
    for c in feats:
        X_latest[c] = pd.to_numeric(X_latest[c], errors="coerce")
    # Replace inf
    X_latest.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Fill with history means (numeric only), fallback to 0 if still NaN
    hist_numeric = df_sorted[feats].apply(pd.to_numeric, errors="coerce")
    col_means = hist_numeric.mean(numeric_only=True)
    col_means = col_means.reindex(feats)
    X_latest = X_latest.fillna(col_means).fillna(0.0)
    return X_latest


def predict_with_lightgbm(df: pd.DataFrame) -> Dict[str, float]:
    feats = _select_feature_columns(df)
    X_latest = _coerce_numeric_impute_latest(df, feats)
    result: Dict[str, float] = {}
    for hz, target in [("hd1", "target_aqi_d1"), ("hd2", "target_aqi_d2"), ("hd3", "target_aqi_d3")]:
        model_path = _latest_file(os.path.join(REGISTRY_DIR, f"lgb_{hz}_*.txt"))
        if not model_path:
            continue
        booster = lgb.Booster(model_file=model_path)
        # LightGBM matches feature names; ensure correct order/names
        pred = float(booster.predict(X_latest)[0])
        result[hz] = pred
    return result


def predict_with_linear(df: pd.DataFrame) -> Dict[str, float]:
    df_sorted = df.sort_values("event_timestamp").reset_index(drop=True)
    result: Dict[str, float] = {}
    for hz, target in [("hd1", "target_aqi_d1"), ("hd2", "target_aqi_d2"), ("hd3", "target_aqi_d3")]:
        payload_path = _latest_file(os.path.join(REGISTRY_DIR, f"linear_{hz}_*.joblib"))
        if not payload_path:
            continue
        payload = joblib.load(payload_path)
        model = payload["model"]
        feats: List[str] = payload["features"]
        X_latest = df_sorted.iloc[[-1]][feats].copy()
        # Fill any remaining NaNs with column means from history (fallback)
        col_means = pd.Series(np.nanmean(df_sorted[feats].values, axis=0), index=feats)
        X_latest = X_latest.fillna(col_means)
        pred = float(model.predict(X_latest)[0])
        result[hz] = pred
    return result




def predict_with_hgbr(df: pd.DataFrame) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for hz in ["hd1", "hd2", "hd3"]:
        payload_path = _latest_file(os.path.join(REGISTRY_DIR, f"hgb_{hz}_*.joblib"))
        if not payload_path:
            continue
        try:
            payload = joblib.load(payload_path)
            model = payload.get("model")
            feats: List[str] = payload.get("features", _select_feature_columns(df))
            X_latest = _coerce_numeric_impute_latest(df, feats)
            pred = float(model.predict(X_latest)[0])
            result[hz] = pred
        except Exception:
            continue
    return result


def _load_blend_weights() -> Dict[str, Dict[str, float]]:
    # Prefer latest registry artifact
    reg_weights = _latest_file(os.path.join(REGISTRY_DIR, "blend_weights_*.json"))
    if reg_weights and os.path.exists(reg_weights):
        try:
            with open(reg_weights, "r", encoding="utf-8") as f:
                blob = json.load(f)
            weights = blob.get("weights", {})
            # keys like y_pred_lgb/y_pred_linear/y_pred_xgb/y_pred_hgb
            return {hz: {k: float(v) for k, v in w.items()} for hz, w in weights.items()}
        except Exception:
            pass
    # Fallback to EDA summary weights (same keys)
    try:
        with open(os.path.join("EDA", "blend_output", "summary.json"), "r", encoding="utf-8") as f:
            blob = json.load(f)
        return blob.get("weights", {})
    except Exception:
        return {}


def main() -> None:
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(f"Parquet not found: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    if df.empty:
        raise RuntimeError("Features parquet is empty; run the pipeline first.")

    lgb_preds = predict_with_lightgbm(df)
    lin_preds = predict_with_linear(df)
    # XGBoost disabled in blend path
    # XGBoost removed
    hgb_preds = predict_with_hgbr(df)

    weights = _load_blend_weights()
    blended: Dict[str, float] = {}
    bases = {
        "y_pred_lgb": lgb_preds,
        "y_pred_linear": lin_preds,
        # "y_pred_xgb": xgb_preds,
        "y_pred_hgb": hgb_preds,
    }
    for hz in ["hd1", "hd2", "hd3"]:
        wz = weights.get(hz) or {}
        # default fallback: prefer lgb, else linear, else xgb, else hgb
        if not wz:
            if hz in lgb_preds:
                blended[hz] = lgb_preds[hz]
                continue
            if hz in lin_preds:
                blended[hz] = lin_preds[hz]
                continue
            if hz in xgb_preds:
                blended[hz] = xgb_preds[hz]
                continue
            if hz in hgb_preds:
                blended[hz] = hgb_preds[hz]
                continue
            continue
        total = 0.0
        value = 0.0
        for key, wv in wz.items():
            pred_map = bases.get(key)
            if pred_map is None:
                continue
            pv = pred_map.get(hz)
            if pv is None:
                continue
            value += float(wv) * float(pv)
            total += float(wv)
        if total > 0:
            blended[hz] = float(value)

    latest_ts = pd.to_datetime(df["event_timestamp"]).max()
    out = {
        "latest_feature_timestamp": str(latest_ts),
        "lightgbm": lgb_preds,
        "linear": lin_preds,
        # "xgboost": {},
        "hgbr": hgb_preds,
        "blend": blended,
    }
    out_path = os.path.join(REGISTRY_DIR, "latest_forecast.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()



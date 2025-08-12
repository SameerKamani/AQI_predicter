import os
import json
import argparse
from datetime import datetime
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
try:
    import shap  # type: ignore
except Exception:
    shap = None
try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None


DEFAULT_PARQUET = os.path.join("Data", "feature_store", "karachi_daily_features.parquet")
DEFAULT_REGISTRY = os.path.join("Models", "registry")
DEFAULT_OUT = os.path.join("EDA", "blend_output")


def select_feature_columns(df: pd.DataFrame) -> List[str]:
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


def time_split(df: pd.DataFrame, target_col: str, holdout_days: int):
    df_sorted = df.sort_values("event_timestamp").reset_index(drop=True)
    if holdout_days <= 0 or holdout_days >= len(df_sorted):
        raise ValueError("holdout_days must be between 1 and number of rows - 1")
    train = df_sorted.iloc[:-holdout_days]
    test = df_sorted.iloc[-holdout_days:]
    feats = select_feature_columns(df_sorted)
    te = test.dropna(subset=[target_col])[feats + [target_col, "event_timestamp"]].copy()
    # Impute feature NaNs using train means for stability
    means = train[feats].mean(numeric_only=True)
    Xte = te[feats].fillna(means)
    yte = te[target_col]
    ts = te["event_timestamp"]
    return Xte, yte, ts, feats


def _aqi_category(values: np.ndarray) -> np.ndarray:
    cats = np.zeros_like(values, dtype=int)
    cats[(values <= 50)] = 1
    cats[(values > 50) & (values <= 100)] = 2
    cats[(values > 100) & (values <= 150)] = 3
    cats[(values > 150) & (values <= 200)] = 4
    cats[(values > 200) & (values <= 300)] = 5
    cats[(values > 300)] = 6
    return cats


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    denom = np.clip(np.abs(y_true), 1.0, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    acc_cat = float(np.mean(_aqi_category(y_true) == _aqi_category(y_pred)))
    within15 = float(np.mean(np.abs(y_true - y_pred) <= 15.0))
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape_pct": mape, "acc_category": acc_cat, "acc_within_15": within15}


def latest_file(pattern: str) -> str:
    files = glob(pattern)
    if not files:
        return ""
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def load_lightgbm_preds(registry_dir: str, horizon: str) -> pd.DataFrame:
    path = latest_file(os.path.join(registry_dir, f"lgb_{horizon}_*_preds.csv"))
    if not path:
        raise FileNotFoundError(f"No LightGBM preds found for {horizon} in {registry_dir}")
    df = pd.read_csv(path)
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True, errors="coerce")
    df = df.rename(columns={"y_pred": "y_pred_lgb", "y_true": "y_true_lgb"})
    return df


def load_generic_preds(registry_dir: str, prefix: str, horizon: str, col_name: str) -> pd.DataFrame:
    path = latest_file(os.path.join(registry_dir, f"{prefix}_{horizon}_*_preds.csv"))
    if not path:
        raise FileNotFoundError(f"No {prefix} preds found for {horizon} in {registry_dir}")
    df = pd.read_csv(path)
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True, errors="coerce")
    df = df.rename(columns={"y_pred": col_name, "y_true": f"y_true_{col_name.split('_', 1)[1]}"})
    return df


def load_linear_model_and_pred(parquet: str, registry_dir: str, horizon: str, holdout_days: int) -> Tuple[pd.DataFrame, str]:
    path = latest_file(os.path.join(registry_dir, f"linear_{horizon}_*.joblib"))
    if not path:
        raise FileNotFoundError(f"No linear model found for {horizon} in {registry_dir}")
    payload = joblib.load(path)
    model = payload["model"]
    feats = payload["features"]

    df = pd.read_parquet(parquet).sort_values("event_timestamp").reset_index(drop=True)
    target_col = {"hd1": "target_aqi_d1", "hd2": "target_aqi_d2", "hd3": "target_aqi_d3"}[horizon]
    Xte_all, yte, ts, _ = time_split(df, target_col, holdout_days)
    Xte = Xte_all[feats].copy()
    y_pred = model.predict(Xte)
    out = pd.DataFrame({
        "event_timestamp": pd.to_datetime(ts, utc=True),
        "y_true_linear": yte.values,
        "y_pred_linear": y_pred,
    })
    return out, path


def _project_to_simplex(weights: np.ndarray) -> np.ndarray:
    # Projects weights onto the probability simplex (non-negative, sum to 1)
    if np.sum(weights) == 1.0 and np.all(weights >= 0):
        return weights
    w = np.array(weights, dtype=float)
    w_sorted = np.sort(w)[::-1]
    cssv = np.cumsum(w_sorted)
    rho = np.nonzero(w_sorted * np.arange(1, len(w) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(w - theta, 0)
    return w


def optimize_weights(y_true: np.ndarray, preds_matrix: np.ndarray, constrain_nonneg: bool, sum_to_one: bool) -> np.ndarray:
    # Unconstrained least squares
    X = preds_matrix
    w, *_ = np.linalg.lstsq(X, y_true, rcond=None)
    if constrain_nonneg and sum_to_one:
        w = _project_to_simplex(w)
    elif constrain_nonneg:
        w = np.maximum(w, 0)
        if w.sum() > 0:
            w = w / w.sum() if sum_to_one else w
    elif sum_to_one:
        s = w.sum()
        if s == 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / s
    return w


def parse_args():
    p = argparse.ArgumentParser(description="Stack LightGBM and Linear predictions per horizon (hd1..hd3) with optimized weights and optional calibration")
    p.add_argument("--parquet", type=str, default=DEFAULT_PARQUET)
    p.add_argument("--registry", type=str, default=DEFAULT_REGISTRY)
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    p.add_argument("--holdout_days", type=int, default=90)
    p.add_argument("--constrain_nonneg", action="store_true", help="Constrain weights to be non-negative")
    p.add_argument("--sum_to_one", action="store_true", help="Constrain weights to sum to one")
    p.add_argument("--calibrate", action="store_true", help="Apply isotonic regression calibration on blended predictions")
    p.add_argument("--metric", type=str, default="rmse", choices=["rmse", "within15"], help="Optimization metric to report")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    summary: Dict[str, Dict[str, float]] = {}
    weights_out: Dict[str, Dict[str, float]] = {}
    artifacts: Dict[str, Dict[str, str]] = {}

    # Load full features once for SHAP alignment
    df_full = pd.read_parquet(args.parquet).sort_values("event_timestamp").reset_index(drop=True)

    for horizon in ["hd1", "hd2", "hd3"]:
        # Load predictions (support LGB, Linear, HGBR, RF)
        lgb_df = load_lightgbm_preds(args.registry, horizon)
        lin_df, lin_model_path = load_linear_model_and_pred(args.parquet, args.registry, horizon, args.holdout_days)
        base_frames = [lgb_df, lin_df]
        base_cols = ["y_pred_lgb", "y_pred_linear"]
        # XGBoost disabled: not loading xgb preds
        # HGBR
        try:
            hgb_df = load_generic_preds(args.registry, "hgb", horizon, "y_pred_hgb")
            base_frames.append(hgb_df)
            base_cols.append("y_pred_hgb")
        except Exception:
            pass

        # RandomForest
        try:
            rf_df = load_generic_preds(args.registry, "rf", horizon, "y_pred_rf")
            base_frames.append(rf_df)
            base_cols.append("y_pred_rf")
        except Exception:
            pass

        # Align by timestamp
        merged = base_frames[0]
        for bf in base_frames[1:]:
            merged = pd.merge(merged, bf, on="event_timestamp", how="inner")
        if merged.empty:
            raise RuntimeError(f"No overlapping timestamps between base models for {horizon}")

        # Sanity: all y_true_* columns should match (allow minor float diffs)
        y_cols = [c for c in merged.columns if c.startswith("y_true_")]
        y_true = merged[y_cols[0]].astype(float).values
        for c in y_cols[1:]:
            if not np.allclose(y_true, merged[c].astype(float).values, atol=1e-6, equal_nan=False):
                raise RuntimeError(f"y_true mismatch between base models for {horizon}")

        # Stack base predictions in consistent order
        X = np.stack([merged[c].astype(float).values for c in base_cols], axis=1)

        # Optimize weights (least squares -> optional constraints)
        w = optimize_weights(y_true, X, args.constrain_nonneg, args.sum_to_one)

        # Blend
        y_blend = X @ w

        # Optional isotonic calibration
        if args.calibrate:
            try:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(y_blend, y_true)
                y_blend = iso.transform(y_blend)
            except Exception:
                pass

        # Clamp to AQI bounds
        y_blend = np.clip(y_blend, 0.0, 500.0)

        # Metrics (full holdout and last-30 slice)
        m_full = metrics(y_true, y_blend)
        # ensure time order
        ord_idx = np.argsort(merged["event_timestamp"].values)
        y_true_ord = y_true[ord_idx]
        y_blend_ord = y_blend[ord_idx]
        last = 30 if len(y_true_ord) >= 30 else len(y_true_ord)
        m_last30 = metrics(y_true_ord[-last:], y_blend_ord[-last:])
        summary[horizon] = {"full": m_full, "last30": m_last30}
        weights_out[horizon] = {name: float(val) for name, val in zip(base_cols, w)}

        # Save blended preds
        out = {
            "event_timestamp": merged["event_timestamp"],
            "y_true": y_true,
            "y_pred_blend": y_blend,
        }
        for c in base_cols:
            out[c] = merged[c].astype(float).values
            out["weight_" + c] = float(w[base_cols.index(c)])
        out_preds = pd.DataFrame(out)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_preds.to_csv(os.path.join(args.out, f"stack_{horizon}_{ts}.csv"), index=False)

        # Record artifacts used
        artifacts[horizon] = {
            "lightgbm_preds": latest_file(os.path.join(args.registry, f"lgb_{horizon}_*_preds.csv")),
            "linear_model": lin_model_path,
            "holdout_days": str(args.holdout_days),
        }

        # --- SHAP explanations (optional) ---
        try:
            if shap is not None:
                # Build holdout feature matrix aligned to merged timestamps
                target_col = {"hd1": "target_aqi_d1", "hd2": "target_aqi_d2", "hd3": "target_aqi_d3"}[horizon]
                # Reuse time_split to get holdout features and timestamps
                Xte_all, _, ts_all, feats = time_split(df_full, target_col, args.holdout_days)
                # Align rows to merged event_timestamp
                Xte_all = Xte_all.copy()
                # Normalize timezone on both sides to UTC-aware for a safe merge
                merged["event_timestamp"] = pd.to_datetime(merged["event_timestamp"], utc=True)
                Xte_all["event_timestamp"] = pd.to_datetime(ts_all, utc=True)
                X_aligned = pd.merge(
                    merged[["event_timestamp"]],
                    Xte_all,
                    on="event_timestamp",
                    how="inner",
                ).drop(columns=["event_timestamp"])  # same order as merged

                # LightGBM SHAP
                shap_lgb = None
                lgb_model_path = latest_file(os.path.join(args.registry, f"lgb_{horizon}_*.txt"))
                if lgb is not None and lgb_model_path:
                    try:
                        booster = lgb.Booster(model_file=lgb_model_path)
                        expl = shap.TreeExplainer(booster)
                        shap_vals = expl.shap_values(X_aligned, check_additivity=False)
                        shap_lgb = shap_vals if not isinstance(shap_vals, list) else shap_vals[0]
                    except Exception:
                        shap_lgb = None

                # Linear SHAP using saved pipeline
                shap_lin = None
                try:
                    payload = joblib.load(lin_model_path)
                    model = payload["model"]
                    scaler = getattr(model, "named_steps", {}).get("scaler")
                    reg = getattr(model, "named_steps", {}).get("reg")
                    if scaler is not None and reg is not None:
                        X_scaled = scaler.transform(X_aligned)
                        lexpl = shap.LinearExplainer(reg, X_scaled)
                        shap_lin = lexpl.shap_values(X_scaled)
                except Exception:
                    shap_lin = None

                # Combine per weights (pre-calibration)
                shap_blend = None
                if shap_lgb is not None or shap_lin is not None:
                    lgb_part = 0.0 if shap_lgb is None else float(w[0]) * shap_lgb
                    lin_part = 0.0 if shap_lin is None else float(w[1]) * shap_lin
                    # Handle cases where one part is None
                    if shap_lgb is None:
                        shap_blend = lin_part
                    elif shap_lin is None:
                        shap_blend = lgb_part
                    else:
                        shap_blend = lgb_part + lin_part

                if shap_blend is not None:
                    mean_abs = np.mean(np.abs(shap_blend), axis=0)
                    shap_df = (
                        pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs})
                        .sort_values("mean_abs_shap", ascending=False)
                        .reset_index(drop=True)
                    )
                    out_shap_dir = os.path.join("EDA", "shap_output")
                    os.makedirs(out_shap_dir, exist_ok=True)
                    shap_path = os.path.join(out_shap_dir, f"shap_global_stack_{horizon}.csv")
                    shap_df.to_csv(shap_path, index=False)
        except Exception as e:
            # Do not break stacking if SHAP fails
            print(f"SHAP (stack) failed for {horizon}: {e}")

    # Write summary and weights
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": summary, "weights": weights_out, "holdout_days": args.holdout_days}, f, indent=2)
    print(f"Saved stacked blend summary -> {os.path.join(args.out, 'summary.json')}")

    # Persist a registry artifact with blend weights for reuse at inference time
    reg_path = os.path.join(DEFAULT_REGISTRY, f"blend_weights_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json")
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump({"weights": weights_out, "artifacts": artifacts, "holdout_days": args.holdout_days}, f, indent=2)
    print(f"Saved blend weights -> {reg_path}")


if __name__ == "__main__":
    main()



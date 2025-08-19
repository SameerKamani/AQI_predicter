import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning
import joblib
import warnings
try:
    import shap  # type: ignore
except Exception:
    shap = None


DEFAULT_PARQUET = os.path.join("..", "Data", "feature_store", "karachi_daily_features.parquet")
DEFAULT_REGISTRY = os.path.join("Models", "registry")
DEFAULT_OUT = os.path.join("EDA", "linear_output")


def select_feature_columns(df: pd.DataFrame) -> List[str]:
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


def time_split(df: pd.DataFrame, target_col: str, holdout_days: int):
    df_sorted = df.sort_values("event_timestamp").reset_index(drop=True)
    if holdout_days <= 0 or holdout_days >= len(df_sorted):
        raise ValueError("holdout_days must be between 1 and number of rows - 1")
    train = df_sorted.iloc[:-holdout_days]
    test = df_sorted.iloc[-holdout_days:]
    feats = select_feature_columns(df_sorted)
    # Keep rows where target is available
    tr = train.dropna(subset=[target_col])[feats + [target_col]].copy()
    te = test.dropna(subset=[target_col])[feats + [target_col]].copy()

    # Impute feature NaNs with train means (columnwise). If a column is all-NaN in train, fill with 0 and drop from both.
    Xtr = tr[feats].copy()
    Xte = te[feats].copy()
    # Coerce to numeric and replace infs
    for dfp in (Xtr, Xte):
        for c in dfp.columns:
            dfp[c] = pd.to_numeric(dfp[c], errors="coerce")
        dfp.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Column-wise means without emitting numpy warnings for all-NaN columns
    means = Xtr.mean(axis=0, skipna=True).values
    # Identify columns with all-NaN in train
    all_nan_cols = np.isnan(means)
    if np.any(all_nan_cols):
        cols_to_keep = [f for f, keep in zip(feats, ~all_nan_cols) if keep]
        feats = cols_to_keep
        Xtr = Xtr[feats]
        Xte = Xte[feats]
        means = np.nanmean(Xtr.values, axis=0)
    # Fill NaNs
    Xtr = Xtr.fillna(pd.Series(means, index=feats))
    Xte = Xte.fillna(pd.Series(means, index=feats))

    return Xtr, Xte, tr[target_col], te[target_col], feats


def _aqi_category(values: np.ndarray) -> np.ndarray:
    cats = np.zeros_like(values, dtype=int)
    cats[(values <= 50)] = 1
    cats[(values > 50) & (values <= 100)] = 2
    cats[(values > 100) & (values <= 150)] = 3
    cats[(values > 150) & (values <= 200)] = 4
    cats[(values > 200) & (values <= 300)] = 5
    cats[(values > 300)] = 6
    return cats


def eval_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    denom = np.clip(np.abs(y_true.values), 1.0, None)
    mape = float(np.mean(np.abs((y_true.values - y_pred) / denom)) * 100.0)
    acc_cat = float(np.mean(_aqi_category(y_true.values) == _aqi_category(y_pred)))
    within15 = float(np.mean(np.abs(y_true.values - y_pred) <= 15.0))
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape_pct": mape, "acc_category": acc_cat, "acc_within_15": within15}


def train_and_select(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, seed: int):
    from sklearn.model_selection import TimeSeriesSplit

    # Grids
    # Avoid extremely tiny alphas which often trigger convergence issues
    ridge_alphas = np.logspace(-2, 3, 12)
    enet_alphas = np.logspace(-2, 2, 11)
    enet_l1 = np.linspace(0.05, 0.95, 7)

    candidates: List[Tuple[str, Pipeline]] = []
    # Ridge candidates
    for a in ridge_alphas:
        candidates.append((f"ridge_a_{a:.4g}", Pipeline([
            ("scaler", RobustScaler()),
            ("reg", Ridge(alpha=float(a), random_state=seed))
        ])))
    # ElasticNet candidates
    for a in enet_alphas:
        for l1 in enet_l1:
            candidates.append((f"enet_a_{a:.4g}_l1_{l1:.2f}", Pipeline([
                ("scaler", RobustScaler()),
                ("reg", ElasticNet(alpha=float(a), l1_ratio=float(l1), random_state=seed, max_iter=10000))
            ])))
    # Huber (single robust baseline)
    candidates.append(("huber_default", Pipeline([
        ("scaler", RobustScaler()),
        ("reg", HuberRegressor())
    ])))

    tscv = TimeSeriesSplit(n_splits=3)
    best_cv_rmse = float("inf")
    best_name = ""
    best_model: Pipeline | None = None

    # CV selection on training window
    for name, pipe in candidates:
        rmses: List[float] = []
        for tr_idx, va_idx in tscv.split(X_train):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_va)
            rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
            rmses.append(rmse)
        cv_rmse = float(np.mean(rmses)) if rmses else float("inf")
        if cv_rmse < best_cv_rmse:
            best_cv_rmse, best_name, best_model = cv_rmse, name, pipe

    # Fit best on full training window and evaluate on test
    assert best_model is not None
    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_test)
    m = eval_metrics(y_test, preds)
    return best_model, best_name, m


def parse_args():
    p = argparse.ArgumentParser(description="Train Ridge/ElasticNet baselines for AQI D+1..D+3 and pick the stronger")
    p.add_argument("--parquet", type=str, default=DEFAULT_PARQUET)
    p.add_argument("--registry", type=str, default=DEFAULT_REGISTRY)
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    p.add_argument("--holdout_days", type=int, default=90)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.parquet):
        raise FileNotFoundError(f"Parquet not found: {args.parquet}")
    df = pd.read_parquet(args.parquet)
    req = {"event_timestamp", "AQI", "AQI_t+1", "AQI_t+2", "AQI_t+3"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    os.makedirs(args.registry, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    results: Dict[str, Dict[str, float]] = {}
    chosen: Dict[str, str] = {}
    for target in ["AQI_t+1", "AQI_t+2", "AQI_t+3"]:
        Xtr, Xte, ytr, yte, feats = time_split(df, target, args.holdout_days)
        model, name, metrics = train_and_select(Xtr, ytr, Xte, yte, args.seed)
        results[target] = metrics
        chosen[target] = name

        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        base = f"linear_{target.replace('target_aqi_','h')}_{ts}"
        model_path = os.path.join(args.registry, base + ".joblib")
        joblib.dump({"model": model, "features": feats}, model_path)

        # SHAP global importance (optional, saved per horizon)
        try:
            if shap is not None:
                # Transform train/test through the scaler inside the pipeline
                scaler = model.named_steps.get("scaler")
                reg = model.named_steps.get("reg")
                if scaler is not None and reg is not None:
                    Xtr_trans = scaler.transform(Xtr)
                    explainer = shap.LinearExplainer(reg, Xtr_trans)
                    Xte_trans = scaler.transform(Xte)
                    shap_vals = explainer.shap_values(Xte_trans)
                    mean_abs = np.mean(np.abs(shap_vals), axis=0)
                    shap_df = (
                        pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs})
                        .sort_values("mean_abs_shap", ascending=False)
                        .reset_index(drop=True)
                    )
                    out_dir = os.path.join("EDA", "shap_output")
                    os.makedirs(out_dir, exist_ok=True)
                    shap_path = os.path.join(out_dir, f"shap_global_linear_{target.replace('target_aqi_','h')}.csv")
                    shap_df.to_csv(shap_path, index=False)
        except Exception as e:
            print(f"SHAP (linear) failed for {target}: {e}")

    # write summary
    summary = {"metrics": results, "chosen": chosen}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved linear summary -> {os.path.join(args.out, 'summary.json')}")


if __name__ == "__main__":
    main()



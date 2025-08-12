import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    import optuna  # type: ignore
except Exception:
    optuna = None


DEFAULT_PARQUET = os.path.join("Data", "feature_store", "karachi_daily_features.parquet")
DEFAULT_REGISTRY = os.path.join("Models", "registry")
DEFAULT_OUT = os.path.join("EDA", "hgbr_output")


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


def time_split(df: pd.DataFrame, target_col: str, holdout_days: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df_sorted = df.sort_values("event_timestamp").reset_index(drop=True)
    if holdout_days <= 0 or holdout_days >= len(df_sorted):
        raise ValueError("holdout_days must be between 1 and number of rows - 1")
    train = df_sorted.iloc[:-holdout_days]
    test = df_sorted.iloc[-holdout_days:]
    feats = select_feature_columns(df_sorted)
    return train[feats], test[feats], train[target_col], test[target_col]


def _coerce_numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    coerced = df.copy()
    for col in coerced.columns:
        coerced[col] = pd.to_numeric(coerced[col], errors="coerce")
    coerced = coerced.replace([np.inf, -np.inf], np.nan)
    return coerced.fillna(coerced.mean(numeric_only=True))


def _rolling_time_slices(n_rows: int, n_splits: int = 3) -> List[Tuple[slice, slice]]:
    if n_rows < n_splits + 2:
        cut = max(2, int(n_rows * 0.8))
        return [(slice(0, cut), slice(cut, n_rows))]
    cuts = [int(n_rows * r) for r in [0.6, 0.75, 0.9]]
    splits: List[Tuple[slice, slice]] = []
    prev_train_end = cuts[0]
    for val_end in cuts[1:] + [n_rows]:
        train_slice = slice(0, prev_train_end)
        val_slice = slice(prev_train_end, val_end)
        if val_end - prev_train_end >= 3 and prev_train_end >= 10:
            splits.append((train_slice, val_slice))
        prev_train_end = int(val_end * 0.85)
    return splits or [(slice(0, int(n_rows * 0.8)), slice(int(n_rows * 0.8), n_rows))]


def _cv_rmse_hgbr(X: pd.DataFrame, y: pd.Series, params: Dict[str, object]) -> float:
    Xn = _coerce_numeric_dataframe(X)
    splits = _rolling_time_slices(len(Xn), n_splits=3)
    rmses: List[float] = []
    for tr, va in splits:
        X_tr, y_tr = Xn.iloc[tr], y.iloc[tr]
        X_va, y_va = Xn.iloc[va], y.iloc[va]
        model = HistGradientBoostingRegressor(**params)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        rmse = float(np.sqrt(((pred - y_va.values) ** 2).mean()))
        rmses.append(rmse)
    return float(np.mean(rmses)) if rmses else float("inf")


def eval_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    denom = np.clip(np.abs(y_true.values), 1.0, None)
    mape = float(np.mean(np.abs((y_true.values - y_pred) / denom)) * 100.0)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape_pct": mape}


def train_hgbr(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int,
    tuned_overrides: Optional[Dict[str, object]] = None,
) -> HistGradientBoostingRegressor:
    params = dict(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=8,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.0,
        early_stopping=True,
        scoring="loss",
        random_state=seed,
    )
    if tuned_overrides:
        params.update(tuned_overrides)
    model = HistGradientBoostingRegressor(**params)
    model.fit(_coerce_numeric_dataframe(X_train), y_train)
    return model


def save_artifacts(model, feature_cols: List[str], horizon: str, registry_dir: str, out_dir: str, metrics: Dict[str, float], preds_df: pd.DataFrame) -> None:
    os.makedirs(registry_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base = f"hgb_{horizon}_{ts}"

    model_path = os.path.join(registry_dir, base + ".joblib")
    meta_path = os.path.join(registry_dir, base + "_meta.json")
    preds_path = os.path.join(registry_dir, base + "_preds.csv")

    try:
        import joblib
        joblib.dump({"model": model, "features": feature_cols}, model_path)
    except Exception:
        pass
    preds_df.to_csv(preds_path, index=False)

    meta = {
        "horizon": horizon,
        "created_utc": ts,
        "model_path": model_path,
        "metrics": metrics,
        "predictions_path": preds_path,
        "library": "sklearn.HGBR",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def train_for_horizon(
    df: pd.DataFrame,
    target_col: str,
    holdout_days: int,
    seed: int,
    registry_dir: str,
    out_dir: str,
    tune: bool = False,
    tune_trials: int = 20,
) -> Dict[str, float]:
    X_train, X_test, y_train, y_test = time_split(df, target_col, holdout_days)
    split_idx = int(len(X_train) * 0.85)
    X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

    tuned_overrides: Optional[Dict[str, object]] = None
    if tune and optuna is not None and len(X_train) >= 50:
        def objective(trial: "optuna.trial.Trial") -> float:
            space = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 16),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 60),
                "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 5.0),
            }
            return _cv_rmse_hgbr(X_train, y_train, space)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=tune_trials, show_progress_bar=False)
        tuned_overrides = study.best_params
        print(f"Best tuned params (HGBR {target_col}): {tuned_overrides}")
    elif tune and optuna is None:
        print("Optuna not installed; skipping HGBR tuning. 'pip install optuna' to enable.")

    model = train_hgbr(X_tr, y_tr, X_val, y_val, seed, tuned_overrides)
    y_pred = model.predict(_coerce_numeric_dataframe(X_test))
    metrics = eval_metrics(y_test, y_pred)

    preds_df = pd.DataFrame(
        {
            "event_timestamp": df.sort_values("event_timestamp").iloc[-len(X_test) :]["event_timestamp"].values,
            "y_true": y_test.values,
            "y_pred": y_pred,
        }
    )
    feature_cols = select_feature_columns(df)
    horizon = target_col.replace("target_aqi_", "h")
    save_artifacts(model, feature_cols, horizon, registry_dir, out_dir, metrics, preds_df)
    print(f"HGBR {horizon} -> RMSE: {metrics['rmse']:.3f} | MAE: {metrics['mae']:.3f} | R2: {metrics['r2']:.3f}")
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HGBR models for AQI D+1..D+3")
    p.add_argument("--parquet", type=str, default=DEFAULT_PARQUET)
    p.add_argument("--registry", type=str, default=DEFAULT_REGISTRY)
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    p.add_argument("--holdout_days", type=int, default=90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tune", action="store_true")
    p.add_argument("--tune_trials", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.parquet):
        raise FileNotFoundError(f"Parquet not found: {args.parquet}")
    df = pd.read_parquet(args.parquet)

    required = {"event_timestamp", "aqi_daily", "target_aqi_d1", "target_aqi_d2", "target_aqi_d3"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    overall: Dict[str, Dict[str, float]] = {}
    for target in ["target_aqi_d1", "target_aqi_d2", "target_aqi_d3"]:
        overall[target] = train_for_horizon(df, target, args.holdout_days, args.seed, args.registry, args.out, tune=args.tune, tune_trials=args.tune_trials)

    os.makedirs(args.out, exist_ok=True)
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)
    print(f"Saved HGBR summary -> {summary_path}")


if __name__ == "__main__":
    main()



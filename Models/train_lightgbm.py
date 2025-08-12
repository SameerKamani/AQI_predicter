import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
try:
    import shap  # type: ignore
except Exception:
    shap = None  # SHAP is optional; training will continue without explanations
try:
    import optuna  # type: ignore
except Exception:
    optuna = None


DEFAULT_PARQUET = os.path.join("Data", "feature_store", "karachi_daily_features.parquet")
DEFAULT_REGISTRY = os.path.join("Models", "registry")
DEFAULT_LGB_OUT = os.path.join("EDA", "lightgbm_output")


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {
        "date",
        "event_timestamp",
        "created",
        "city",
        "karachi_id",
        # targets
        "target_aqi_d1",
        "target_aqi_d2",
        "target_aqi_d3",
    }
    return [c for c in df.columns if c not in exclude]


def time_split(
    df: pd.DataFrame, target_col: str, holdout_days: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df_sorted = df.sort_values("event_timestamp").reset_index(drop=True)
    if holdout_days <= 0 or holdout_days >= len(df_sorted):
        raise ValueError("holdout_days must be between 1 and number of rows - 1")
    train = df_sorted.iloc[:-holdout_days]
    test = df_sorted.iloc[-holdout_days:]
    feats = select_feature_columns(df_sorted)
    return train[feats], test[feats], train[target_col], test[target_col]


def _coerce_numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    coerced = df.copy()
    # Convert object columns to numeric; keep bool and numeric as-is
    for col in coerced.columns:
        if coerced[col].dtype == object:
            coerced[col] = pd.to_numeric(coerced[col], errors="coerce")
    # Replace inf with NaN
    coerced = coerced.replace([np.inf, -np.inf], np.nan)
    return coerced


def _rolling_time_slices(n_rows: int, n_splits: int = 3) -> List[Tuple[slice, slice]]:
    if n_rows < n_splits + 2:
        # Fallback to a single split
        cut = max(2, int(n_rows * 0.8))
        return [(slice(0, cut), slice(cut, n_rows))]
    # Expanding window style
    cuts = [int(n_rows * r) for r in [0.6, 0.75, 0.9]]
    splits: List[Tuple[slice, slice]] = []
    prev_train_end = cuts[0]
    for val_end in cuts[1:] + [n_rows]:
        train_slice = slice(0, prev_train_end)
        val_slice = slice(prev_train_end, val_end)
        if val_end - prev_train_end >= 3 and prev_train_end >= 10:
            splits.append((train_slice, val_slice))
        prev_train_end = int(val_end * 0.85)  # grow train window
    return splits or [(slice(0, int(n_rows * 0.8)), slice(int(n_rows * 0.8), n_rows))]


def _cv_rmse_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    base_params: Dict[str, object],
    seed: int,
) -> float:
    Xn = _coerce_numeric_dataframe(X)
    splits = _rolling_time_slices(len(Xn), n_splits=3)
    rmses: List[float] = []
    for tr, va in splits:
        X_tr, y_tr = Xn.iloc[tr], y.iloc[tr]
        X_va, y_va = Xn.iloc[va], y.iloc[va]
        dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=list(Xn.columns))
        dva = lgb.Dataset(X_va, label=y_va, feature_name=list(Xn.columns), reference=dtr)
        params = dict(base_params)
        params.setdefault("objective", "regression")
        params.setdefault("metric", "rmse")
        params.setdefault("verbosity", -1)
        params.setdefault("seed", seed)
        params.setdefault("force_row_wise", True)
        callbacks = []
        try:
            callbacks.append(lgb.early_stopping(stopping_rounds=200))
            callbacks.append(lgb.log_evaluation(period=0))
        except Exception:
            callbacks = None
        booster = lgb.train(
            params,
            dtr,
            num_boost_round=3000,
            valid_sets=[dtr, dva],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )
        num_iter = None
        try:
            num_iter = booster.best_iteration or booster.current_iteration()
        except Exception:
            num_iter = None
        pred = booster.predict(X_va, num_iteration=num_iter)
        rmse = float(np.sqrt(((pred - y_va.values) ** 2).mean()))
        rmses.append(rmse)
    return float(np.mean(rmses)) if rmses else float("inf")


def _aqi_category(values: np.ndarray) -> np.ndarray:
    # EPA breakpoints mapped to category 1..6 (Good..Hazardous)
    # Using inclusive upper bounds
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
    # MAPE with protection against division by zero (clip denominator to >=1)
    denom = np.clip(np.abs(y_true.values), 1.0, None)
    mape = float(np.mean(np.abs((y_true.values - y_pred) / denom)) * 100.0)
    # Category accuracy based on EPA bins
    acc_cat = float(np.mean(_aqi_category(y_true.values) == _aqi_category(y_pred)))
    # Hit rate within +/- 15 AQI units
    within15 = float(np.mean(np.abs(y_true.values - y_pred) <= 15.0))
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape_pct": mape, "acc_category": acc_cat, "acc_within_15": within15}


def train_lgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int,
) -> lgb.Booster:
    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=list(X_train.columns))
    lgb_val = lgb.Dataset(X_val, label=y_val, feature_name=list(X_train.columns), reference=lgb_train)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "seed": seed,
        "verbosity": -1,
        "force_row_wise": True,
    }
    # Use callbacks for early stopping/log suppression across lightgbm versions
    callbacks = []
    try:
        callbacks.append(lgb.early_stopping(stopping_rounds=200))
    except Exception:
        callbacks = None
    # Always suppress eval logging
    try:
        if callbacks is not None:
            callbacks.append(lgb.log_evaluation(period=0))
        else:
            callbacks = [lgb.log_evaluation(period=0)]
    except Exception:
        pass
    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=3000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )
    return booster


def save_artifacts(
    booster: lgb.Booster,
    feature_cols: List[str],
    horizon: str,
    registry_dir: str,
    lgb_out_dir: str,
    metrics: Dict[str, float],
    preds_df: pd.DataFrame,
) -> None:
    os.makedirs(registry_dir, exist_ok=True)
    os.makedirs(lgb_out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base = f"lgb_{horizon}_{ts}"

    # Registry artifacts
    model_path = os.path.join(registry_dir, base + ".txt")
    meta_path = os.path.join(registry_dir, base + "_meta.json")
    preds_path = os.path.join(registry_dir, base + "_preds.csv")
    fi_path_registry = os.path.join(registry_dir, base + "_feature_importance.csv")

    booster.save_model(model_path, num_iteration=booster.best_iteration or booster.current_iteration())
    preds_df.to_csv(preds_path, index=False)

    # Feature importance (gain)
    gains = booster.feature_importance(importance_type="gain")
    fi_df = (
        pd.DataFrame({"feature": feature_cols, "gain": gains})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    fi_df.to_csv(fi_path_registry, index=False)

    # LightGBM output folder artifacts
    fi_path_lgb = os.path.join(lgb_out_dir, f"feature_importance_{horizon}.csv")
    fi_df.to_csv(fi_path_lgb, index=False)

    meta = {
        "horizon": horizon,
        "created_utc": ts,
        "model_path": model_path,
        "metrics": metrics,
        "predictions_path": preds_path,
        "feature_importance_registry": fi_path_registry,
        "feature_importance_lgb": fi_path_lgb,
        "library": "lightgbm",
        "lightgbm_version": lgb.__version__,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def train_for_horizon(
    df: pd.DataFrame,
    target_col: str,
    holdout_days: int,
    seed: int,
    registry_dir: str,
    lgb_out_dir: str,
    tune: bool = False,
    tune_trials: int = 25,
) -> Dict[str, float]:
    X_train, X_test, y_train, y_test = time_split(df, target_col, holdout_days)
    split_idx = int(len(X_train) * 0.85)
    X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

    # Ensure numeric dtypes for LightGBM (handles visibility_mean object, etc.)
    X_tr = _coerce_numeric_dataframe(X_tr)
    X_val = _coerce_numeric_dataframe(X_val)
    X_test = _coerce_numeric_dataframe(X_test)

    # Optional tiny tuner over a small param space using rolling time CV on full training window
    tuned_overrides: Dict[str, object] = {}
    if tune and optuna is not None and len(X_train) >= 50:
        def objective(trial: "optuna.trial.Trial") -> float:
            space = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 31, 255, step=8),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "bagging_freq": 1,
                "max_depth": -1,
            }
            base = {
                "objective": "regression",
                "metric": "rmse",
                "verbosity": -1,
                "seed": seed,
                "force_row_wise": True,
            }
            base.update(space)
            return _cv_rmse_lightgbm(X_train, y_train, base, seed)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=tune_trials, show_progress_bar=False)
        tuned_overrides = study.best_params
        # Ensure required keys are present
        tuned_overrides.setdefault("bagging_freq", 1)
        tuned_overrides.setdefault("max_depth", -1)
        print(f"Best tuned params ({target_col}): {tuned_overrides}")
    elif tune and optuna is None:
        print("Optuna not installed; skipping tuning. 'pip install optuna' to enable.")

    # Train final booster using tuned params where provided (on the same train/val split for early stopping)
    if tuned_overrides:
        # Build params by overriding defaults inside train_lgb via a small wrapper
        def _train_with_overrides(Xa, ya, Xb, yb, seed_):
            lgb_train = lgb.Dataset(Xa, label=ya, feature_name=list(Xa.columns))
            lgb_val = lgb.Dataset(Xb, label=yb, feature_name=list(Xa.columns), reference=lgb_train)
            params = {
                "objective": "regression",
                "metric": "rmse",
                "learning_rate": tuned_overrides.get("learning_rate", 0.03),
                "num_leaves": tuned_overrides.get("num_leaves", 63),
                "max_depth": tuned_overrides.get("max_depth", -1),
                "feature_fraction": tuned_overrides.get("feature_fraction", 0.9),
                "bagging_fraction": tuned_overrides.get("bagging_fraction", 0.9),
                "bagging_freq": tuned_overrides.get("bagging_freq", 1),
                "lambda_l2": tuned_overrides.get("lambda_l2", 1.0),
                "min_data_in_leaf": tuned_overrides.get("min_data_in_leaf", 20),
                "seed": seed_,
                "verbosity": -1,
                "force_row_wise": True,
            }
            callbacks = []
            try:
                callbacks.append(lgb.early_stopping(stopping_rounds=200))
                callbacks.append(lgb.log_evaluation(period=0))
            except Exception:
                callbacks = None
            return lgb.train(
                params,
                lgb_train,
                num_boost_round=3000,
                valid_sets=[lgb_train, lgb_val],
                valid_names=["train", "valid"],
                callbacks=callbacks,
            )

        booster = _train_with_overrides(X_tr, y_tr, X_val, y_val, seed)
    else:
        booster = train_lgb(X_tr, y_tr, X_val, y_val, seed)
    # Predict with best iteration if available
    num_iter = None
    try:
        if booster.best_iteration:
            num_iter = booster.best_iteration
    except Exception:
        try:
            num_iter = booster.current_iteration()
        except Exception:
            num_iter = None
    y_pred = booster.predict(X_test, num_iteration=num_iter)
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
    save_artifacts(booster, feature_cols, horizon, registry_dir, lgb_out_dir, metrics, preds_df)

    # SHAP global importance on test split (minimal addition)
    try:
        if shap is not None:
            X_shap = X_test.copy()
            # Limit size for speed if very large
            if len(X_shap) > 5000:
                X_shap = X_shap.sample(5000, random_state=seed)
            explainer = shap.TreeExplainer(booster)
            shap_values = explainer.shap_values(X_shap, check_additivity=False)
            # For regression, shap_values is a 2D array; handle list just in case
            if isinstance(shap_values, list):
                shap_arr = shap_values[0]
            else:
                shap_arr = shap_values
            mean_abs = np.mean(np.abs(shap_arr), axis=0)
            shap_df = (
                pd.DataFrame({"feature": X_shap.columns, "mean_abs_shap": mean_abs})
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )
            shap_out_dir = os.path.join("EDA", "shap_output")
            os.makedirs(shap_out_dir, exist_ok=True)
            shap_path = os.path.join(shap_out_dir, f"shap_global_{horizon}.csv")
            shap_df.to_csv(shap_path, index=False)
            print(f"Saved SHAP global importance -> {shap_path}")
        else:
            print("SHAP not installed; skipping SHAP explanations. 'pip install shap' to enable.")
    except Exception as e:
        print(f"SHAP computation failed: {e}")
    print(f"LightGBM {horizon} -> RMSE: {metrics['rmse']:.3f} | MAE: {metrics['mae']:.3f} | R2: {metrics['r2']:.3f}")
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LightGBM models for AQI D+1..D+3")
    p.add_argument("--parquet", type=str, default=DEFAULT_PARQUET)
    p.add_argument("--registry", type=str, default=DEFAULT_REGISTRY)
    p.add_argument("--lgb_out", type=str, default=DEFAULT_LGB_OUT)
    p.add_argument("--holdout_days", type=int, default=90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tune", action="store_true", help="Enable small Optuna tuning on training window")
    p.add_argument("--tune_trials", type=int, default=25, help="Number of Optuna trials (default 25)")
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
        overall[target] = train_for_horizon(
            df,
            target,
            args.holdout_days,
            args.seed,
            args.registry,
            args.lgb_out,
            tune=args.tune,
            tune_trials=args.tune_trials,
        )

    # Write a LightGBM summary in the output folder
    os.makedirs(args.lgb_out, exist_ok=True)
    summary_path = os.path.join(args.lgb_out, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)
    print(f"Saved LightGBM summary -> {summary_path}")


if __name__ == "__main__":
    main()



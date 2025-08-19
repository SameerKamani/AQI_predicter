import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import shap  # type: ignore
except Exception:
    shap = None

try:
    import optuna  # type: ignore
except Exception:
    optuna = None


DEFAULT_PARQUET = os.path.join("..", "Data", "feature_store", "karachi_daily_features.parquet")
DEFAULT_REGISTRY = os.path.join("Models", "registry")
DEFAULT_OUT = os.path.join("EDA", "rf_output")


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {
        "date",
        "event_timestamp",
        "created",
        "city",
        "karachi_id",
        # targets
        "AQI_t+1",
        "AQI_t+2",
        "AQI_t+3",
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
    return coerced.fillna(0.0)


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


def _cv_rmse_rf(X: pd.DataFrame, y: pd.Series, params: Dict[str, object], seed: int) -> float:
    Xn = _coerce_numeric_dataframe(X)
    splits = _rolling_time_slices(len(Xn), n_splits=3)
    rmses: List[float] = []
    for tr, va in splits:
        X_tr, y_tr = Xn.iloc[tr], y.iloc[tr]
        X_va, y_va = Xn.iloc[va], y.iloc[va]
        model = RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 400)),
            max_depth=params.get("max_depth", None),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        rmse = float(np.sqrt(((pred - y_va.values) ** 2).mean()))
        rmses.append(rmse)
    return float(np.mean(rmses)) if rmses else float("inf")


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


def train_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
    tuned_overrides: Optional[Dict[str, object]] = None,
) -> RandomForestRegressor:
    params: Dict[str, object] = dict(
        n_estimators=600,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
    )
    if tuned_overrides:
        params.update(tuned_overrides)
    model = RandomForestRegressor(**params)
    model.fit(_coerce_numeric_dataframe(X_train), y_train)
    return model


def save_artifacts(
    model: RandomForestRegressor,
    feature_cols: List[str],
    horizon: str,
    registry_dir: str,
    out_dir: str,
    metrics: Dict[str, float],
    preds_df: pd.DataFrame,
) -> None:
    os.makedirs(registry_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base = f"rf_{horizon}_{ts}"

    model_path = os.path.join(registry_dir, base + ".joblib")
    meta_path = os.path.join(registry_dir, base + "_meta.json")
    preds_path = os.path.join(registry_dir, base + "_preds.csv")
    fi_path_registry = os.path.join(registry_dir, base + "_feature_importance.csv")

    try:
        import joblib  # type: ignore

        joblib.dump({"model": model, "features": feature_cols}, model_path)
    except Exception:
        pass
    preds_df.to_csv(preds_path, index=False)

    # Feature importance by impurity decrease
    try:
        gains = getattr(model, "feature_importances_", None)
        if gains is not None:
            fi_df = (
                pd.DataFrame({"feature": feature_cols, "importance": gains})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
            fi_df.to_csv(fi_path_registry, index=False)
            # Also write to model-specific EDA folder for convenience
            fi_path_out = os.path.join(out_dir, f"feature_importance_{horizon}.csv")
            fi_df.to_csv(fi_path_out, index=False)
    except Exception:
        pass

    meta = {
        "horizon": horizon,
        "created_utc": ts,
        "model_path": model_path,
        "metrics": metrics,
        "predictions_path": preds_path,
        "feature_importance_registry": fi_path_registry,
        "library": "sklearn.RandomForestRegressor",
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
    tune_trials: int = 25,
) -> Dict[str, float]:
    X_train, X_test, y_train, y_test = time_split(df, target_col, holdout_days)

    tuned_overrides: Optional[Dict[str, object]] = None
    if tune and optuna is not None and len(X_train) >= 50:
        def objective(trial: "optuna.trial.Trial") -> float:
            space = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
                "max_depth": trial.suggest_int("max_depth", 4, 32),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.8, 0.6]),
            }
            return _cv_rmse_rf(X_train, y_train, space, seed)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=tune_trials, show_progress_bar=False)
        tuned_overrides = study.best_params
        print(f"Best tuned params (RF {target_col}): {tuned_overrides}")
    elif tune and optuna is None:
        print("Optuna not installed; skipping RF tuning. 'pip install optuna' to enable.")

    model = train_rf(X_train, y_train, seed, tuned_overrides)
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

    # SHAP global importance (optional)
    try:
        if shap is not None:
            X_shap = _coerce_numeric_dataframe(X_test.copy())
            if len(X_shap) > 5000:
                X_shap = X_shap.sample(5000, random_state=seed)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)
            shap_arr = shap_values if not isinstance(shap_values, list) else shap_values[0]
            mean_abs = np.mean(np.abs(shap_arr), axis=0)
            shap_df = (
                pd.DataFrame({"feature": X_shap.columns, "mean_abs_shap": mean_abs})
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )
            shap_out_dir = os.path.join("EDA", "shap_output")
            os.makedirs(shap_out_dir, exist_ok=True)
            shap_path = os.path.join(shap_out_dir, f"shap_global_rf_{horizon}.csv")
            shap_df.to_csv(shap_path, index=False)
            print(f"Saved SHAP (RF) global importance -> {shap_path}")
        else:
            print("SHAP not installed; skipping SHAP explanations. 'pip install shap' to enable.")
    except Exception as e:
        print(f"SHAP (RF) computation failed: {e}")

    print(
        f"RandomForest {horizon} -> RMSE: {metrics['rmse']:.3f} | MAE: {metrics['mae']:.3f} | R2: {metrics['r2']:.3f}"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RandomForest models for AQI D+1..D+3")
    p.add_argument("--parquet", type=str, default=DEFAULT_PARQUET)
    p.add_argument("--registry", type=str, default=DEFAULT_REGISTRY)
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    p.add_argument("--holdout_days", type=int, default=90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tune", action="store_true", help="Enable Optuna tuning")
    p.add_argument("--tune_trials", type=int, default=25)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.parquet):
        raise FileNotFoundError(f"Parquet not found: {args.parquet}")
    df = pd.read_parquet(args.parquet)

    required = {"event_timestamp", "AQI", "AQI_t+1", "AQI_t+2", "AQI_t+3"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    overall: Dict[str, Dict[str, float]] = {}
    for target in ["AQI_t+1", "AQI_t+2", "AQI_t+3"]:
        overall[target] = train_for_horizon(
            df,
            target,
            args.holdout_days,
            args.seed,
            args.registry,
            args.out,
            tune=args.tune,
            tune_trials=args.tune_trials,
        )

    os.makedirs(args.out, exist_ok=True)
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)
    print(f"Saved RF summary -> {summary_path}")


if __name__ == "__main__":
    main()




from __future__ import annotations

import os
from typing import Optional

import pandas as pd


def get_latest_offline_row(features_parquet: str) -> Optional[pd.Series]:
    if not os.path.exists(features_parquet):
        return None
    df = pd.read_parquet(features_parquet)
    if df.empty:
        return None
    df = df.sort_values("event_timestamp").reset_index(drop=True)
    return df.iloc[-1]


def get_latest_features(feast_repo_path: str, features_parquet: str) -> Optional[pd.Series]:
    """Try Feast online first; fallback to offline last row.

    Returns a pandas Series representing a single latest feature row.
    """
    try:
        from feast import FeatureStore  # type: ignore

        store = FeatureStore(repo_path=feast_repo_path)
        features = [
            "karachi_air_quality_daily:aqi_daily",
            "karachi_air_quality_daily:pm2_5_mean",
            "karachi_air_quality_daily:pm10_mean",
            "karachi_air_quality_daily:co_mean",
            "karachi_air_quality_daily:no_mean",
            "karachi_air_quality_daily:no2_mean",
            "karachi_air_quality_daily:o3_mean",
            "karachi_air_quality_daily:so2_mean",
            "karachi_air_quality_daily:nh3_mean",
            "karachi_air_quality_daily:day_of_week",
            "karachi_air_quality_daily:day_of_year",
            "karachi_air_quality_daily:month",
            "karachi_air_quality_daily:is_weekend",
            "karachi_air_quality_daily:aqi_daily_lag1",
            "karachi_air_quality_daily:aqi_daily_lag2",
            "karachi_air_quality_daily:aqi_daily_lag3",
            "karachi_air_quality_daily:pm2_5_mean_lag1",
            "karachi_air_quality_daily:pm2_5_mean_lag2",
            "karachi_air_quality_daily:pm2_5_mean_lag3",
            "karachi_air_quality_daily:pm10_mean_lag1",
            "karachi_air_quality_daily:pm10_mean_lag2",
            "karachi_air_quality_daily:pm10_mean_lag3",
            "karachi_air_quality_daily:aqi_roll_mean_3",
            "karachi_air_quality_daily:aqi_roll_std_7",
            "karachi_air_quality_daily:aqi_change_rate",
        ]
        online = store.get_online_features(features=features, entity_rows=[{"karachi_id": 1}]).to_dict()
        # Feast returns dict of lists → create a 1-row Series
        row = {k.split(":", 1)[-1]: (v[0] if isinstance(v, list) else v) for k, v in online.items()}
        # Add timestamp if available offline
        off = get_latest_offline_row(features_parquet)
        if off is not None and "event_timestamp" in off.index:
            row["event_timestamp"] = off["event_timestamp"]
        return pd.Series(row)
    except Exception:
        return get_latest_offline_row(features_parquet)



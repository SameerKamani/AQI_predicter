from __future__ import annotations

import os
from typing import Optional

import pandas as pd


def get_latest_offline_row(features_path: str) -> Optional[pd.Series]:
    if not os.path.exists(features_path):
        return None
    
    # Handle both CSV and parquet files
    if features_path.endswith('.csv'):
        df = pd.read_csv(features_path)
    elif features_path.endswith('.parquet'):
        df = pd.read_parquet(features_path)
    else:
        # Try to auto-detect
        try:
            df = pd.read_csv(features_path)
        except:
            try:
                df = pd.read_parquet(features_path)
            except:
                return None
    
    if df.empty:
        return None
    df = df.sort_values("event_timestamp").reset_index(drop=True)
    return df.iloc[-1]


def get_latest_features(feast_repo_path: str, features_path: str) -> Optional[pd.Series]:
    """Try Feast online first; fallback to offline last row.

    Returns a pandas Series representing a single latest feature row.
    """
    try:
        from feast import FeatureStore  # type: ignore

        store = FeatureStore(repo_path=feast_repo_path)
        features = [
            # Core daily means - match the actual data structure from data_fetch.py
            "karachi_air_quality_daily:AQI",
            "karachi_air_quality_daily:PM2.5",
            "karachi_air_quality_daily:PM10",
            "karachi_air_quality_daily:NO2",
            "karachi_air_quality_daily:SO2",
            "karachi_air_quality_daily:CO",
            "karachi_air_quality_daily:O3",
            "karachi_air_quality_daily:Temperature",
            "karachi_air_quality_daily:Humidity",
            "karachi_air_quality_daily:Precipitation",

            # Temporal features
            "karachi_air_quality_daily:month",
            "karachi_air_quality_daily:log_PM2.5",
            "karachi_air_quality_daily:log_CO",
            
            # Lag features
            "karachi_air_quality_daily:AQI_lag_1",
            "karachi_air_quality_daily:AQI_lag_2",
            
            # Rolling features
            "karachi_air_quality_daily:AQI_roll_mean_3",
            "karachi_air_quality_daily:AQI_roll_std_3",
            
            # Difference features
            "karachi_air_quality_daily:AQI_diff",
            
            # Target variables (for training, not prediction)
            "karachi_air_quality_daily:AQI_t+1",
            "karachi_air_quality_daily:AQI_t+2",
            "karachi_air_quality_daily:AQI_t+3",
            
            # One-hot encoded features
            "karachi_air_quality_daily:season_Spring",
            "karachi_air_quality_daily:season_Summer",
            "karachi_air_quality_daily:season_Winter",
            "karachi_air_quality_daily:weekday_1",
            "karachi_air_quality_daily:weekday_2",
            "karachi_air_quality_daily:weekday_3",
            "karachi_air_quality_daily:weekday_4",
            "karachi_air_quality_daily:weekday_5",
            "karachi_air_quality_daily:weekday_6",
        ]
        online = store.get_online_features(features=features, entity_rows=[{"karachi_id": "karachi_001"}]).to_dict()
        # Feast returns dict of lists → create a 1-row Series
        row = {k.split(":", 1)[-1]: (v[0] if isinstance(v, list) else v) for k, v in online.items()}
        # Add timestamp if available offline
        off = get_latest_offline_row(features_path)
        if off is not None and "event_timestamp" in off.index:
            row["event_timestamp"] = off["event_timestamp"]
        return pd.Series(row)
    except Exception:
        return get_latest_offline_row(features_path)



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
            # Core daily means
            "karachi_air_quality_daily:aqi_daily",
            "karachi_air_quality_daily:pm2_5_mean",
            "karachi_air_quality_daily:pm10_mean",
            "karachi_air_quality_daily:co_mean",
            "karachi_air_quality_daily:no_mean",
            "karachi_air_quality_daily:no2_mean",
            "karachi_air_quality_daily:o3_mean",
            "karachi_air_quality_daily:so2_mean",
            "karachi_air_quality_daily:nh3_mean",
            "karachi_air_quality_daily:o3_8h_max_ppb",

            # Calendar/time features
            "karachi_air_quality_daily:day_of_week",
            "karachi_air_quality_daily:day_of_year",
            "karachi_air_quality_daily:month",
            "karachi_air_quality_daily:is_weekend",
            "karachi_air_quality_daily:week_of_year",
            "karachi_air_quality_daily:doy_sin",
            "karachi_air_quality_daily:doy_cos",

            # Lags for AQI and particulates (1..7)
            "karachi_air_quality_daily:aqi_daily_lag1",
            "karachi_air_quality_daily:aqi_daily_lag2",
            "karachi_air_quality_daily:aqi_daily_lag3",
            "karachi_air_quality_daily:aqi_daily_lag4",
            "karachi_air_quality_daily:aqi_daily_lag5",
            "karachi_air_quality_daily:aqi_daily_lag6",
            "karachi_air_quality_daily:aqi_daily_lag7",

            "karachi_air_quality_daily:pm2_5_mean_lag1",
            "karachi_air_quality_daily:pm2_5_mean_lag2",
            "karachi_air_quality_daily:pm2_5_mean_lag3",
            "karachi_air_quality_daily:pm2_5_mean_lag4",
            "karachi_air_quality_daily:pm2_5_mean_lag5",
            "karachi_air_quality_daily:pm2_5_mean_lag6",
            "karachi_air_quality_daily:pm2_5_mean_lag7",

            "karachi_air_quality_daily:pm10_mean_lag1",
            "karachi_air_quality_daily:pm10_mean_lag2",
            "karachi_air_quality_daily:pm10_mean_lag3",
            "karachi_air_quality_daily:pm10_mean_lag4",
            "karachi_air_quality_daily:pm10_mean_lag5",
            "karachi_air_quality_daily:pm10_mean_lag6",
            "karachi_air_quality_daily:pm10_mean_lag7",

            # Rolling stats (AQI, PM2.5, PM10) for 3/7/14/30
            "karachi_air_quality_daily:aqi_roll_mean_3",
            "karachi_air_quality_daily:aqi_roll_mean_7",
            "karachi_air_quality_daily:aqi_roll_mean_14",
            "karachi_air_quality_daily:aqi_roll_mean_30",
            "karachi_air_quality_daily:aqi_roll_std_3",
            "karachi_air_quality_daily:aqi_roll_std_7",
            "karachi_air_quality_daily:aqi_roll_std_14",
            "karachi_air_quality_daily:aqi_roll_std_30",

            "karachi_air_quality_daily:pm25_roll_mean_3",
            "karachi_air_quality_daily:pm25_roll_mean_7",
            "karachi_air_quality_daily:pm25_roll_mean_14",
            "karachi_air_quality_daily:pm25_roll_mean_30",
            "karachi_air_quality_daily:pm25_roll_std_3",
            "karachi_air_quality_daily:pm25_roll_std_7",
            "karachi_air_quality_daily:pm25_roll_std_14",
            "karachi_air_quality_daily:pm25_roll_std_30",

            "karachi_air_quality_daily:pm10_roll_mean_3",
            "karachi_air_quality_daily:pm10_roll_mean_7",
            "karachi_air_quality_daily:pm10_roll_mean_14",
            "karachi_air_quality_daily:pm10_roll_mean_30",
            "karachi_air_quality_daily:pm10_roll_std_3",
            "karachi_air_quality_daily:pm10_roll_std_7",
            "karachi_air_quality_daily:pm10_roll_std_14",
            "karachi_air_quality_daily:pm10_roll_std_30",

            # Weather aggregates (if present)
            "karachi_air_quality_daily:temp_mean",
            "karachi_air_quality_daily:temp_min",
            "karachi_air_quality_daily:temp_max",
            "karachi_air_quality_daily:humidity_mean",
            "karachi_air_quality_daily:dew_point_mean",
            "karachi_air_quality_daily:pressure_mean",
            "karachi_air_quality_daily:wind_speed_mean",
            "karachi_air_quality_daily:wind_speed_max",
            "karachi_air_quality_daily:wind_gust_max",
            "karachi_air_quality_daily:clouds_mean",
            "karachi_air_quality_daily:visibility_mean",
            "karachi_air_quality_daily:rain_sum",
            "karachi_air_quality_daily:snow_sum",
            "karachi_air_quality_daily:wind_u_mean",
            "karachi_air_quality_daily:wind_v_mean",
            "karachi_air_quality_daily:temp_range",

            # Weather lags/rollings for selected vars
            "karachi_air_quality_daily:temp_mean_lag1",
            "karachi_air_quality_daily:temp_mean_lag2",
            "karachi_air_quality_daily:temp_mean_lag3",
            "karachi_air_quality_daily:temp_mean_roll_mean_3",
            "karachi_air_quality_daily:temp_mean_roll_mean_7",
            "karachi_air_quality_daily:temp_mean_roll_std_3",
            "karachi_air_quality_daily:temp_mean_roll_std_7",

            "karachi_air_quality_daily:humidity_mean_lag1",
            "karachi_air_quality_daily:humidity_mean_lag2",
            "karachi_air_quality_daily:humidity_mean_lag3",
            "karachi_air_quality_daily:humidity_mean_roll_mean_3",
            "karachi_air_quality_daily:humidity_mean_roll_mean_7",
            "karachi_air_quality_daily:humidity_mean_roll_std_3",
            "karachi_air_quality_daily:humidity_mean_roll_std_7",

            "karachi_air_quality_daily:wind_u_mean_lag1",
            "karachi_air_quality_daily:wind_u_mean_lag2",
            "karachi_air_quality_daily:wind_u_mean_lag3",
            "karachi_air_quality_daily:wind_u_mean_roll_mean_3",
            "karachi_air_quality_daily:wind_u_mean_roll_mean_7",
            "karachi_air_quality_daily:wind_u_mean_roll_std_3",
            "karachi_air_quality_daily:wind_u_mean_roll_std_7",

            "karachi_air_quality_daily:wind_v_mean_lag1",
            "karachi_air_quality_daily:wind_v_mean_lag2",
            "karachi_air_quality_daily:wind_v_mean_lag3",
            "karachi_air_quality_daily:wind_v_mean_roll_mean_3",
            "karachi_air_quality_daily:wind_v_mean_roll_mean_7",
            "karachi_air_quality_daily:wind_v_mean_roll_std_3",
            "karachi_air_quality_daily:wind_v_mean_roll_std_7",

            # Dynamics and targets
            "karachi_air_quality_daily:aqi_change_rate",
            "karachi_air_quality_daily:target_aqi_d1",
            "karachi_air_quality_daily:target_aqi_d2",
            "karachi_air_quality_daily:target_aqi_d3",

            # Data quality and completeness features
            "karachi_air_quality_daily:num_hours",
            "karachi_air_quality_daily:is_complete_day",
            "karachi_air_quality_daily:imputed",
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



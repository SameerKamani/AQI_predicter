from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, ValueType


# Offline source points to the Parquet we generate under Data/feature_store
# Note: path is relative to this repo directory (feature_repo)
karachi_source = FileSource(
    path="../Data/feature_store/karachi_daily_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)


# Entity for Karachi; we use a static key column "karachi_id" present in the dataset
karachi = Entity(
    name="karachi_id",
    value_type=ValueType.INT64,
    join_keys=["karachi_id"],
    description="Karachi city identifier",
)


karachi_fv = FeatureView(
    name="karachi_air_quality_daily",
    entities=[karachi],
    ttl=timedelta(days=2),
    schema=[
        # Core daily means
        Field(name="aqi_daily", dtype=Float32),
        Field(name="pm2_5_mean", dtype=Float32),
        Field(name="pm10_mean", dtype=Float32),
        Field(name="co_mean", dtype=Float32),
        Field(name="no_mean", dtype=Float32),
        Field(name="no2_mean", dtype=Float32),
        Field(name="o3_mean", dtype=Float32),
        Field(name="so2_mean", dtype=Float32),
        Field(name="nh3_mean", dtype=Float32),
        Field(name="o3_8h_max_ppb", dtype=Float32),

        # Calendar/time features
        Field(name="day_of_week", dtype=Float32),
        Field(name="day_of_year", dtype=Float32),
        Field(name="month", dtype=Float32),
        Field(name="is_weekend", dtype=Float32),
        Field(name="week_of_year", dtype=Float32),
        Field(name="doy_sin", dtype=Float32),
        Field(name="doy_cos", dtype=Float32),

        # Lags for AQI and particulates (1..7)
        Field(name="aqi_daily_lag1", dtype=Float32),
        Field(name="aqi_daily_lag2", dtype=Float32),
        Field(name="aqi_daily_lag3", dtype=Float32),
        Field(name="aqi_daily_lag4", dtype=Float32),
        Field(name="aqi_daily_lag5", dtype=Float32),
        Field(name="aqi_daily_lag6", dtype=Float32),
        Field(name="aqi_daily_lag7", dtype=Float32),

        Field(name="pm2_5_mean_lag1", dtype=Float32),
        Field(name="pm2_5_mean_lag2", dtype=Float32),
        Field(name="pm2_5_mean_lag3", dtype=Float32),
        Field(name="pm2_5_mean_lag4", dtype=Float32),
        Field(name="pm2_5_mean_lag5", dtype=Float32),
        Field(name="pm2_5_mean_lag6", dtype=Float32),
        Field(name="pm2_5_mean_lag7", dtype=Float32),

        Field(name="pm10_mean_lag1", dtype=Float32),
        Field(name="pm10_mean_lag2", dtype=Float32),
        Field(name="pm10_mean_lag3", dtype=Float32),
        Field(name="pm10_mean_lag4", dtype=Float32),
        Field(name="pm10_mean_lag5", dtype=Float32),
        Field(name="pm10_mean_lag6", dtype=Float32),
        Field(name="pm10_mean_lag7", dtype=Float32),

        # Rolling stats (AQI, PM2.5, PM10) for 3/7/14/30
        Field(name="aqi_roll_mean_3", dtype=Float32),
        Field(name="aqi_roll_mean_7", dtype=Float32),
        Field(name="aqi_roll_mean_14", dtype=Float32),
        Field(name="aqi_roll_mean_30", dtype=Float32),
        Field(name="aqi_roll_std_3", dtype=Float32),
        Field(name="aqi_roll_std_7", dtype=Float32),
        Field(name="aqi_roll_std_14", dtype=Float32),
        Field(name="aqi_roll_std_30", dtype=Float32),

        Field(name="pm25_roll_mean_3", dtype=Float32),
        Field(name="pm25_roll_mean_7", dtype=Float32),
        Field(name="pm25_roll_mean_14", dtype=Float32),
        Field(name="pm25_roll_mean_30", dtype=Float32),
        Field(name="pm25_roll_std_3", dtype=Float32),
        Field(name="pm25_roll_std_7", dtype=Float32),
        Field(name="pm25_roll_std_14", dtype=Float32),
        Field(name="pm25_roll_std_30", dtype=Float32),

        Field(name="pm10_roll_mean_3", dtype=Float32),
        Field(name="pm10_roll_mean_7", dtype=Float32),
        Field(name="pm10_roll_mean_14", dtype=Float32),
        Field(name="pm10_roll_mean_30", dtype=Float32),
        Field(name="pm10_roll_std_3", dtype=Float32),
        Field(name="pm10_roll_std_7", dtype=Float32),
        Field(name="pm10_roll_std_14", dtype=Float32),
        Field(name="pm10_roll_std_30", dtype=Float32),

        # Weather aggregates (if present)
        Field(name="temp_mean", dtype=Float32),
        Field(name="temp_min", dtype=Float32),
        Field(name="temp_max", dtype=Float32),
        Field(name="humidity_mean", dtype=Float32),
        Field(name="dew_point_mean", dtype=Float32),
        Field(name="pressure_mean", dtype=Float32),
        Field(name="wind_speed_mean", dtype=Float32),
        Field(name="wind_speed_max", dtype=Float32),
        Field(name="wind_gust_max", dtype=Float32),
        Field(name="clouds_mean", dtype=Float32),
        Field(name="visibility_mean", dtype=Float32),
        Field(name="rain_sum", dtype=Float32),
        Field(name="snow_sum", dtype=Float32),
        Field(name="wind_u_mean", dtype=Float32),
        Field(name="wind_v_mean", dtype=Float32),
        Field(name="temp_range", dtype=Float32),

        # Weather lags/rollings for selected vars
        Field(name="temp_mean_lag1", dtype=Float32),
        Field(name="temp_mean_lag2", dtype=Float32),
        Field(name="temp_mean_lag3", dtype=Float32),
        Field(name="temp_mean_roll_mean_3", dtype=Float32),
        Field(name="temp_mean_roll_mean_7", dtype=Float32),
        Field(name="temp_mean_roll_std_3", dtype=Float32),
        Field(name="temp_mean_roll_std_7", dtype=Float32),

        Field(name="humidity_mean_lag1", dtype=Float32),
        Field(name="humidity_mean_lag2", dtype=Float32),
        Field(name="humidity_mean_lag3", dtype=Float32),
        Field(name="humidity_mean_roll_mean_3", dtype=Float32),
        Field(name="humidity_mean_roll_mean_7", dtype=Float32),
        Field(name="humidity_mean_roll_std_3", dtype=Float32),
        Field(name="humidity_mean_roll_std_7", dtype=Float32),

        Field(name="wind_u_mean_lag1", dtype=Float32),
        Field(name="wind_u_mean_lag2", dtype=Float32),
        Field(name="wind_u_mean_lag3", dtype=Float32),
        Field(name="wind_u_mean_roll_mean_3", dtype=Float32),
        Field(name="wind_u_mean_roll_mean_7", dtype=Float32),
        Field(name="wind_u_mean_roll_std_3", dtype=Float32),
        Field(name="wind_u_mean_roll_std_7", dtype=Float32),

        Field(name="wind_v_mean_lag1", dtype=Float32),
        Field(name="wind_v_mean_lag2", dtype=Float32),
        Field(name="wind_v_mean_lag3", dtype=Float32),
        Field(name="wind_v_mean_roll_mean_3", dtype=Float32),
        Field(name="wind_v_mean_roll_mean_7", dtype=Float32),
        Field(name="wind_v_mean_roll_std_3", dtype=Float32),
        Field(name="wind_v_mean_roll_std_7", dtype=Float32),

        # Dynamics and targets
        Field(name="aqi_change_rate", dtype=Float32),
        Field(name="target_aqi_d1", dtype=Float32),
        Field(name="target_aqi_d2", dtype=Float32),
        Field(name="target_aqi_d3", dtype=Float32),

        # Data quality and completeness features
        Field(name="num_hours", dtype=Float32),
        Field(name="is_complete_day", dtype=Float32),
        Field(name="imputed", dtype=Float32),
    ],
    source=karachi_source,
)



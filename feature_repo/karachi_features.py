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
        Field(name="aqi_daily", dtype=Float32),
        Field(name="pm2_5_mean", dtype=Float32),
        Field(name="pm10_mean", dtype=Float32),
        Field(name="co_mean", dtype=Float32),
        Field(name="no_mean", dtype=Float32),
        Field(name="no2_mean", dtype=Float32),
        Field(name="o3_mean", dtype=Float32),
        Field(name="so2_mean", dtype=Float32),
        Field(name="nh3_mean", dtype=Float32),
        Field(name="day_of_week", dtype=Float32),
        Field(name="day_of_year", dtype=Float32),
        Field(name="month", dtype=Float32),
        Field(name="is_weekend", dtype=Float32),
        Field(name="aqi_daily_lag1", dtype=Float32),
        Field(name="aqi_daily_lag2", dtype=Float32),
        Field(name="aqi_daily_lag3", dtype=Float32),
        Field(name="pm2_5_mean_lag1", dtype=Float32),
        Field(name="pm2_5_mean_lag2", dtype=Float32),
        Field(name="pm2_5_mean_lag3", dtype=Float32),
        Field(name="pm10_mean_lag1", dtype=Float32),
        Field(name="pm10_mean_lag2", dtype=Float32),
        Field(name="pm10_mean_lag3", dtype=Float32),
        Field(name="aqi_roll_mean_3", dtype=Float32),
        Field(name="aqi_roll_std_7", dtype=Float32),
        Field(name="aqi_change_rate", dtype=Float32),
        Field(name="target_aqi_d1", dtype=Float32),
        Field(name="target_aqi_d2", dtype=Float32),
        Field(name="target_aqi_d3", dtype=Float32),
    ],
    source=karachi_source,
)



from datetime import timedelta
import os
import pandas as pd

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String, ValueType
from feast.data_format import ParquetFormat


def convert_csv_to_parquet(csv_file_path):
    """
    Converts a CSV file to Parquet format if the corresponding Parquet file doesn't exist.
    Also fixes timestamp columns to be proper datetime objects with timezone info.
    
    Parameters:
    - csv_file_path (str): Path to the CSV file.
    
    Returns:
    - str: Path to the Parquet file.
    """
    # Determine the Parquet file path
    parquet_file_path = csv_file_path.replace('.csv', '.parquet')
    
    # Check if the Parquet file already exists
    if not os.path.exists(parquet_file_path):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)
        
        # Fix timestamp columns - convert strings to proper datetime objects
        if 'event_timestamp' in df.columns:
            # Handle date-only format like "2022-09-03"
            df['event_timestamp'] = pd.to_datetime(df['event_timestamp']).dt.tz_localize('UTC')
        
        if 'created' in df.columns:
            # Handle datetime format like "2025-08-19 01:18:29.762929"
            df['created'] = pd.to_datetime(df['created']).dt.tz_localize('UTC')
        
        # Write the DataFrame to a Parquet file with proper timestamp format
        df.to_parquet(parquet_file_path, index=False, engine='pyarrow')
        
        print(f"Converted {csv_file_path} to {parquet_file_path}")
        print(f"Fixed timestamp columns: event_timestamp and created are now proper datetime objects")
    else:
        print(f"Parquet file {parquet_file_path} already exists. Skipping conversion.")
    
    return parquet_file_path


# Convert CSV to Parquet if needed
csv_path = "../Data/feature_store/karachi_daily_features.csv"
parquet_path = convert_csv_to_parquet(csv_path)


# Offline source points to the Parquet file we generate under Data/feature_store
# Note: path is relative to this repo directory (feature_repo)
karachi_source = FileSource(
    path=parquet_path,
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
    file_format=ParquetFormat()
)


# Entity for Karachi; we use a static key column "karachi_id" present in the dataset
karachi = Entity(
    name="karachi_id",
    value_type=ValueType.STRING,  # Changed to STRING since your script uses "karachi_001"
    join_keys=["karachi_id"],
    description="Karachi city identifier",
)


karachi_fv = FeatureView(
    name="karachi_air_quality_daily",
    entities=[karachi],
    ttl=timedelta(days=2),
    schema=[
        # =============================================================================
        # CORE FEATURES (Exactly what data_fetch.py creates)
        # =============================================================================
        Field(name="AQI", dtype=Float32),
        Field(name="PM2.5", dtype=Float32),
        Field(name="PM10", dtype=Float32),
        Field(name="NO2", dtype=Float32),
        Field(name="SO2", dtype=Float32),
        Field(name="CO", dtype=Float32),
        Field(name="O3", dtype=Float32),
        Field(name="Temperature", dtype=Float32),
        Field(name="Humidity", dtype=Float32),
        Field(name="Precipitation", dtype=Float32),
        
        # =============================================================================
        # TEMPORAL FEATURES (Exactly what data_fetch.py creates)
        # =============================================================================
        Field(name="date", dtype=String),
        Field(name="month", dtype=Int64),
        
        # =============================================================================
        # TRANSFORMED FEATURES (Exactly what data_fetch.py creates)
        # =============================================================================
        Field(name="log_PM2.5", dtype=Float32),
        Field(name="log_CO", dtype=Float32),
        
        # =============================================================================
        # LAG FEATURES (Exactly what data_fetch.py creates)
        # =============================================================================
        Field(name="AQI_lag_1", dtype=Float32),
        Field(name="AQI_lag_2", dtype=Float32),
        
        # =============================================================================
        # ROLLING FEATURES (Exactly what data_fetch.py creates)
        # =============================================================================
        Field(name="AQI_roll_mean_3", dtype=Float32),
        Field(name="AQI_roll_std_3", dtype=Float32),
        
        # =============================================================================
        # DIFFERENCE FEATURES (Exactly what data_fetch.py creates)
        # =============================================================================
        Field(name="AQI_diff", dtype=Float32),
        
        # =============================================================================
        # TARGET VARIABLES (Exactly what data_fetch.py creates)
        # =============================================================================
        Field(name="AQI_t+1", dtype=Float32),
        Field(name="AQI_t+2", dtype=Float32),
        Field(name="AQI_t+3", dtype=Float32),
        
        # =============================================================================
        # ONE-HOT ENCODED FEATURES (Exactly what data_fetch.py creates)
        # =============================================================================
        Field(name="season_Spring", dtype=Int64),
        Field(name="season_Summer", dtype=Int64),
        Field(name="season_Winter", dtype=Int64),
        Field(name="weekday_1", dtype=Int64),
        Field(name="weekday_2", dtype=Int64),
        Field(name="weekday_3", dtype=Int64),
        Field(name="weekday_4", dtype=Int64),
        Field(name="weekday_5", dtype=Int64),
        Field(name="weekday_6", dtype=Int64),  # Sunday
        
        # =============================================================================
        # METADATA (Required by Feast)
        # =============================================================================
        Field(name="karachi_id", dtype=String),
        Field(name="event_timestamp", dtype=String),
        Field(name="created", dtype=String),
    ],
    source=karachi_source,
)

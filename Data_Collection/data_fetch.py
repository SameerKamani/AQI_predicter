import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# UTILITY FUNCTIONS (like Sheema)
# =============================================================================

def iqr_cap(df, column):
    """Apply IQR capping to remove outliers (like Sheema)"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[column].clip(lower, upper)

# =============================================================================
# CONFIGURATION
# =============================================================================
LAT, LON = 24.8607, 67.0011
TIMEZONE = "Asia/Karachi"

# Data paths
DATA_DIR = "Data"
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "karachi_features_enhanced.csv")
FEATURE_STORE_PATH = os.path.join(DATA_DIR, "feature_store", "karachi_daily_features.csv")

# =============================================================================
# DATA FETCHING FUNCTIONS
# =============================================================================

def fetch_day_data(day):
    """Fetch air quality and weather data for a specific day"""
    try:
        # Air Quality API
        air_url = (
            "https://air-quality-api.open-meteo.com/v1/air-quality"
            f"?latitude={LAT}&longitude={LON}"
            f"&start_date={day}&end_date={day}"
            "&hourly=us_aqi,pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone"
            f"&timezone={TIMEZONE}"
        )
        air_response = requests.get(air_url, timeout=30)
        if air_response.status_code != 200 or air_response.text.strip() == "":
            raise ValueError(f"Empty or failed air API response for {day}")

        air_data = air_response.json()["hourly"]

        # Weather API (only what Sheema uses)
        weather_url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={LAT}&longitude={LON}"
            f"&start_date={day}&end_date={day}"
            "&hourly=temperature_2m,relative_humidity_2m,precipitation"
            f"&timezone={TIMEZONE}"
        )
        weather_response = requests.get(weather_url, timeout=30)
        if weather_response.status_code != 200 or weather_response.text.strip() == "":
            raise ValueError(f"Empty or failed weather API response for {day}")

        weather_data = weather_response.json()["hourly"]

        # Merge data
        df_air = pd.DataFrame(air_data)
        df_weather = pd.DataFrame(weather_data)
        df = pd.merge(df_air, df_weather, on='time', how='outer')
        df["time"] = pd.to_datetime(df["time"])
        
        return df

    except Exception as e:
        print(f"ERROR: Failed on {day}: {e}")
        return None

def process_daily(day, df):
    """Process hourly data to daily aggregates (like Sheema)"""
    try:
        daily = df.set_index("time").resample("D").agg({
            'us_aqi': 'mean',           # AQI
            'pm2_5': 'mean',            # PM2.5
            'pm10': 'mean',             # PM10
            'nitrogen_dioxide': 'mean', # NO2
            'sulphur_dioxide': 'mean',  # SO2
            'carbon_monoxide': 'mean',  # CO
            'ozone': 'mean',            # O3
            'temperature_2m': 'mean',   # Temperature
            'relative_humidity_2m': 'mean', # Humidity
            'precipitation': 'sum'      # Precipitation
        }).round(3)
        
        # Rename columns to match expected names
        daily = daily.rename(columns={
            'us_aqi': 'AQI',
            'pm2_5': 'PM2.5',
            'pm10': 'PM10',
            'nitrogen_dioxide': 'NO2',
            'sulphur_dioxide': 'SO2',
            'carbon_monoxide': 'CO',
            'ozone': 'O3',
            'temperature_2m': 'Temperature',
            'relative_humidity_2m': 'Humidity',
            'precipitation': 'Precipitation'
        })
        
        # Just use event_timestamp - no need for separate date column
        daily["event_timestamp"] = pd.to_datetime(day)
        daily = daily.reset_index(drop=True)
        
        # Safety check: ensure we have a valid DataFrame
        if daily.empty:
            print(f"ERROR: Daily aggregation resulted in empty DataFrame for {day}")
            return None
        
        if 'event_timestamp' not in daily.columns:
            print(f"ERROR: event_timestamp column missing in daily data for {day}")
            return None
        
        # Ensure event_timestamp is datetime
        daily['event_timestamp'] = pd.to_datetime(daily['event_timestamp'])
        
        return daily

    except Exception as e:
        print(f"ERROR: Failed to process {day}: {e}")
        return None

# =============================================================================
# FEATURE ENGINEERING FUNCTIONS
# =============================================================================

def create_temporal_features(df):
    """Create temporal features (exactly like Sheema)"""
    print("Creating temporal features...")
    
    # Convert event_timestamp to datetime if it's not already
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    
    # Extract month
    df["month"] = df["event_timestamp"].dt.month
    
    # Create season (exactly like Sheema)
    df["season"] = df["month"].apply(lambda x: "Spring" if x in [3, 4, 5] else "Summer" if x in [6, 7, 8] else "Winter")
    
    # Create weekday (exactly like Sheema)
    df["weekday"] = df["event_timestamp"].dt.day_name()
    
    print("Temporal features created")
    return df

def create_statistical_features(df):
    """Create statistical features (exactly like Sheema)"""
    print("Creating statistical features...")
    
    # Logarithmic transformations (exactly like Sheema)
    df["log_PM2.5"] = np.log1p(df["PM2.5"])
    df["log_CO"] = np.log1p(df["CO"])
    
    # AQI lag features (exactly like Sheema)
    df["AQI_lag_1"] = df["AQI"].shift(1)
    df["AQI_lag_2"] = df["AQI"].shift(2)
    
    # Rolling statistics for AQI
    df["AQI_roll_mean_3"] = df["AQI"].rolling(window=3, min_periods=1).mean()
    df["AQI_roll_std_3"] = df["AQI"].rolling(window=3, min_periods=1).std()
    
    # Difference features
    df["AQI_diff"] = df["AQI"].diff()
    
    print("Statistical features created")
    return df

def create_target_features(df):
    """Create target variables (like Sheema)"""
    print("Creating target features...")
    
    # Future AQI targets (like Sheema)
    df["AQI_t+1"] = df["AQI"].shift(-1)
    df["AQI_t+2"] = df["AQI"].shift(-2)
    df["AQI_t+3"] = df["AQI"].shift(-3)
    
    return df

def clean_and_validate_data(df):
    """Clean and validate the final dataset (exactly like Sheema)"""
    print("Cleaning and validating data...")
    
    # Normalize timestamp column first
    if "event_timestamp" in df.columns:
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="coerce", utc=True)
    elif "date" in df.columns:
        # Fallback if upstream sent legacy column
        df["event_timestamp"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    else:
        print("ERROR: No timestamp column present!")
        return None
    
    # Remove legacy 'date' column if present
    if "date" in df.columns:
        df = df.drop(columns=["date"]) 
        print("Removed legacy 'date' column")
    
    # Apply IQR capping (exactly like Sheema)
    print("Applying IQR capping...")
    for col in ["PM10", "SO2", "NO2", "O3", "Temperature", "Humidity", "Precipitation"]:
        if col in df.columns:
            df[col] = iqr_cap(df, col)
            print(f"IQR capped: {col}")
    
    # One-hot encode season and weekday (exactly like Sheema)
    print("One-hot encoding season and weekday...")
    if "season" in df.columns and "weekday" in df.columns:
        df = pd.get_dummies(df, columns=["season", "weekday"], drop_first=True)
    
    # Do NOT drop rows due to missing targets for production feature store
    print("Leaving rows with missing target values intact for feature store")
    
    # Forward fill remaining missing values for non-target columns
    df = df.ffill()
    
    # Sort by timestamp
    df = df.sort_values("event_timestamp").reset_index(drop=True)
    
    # Add metadata (like Sheema)
    df["karachi_id"] = "karachi_001"
    df["created"] = pd.Timestamp.now(tz="UTC")
    
    # Final safety check - ensure event_timestamp exists and is datetime
    if "event_timestamp" not in df.columns:
        print("ERROR: event_timestamp column missing after processing!")
        return None
    
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="coerce", utc=True)
    
    print(f"Final dataset shape: {df.shape}")
    print(f"Date range: {df['event_timestamp'].min()} to {df['event_timestamp'].max()}")
    
    return df

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("Starting Data Fetch and Feature Extraction Pipeline")
    print("=" * 60)
    
    # Check if existing data exists
    existing_data = None
    if os.path.exists(FEATURE_STORE_PATH):
        try:
            existing_data = pd.read_csv(FEATURE_STORE_PATH)
            print(f"Found existing data: {len(existing_data)} records")
            
            # Handle existing CSV structure - it might have both 'date' and 'event_timestamp'
            if 'event_timestamp' in existing_data.columns:
                print(f"Data range: {existing_data['event_timestamp'].min()} to {existing_data['event_timestamp'].max()}")
            elif 'date' in existing_data.columns:
                # Convert old 'date' column to 'event_timestamp' for consistency
                existing_data['event_timestamp'] = pd.to_datetime(existing_data['date'])
                print(f"Converted old 'date' column to 'event_timestamp'")
                print(f"Data range: {existing_data['event_timestamp'].min()} to {existing_data['event_timestamp'].max()}")
            else:
                print("Warning: No date column found in existing data")
                
        except Exception as e:
            print(f"Warning: Could not read existing data: {e}")
            existing_data = None
    
    # Step 1: Fetch new data from OpenMeteo APIs
    print("\nSTEP 1: Fetching new data from OpenMeteo APIs...")
    
    # Get current date and fetch data for the last few days
    current_date = date.today()
    dates_to_fetch = [current_date - timedelta(days=i) for i in range(7)]  # Last 7 days
    
    # If we have existing data, only fetch dates we don't have
    if existing_data is not None and 'event_timestamp' in existing_data.columns:
        # Use the correct column name: 'event_timestamp'
        existing_dates = set(pd.to_datetime(existing_data['event_timestamp']).dt.date)
        dates_to_fetch = [d for d in dates_to_fetch if d not in existing_dates]
        print(f"Existing data covers: {sorted(existing_dates)}")
        print(f"Fetching missing dates: {sorted(dates_to_fetch)}")
        
        if not dates_to_fetch:
            print("All data is up to date! No new data needed.")
            return
    
    all_daily_data = []
    for day in dates_to_fetch:
        print(f"Fetching data for {day}...")
        hourly_data = fetch_day_data(day)
        if hourly_data is not None:
            daily_data = process_daily(day, hourly_data)
            if daily_data is not None:
                all_daily_data.append(daily_data)
                print(f"SUCCESS: Data fetched for {day}")
            else:
                print(f"ERROR: Failed to process data for {day}")
        else:
            print(f"ERROR: Failed to fetch data for {day}")
    
    if not all_daily_data:
        print("No new data fetched. Exiting.")
        return
    
    # Combine new data
    df_new = pd.concat(all_daily_data, ignore_index=True)
    print(f"Fetched {len(df_new)} new daily records")
    
    # Step 2: Feature engineering on new data
    print("\nSTEP 2: Feature engineering...")
    df_features = create_temporal_features(df_new)
    df_features = create_statistical_features(df_features)
    df_features = create_target_features(df_features)
    
    # Ensure event_timestamp is datetime before proceeding
    df_features['event_timestamp'] = pd.to_datetime(df_features['event_timestamp'])
    
    # Step 3: Clean and validate new data
    print("\nSTEP 3: Cleaning and validation...")
    df_final = clean_and_validate_data(df_features)
    
    # Check if cleaning failed
    if df_final is None:
        print("ERROR: Data cleaning and validation failed!")
        return
    
    # Ensure df_final event_timestamp is datetime
    df_final['event_timestamp'] = pd.to_datetime(df_final['event_timestamp'])
    
    # Step 4: Merge with existing data if available
    if existing_data is not None and 'event_timestamp' in existing_data.columns:
        print("\nSTEP 4: Merging with existing data...")
        # Convert existing data date to datetime for comparison
        existing_data['event_timestamp'] = pd.to_datetime(existing_data['event_timestamp'])
        
        # Debug: Check data types before accessing .dt
        print(f"DEBUG: existing_data['event_timestamp'] dtype: {existing_data['event_timestamp'].dtype}")
        print(f"DEBUG: df_final['event_timestamp'] dtype: {df_final['event_timestamp'].dtype}")
        print(f"DEBUG: df_final shape: {df_final.shape}")
        print(f"DEBUG: df_final columns: {list(df_final.columns)}")
        
        # Safety check: ensure event_timestamp is a Series with datetime dtype
        if not isinstance(df_final['event_timestamp'], pd.Series):
            print("ERROR: event_timestamp is not a pandas Series!")
            return
        
        if not pd.api.types.is_datetime64_any_dtype(df_final['event_timestamp']):
            print("ERROR: event_timestamp is not datetime dtype!")
            return
        
        # Now safely access .dt methods
        try:
            existing_dates = sorted(existing_data['event_timestamp'].dt.date.unique())
            new_dates = sorted(df_final['event_timestamp'].dt.date.unique())
            print(f"DEBUG: Existing data dates: {existing_dates}")
            print(f"DEBUG: New data dates: {new_dates}")
        except Exception as e:
            print(f"ERROR accessing .dt methods: {e}")
            return
        
        # Remove any overlapping dates from existing data
        existing_data = existing_data[~existing_data['event_timestamp'].isin(df_final['event_timestamp'])]
        
        # Combine existing and new data
        df_combined = pd.concat([existing_data, df_final], ignore_index=True)
        df_final = df_combined.sort_values('event_timestamp').reset_index(drop=True)
        print(f"Combined dataset: {len(df_final)} total records")
        
        # Final debug check after merge
        try:
            final_dates = sorted(df_final['event_timestamp'].dt.date.unique())
            print(f"DEBUG: Final dataset dates: {final_dates}")
        except Exception as e:
            print(f"ERROR accessing final dates: {e}")
            return
    
    # Final safety check - ensure event_timestamp is datetime
    df_final['event_timestamp'] = pd.to_datetime(df_final['event_timestamp'])
    
    # Comprehensive safety checks before any .dt operations
    print(f"DEBUG: df_final shape: {df_final.shape}")
    print(f"DEBUG: df_final columns: {list(df_final.columns)}")
    
    if 'event_timestamp' not in df_final.columns:
        print("ERROR: event_timestamp column missing!")
        return
    
    if not isinstance(df_final['event_timestamp'], pd.Series):
        print("ERROR: event_timestamp is not a pandas Series!")
        return
    
    if not pd.api.types.is_datetime64_any_dtype(df_final['event_timestamp']):
        print(f"ERROR: event_timestamp dtype is {df_final['event_timestamp'].dtype}, not datetime64[ns]!")
        return
    
    if df_final.empty:
        print("ERROR: df_final is empty!")
        return
    
    # Now safely access .dt methods
    try:
        print(f"DEBUG: df_final['event_timestamp'] dtype: {df_final['event_timestamp'].dtype}")
        print(f"DEBUG: df_final['event_timestamp'] sample: {df_final['event_timestamp'].head()}")
        print(f"DEBUG: df_final['event_timestamp'] min: {df_final['event_timestamp'].min()}")
        print(f"DEBUG: df_final['event_timestamp'] max: {df_final['event_timestamp'].max()}")
    except Exception as e:
        print(f"ERROR accessing event_timestamp: {e}")
        return
    
    # Step 5: Save processed data
    print("\nSTEP 5: Saving processed data...")
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df_final.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved: {PROCESSED_DATA_PATH}")
    
    # Step 6: Save feature store version
    print("\nSTEP 6: Saving feature store version...")
    os.makedirs(os.path.dirname(FEATURE_STORE_PATH), exist_ok=True)
    df_final.to_csv(FEATURE_STORE_PATH, index=False)
    print(f"Feature store saved: {FEATURE_STORE_PATH}")
    
    # Also save parquet for consumers expecting parquet
    try:
        feature_parquet_path = FEATURE_STORE_PATH.replace('.csv', '.parquet')
        df_final.to_parquet(feature_parquet_path, index=False)
        print(f"Feature store parquet saved: {feature_parquet_path}")
    except Exception as e:
        print(f"WARNING: Could not save parquet feature store: {e}")
    
    # Verify the file was written correctly
    print("\nSTEP 7: Verifying saved data...")
    try:
        verification_df = pd.read_csv(FEATURE_STORE_PATH)
        print(f"VERIFICATION: File contains {len(verification_df)} records")
        
        # Safety check before accessing .dt methods
        if 'event_timestamp' in verification_df.columns:
            verification_df['event_timestamp'] = pd.to_datetime(verification_df['event_timestamp'])
            if verification_df['event_timestamp'].dtype == 'datetime64[ns]':
                print(f"VERIFICATION: Date range: {verification_df['event_timestamp'].min()} to {verification_df['event_timestamp'].max()}")
                print(f"VERIFICATION: Latest date: {verification_df['event_timestamp'].max()}")
            else:
                print(f"VERIFICATION WARNING: event_timestamp dtype is {verification_df['event_timestamp'].dtype}")
        else:
            print("VERIFICATION WARNING: event_timestamp column not found in saved file")
            
    except Exception as e:
        print(f"VERIFICATION ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Total records: {len(df_final)}")
    print(f"Total features: {len(df_final.columns)}")
    
    # Safe date range display
    try:
        if 'event_timestamp' in df_final.columns and df_final['event_timestamp'].dtype == 'datetime64[ns]':
            print(f"Date range: {df_final['event_timestamp'].min()} to {df_final['event_timestamp'].max()}")
        else:
            print("Date range: Unable to determine (datetime conversion issue)")
    except Exception as e:
        print(f"Date range: Error accessing dates - {e}")
    
    print(f"Targets created: AQI_t+1, AQI_t+2, AQI_t+3")
    print("\nFiles created:")
    print(f"   • Raw data: {PROCESSED_DATA_PATH}")
    print(f"   • Feature store: {FEATURE_STORE_PATH}")

if __name__ == "__main__":
    main()

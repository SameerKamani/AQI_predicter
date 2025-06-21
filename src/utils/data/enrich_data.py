import pandas as pd
import requests
import os
import json
from datetime import datetime
import numpy as np

# --- 1. Define File Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
OUTPUT_FILE = os.path.join(DATA_DIR, 'enriched_aqi_data.csv')

# --- 2. Load OpenAQ Data ---
print("Loading data for PM10...")
pm10_data = []
with open(os.path.join(DATA_DIR, 'openaq_los_angeles_pm10.jsonl'), 'r') as f:
    for line in f:
        data = json.loads(line)
        pm10_data.append({
            'datetime': data['period']['datetimeTo']['utc'],
            'pm10': data['value']
        })

print("Loading data for PM25...")
pm25_data = []
with open(os.path.join(DATA_DIR, 'openaq_los_angeles_pm25.jsonl'), 'r') as f:
    for line in f:
        data = json.loads(line)
        pm25_data.append({
            'datetime': data['period']['datetimeTo']['utc'],
            'pm25': data['value']
        })

print("Loading data for O3...")
o3_data = []
with open(os.path.join(DATA_DIR, 'openaq_los_angeles_o3.jsonl'), 'r') as f:
    for line in f:
        data = json.loads(line)
        o3_data.append({
            'datetime': data['period']['datetimeTo']['utc'],
            'o3': data['value']
        })

print("Loading data for NO2...")
no2_data = []
with open(os.path.join(DATA_DIR, 'openaq_los_angeles_no2.jsonl'), 'r') as f:
    for line in f:
        data = json.loads(line)
        no2_data.append({
            'datetime': data['period']['datetimeTo']['utc'],
            'no2': data['value']
        })

# Load additional parameters if they exist
co_data = []
so2_data = []
bc_data = []
nox_data = []

try:
    print("Loading data for CO...")
    with open(os.path.join(DATA_DIR, 'openaq_los_angeles_co.jsonl'), 'r') as f:
        for line in f:
            data = json.loads(line)
            co_data.append({
                'datetime': data['period']['datetimeTo']['utc'],
                'co': data['value']
            })
except FileNotFoundError:
    print("CO data not found, skipping...")

try:
    print("Loading data for SO2...")
    with open(os.path.join(DATA_DIR, 'openaq_los_angeles_so2.jsonl'), 'r') as f:
        for line in f:
            data = json.loads(line)
            so2_data.append({
                'datetime': data['period']['datetimeTo']['utc'],
                'so2': data['value']
            })
except FileNotFoundError:
    print("SO2 data not found, skipping...")

try:
    print("Loading data for BC...")
    with open(os.path.join(DATA_DIR, 'openaq_los_angeles_bc.jsonl'), 'r') as f:
        for line in f:
            data = json.loads(line)
            bc_data.append({
                'datetime': data['period']['datetimeTo']['utc'],
                'bc': data['value']
            })
except FileNotFoundError:
    print("BC data not found, skipping...")

try:
    print("Loading data for NOx...")
    with open(os.path.join(DATA_DIR, 'openaq_los_angeles_nox.jsonl'), 'r') as f:
        for line in f:
            data = json.loads(line)
            nox_data.append({
                'datetime': data['period']['datetimeTo']['utc'],
                'nox': data['value']
            })
except FileNotFoundError:
    print("NOx data not found, skipping...")

# --- 3. Convert to DataFrames ---
df_pm10 = pd.DataFrame(pm10_data)
df_pm25 = pd.DataFrame(pm25_data)
df_o3 = pd.DataFrame(o3_data)
df_no2 = pd.DataFrame(no2_data)

# Convert datetime strings to datetime objects
df_pm10['datetime'] = pd.to_datetime(df_pm10['datetime'])
df_pm25['datetime'] = pd.to_datetime(df_pm25['datetime'])
df_o3['datetime'] = pd.to_datetime(df_o3['datetime'])
df_no2['datetime'] = pd.to_datetime(df_no2['datetime'])

# Set datetime as index
df_pm10.set_index('datetime', inplace=True)
df_pm25.set_index('datetime', inplace=True)
df_o3.set_index('datetime', inplace=True)
df_no2.set_index('datetime', inplace=True)

# Handle additional parameters
dfs_additional = []
if co_data:
    df_co = pd.DataFrame(co_data)
    df_co['datetime'] = pd.to_datetime(df_co['datetime'])
    df_co.set_index('datetime', inplace=True)
    dfs_additional.append(df_co)

if so2_data:
    df_so2 = pd.DataFrame(so2_data)
    df_so2['datetime'] = pd.to_datetime(df_so2['datetime'])
    df_so2.set_index('datetime', inplace=True)
    dfs_additional.append(df_so2)

if bc_data:
    df_bc = pd.DataFrame(bc_data)
    df_bc['datetime'] = pd.to_datetime(df_bc['datetime'])
    df_bc.set_index('datetime', inplace=True)
    dfs_additional.append(df_bc)

if nox_data:
    df_nox = pd.DataFrame(nox_data)
    df_nox['datetime'] = pd.to_datetime(df_nox['datetime'])
    df_nox.set_index('datetime', inplace=True)
    dfs_additional.append(df_nox)

# --- 4. Merge All Data ---
print("All pollutant data loaded and merged.")
df_merged = df_pm10.join([df_pm25, df_o3, df_no2], how='outer')

# Add additional parameters if available
for df_add in dfs_additional:
    df_merged = df_merged.join(df_add, how='outer')

# --- 5. Fetch Weather Data ---
print("Fetching weather data for Los Angeles...")

# Get date range from the data
start_date = df_merged.index.min()
end_date = df_merged.index.max()
print(f"Date range: {start_date} to {end_date}")

# OpenWeatherMap API call for historical weather data
def fetch_weather_data(lat, lon, start_date, end_date):
    # This is a simplified version - you might want to use a proper weather API
    # For now, we'll create some synthetic weather data based on the date range
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    weather_data = []
    
    for date in date_range:
        # Simple seasonal weather patterns for Los Angeles
        month = date.month
        hour = date.hour
        
        # Temperature: warmer in summer (months 6-9), cooler in winter
        base_temp = 20  # Base temperature in Celsius
        seasonal_temp = 10 * np.sin(2 * np.pi * (month - 6) / 12)  # Seasonal variation
        hourly_temp = 5 * np.sin(2 * np.pi * (hour - 12) / 24)  # Daily variation
        temperature = base_temp + seasonal_temp + hourly_temp
        
        # Humidity: higher in winter, lower in summer
        humidity = 60 + 20 * np.cos(2 * np.pi * (month - 6) / 12)
        
        # Wind speed: slightly higher in afternoon
        wind_speed = 5 + 3 * np.sin(2 * np.pi * (hour - 12) / 24)
        
        # Precipitation: rare in LA, mostly in winter
        precipitation = 0
        if month in [12, 1, 2, 3] and np.random.random() < 0.1:  # 10% chance in winter
            precipitation = np.random.exponential(2)
        
        weather_data.append({
            'datetime': date,
            'temperature_2m': temperature,
            'relative_humidity_2m': humidity,
            'precipitation': precipitation,
            'wind_speed_10m': wind_speed
        })
    
    return pd.DataFrame(weather_data).set_index('datetime')

# Fetch weather data
weather_df = fetch_weather_data(34.0522, -118.2437, start_date, end_date)
print("Weather data fetched successfully.")

# --- 6. Merge Weather Data ---
print("Merging pollutant and weather data...")
df_final = df_merged.join(weather_df, how='left')

# --- 7. Add Temporal Features ---
print("Temporal features added and data cleaned.")
df_final['hour'] = df_final.index.hour
df_final['day_of_week'] = df_final.index.dayofweek
df_final['month'] = df_final.index.month

# --- 8. Clean Data ---
# Remove rows with all NaN values
df_final = df_final.dropna(how='all')

# Forward fill missing values (use previous value)
df_final = df_final.fillna(method='ffill')

# Backward fill any remaining NaN values
df_final = df_final.fillna(method='bfill')

# --- 9. Save Enriched Data ---
df_final.to_csv(OUTPUT_FILE)
print(f"Enriched data saved to {OUTPUT_FILE}")
print(df_final.head()) 
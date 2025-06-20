import pandas as pd
import requests
import os
import json

# --- 1. Define file paths and parameters ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
OUTPUT_FILE = os.path.join(DATA_DIR, 'enriched_aqi_data.csv')
PARAMETERS = ['pm25', 'pm10', 'o3', 'no2']

# --- 2. Load and process each pollutant's data ---
all_pollutants_df = pd.DataFrame()

for param in PARAMETERS:
    input_file = os.path.join(DATA_DIR, f'openaq_los_angeles_{param}.jsonl')
    if not os.path.exists(input_file):
        print(f"Warning: Data file not found for {param.upper()} at {input_file}. Skipping.")
        continue
    
    print(f"Loading data for {param.upper()}...")
    records = []
    with open(input_file, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    
    if not records:
        print(f"No records found for {param.upper()}. Skipping.")
        continue

    df = pd.DataFrame(records)
    df['datetime'] = df['period'].apply(lambda x: x['datetimeTo']['utc'])
    df[param] = df['value']
    df = df[['datetime', param]]
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    df = df.resample('h').mean()
    
    if all_pollutants_df.empty:
        all_pollutants_df = df
    else:
        all_pollutants_df = all_pollutants_df.join(df, how='outer')

print("All pollutant data loaded and merged.")

# --- 3. Fetch Historical Weather Data ---
start_date = all_pollutants_df.index.min().strftime('%Y-%m-%d')
end_date = all_pollutants_df.index.max().strftime('%Y-%m-%d')
print(f"Fetching weather data for Los Angeles from {start_date} to {end_date}...")

WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 34.05,
    "longitude": -118.24,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
    "timezone": "America/Los_Angeles"
}
response = requests.get(WEATHER_URL, params=params)
weather_data = response.json()

weather_df = pd.DataFrame(weather_data['hourly'])
weather_df = weather_df.rename(columns={'time': 'datetime'})
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'], utc=True)
weather_df = weather_df.set_index('datetime')
print("Weather data fetched successfully.")

# --- 4. Merge Pollutants with Weather Data ---
print("Merging pollutant and weather data...")
final_df = all_pollutants_df.join(weather_df, how='inner')

# --- 5. Add Temporal Features and Clean Data ---
final_df['hour'] = final_df.index.hour
final_df['day_of_week'] = final_df.index.dayofweek
final_df['month'] = final_df.index.month
# Forward fill to handle missing values, then backfill
final_df = final_df.ffill().bfill()
print("Temporal features added and data cleaned.")

# --- 6. Save Enriched Data ---
final_df.to_csv(OUTPUT_FILE)
print(f"Enriched data saved to {OUTPUT_FILE}. Shape: {final_df.shape}")
print("\nFinal DataFrame head:")
print(final_df.head()) 
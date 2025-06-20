import os
import requests
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
JSONL_FILE = os.path.join(DATA_DIR, 'openaq_los_angeles_pm25.jsonl')
LOCATIONS_URL = 'https://api.openaq.org/v3/locations'
SENSORS_URL_TPL = 'https://api.openaq.org/v3/sensors/{sensor_id}/measurements'

COORDINATES = '34.0522,-118.2437' # Los Angeles
RADIUS = 25000 
PARAMETER = 'pm25'
LIMIT = 500
TOTAL_RECORDS = 500


def get_pm25_sensor_ids(coordinates, radius, api_key):
    headers = {'X-API-Key': api_key}
    params = {
        'coordinates': coordinates, 
        'radius': radius, 
        'parameter': PARAMETER,
        'limit': 20
    }
    response = requests.get(LOCATIONS_URL, params=params, headers=headers)
    if response.status_code != 200:
        print(f"Error finding locations: {response.status_code}, {response.text}")
        return []

    locations = response.json().get('results', [])
    sensor_ids = []
    for loc in locations:
        for sensor in loc.get('sensors', []):
            if sensor.get('parameter', {}).get('name') == PARAMETER:
                sensor_ids.append(sensor['id'])
    
    print(f"Found {len(sensor_ids)} sensors with '{PARAMETER}' near {coordinates}.")
    return sensor_ids

def fetch_openaq_data(sensor_id, api_key, total_records=500, limit=500):
    records = []
    page = 1
    headers = {'X-API-Key': api_key}
    
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=90)
    
    url = SENSORS_URL_TPL.format(sensor_id=sensor_id)

    while len(records) < total_records:
        params = {
            'limit': limit,
            'page': page,
            'sort': 'desc',
            'order_by': 'datetime',
            'date_from': date_from.isoformat(),
            'date_to': date_to.isoformat()
        }
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            print(f"Error fetching measurements for sensor {sensor_id}: {response.status_code}, {response.text}")
            break
        data = response.json()
        results = data.get('results', [])
        if not results:
            break
        records.extend(results)
        print(f"Fetched {len(records)} records for sensor {sensor_id}...")
        page += 1
    return records[:total_records]

def save_to_jsonl(records, jsonl_file):
    os.makedirs(os.path.dirname(jsonl_file), exist_ok=True)
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')
    print(f"Saved {len(records)} records to {jsonl_file}.")

if __name__ == "__main__":
    load_dotenv()
    OPENAQ_API_KEY = os.getenv('OPENAQ_API_KEY')
    if not OPENAQ_API_KEY:
        print("Please set your OPENAQ_API_KEY in a .env file.")
    else:
        sensor_ids = get_pm25_sensor_ids(COORDINATES, RADIUS, OPENAQ_API_KEY)
        if not sensor_ids:
            print(f"Could not find any sensors for {PARAMETER} near {COORDINATES}.")
        else:
            all_records = None
            for sensor_id in sensor_ids:
                print(f"\n--- Trying Sensor ID: {sensor_id} ---")
                records = fetch_openaq_data(sensor_id, OPENAQ_API_KEY, total_records=TOTAL_RECORDS, limit=LIMIT)
                if records:
                    all_records = records
                    print(f"Success! Found data for sensor {sensor_id}.")
                    break
            
            if all_records:
                save_to_jsonl(all_records, JSONL_FILE)
            else:
                print(f"\nCould not retrieve '{PARAMETER}' data for any of the found sensors.") 
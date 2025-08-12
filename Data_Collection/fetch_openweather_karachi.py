import os
import sys
import time
import json
import argparse
from datetime import datetime, timedelta, timezone
import requests
import pandas as pd

LATITUDE = 24.8607   # Karachi
LONGITUDE = 67.0011  # Karachi
API_URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"

# Defaults: last 30 days to today (UTC)
DEFAULT_DAYS = 30

def parse_args():
    parser = argparse.ArgumentParser(description="Fetch OpenWeather Air Pollution data for Karachi and save to CSV.")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD) in UTC")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD) in UTC (inclusive)")
    parser.add_argument("--out", type=str, default=os.path.join("Data", "karachi_air_pollution.csv"),
                        help="Output CSV path (will be created if missing)")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep seconds between API calls to avoid rate limits")
    parser.add_argument("--append", action="store_true", help="Append to existing CSV (deduplicates by datetime)")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP request timeout seconds")
    return parser.parse_args()

def to_unix_utc(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

def day_bounds_unix(date: datetime):
    start_dt = datetime(date.year, date.month, date.day, tzinfo=timezone.utc)
    end_dt = start_dt + timedelta(days=1) - timedelta(seconds=1)
    return int(start_dt.timestamp()), int(end_dt.timestamp())

def fetch_day(session: requests.Session, api_key: str, lat: float, lon: float, start_unix: int, end_unix: int, timeout: float):
    params = {
        "lat": lat,
        "lon": lon,
        "start": start_unix,
        "end": end_unix,
        "appid": api_key
    }
    resp = session.get(API_URL, params=params, timeout=timeout)
    if resp.status_code == 200:
        return resp.json()
    # Basic backoff for 429 or transient errors
    if resp.status_code in (429, 500, 502, 503, 504):
        time.sleep(2)
        resp = session.get(API_URL, params=params, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    raise RuntimeError(f"API error {resp.status_code}: {resp.text[:200]}")

def normalize_response(payload: dict):
    """
    Convert OpenWeather response to flat records.
    """
    if not payload or "list" not in payload:
        return []
    rows = []
    for entry in payload["list"]:
        dt_unix = entry.get("dt")
        if dt_unix is None:
            continue
        dt = datetime.fromtimestamp(dt_unix, tz=timezone.utc)
        comp = entry.get("components", {}) or {}
        main = entry.get("main", {}) or {}
        rows.append({
            "datetime_utc": dt.isoformat().replace("+00:00", "Z"),
            "lat": payload.get("coord", {}).get("lat", None),
            "lon": payload.get("coord", {}).get("lon", None),
            "pm2_5": comp.get("pm2_5"),
            "pm10": comp.get("pm10"),
            "co": comp.get("co"),
            "no": comp.get("no"),
            "no2": comp.get("no2"),
            "o3": comp.get("o3"),
            "so2": comp.get("so2"),
            "nh3": comp.get("nh3"),
            "ow_aqi_1to5": main.get("aqi"),  # OpenWeather AQI category (1..5)
            "source": "openweather",
            "city": "Karachi",
        })
    return rows

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def daterange(start_date: datetime, end_date: datetime):
    cur = start_date
    while cur <= end_date:
        yield cur
        cur += timedelta(days=1)

def main():
    # Load .env if present (from project root or parent dirs)
    try:
        from dotenv import load_dotenv, find_dotenv  # type: ignore
        load_dotenv(find_dotenv(), override=False)
    except Exception:
        pass

    args = parse_args()
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        print("ERROR: Please set environment variable OPENWEATHER_API_KEY.", file=sys.stderr)
        sys.exit(1)

    # Determine date range
    if args.start and args.end:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=DEFAULT_DAYS - 1)

    ensure_dir(args.out)

    all_rows = []
    with requests.Session() as session:
        for day in daterange(start_date, end_date):
            start_unix, end_unix = day_bounds_unix(day)
            date_str = day.strftime("%Y-%m-%d")
            try:
                payload = fetch_day(session, api_key, LATITUDE, LONGITUDE, start_unix, end_unix, args.timeout)
                rows = normalize_response(payload)
                if rows:
                    all_rows.extend(rows)
                    print(f"{date_str}: fetched {len(rows)} records")
                else:
                    print(f"{date_str}: no records")
            except Exception as e:
                print(f"{date_str}: failed - {e}")
            time.sleep(args.sleep)

    if not all_rows:
        print("No data collected; exiting.")
        sys.exit(0)

    df = pd.DataFrame(all_rows)
    # Deduplicate by timestamp just in case
    df = df.drop_duplicates(subset=["datetime_utc"]).sort_values("datetime_utc").reset_index(drop=True)

    # Append or write new
    if args.append and os.path.exists(args.out):
        try:
            existing = pd.read_csv(args.out)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["datetime_utc"]).sort_values("datetime_utc").reset_index(drop=True)
            combined.to_csv(args.out, index=False)
            print(f"Wrote {len(combined)} total rows to {args.out} (appended).")
        except Exception as e:
            print(f"Append failed, writing fresh CSV. Reason: {e}")
            df.to_csv(args.out, index=False)
            print(f"Wrote {len(df)} rows to {args.out}.")
    else:
        df.to_csv(args.out, index=False)
        print(f"Wrote {len(df)} rows to {args.out}.")

if __name__ == "__main__":
    main()
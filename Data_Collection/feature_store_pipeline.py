import os
import sys
import time
import argparse
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple
import numpy as np

import requests
import pandas as pd


LATITUDE = 24.8607   # Karachi
LONGITUDE = 67.0011  # Karachi
API_URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"
# Open-Meteo ERA5 archive (historical hourly weather)
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/era5"

# Defaults: last 90 days to today (UTC) for a better feature history
DEFAULT_DAYS = 90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch OpenWeather Air Pollution data for Karachi, compute daily features/targets, "
            "and store them in a local feature store under Data/feature_store/."
        )
    )
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD) in UTC")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD) in UTC (inclusive)")
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=(
            "Number of past days to fetch (excluding today) when --start/--end are not provided. "
            f"Default: {DEFAULT_DAYS}."
        ),
    )
    parser.add_argument(
        "--raw_out",
        type=str,
        default=os.path.join("Data", "raw", "karachi_air_pollution.csv"),
        help="Optional: path to write raw hourly CSV (created if missing)",
    )
    parser.add_argument(
        "--features_out",
        type=str,
        default=os.path.join("Data", "feature_store", "karachi_daily_features.csv"),
        help="Path to write daily features CSV (created if missing)",
    )
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep seconds between API calls")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP request timeout seconds")
    parser.add_argument("--append", action="store_true", help="Append to existing feature CSV with dedup by date")
    parser.add_argument(
        "--min_hours_per_day",
        type=int,
        default=20,
        help="Minimum hourly observations required to treat a day as valid before imputation. Default: 20",
    )
    parser.add_argument(
        "--impute_short_gaps",
        action="store_true",
        help="If set, interpolate single missing calendar days at daily granularity",
    )
    return parser.parse_args()


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def day_bounds_unix(date: datetime) -> Tuple[int, int]:
    start_dt = datetime(date.year, date.month, date.day, tzinfo=timezone.utc)
    end_dt = start_dt + timedelta(days=1) - timedelta(seconds=1)
    return int(start_dt.timestamp()), int(end_dt.timestamp())


def daterange(start_date: datetime, end_date: datetime):
    cur = start_date
    while cur <= end_date:
        yield cur
        cur += timedelta(days=1)


def fetch_day(
    session: requests.Session,
    api_key: str,
    lat: float,
    lon: float,
    start_unix: int,
    end_unix: int,
    timeout: float,
):
    params = {"lat": lat, "lon": lon, "start": start_unix, "end": end_unix, "appid": api_key}
    resp = session.get(API_URL, params=params, timeout=timeout)
    if resp.status_code == 200:
        return resp.json()
    if resp.status_code in (429, 500, 502, 503, 504):  # simple retry for transient errors
        time.sleep(2)
        resp = session.get(API_URL, params=params, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    raise RuntimeError(f"API error {resp.status_code}: {resp.text[:200]}")


def normalize_response(payload: dict) -> List[dict]:
    if not payload or "list" not in payload:
        return []
    rows: List[dict] = []
    coord = payload.get("coord", {}) or {}
    for entry in payload["list"]:
        dt_unix = entry.get("dt")
        if dt_unix is None:
            continue
        dt = datetime.fromtimestamp(dt_unix, tz=timezone.utc)
        comp = entry.get("components", {}) or {}
        main = entry.get("main", {}) or {}
        rows.append(
            {
                "datetime_utc": dt.isoformat().replace("+00:00", "Z"),
                "date": dt.date().isoformat(),  # convenience for grouping
                "lat": coord.get("lat"),
                "lon": coord.get("lon"),
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
            }
        )
    return rows


# --- Open-Meteo helpers (historical hourly weather) ---
def fetch_openmeteo_hourly(
    start_date: datetime,
    end_date: datetime,
    lat: float,
    lon: float,
    timeout: float,
) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.date().isoformat(),
        "end_date": end_date.date().isoformat(),
        "timezone": "UTC",
        "hourly": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "dew_point_2m",
                "surface_pressure",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_gusts_10m",
                "precipitation",
                "rain",
                "snowfall",
                "cloud_cover",
                "visibility",
            ]
        ),
    }
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        js = resp.json()
    except Exception as e:
        print(f"Open-Meteo fetch failed: {e}")
        return pd.DataFrame()

    hourly = js.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return pd.DataFrame()

    def _col(name: str):
        v = hourly.get(name)
        return v if isinstance(v, list) else [None] * len(times)

    df = pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(times, utc=True),
            "temperature_2m": _col("temperature_2m"),
            "relative_humidity_2m": _col("relative_humidity_2m"),
            "dew_point_2m": _col("dew_point_2m"),
            "surface_pressure": _col("surface_pressure"),
            "wind_speed_10m": _col("wind_speed_10m"),
            "wind_direction_10m": _col("wind_direction_10m"),
            "wind_gusts_10m": _col("wind_gusts_10m"),
            "precipitation": _col("precipitation"),
            "rain": _col("rain"),
            "snowfall": _col("snowfall"),
            "cloud_cover": _col("cloud_cover"),
            "visibility": _col("visibility"),
        }
    )
    df["date"] = df["datetime_utc"].dt.date.astype(str)
    # Wind components from speed and direction
    try:
        rad = np.deg2rad(pd.to_numeric(df["wind_direction_10m"], errors="coerce"))
        spd = pd.to_numeric(df["wind_speed_10m"], errors="coerce")
        df["wind_u"] = spd * np.cos(rad)
        df["wind_v"] = spd * np.sin(rad)
    except Exception:
        df["wind_u"] = np.nan
        df["wind_v"] = np.nan
    return df

## Removed One Call helpers to revert to previous behavior
# --- EPA AQI computation (continuous 0..500) using PM2.5 and PM10 24h breakpoints ---

def aqi_from_concentration(conc: float, breakpoints: List[Tuple[float, float, int, int]]) -> Optional[float]:
    if conc is None or pd.isna(conc):
        return None
    for bp_low, bp_high, i_low, i_high in breakpoints:
        if bp_low <= conc <= bp_high:
            # Linear interpolation
            return (i_high - i_low) / (bp_high - bp_low) * (conc - bp_low) + i_low
    return None


# PM2.5 (µg/m3) 24h avg
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

# PM10 (µg/m3) 24h avg
PM10_BREAKPOINTS = [
    (0.0, 54.0, 0, 50),
    (55.0, 154.0, 51, 100),
    (155.0, 254.0, 101, 150),
    (255.0, 354.0, 151, 200),
    (355.0, 424.0, 201, 300),
    (425.0, 504.0, 301, 400),
    (505.0, 604.0, 401, 500),
]

# O3 8-hour (ppb) max daily
O3_8H_BREAKPOINTS_PPB = [
    (0.0, 54.0, 0, 50),
    (55.0, 70.0, 51, 100),
    (71.0, 85.0, 101, 150),
    (86.0, 105.0, 151, 200),
    (106.0, 200.0, 201, 300),
]


def compute_daily_label_aqi(row: pd.Series) -> Optional[float]:
    aqi_pm25 = aqi_from_concentration(row.get("pm2_5_mean"), PM25_BREAKPOINTS)
    aqi_pm10 = aqi_from_concentration(row.get("pm10_mean"), PM10_BREAKPOINTS)
    aqi_o3_8h = aqi_from_concentration(row.get("o3_8h_max_ppb"), O3_8H_BREAKPOINTS_PPB)
    candidates = [v for v in [aqi_pm25, aqi_pm10, aqi_o3_8h] if v is not None]
    return max(candidates) if candidates else None


def build_daily_features(
    raw_df: pd.DataFrame,
    min_hours_per_day: int,
    impute_short_gaps: bool,
    weather_hourly_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    # Ensure datetime types
    df = raw_df.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df["date"] = pd.to_datetime(df["date"])  # naive date

    # Clean raw pollutant readings: treat negative numbers (e.g., -9999 sentinels) as missing
    pollutant_cols = [
        "pm2_5",
        "pm10",
        "co",
        "no",
        "no2",
        "o3",
        "so2",
        "nh3",
    ]
    for col in pollutant_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] < 0, col] = np.nan

    # Build continuous daily calendar between min and max date
    all_days = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="D")

    # Compute O3 daily max of 8h rolling mean (convert µg/m³ -> ppb with factor ~0.0205)
    df_sorted = df.sort_values("datetime_utc").copy()
    # 8-hour rolling mean in µg/m³
    df_sorted["o3_roll8h_ugm3"] = (
        df_sorted["o3"].rolling(window=8, min_periods=8).mean()
    )
    # Convert to ppb at 1 atm, 25°C: ppb = (µg/m³) * 24.45 / 48 ≈ 0.509
    df_sorted["o3_roll8h_ppb"] = df_sorted["o3_roll8h_ugm3"] * 0.509
    o3_8h_daily = (
        df_sorted.dropna(subset=["o3_roll8h_ppb"]).groupby(df_sorted["date"]).agg(
            o3_8h_max_ppb=("o3_roll8h_ppb", "max")
        )
    )

    # Aggregate hourly to daily means and counts (NaNs ignored in mean)
    daily = (
        df.groupby("date")
        .agg(
            pm2_5_mean=("pm2_5", "mean"),
            pm10_mean=("pm10", "mean"),
            co_mean=("co", "mean"),
            no_mean=("no", "mean"),
            no2_mean=("no2", "mean"),
            o3_mean=("o3", "mean"),
            so2_mean=("so2", "mean"),
            nh3_mean=("nh3", "mean"),
            num_hours=("datetime_utc", "count"),
        )
        .reindex(all_days, fill_value=np.nan)
    )
    daily.index.name = "date"
    daily = daily.reset_index()

    # Clip any residual negative daily means to zero for safety
    daily_means = [
        "pm2_5_mean",
        "pm10_mean",
        "co_mean",
        "no_mean",
        "no2_mean",
        "o3_mean",
        "so2_mean",
        "nh3_mean",
    ]
    for col in daily_means:
        if col in daily.columns:
            daily[col] = daily[col].clip(lower=0)

    # Merge O3 8h metric
    daily = daily.merge(o3_8h_daily, how="left", left_on="date", right_index=True)

    # Merge Open-Meteo weather (if provided)
    if weather_hourly_df is not None and not weather_hourly_df.empty:
        wdf = weather_hourly_df.copy()
        wdf["datetime_utc"] = pd.to_datetime(wdf["datetime_utc"], utc=True)
        wdf["date"] = pd.to_datetime(wdf["date"])  # naive date
        weather_daily = (
            wdf.groupby("date")
            .agg(
                temp_mean=("temperature_2m", "mean"),
                temp_min=("temperature_2m", "min"),
                temp_max=("temperature_2m", "max"),
                humidity_mean=("relative_humidity_2m", "mean"),
                dew_point_mean=("dew_point_2m", "mean"),
                pressure_mean=("surface_pressure", "mean"),
                wind_speed_mean=("wind_speed_10m", "mean"),
                wind_speed_max=("wind_speed_10m", "max"),
                wind_gust_max=("wind_gusts_10m", "max"),
                clouds_mean=("cloud_cover", "mean"),
                visibility_mean=("visibility", "mean"),
                rain_sum=("rain", "sum"),
                snow_sum=("snowfall", "sum"),
                wind_u_mean=("wind_u", "mean"),
                wind_v_mean=("wind_v", "mean"),
            )
        )
        weather_daily.index.name = "date"
        weather_daily = weather_daily.reset_index()
        weather_daily["temp_range"] = weather_daily["temp_max"] - weather_daily["temp_min"]
        daily = daily.merge(weather_daily, on="date", how="left")

    # No weather merge in reverted version

    # Mark completeness and optionally impute single-day gaps at daily granularity
    daily["is_complete_day"] = (daily["num_hours"] >= min_hours_per_day).astype(int)
    daily["imputed"] = 0

    # Compute AQI (before imputation to detect which rows are missing)
    daily["aqi_daily"] = daily.apply(compute_daily_label_aqi, axis=1)

    if impute_short_gaps:
        # Interpolate only 1-day gaps on daily means and AQI
        interp_cols = [
            "pm2_5_mean",
            "pm10_mean",
            "co_mean",
            "no_mean",
            "no2_mean",
            "o3_mean",
            "so2_mean",
            "nh3_mean",
            "aqi_daily",
        ]
        daily = daily.sort_values("date").set_index("date")
        before = daily[interp_cols].isna().copy()
        daily[interp_cols] = daily[interp_cols].interpolate(method="time", limit=1, limit_direction="both")
        filled_mask = before & daily[interp_cols].notna()
        any_filled = filled_mask.any(axis=1)
        daily.loc[any_filled, "imputed"] = 1
        daily = daily.reset_index()

    # Filter out days with insufficient coverage that were not imputed
    daily_valid = daily[(daily["is_complete_day"] == 1) | (daily["imputed"] == 1)].copy()

    # Validate: no negative pollutant means should remain
    if (
        daily_valid[[c for c in daily_means if c in daily_valid.columns]]
        .lt(0)
        .any()
        .any()
    ):
        raise ValueError("Negative pollutant means detected after cleaning.")

    # Time-based features
    daily_valid["day_of_week"] = daily_valid["date"].dt.dayofweek
    daily_valid["day_of_year"] = daily_valid["date"].dt.dayofyear
    daily_valid["month"] = daily_valid["date"].dt.month
    daily_valid["is_weekend"] = (daily_valid["day_of_week"] >= 5).astype(int)
    # Additional calendar features
    try:
        daily_valid["week_of_year"] = daily_valid["date"].dt.isocalendar().week.astype(int)
    except Exception:
        daily_valid["week_of_year"] = daily_valid["date"].dt.week
    daily_valid["doy_sin"] = np.sin(2 * np.pi * daily_valid["day_of_year"] / 365.0)
    daily_valid["doy_cos"] = np.cos(2 * np.pi * daily_valid["day_of_year"] / 365.0)

    # Derivatives & lags computed AFTER calendar reindex/filter to keep alignment
    for col in ["aqi_daily", "pm2_5_mean", "pm10_mean"]:
        for lag in [1, 2, 3, 4, 5, 6, 7]:
            daily_valid[f"{col}_lag{lag}"] = daily_valid[col].shift(lag)

    # Rolling stats (on AQI)
    # AQI rolling windows
    for w in [3, 7, 14, 30]:
        daily_valid[f"aqi_roll_mean_{w}"] = daily_valid["aqi_daily"].rolling(window=w, min_periods=max(2, w//3)).mean()
        daily_valid[f"aqi_roll_std_{w}"] = daily_valid["aqi_daily"].rolling(window=w, min_periods=max(2, w//3)).std()
    # PM rollings
    for base, col in [("pm25", "pm2_5_mean"), ("pm10", "pm10_mean")]:
        for w in [3, 7, 14, 30]:
            daily_valid[f"{base}_roll_mean_{w}"] = daily_valid[col].rolling(window=w, min_periods=max(2, w//3)).mean()
            daily_valid[f"{base}_roll_std_{w}"] = daily_valid[col].rolling(window=w, min_periods=max(2, w//3)).std()

    # Weather lags/rollings
    for base_col in ["temp_mean", "humidity_mean", "wind_u_mean", "wind_v_mean"]:
        if base_col in daily_valid.columns:
            for lag in [1, 2, 3]:
                daily_valid[f"{base_col}_lag{lag}"] = daily_valid[base_col].shift(lag)
            for w in [3, 7]:
                daily_valid[f"{base_col}_roll_mean_{w}"] = (
                    daily_valid[base_col].rolling(window=w, min_periods=max(2, w//3)).mean()
                )
                daily_valid[f"{base_col}_roll_std_{w}"] = (
                    daily_valid[base_col].rolling(window=w, min_periods=max(2, w//3)).std()
                )

    # No weather lags/rollings in reverted version

    # Change rate
    daily_valid["aqi_change_rate"] = daily_valid["aqi_daily"] - daily_valid["aqi_daily"].shift(1)

    # Targets for t+1..t+3 days (calendar aligned)
    daily_valid["target_aqi_d1"] = daily_valid["aqi_daily"].shift(-1)
    daily_valid["target_aqi_d2"] = daily_valid["aqi_daily"].shift(-2)
    daily_valid["target_aqi_d3"] = daily_valid["aqi_daily"].shift(-3)

    # Metadata
    daily_valid["city"] = "Karachi"
    daily_valid["karachi_id"] = 1

    # Feast-compatible timestamps
    daily_valid["event_timestamp"] = pd.to_datetime(daily_valid["date"]).dt.tz_localize("UTC")
    _now_utc = pd.Timestamp.now(tz="UTC")
    daily_valid["created"] = _now_utc

    # Drop rows that cannot be used due to missing critical fields after alignment
    feature_cols_required = [
        "aqi_daily",
        "aqi_daily_lag1",
        "pm2_5_mean_lag1",
        "pm10_mean_lag1",
        "target_aqi_d1",
        "target_aqi_d2",
        "target_aqi_d3",
    ]
    daily_clean = daily_valid.dropna(subset=feature_cols_required)

    return daily_clean.reset_index(drop=True)


def fetch_raw_hourly_dataframe(api_key: str, start_date: datetime, end_date: datetime, timeout: float, sleep_s: float) -> pd.DataFrame:
    rows: List[dict] = []
    with requests.Session() as session:
        for day in daterange(start_date, end_date):
            start_unix, end_unix = day_bounds_unix(day)
            date_str = day.strftime("%Y-%m-%d")
            try:
                payload = fetch_day(session, api_key, LATITUDE, LONGITUDE, start_unix, end_unix, timeout)
                day_rows = normalize_response(payload)
                if day_rows:
                    rows.extend(day_rows)
                    print(f"{date_str}: fetched {len(day_rows)} hourly rows")
                else:
                    print(f"{date_str}: no rows")
            except Exception as e:
                print(f"{date_str}: failed - {e}")
            time.sleep(sleep_s)

    if not rows:
        return pd.DataFrame(columns=[
            "datetime_utc","date","lat","lon","pm2_5","pm10","co","no","no2","o3","so2","nh3","ow_aqi_1to5","source","city"
        ])
    df = pd.DataFrame(rows)
    # Deduplicate by timestamp
    df = df.drop_duplicates(subset=["datetime_utc"]).sort_values("datetime_utc").reset_index(drop=True)
    return df


def write_csv_dedup(df: pd.DataFrame, out_path: str, key_cols: List[str]) -> None:
    ensure_parent_dir(out_path)
    
    def _coerce_key_cols_to_str(frame: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
        coerced = frame.copy()
        for k in keys:
            if k in coerced.columns:
                coerced[k] = coerced[k].astype(str)
        return coerced

    df = _coerce_key_cols_to_str(df, key_cols)
    if os.path.exists(out_path):
        try:
            existing = pd.read_csv(out_path)
            existing = _coerce_key_cols_to_str(existing, key_cols)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=key_cols).sort_values(key_cols).reset_index(drop=True)
            combined.to_csv(out_path, index=False)
            print(f"Wrote {len(combined)} total rows to {out_path} (merged).")
            return
        except Exception as e:
            print(f"Merge failed; writing fresh CSV. Reason: {e}")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}.")


def write_parquet_dedup(df: pd.DataFrame, out_path: str, key_cols: List[str]) -> None:
    ensure_parent_dir(out_path)
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
    except Exception as e:
        print(
            f"Parquet dependencies missing (pyarrow). Skipping Parquet write to {out_path}. Reason: {e}"
        )
        return
    try:
        if os.path.exists(out_path):
            existing = pd.read_parquet(out_path)
            # Normalize key column dtypes to avoid Timestamp vs str compare during sort_values
            for k in key_cols:
                if k in existing.columns and k in df.columns:
                    # Prefer datetime if both convertible, else cast to string for stable sort
                    try:
                        existing[k] = pd.to_datetime(existing[k])
                        df[k] = pd.to_datetime(df[k])
                    except Exception:
                        existing[k] = existing[k].astype(str)
                        df[k] = df[k].astype(str)

            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=key_cols).sort_values(key_cols).reset_index(drop=True)
            combined.to_parquet(out_path, index=False)
            print(f"Wrote {len(combined)} total rows to {out_path} (merged Parquet).")
        else:
            df.to_parquet(out_path, index=False)
            print(f"Wrote {len(df)} rows to {out_path} (Parquet).")
    except Exception as e:
        print(f"Parquet write failed for {out_path}. Reason: {e}")


def main() -> None:
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
        if end_date < start_date:
            print("ERROR: end date must be >= start date", file=sys.stderr)
            sys.exit(1)
    else:
        if args.days is None or args.days < 1:
            print("ERROR: --days must be >= 1 when --start/--end are not provided", file=sys.stderr)
            sys.exit(1)
        # Exclude today: set end_date to yesterday 00:00 UTC
        today_midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = today_midnight - timedelta(days=1)
        start_date = end_date - timedelta(days=args.days - 1)

    # 1) Fetch raw hourly
    raw_df = fetch_raw_hourly_dataframe(api_key, start_date, end_date, timeout=args.timeout, sleep_s=args.sleep)
    if raw_df.empty:
        print("No raw data collected; exiting.")
        sys.exit(0)

    # Optional write raw
    if args.raw_out:
        write_csv_dedup(raw_df, args.raw_out, key_cols=["datetime_utc"])

    # 2) Compute features and targets (daily grain)
    # Fetch Open-Meteo hourly weather for the same date span
    try:
        start_d = pd.to_datetime(raw_df["date"].min()).tz_localize("UTC")
        end_d = pd.to_datetime(raw_df["date"].max()).tz_localize("UTC")
    except Exception:
        start_d = pd.Timestamp.now(tz="UTC")
        end_d = start_d
    weather_df = fetch_openmeteo_hourly(start_d, end_d, LATITUDE, LONGITUDE, timeout=args.timeout)

    features_df = build_daily_features(
        raw_df,
        min_hours_per_day=args.min_hours_per_day,
        impute_short_gaps=args.impute_short_gaps,
        weather_hourly_df=weather_df if not weather_df.empty else None,
    )
    if features_df.empty:
        print("No feature rows produced after aggregation; exiting.")
        sys.exit(0)

    # 3) Store features in local feature store (CSV + Parquet under Data/feature_store)
    # Dedup by date as primary key
    write_csv_dedup(features_df, args.features_out, key_cols=["date"])
    parquet_path = os.path.splitext(args.features_out)[0] + ".parquet"
    write_parquet_dedup(features_df, parquet_path, key_cols=["date"])


if __name__ == "__main__":
    main()



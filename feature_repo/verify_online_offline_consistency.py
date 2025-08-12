import os
import sys
from typing import List

import pandas as pd
from feast import FeatureStore


def main() -> None:
    # Paths relative to project root
    repo_path = os.path.join(os.path.dirname(__file__))
    parquet_path = os.path.join(os.path.dirname(repo_path), "Data", "feature_store", "karachi_daily_features.parquet")

    # Read offline latest row
    offline = pd.read_parquet(parquet_path)
    if offline.empty:
        print("ERROR: Offline feature Parquet is empty.")
        sys.exit(1)

    offline_sorted = offline.sort_values("event_timestamp")
    latest = offline_sorted.iloc[-1]

    # Fetch online features for the same entity
    store = FeatureStore(repo_path=repo_path)
    features: List[str] = [
        "karachi_air_quality_daily:aqi_daily",
        "karachi_air_quality_daily:pm2_5_mean",
        "karachi_air_quality_daily:pm10_mean",
        "karachi_air_quality_daily:co_mean",
        "karachi_air_quality_daily:no_mean",
        "karachi_air_quality_daily:no2_mean",
        "karachi_air_quality_daily:o3_mean",
        "karachi_air_quality_daily:so2_mean",
        "karachi_air_quality_daily:nh3_mean",
    ]

    online = store.get_online_features(
        features=features,
        entity_rows=[{"karachi_id": 1}],
    ).to_dict()

    # Compare selected fields
    mismatches = []
    for f in [
        "aqi_daily",
        "pm2_5_mean",
        "pm10_mean",
        "co_mean",
        "no_mean",
        "no2_mean",
        "o3_mean",
        "so2_mean",
        "nh3_mean",
    ]:
        online_val = online.get(f)
        # get_online_features returns lists per key; take first element
        if isinstance(online_val, list):
            online_val = online_val[0]
        offline_val = latest[f]

        def is_close(a, b, atol=1e-5, rtol=1e-6):
            if pd.isna(a) and pd.isna(b):
                return True
            if pd.isna(a) or pd.isna(b):
                return False
            a = float(a)
            b = float(b)
            return abs(a - b) <= (atol + rtol * max(abs(a), abs(b)))

        if not is_close(online_val, offline_val):
            mismatches.append((f, offline_val, online_val))

    if mismatches:
        print("CONSISTENCY CHECK: FAIL")
        for f, off_v, on_v in mismatches:
            print(f" - {f}: offline={off_v} online={on_v}")
        sys.exit(2)
    else:
        print("CONSISTENCY CHECK: PASS (online matches latest offline row)")


if __name__ == "__main__":
    main()



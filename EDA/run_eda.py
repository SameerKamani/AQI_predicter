import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(__file__))
    parquet_path = os.path.join(project_root, "Data", "feature_store", "karachi_daily_features.parquet")
    output_dir = os.path.join(project_root, "EDA", "output")
    ensure_dir(output_dir)

    # Load data
    if not os.path.exists(parquet_path):
        print(f"ERROR: Parquet not found at {parquet_path}. Run the feature pipeline first.")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)

    # Basic info & summary
    summary_txt = os.path.join(output_dir, "summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Columns:\n")
        f.write(", ".join(df.columns) + "\n\n")
        f.write("DTypes:\n")
        f.write(str(df.dtypes) + "\n\n")
        f.write("Describe (numeric):\n")
        f.write(str(df.describe()) + "\n\n")
        f.write("Missing values per column:\n")
        f.write(str(df.isna().sum()) + "\n")

    # Ensure time ordering
    if "event_timestamp" in df.columns:
        df = df.sort_values("event_timestamp")

    # Time series of AQI
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=df, x="event_timestamp", y="aqi_daily")
    plt.title("Daily AQI over time (Karachi)")
    plt.xlabel("Date")
    plt.ylabel("AQI (EPA 0-500)")
    save_fig(os.path.join(output_dir, "aqi_timeseries.png"))

    # Distributions of key pollutants
    for col in ["pm2_5_mean", "pm10_mean", "o3_mean", "no2_mean", "so2_mean", "co_mean"]:
        if col in df.columns:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col].dropna(), bins=30, kde=True)
            plt.title(f"Distribution: {col}")
            save_fig(os.path.join(output_dir, f"dist_{col}.png"))

    # Seasonality by month and day of week
    if "month" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df["month"], y=df["aqi_daily"])
        plt.title("AQI by month")
        save_fig(os.path.join(output_dir, "aqi_by_month.png"))

    if "day_of_week" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df["day_of_week"], y=df["aqi_daily"])
        plt.title("AQI by day of week")
        save_fig(os.path.join(output_dir, "aqi_by_day_of_week.png"))

    # Correlation heatmap (numeric cols only)
    num_df = df.select_dtypes(include=[np.number]).copy()
    if not num_df.empty:
        corr = num_df.corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Correlation heatmap (numeric features)")
        save_fig(os.path.join(output_dir, "correlation_heatmap.png"))

    print(f"EDA complete. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()



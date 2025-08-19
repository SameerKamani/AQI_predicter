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
    print(f"✅ Loaded data with {len(df)} rows and {len(df.columns)} columns")

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
        f.write(f"\nData shape: {df.shape}\n")
        f.write(f"Date range: {df['event_timestamp'].min()} to {df['event_timestamp'].max()}\n")

    # Ensure time ordering
    if "event_timestamp" in df.columns:
        df = df.sort_values("event_timestamp")

    # Time series of AQI
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="event_timestamp", y="aqi_daily")
    plt.title("Daily AQI over time (Karachi) - Enhanced Dataset")
    plt.xlabel("Date")
    plt.ylabel("AQI (EPA 0-500)")
    plt.xticks(rotation=45)
    save_fig(os.path.join(output_dir, "aqi_timeseries.png"))

    # Distributions of key pollutants (using correct column names)
    pollutant_cols = ["pm2_5_mean", "pm10_mean", "o3_mean", "no2_mean", "so2_mean", "co_mean"]
    for col in pollutant_cols:
        if col in df.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col].dropna(), bins=30, kde=True)
            plt.title(f"Distribution: {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            save_fig(os.path.join(output_dir, f"dist_{col}.png"))

    # Seasonality analysis
    if "month" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df["month"], y=df["aqi_daily"])
        plt.title("AQI by Month - Seasonal Patterns")
        plt.xlabel("Month")
        plt.ylabel("AQI")
        save_fig(os.path.join(output_dir, "aqi_by_month.png"))

    if "day_of_week" in df.columns:
        plt.figure(figsize=(10, 6))
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        sns.boxplot(x=df["day_of_week"], y=df["aqi_daily"])
        plt.title("AQI by Day of Week")
        plt.xlabel("Day of Week")
        plt.ylabel("AQI")
        plt.xticks(range(7), day_names)
        save_fig(os.path.join(output_dir, "aqi_by_day_of_week.png"))

    # Seasonal encoding analysis
    if "season_Spring" in df.columns:
        plt.figure(figsize=(12, 5))
        seasons = ["Spring", "Summer", "Winter"]
        season_cols = ["season_Spring", "season_Summer", "season_Winter"]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, (season, col) in enumerate(zip(seasons, season_cols)):
            if col in df.columns:
                season_data = df[df[col] == 1]["aqi_daily"]
                axes[i].hist(season_data, bins=20, alpha=0.7, color=f"C{i}")
                axes[i].set_title(f"AQI Distribution - {season}")
                axes[i].set_xlabel("AQI")
                axes[i].set_ylabel("Frequency")
        plt.tight_layout()
        save_fig(os.path.join(output_dir, "aqi_by_season.png"))

    # Rolling means analysis (key feature from Sheema's approach)
    rolling_cols = [col for col in df.columns if "roll_mean" in col and "aqi" in col]
    if rolling_cols:
        plt.figure(figsize=(12, 8))
        for col in rolling_cols:
            sns.lineplot(data=df, x="event_timestamp", y=col, label=col)
        plt.title("AQI Rolling Means Over Time")
        plt.xlabel("Date")
        plt.ylabel("AQI")
        plt.legend()
        plt.xticks(rotation=45)
        save_fig(os.path.join(output_dir, "aqi_rolling_means.png"))

    # Lag features analysis
    lag_cols = [col for col in df.columns if "lag" in col and "aqi" in col]
    if lag_cols:
        plt.figure(figsize=(12, 8))
        for col in lag_cols:
            sns.lineplot(data=df, x="event_timestamp", y=col, label=col)
        plt.title("AQI Lag Features Over Time")
        plt.xlabel("Date")
        plt.ylabel("AQI")
        plt.legend()
        plt.xticks(rotation=45)
        save_fig(os.path.join(output_dir, "aqi_lag_features.png"))

    # Target variables analysis
    target_cols = ["target_aqi_d1", "target_aqi_d2", "target_aqi_d3"]
    if all(col in df.columns for col in target_cols):
        plt.figure(figsize=(12, 6))
        for col in target_cols:
            sns.lineplot(data=df, x="event_timestamp", y=col, label=col)
        plt.title("Target Variables (D+1, D+2, D+3) Over Time")
        plt.xlabel("Date")
        plt.ylabel("AQI")
        plt.legend()
        plt.xticks(rotation=45)
        save_fig(os.path.join(output_dir, "target_variables.png"))

    # Correlation heatmap (focus on key features)
    key_features = [
        "aqi_daily", "pm2_5_mean", "pm10_mean", "o3_mean", "no2_mean", "so2_mean", "co_mean",
        "temp_mean", "humidity_mean", "wind_speed_mean", "pressure_mean",
        "target_aqi_d1", "target_aqi_d2", "target_aqi_d3"
    ]
    
    available_features = [col for col in key_features if col in df.columns]
    if available_features:
        corr_df = df[available_features].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_df, annot=True, cmap="coolwarm", center=0, fmt=".2f")
        plt.title("Correlation Heatmap - Key Features")
        save_fig(os.path.join(output_dir, "correlation_heatmap_key.png"))

    # Full correlation heatmap (all numeric features)
    num_df = df.select_dtypes(include=[np.number]).copy()
    if not num_df.empty:
        # Sample for visualization (too many features for readable heatmap)
        if len(num_df.columns) > 50:
            # Take first 50 columns for visualization
            sample_cols = num_df.columns[:50]
            corr = num_df[sample_cols].corr(numeric_only=True)
            plt.figure(figsize=(16, 12))
            sns.heatmap(corr, cmap="coolwarm", center=0, xticklabels=True, yticklabels=True)
            plt.title("Correlation Heatmap - Sample of Numeric Features (First 50)")
            save_fig(os.path.join(output_dir, "correlation_heatmap_sample.png"))
        else:
            corr = num_df.corr(numeric_only=True)
            plt.figure(figsize=(16, 12))
            sns.heatmap(corr, cmap="coolwarm", center=0)
            plt.title("Correlation Heatmap - All Numeric Features")
            save_fig(os.path.join(output_dir, "correlation_heatmap_full.png"))

    # Feature importance summary
    feature_summary = os.path.join(output_dir, "feature_summary.txt")
    with open(feature_summary, "w", encoding="utf-8") as f:
        f.write("ENHANCED DATASET FEATURE SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total Features: {len(df.columns)}\n")
        f.write(f"Total Rows: {len(df)}\n")
        f.write(f"Date Range: {df['event_timestamp'].min()} to {df['event_timestamp'].max()}\n\n")
        
        f.write("FEATURE CATEGORIES:\n")
        f.write("-" * 30 + "\n")
        
        # Core features
        core_features = [col for col in df.columns if any(x in col for x in ["aqi", "pm", "o3", "no2", "so2", "co"]) and "roll" not in col and "lag" not in col]
        f.write(f"Core Air Quality Features: {len(core_features)}\n")
        
        # Weather features
        weather_features = [col for col in df.columns if any(x in col for x in ["temp", "humidity", "wind", "pressure", "rain"]) and "roll" not in col and "lag" not in col]
        f.write(f"Weather Features: {len(weather_features)}\n")
        
        # Rolling features
        rolling_features = [col for col in df.columns if "roll" in col]
        f.write(f"Rolling Statistics Features: {len(rolling_features)}\n")
        
        # Lag features
        lag_features = [col for col in df.columns if "lag" in col]
        f.write(f"Lag Features: {len(lag_features)}\n")
        
        # Temporal features
        temporal_features = [col for col in df.columns if any(x in col for x in ["year", "month", "day", "week", "season", "weekday"])]
        f.write(f"Temporal Features: {len(temporal_features)}\n")
        
        # Target features
        target_features = [col for col in df.columns if "target" in col]
        f.write(f"Target Variables: {len(target_features)}\n")

    print(f"✅ Enhanced EDA complete! Outputs written to: {output_dir}")
    print(f"📊 Generated {len(os.listdir(output_dir))} analysis files")
    print(f"🔍 Analyzed {len(df.columns)} features from {len(df)} data points")


if __name__ == "__main__":
    main()



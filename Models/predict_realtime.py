#!/usr/bin/env python3
"""
Real-time AQI Prediction Script
Ensures predictions are always for current date + next 3 days
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from WebApp.Backend.app import model_loader, feast_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Models/Models/registry/realtime_predictions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_current_utc_date():
    """Get current UTC date"""
    return datetime.now(timezone.utc).date()

def validate_feature_freshness(features_path: str) -> bool:
    """Validate that features are current (have today's data)"""
    try:
        if not os.path.exists(features_path):
            logger.error(f"Features file not found: {features_path}")
            return False
        
        # Read the latest features - try CSV first, then parquet as fallback
        df = None
        csv_path = str(features_path).replace('.parquet', '.csv')
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                logger.info(f"Using CSV features file: {csv_path}")
            except Exception as e:
                logger.warning(f"Could not read CSV file: {e}, trying parquet...")
        
        if df is None and os.path.exists(features_path):
            try:
                df = pd.read_parquet(features_path)
                logger.info(f"Using parquet features file: {features_path}")
            except Exception as e:
                logger.error(f"Could not read parquet file: {e}")
                return False
        
        if df is None or df.empty:
            logger.error("No features data available")
            return False
        
        # Get latest timestamp
        latest_ts = pd.to_datetime(df['event_timestamp'].max())
        current_ts = datetime.now(timezone.utc)
        
        # Check if we have today's data (not just hours old)
        latest_date = latest_ts.date()
        current_date = current_ts.date()
        
        logger.info(f"Latest feature date: {latest_date}")
        logger.info(f"Current date: {current_date}")
        
        if latest_date >= current_date:
            logger.info("SUCCESS: Features are current - have today's data")
            return True
        else:
            logger.warning(f"Features are outdated - last: {latest_date}, current: {current_date}")
            return False
            
    except Exception as e:
        logger.error(f"Error validating feature freshness: {e}")
        return False

def generate_realtime_predictions():
    """Generate real-time predictions for current date + next 3 days"""
    try:
        logger.info("Starting real-time AQI predictions...")
        
        # Get current date
        current_date = get_current_utc_date()
        logger.info(f"Current UTC date: {current_date}")
        
        # Validate feature freshness - use CSV first, then parquet
        csv_path = project_root / "Data" / "feature_store" / "karachi_daily_features.csv"
        parquet_path = project_root / "Data" / "feature_store" / "karachi_daily_features.parquet"
        
        # Try CSV first (updated by data pipeline)
        if csv_path.exists():
            if validate_feature_freshness(str(csv_path)):
                features_path = csv_path
                logger.info("SUCCESS: Using CSV features file for validation")
            else:
                # Fallback to parquet if CSV validation fails
                if parquet_path.exists() and validate_feature_freshness(str(parquet_path)):
                    features_path = parquet_path
                    logger.info("SUCCESS: Using parquet features file for validation")
                else:
                    logger.error("ERROR: No current features available in either CSV or parquet")
                    return False
        else:
            # Only parquet available
            if not validate_feature_freshness(str(parquet_path)):
                logger.error("ERROR: No current features available")
                return False
            features_path = parquet_path
        
        # Initialize model registry
        registry_dir = project_root / "Models" / "Models" / "registry"
        model_loader.initialize_registry(str(registry_dir))
        
        # Get latest features from Feast
        feast_repo_path = project_root / "feature_repo"
        latest_features = feast_client.get_latest_features(str(feast_repo_path), str(features_path))
        
        if latest_features is None:
            logger.error("ERROR: No features available for prediction")
            return False
        
        # Generate predictions
        logger.info("Generating predictions...")
        predictions = model_loader.predict_all_from_series(latest_features)
        
        # Add metadata
        predictions['prediction_generated_at'] = datetime.now(timezone.utc).isoformat()
        predictions['prediction_date'] = current_date.isoformat()
        predictions['forecast_dates'] = [
            (current_date + timedelta(days=i)).isoformat() 
            for i in range(1, 4)
        ]
        
        # Save predictions
        output_path = registry_dir / "latest_forecast.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        
        logger.info(f"SUCCESS: Predictions saved to: {output_path}")
        
        # Log prediction summary
        if 'blend' in predictions:
            blend = predictions['blend']
            logger.info("PREDICTION SUMMARY:")
            logger.info(f"  Day 1 ({predictions['forecast_dates'][0]}): {blend.get('hd1', 'N/A'):.2f}")
            logger.info(f"  Day 2 ({predictions['forecast_dates'][1]}): {blend.get('hd2', 'N/A'):.2f}")
            logger.info(f"  Day 3 ({predictions['forecast_dates'][2]}): {blend.get('hd3', 'N/A'):.2f}")
        
        # Check for hazardous AQI levels
        if 'blend' in predictions and 'hd1' in predictions['blend']:
            hd1 = predictions['blend']['hd1']
            if hd1 >= 200:
                logger.warning(f"HAZARDOUS AQI ALERT: Day 1 AQI = {hd1:.2f} >= 200")
            elif hd1 >= 150:
                logger.warning(f"UNHEALTHY AQI: Day 1 AQI = {hd1:.2f} >= 150")
            else:
                logger.info(f"SUCCESS: AQI levels are acceptable: Day 1 AQI = {hd1:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"ERROR: Error generating real-time predictions: {e}")
        return False

def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("REAL-TIME AQI PREDICTION SYSTEM")
    logger.info("=" * 60)
    
    success = generate_realtime_predictions()
    
    if success:
        logger.info("SUCCESS: Real-time predictions completed successfully!")
        sys.exit(0)
    else:
        logger.error("Real-time predictions failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import time
import threading
import queue
import json
import os
from datetime import datetime, timedelta
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Import our prediction models
from predict import PM10Predictor
from model.ensemble import PM10EnsemblePredictor

class RealTimePM10Predictor:
    def __init__(self, prediction_interval=300, history_window=24):
        """
        Initialize real-time PM10 predictor
        
        Args:
            prediction_interval: Time between predictions in seconds (default: 5 minutes)
            history_window: Number of hours to keep in history (default: 24 hours)
        """
        self.prediction_interval = prediction_interval
        self.history_window = history_window
        
        # Initialize predictors
        try:
            self.single_predictor = PM10Predictor()
            self.ensemble_predictor = PM10EnsemblePredictor()
            print("✅ Real-time predictors initialized")
        except Exception as e:
            print(f"❌ Error initializing predictors: {e}")
            self.single_predictor = None
            self.ensemble_predictor = None
        
        # Data storage
        self.pm10_history = deque(maxlen=history_window)
        self.temperature_history = deque(maxlen=history_window)
        self.predictions_history = deque(maxlen=100)  # Keep last 100 predictions
        
        # Real-time data queue
        self.data_queue = queue.Queue()
        self.prediction_queue = queue.Queue()
        
        # Control flags
        self.is_running = False
        self.alert_thresholds = {
            'warning': 50,    # Moderate
            'alert': 100,     # Unhealthy for sensitive groups
            'critical': 150   # Unhealthy
        }
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'alerts_triggered': 0,
            'avg_prediction_time': 0,
            'last_prediction': None,
            'system_uptime': None
        }
        
        # Initialize with sample data if empty
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Initialize with sample data for testing"""
        if len(self.pm10_history) == 0:
            # Generate 24 hours of sample data
            base_pm10 = 30
            for hour in range(24):
                variation = np.sin(hour * np.pi / 12) * 10
                noise = np.random.normal(0, 5)
                pm10_value = max(10, base_pm10 + variation + noise)
                self.pm10_history.append(round(pm10_value, 1))
                self.temperature_history.append(20 + np.random.normal(0, 2))
            
            print(f"📊 Initialized with {len(self.pm10_history)} hours of sample data")
    
    def add_real_time_data(self, pm10_value, temperature, timestamp=None):
        """
        Add new real-time data to the system
        
        Args:
            pm10_value: Current PM10 reading
            temperature: Current temperature
            timestamp: Timestamp of the reading
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Add to history
        self.pm10_history.append(pm10_value)
        self.temperature_history.append(temperature)
        
        # Add to processing queue
        self.data_queue.put({
            'pm10': pm10_value,
            'temperature': temperature,
            'timestamp': timestamp
        })
        
        print(f"📈 Added data: PM10={pm10_value:.1f}, Temp={temperature:.1f}°C")
    
    def get_aqi_category(self, pm10_value):
        """Get AQI category and color"""
        if pm10_value <= 50:
            return "Good", "#00E400", "🟢"
        elif pm10_value <= 100:
            return "Moderate", "#FFFF00", "🟡"
        elif pm10_value <= 150:
            return "Unhealthy for Sensitive Groups", "#FF7E00", "🟠"
        elif pm10_value <= 200:
            return "Unhealthy", "#FF0000", "🔴"
        elif pm10_value <= 300:
            return "Very Unhealthy", "#8F3F97", "🟣"
        else:
            return "Hazardous", "#7E0023", "⚫"
    
    def check_alerts(self, prediction, current_pm10):
        """Check if alerts should be triggered"""
        alerts = []
        
        # Check prediction-based alerts
        if prediction > self.alert_thresholds['critical']:
            alerts.append({
                'level': 'CRITICAL',
                'message': f'Predicted PM10 ({prediction:.1f}) exceeds critical threshold',
                'timestamp': datetime.now(),
                'type': 'prediction'
            })
        elif prediction > self.alert_thresholds['alert']:
            alerts.append({
                'level': 'ALERT',
                'message': f'Predicted PM10 ({prediction:.1f}) exceeds alert threshold',
                'timestamp': datetime.now(),
                'type': 'prediction'
            })
        elif prediction > self.alert_thresholds['warning']:
            alerts.append({
                'level': 'WARNING',
                'message': f'Predicted PM10 ({prediction:.1f}) exceeds warning threshold',
                'timestamp': datetime.now(),
                'type': 'prediction'
            })
        
        # Check current reading alerts
        if current_pm10 > self.alert_thresholds['critical']:
            alerts.append({
                'level': 'CRITICAL',
                'message': f'Current PM10 ({current_pm10:.1f}) exceeds critical threshold',
                'timestamp': datetime.now(),
                'type': 'current'
            })
        
        return alerts
    
    def make_prediction(self):
        """Make a prediction using available models"""
        if len(self.pm10_history) < 24:
            return None, "Insufficient historical data"
        
        start_time = time.time()
        
        try:
            # Get current data
            current_pm10 = self.pm10_history[-1]
            current_temp = self.temperature_history[-1]
            
            # Convert history to list
            pm10_list = list(self.pm10_history)
            temp_list = list(self.temperature_history)
            
            # Make ensemble prediction
            if self.ensemble_predictor and self.ensemble_predictor.is_loaded:
                ensemble_result = self.ensemble_predictor.predict_ensemble(
                    pm10_list, current_temp
                )
                prediction = ensemble_result['ensemble_prediction']
                confidence = ensemble_result['confidence']
                individual_predictions = ensemble_result['individual_predictions']
            else:
                # Fallback to single model
                prediction = self.single_predictor.predict_single(
                    pm10_list, current_temp
                )
                confidence = self.single_predictor.get_prediction_confidence(pm10_list)
                individual_predictions = {}
            
            # Calculate prediction time
            prediction_time = time.time() - start_time
            
            # Create prediction result
            result = {
                'timestamp': datetime.now(),
                'prediction': prediction,
                'current_pm10': current_pm10,
                'current_temperature': current_temp,
                'confidence': confidence,
                'prediction_time': prediction_time,
                'individual_predictions': individual_predictions,
                'aqi_category': self.get_aqi_category(prediction)[0],
                'alerts': self.check_alerts(prediction, current_pm10)
            }
            
            # Update statistics
            self.stats['total_predictions'] += 1
            self.stats['last_prediction'] = datetime.now()
            self.stats['avg_prediction_time'] = (
                (self.stats['avg_prediction_time'] * (self.stats['total_predictions'] - 1) + prediction_time) /
                self.stats['total_predictions']
            )
            
            if result['alerts']:
                self.stats['alerts_triggered'] += len(result['alerts'])
            
            # Add to prediction history
            self.predictions_history.append(result)
            
            return result, None
            
        except Exception as e:
            return None, str(e)
    
    def prediction_worker(self):
        """Worker thread for making predictions"""
        while self.is_running:
            try:
                # Wait for prediction interval
                time.sleep(self.prediction_interval)
                
                if not self.is_running:
                    break
                
                # Make prediction
                result, error = self.make_prediction()
                
                if result:
                    # Add to prediction queue
                    self.prediction_queue.put(result)
                    
                    # Print prediction
                    category, color, emoji = self.get_aqi_category(result['prediction'])
                    print(f"\n🔮 Real-time Prediction ({result['timestamp'].strftime('%H:%M:%S')}):")
                    print(f"   Current PM10: {result['current_pm10']:.1f} μg/m³")
                    print(f"   Predicted PM10: {result['prediction']:.1f} μg/m³ {emoji} ({category})")
                    print(f"   Confidence: {result['confidence']}")
                    print(f"   Prediction time: {result['prediction_time']:.3f}s")
                    
                    # Show alerts
                    if result['alerts']:
                        print(f"   ⚠️  Alerts triggered: {len(result['alerts'])}")
                        for alert in result['alerts']:
                            print(f"      {alert['level']}: {alert['message']}")
                else:
                    print(f"❌ Prediction failed: {error}")
                    
            except Exception as e:
                print(f"❌ Error in prediction worker: {e}")
    
    def data_worker(self):
        """Worker thread for processing incoming data"""
        while self.is_running:
            try:
                # Get data from queue (non-blocking)
                try:
                    data = self.data_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Process data
                pm10 = data['pm10']
                temp = data['temperature']
                timestamp = data['timestamp']
                
                # Check for immediate alerts
                category, color, emoji = self.get_aqi_category(pm10)
                if pm10 > self.alert_thresholds['warning']:
                    print(f"⚠️  {emoji} High PM10 detected: {pm10:.1f} μg/m³ ({category})")
                
            except Exception as e:
                print(f"❌ Error in data worker: {e}")
    
    def start(self):
        """Start the real-time prediction system"""
        if self.is_running:
            print("⚠️  System already running")
            return
        
        self.is_running = True
        self.stats['system_uptime'] = datetime.now()
        
        # Start worker threads
        self.prediction_thread = threading.Thread(target=self.prediction_worker, daemon=True)
        self.data_thread = threading.Thread(target=self.data_worker, daemon=True)
        
        self.prediction_thread.start()
        self.data_thread.start()
        
        print("🚀 Real-time PM10 prediction system started")
        print(f"   Prediction interval: {self.prediction_interval}s")
        print(f"   History window: {self.history_window}h")
        print(f"   Alert thresholds: {self.alert_thresholds}")
    
    def stop(self):
        """Stop the real-time prediction system"""
        self.is_running = False
        print("🛑 Real-time prediction system stopped")
    
    def get_status(self):
        """Get system status"""
        uptime = None
        if self.stats['system_uptime']:
            uptime = datetime.now() - self.stats['system_uptime']
        
        return {
            'is_running': self.is_running,
            'stats': self.stats,
            'uptime': str(uptime) if uptime else None,
            'history_size': len(self.pm10_history),
            'prediction_queue_size': self.prediction_queue.qsize(),
            'data_queue_size': self.data_queue.qsize(),
            'alert_thresholds': self.alert_thresholds
        }
    
    def get_recent_predictions(self, count=10):
        """Get recent predictions"""
        return list(self.predictions_history)[-count:]
    
    def get_current_data(self):
        """Get current data state"""
        if len(self.pm10_history) == 0:
            return None
        
        current_pm10 = self.pm10_history[-1]
        current_temp = self.temperature_history[-1]
        category, color, emoji = self.get_aqi_category(current_pm10)
        
        return {
            'pm10': current_pm10,
            'temperature': current_temp,
            'aqi_category': category,
            'timestamp': datetime.now(),
            'history_length': len(self.pm10_history)
        }

# Example usage and simulation
def simulate_real_time_data(predictor, duration_minutes=10):
    """Simulate real-time data for testing"""
    print(f"🎭 Simulating real-time data for {duration_minutes} minutes...")
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    while time.time() < end_time and predictor.is_running:
        # Generate realistic PM10 data
        base_pm10 = 30 + np.sin(time.time() / 3600) * 20  # Daily pattern
        noise = np.random.normal(0, 8)
        pm10_value = max(10, base_pm10 + noise)
        
        # Generate realistic temperature
        base_temp = 22 + np.sin(time.time() / 3600) * 5
        temp_noise = np.random.normal(0, 2)
        temperature = base_temp + temp_noise
        
        # Add data to predictor
        predictor.add_real_time_data(pm10_value, temperature)
        
        # Wait 30 seconds before next data point
        time.sleep(30)
    
    print("🎭 Simulation completed")

if __name__ == "__main__":
    # Initialize real-time predictor
    predictor = RealTimePM10Predictor(prediction_interval=60)  # Predict every minute
    
    # Start the system
    predictor.start()
    
    try:
        # Run simulation for 5 minutes
        simulate_real_time_data(predictor, duration_minutes=5)
        
        # Keep running for a bit more
        time.sleep(60)
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping system...")
    finally:
        predictor.stop()
        
        # Print final statistics
        status = predictor.get_status()
        print(f"\n📊 Final Statistics:")
        print(f"   Total predictions: {status['stats']['total_predictions']}")
        print(f"   Alerts triggered: {status['stats']['alerts_triggered']}")
        print(f"   Average prediction time: {status['stats']['avg_prediction_time']:.3f}s")
        print(f"   System uptime: {status['uptime']}") 
#!/usr/bin/env python3
"""
Production startup script for PM10 Prediction System

This script launches both the FastAPI backend and Gradio frontend
in separate processes for a complete production deployment.
"""

import subprocess
import sys
import os
import time
import signal
import threading
from pathlib import Path

def run_api():
    """Run the FastAPI backend"""
    print("🚀 Starting FastAPI backend...")
    api_path = Path(__file__).parent / "src" / "api" / "main.py"
    try:
        subprocess.run([sys.executable, str(api_path)], check=True)
    except KeyboardInterrupt:
        print("\n🛑 API server stopped")
    except Exception as e:
        print(f"❌ Error starting API: {e}")

def run_frontend():
    """Run the Gradio frontend"""
    print("🌐 Starting Gradio frontend...")
    frontend_path = Path(__file__).parent / "src" / "gradio_app.py"
    try:
        subprocess.run([sys.executable, str(frontend_path)], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Frontend server stopped")
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'gradio', 'numpy', 'pandas', 
        'scikit-learn', 'joblib', 'plotly', 'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install dependencies with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def main():
    """Main function to start the production system"""
    print("🌬️ PM10 Air Quality Prediction System")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check if model files exist
    model_dir = Path(__file__).parent / "models"
    required_files = ["best_pm10_model_lr.pkl", "scaler_balanced.pkl"]
    
    missing_files = []
    for file in required_files:
        if not (model_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing model files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n🔧 Please train the models first using:")
        print("   python src/model/train_improved.py")
        sys.exit(1)
    
    print("✅ Model files found")
    
    # Start API in a separate thread
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    
    # Wait a moment for API to start
    print("⏳ Waiting for API to start...")
    time.sleep(3)
    
    # Start frontend
    try:
        run_frontend()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        print("👋 Goodbye!")

if __name__ == "__main__":
    main() 
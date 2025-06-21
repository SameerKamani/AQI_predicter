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
    print("🚀 Starting FastAPI backend on port 8001...")
    api_path = Path(__file__).parent / "src" / "web" / "api" / "main.py"
    try:
        # Set environment variable for different port
        env = os.environ.copy()
        env['API_PORT'] = '8001'  # Use port 8001 instead of 8000
        print(f"   API will run on port: {env['API_PORT']}")
        subprocess.run([sys.executable, str(api_path)], check=True, env=env)
    except KeyboardInterrupt:
        print("\n🛑 API server stopped")
    except Exception as e:
        print(f"❌ Error starting API: {e}")

def run_frontend():
    """Run the Gradio frontend"""
    print("🌐 Starting Gradio frontend on port 7861...")
    frontend_path = Path(__file__).parent / "src" / "web" / "gradio_interface.py"
    try:
        # Set environment variable for different port
        env = os.environ.copy()
        env['GRADIO_SERVER_PORT'] = '7861'  # Use port 7861 instead of 7860
        env['API_PORT'] = '8001'  # Also set API port for Gradio to connect to
        print(f"   Frontend will run on port: {env['GRADIO_SERVER_PORT']}")
        subprocess.run([sys.executable, str(frontend_path)], check=True, env=env)
    except KeyboardInterrupt:
        print("\n🛑 Frontend server stopped")
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'gradio', 'numpy', 'pandas', 
        'sklearn', 'joblib', 'plotly', 'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n❌ Missing required packages: {missing_packages}")
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
        print("   python src/ml/train_pm10_models.py")
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
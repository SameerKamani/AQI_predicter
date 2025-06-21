from apscheduler.schedulers.blocking import BlockingScheduler
import subprocess
import os

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), 'aqi_collector.py')

def run_aqi_script():
    print("Running AQI data collection script...")
    subprocess.run(["python", SCRIPT_PATH])

if __name__ == "__main__":
    scheduler = BlockingScheduler()
    scheduler.add_job(run_aqi_script, 'interval', days=3)
    print("Scheduler started. The AQI script will run every 3 days.")
    run_aqi_script()  # Run once at startup
    scheduler.start() 
import pandas as pd
import os

# --- 1. Define File Path ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
INPUT_FILE = os.path.join(DATA_DIR, 'enriched_aqi_data.csv')

# --- 2. Load the Data ---
print(f"Loading data from {INPUT_FILE} for analysis...")
df = pd.read_csv(INPUT_FILE, index_col='datetime', parse_dates=True)

# --- 3. Display Descriptive Statistics ---
print("\n" + "="*50)
print("DESCRIPTIVE STATISTICS")
print("="*50)
print("This shows the mean, standard deviation, and value ranges for each column.")
print("Look for columns with a standard deviation (std) of 0, or min/max values that seem strange.")
print("-"*50)
print(df.describe())

# --- 4. Display Correlation Matrix ---
print("\n" + "="*50)
print("CORRELATION MATRIX")
print("="*50)
print("This shows the linear relationship between each feature and the target (pm25).")
print("Values are from -1 to 1. Closer to 1 or -1 means a stronger relationship.")
print("If all feature correlations with pm25 are near 0, the model cannot learn.")
print("-"*50)
# Calculate and print correlations
print(df.corr())

# --- 5. Check for Missing Values ---
print("\n" + "="*50)
print("MISSING VALUES CHECK")
print("="*50)
print("This shows the number of missing (NaN) values in each column.")
print("There should be zero missing values at this stage.")
print("-"*50)
print(df.isnull().sum()) 
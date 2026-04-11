import json
import os

def create_notebook(cells, filename):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)

def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}

def code_cell(code):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [code]}

# NB00
nb00_cells = [
    md_cell("# 00. Project Setup & Data Download\n\nThis notebook configures the local working environment and downloads the raw AAPL dataset using `yfinance`."),
    code_cell("!pip install yfinance xgboost scikit-learn"),
    md_cell("## 1. Local Environment Setup\n\nWe structure the project using local paths following the implementation plan."),
    code_cell("""# ── LOCAL DEV SETUP ──────────────────────────────────────────────────────────
import os

PROJECT_ROOT = os.path.abspath('.')
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
FIGURES_DIR  = os.path.join(PROJECT_ROOT, 'outputs', 'figures')
METRICS_DIR  = os.path.join(PROJECT_ROOT, 'outputs', 'metrics')
PREDS_DIR    = os.path.join(PROJECT_ROOT, 'outputs', 'predictions')

for d in [DATA_DIR, FIGURES_DIR, METRICS_DIR, PREDS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── PROJECT CONSTANTS ─────────────────────────────────────────────────────────
TICKER       = 'AAPL'
# Flaw Fix: We pad the start date slightly into 2016 to absorb the 26-day MACD burn-in.
# This ensures that our actual clean dataset begins precisely on 2017-01-01.
FETCH_START  = '2016-11-15' 
START_DATE   = '2017-01-01'
END_DATE     = '2024-12-31'
TRAIN_FRAC   = 0.80
RANDOM_STATE = 42

print(f"Data Directory configured at: {DATA_DIR}")"""),
    md_cell("## 2. Download Raw Data\n\nWe download the daily prices. Modern `yfinance` returns a MultiIndex format which we will flatten for compatibility."),
    code_cell("""import yfinance as yf
import pandas as pd

print(f"Downloading {TICKER} from {FETCH_START} to {END_DATE}...")
df_raw = yf.download(TICKER, start=FETCH_START, end=END_DATE)

# Flaw Fix: yfinance recently changed output defaults. It now returns a MultiIndex.
# We flatten the columns to match expected standard OHLCV.
if isinstance(df_raw.columns, pd.MultiIndex):
    df_raw.columns = df_raw.columns.get_level_values(0)

# Ensure only necessary columns are kept
expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
df_raw = df_raw[expected_cols]

# Reset index so 'Date' becomes a column
df_raw.reset_index(inplace=True)

df_raw.head()"""),
    md_cell("## 3. Data Diagnostics"),
    code_cell("""print("Shape of raw data:", df_raw.shape)
print("\\nData Types:\\n", df_raw.dtypes)
print("\\nMissing Values:\\n", df_raw.isnull().sum())"""),
    md_cell("## 4. Save to Disk"),
    code_cell("""# Save the raw dataset
raw_data_path = os.path.join(DATA_DIR, 'raw_stock_data.csv')
df_raw.to_csv(raw_data_path, index=False)
print(f"✅ Successfully saved to {raw_data_path}")""")
]

# NB01
nb01_cells = [
    md_cell("# 01. Data Cleaning & Feature Engineering\n\nThis notebook computes the 12 primary features, prepares the Target variable, handles NaN burn-in, and enforces the standard chronological 80/20 train/test split."),
    code_cell("""import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── LOCAL DEV SETUP ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath('.')
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')

# Constants required for feature extraction
START_DATE   = '2017-01-01'
TRAIN_FRAC   = 0.80"""),
    md_cell("## 1. Data Validation"),
    code_cell("""raw_data_path = os.path.join(DATA_DIR, 'raw_stock_data.csv')
df = pd.read_csv(raw_data_path)

# Ensure Date is parsed correctly and sort fundamentally
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by='Date', ascending=True, inplace=True)

# Forward-fill any gaps due to data artifacts
df.ffill(inplace=True)

print(f"Raw shape: {df.shape}")
assert not df['Date'].duplicated().any(), "Duplicate dates found!"
"""),
    md_cell("## 2. Feature Engineering\n\nWe manually compute all momentum, tracking, and volatility oscillators systematically without external indicators packages."),
    code_cell("""# ----------------------------------------------------
# A. Intermediate computations
# ----------------------------------------------------
df['MA5'] = df['Close'].rolling(window=5).mean()
df['MA20'] = df['Close'].rolling(window=20).mean()
df['Std20'] = df['Close'].rolling(window=20).std()

df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()

# MACD base
df['MACD_Line'] = df['EMA12'] - df['EMA26']
df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()

# Bollinger base
df['BB_Upper'] = df['MA20'] + 2 * df['Std20']
df['BB_Lower'] = df['MA20'] - 2 * df['Std20']

# Volume/OBV base
df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

# RSI base
delta = df['Close'].diff()
gain = delta.clip(lower=0)
loss = (-delta).clip(lower=0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss

# ----------------------------------------------------
# B. Final 12 Model Features
# ----------------------------------------------------
# Feature 1: Close already exists
df['Daily_Return'] = df['Close'].pct_change()                     # Feature 2
df['Rolling_Std5'] = df['Daily_Return'].rolling(window=5).std()   # Feature 3
df['HL_Pct'] = (df['High'] - df['Low']) / df['Close']             # Feature 4
df['RSI_14'] = 100 - (100 / (1 + rs))                             # Feature 5
df['MACD_Hist'] = df['MACD_Line'] - df['MACD_Signal']             # Feature 6
df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['MA20']   # Feature 7
df['Vol_Ratio'] = df['Volume'] / df['Volume_MA5']                 # Feature 8
df['OBV_Change'] = df['OBV'].diff()                               # Feature 9
df['Close_MA5_Pct'] = df['Close'] / df['MA5'] - 1                 # Feature 10
df['Close_MA20_Pct'] = df['Close'] / df['MA20'] - 1               # Feature 11
df['EMA_Spread'] = df['EMA10'] - df['EMA20']                      # Feature 12
"""),
    md_cell("## 3. Target Alignment & Null Filtering\n\nWe set our target variable out 1-step and handle the NaN burn-in naturally created by our 26-day EMAs and shifted targets."),
    code_cell("""# Define Target predicting T+1
df['Target'] = df['Close'].shift(-1)

# Inspect NaN creation before dropping
print("Rows before dropna:", len(df))

# Drop burn-in rows
df.dropna(inplace=True)

# Now apply our hard start date filter since we theoretically padded 
# into 2016 to let MACD spool up.
df = df[df['Date'] >= START_DATE].reset_index(drop=True)

print("Rows after dropping initial burn-in and slicing right boundary:", len(df))

# Confirm clean scope
assert df.isnull().sum().sum() == 0, "Nulls remain in dataset!"
"""),
    md_cell("## 4. Export Outputs (EDA Set + Train/Test Split)"),
    code_cell("""# 1. Save Full Processed Features set for Exploratory Analytics (NB02)
processed_path = os.path.join(DATA_DIR, 'processed_features.csv')
df.to_csv(processed_path, index=False)

# 2. Extract strictly relevant columns for Modelling
FEATURE_COLS = [
    'Close', 'Daily_Return', 'Rolling_Std5', 'HL_Pct', 'RSI_14', 
    'MACD_Hist', 'BB_Width', 'Vol_Ratio', 'OBV_Change', 
    'Close_MA5_Pct', 'Close_MA20_Pct', 'EMA_Spread'
]
MODEL_COLS = ['Date'] + FEATURE_COLS + ['Target']

df_model = df[MODEL_COLS].copy()

# 3. Chronological 80/20 Split
split_idx = int(len(df_model) * TRAIN_FRAC)

train_df = df_model.iloc[:split_idx]
test_df  = df_model.iloc[split_idx:]

train_path = os.path.join(DATA_DIR, 'train.csv')
test_path  = os.path.join(DATA_DIR, 'test.csv')

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"Data generation complete!")
print(f"Train Shape: {train_df.shape} ({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
print(f"Test Shape:  {test_df.shape} ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")
""")
]

create_notebook(nb00_cells, 'c:/Users/yeapz/OneDrive/Documents/data mining assignment/00_project_setup_and_data_download.ipynb')
create_notebook(nb01_cells, 'c:/Users/yeapz/OneDrive/Documents/data mining assignment/01_data_cleaning_and_feature_engineering.ipynb')

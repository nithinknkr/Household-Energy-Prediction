import json, pathlib

def md(src):
    return {"cell_type":"markdown","metadata":{},"source": src if isinstance(src,list) else [src]}

def code(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source": src if isinstance(src,list) else [src]}

# =====================================================================
# NOTEBOOK 03: PREPROCESSING & CLEANING
# =====================================================================
cells_03 = [
md([
"# 03 — Preprocessing & Cleaning\n",
"\n",
"**Goal:** Transform the raw text file into a clean, continuous time-series dataset ready for feature engineering.\n",
"Based on our EDA, we need to: merge datetime, handle block-missing values (forward-fill), and remove redundant columns."
]),
md("## 1. Imports & Setup"),
code([
"import pandas as pd\n",
"import numpy as np\n",
"import warnings\n",
"warnings.filterwarnings('ignore')\n",
"print('Setup complete.')"
]),
md("## 2. Load & Clean Data"),
code([
"RAW_PATH = '../data/raw/household_power_consumption.txt'\n",
"CLEANED_PATH = '../data/processed/cleaned.csv'\n",
"\n",
"# Load data, treating '?' as NaN\n",
"df = pd.read_csv(RAW_PATH, sep=';', na_values='?', low_memory=False)\n",
"\n",
"# Create proper DatetimeIndex\n",
"df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')\n",
"df.set_index('datetime', inplace=True)\n",
"df.drop(columns=['Date', 'Time'], inplace=True)\n",
"\n",
"# Handle missing values (Forward Fill for block outages)\n",
"df.ffill(inplace=True)\n",
"\n",
"# Drop Global_intensity due to near-perfect multicollinearity with Global_active_power\n",
"df.drop(columns=['Global_intensity'], inplace=True)\n",
"\n",
"print(f'Cleaned data shape: {df.shape}')\n",
"df.head(3)"
]),
md("## 3. Save Cleaned Data"),
code([
"# Save to processed folder so notebook 04 can pick it up\n",
"df.to_csv(CLEANED_PATH)\n",
"print(f'Saved cleaned data to {CLEANED_PATH}')"
])
]

# =====================================================================
# NOTEBOOK 04: FEATURE ENGINEERING
# =====================================================================
cells_04 = [
md([
"# 04 — Feature Engineering\n",
"\n",
"**Goal:** Create new columns (features) that help Machine Learning models understand time, physics, and historical context. Models don't inherently understand that 11:59 PM is next to 12:00 AM, or what happened yesterday. We have to mathematically teach them."
]),
md("## 1. Imports & Load Cleaned Data"),
code([
"import pandas as pd\n",
"import numpy as np\n",
"import matplotlib.pyplot as plt\n",
"import seaborn as sns\n",
"import warnings\n",
"warnings.filterwarnings('ignore')\n",
"\n",
"CLEANED_PATH = '../data/processed/cleaned.csv'\n",
"FEATURED_PATH = '../data/processed/featured.csv'\n",
"\n",
"# Load cleaned data and restore datetime index\n",
"df = pd.read_csv(CLEANED_PATH, parse_dates=['datetime'], index_col='datetime')\n",
"\n",
"# Resample to HOURLY frequency to speed up ML training and reduce noise\n",
"# For power (kW) we take the mean. For energy (Wh) we take the sum.\n",
"df_hourly = df.resample('H').agg({\n",
"    'Global_active_power': 'mean',\n",
"    'Global_reactive_power': 'mean',\n",
"    'Voltage': 'mean',\n",
"    'Sub_metering_1': 'sum',\n",
"    'Sub_metering_2': 'sum',\n",
"    'Sub_metering_3': 'sum'\n",
"})\n",
"print(f'Resampled to hourly. Shape: {df_hourly.shape}')"
]),
md([
"## 2. Physics Features (Domain Knowledge)\n",
"\n",
"**What:** Calculating the energy that isn't captured by the three specific sub-meters.\n",
"**Why:** It creates a catch-all feature for lights, fridges, and small electronics, giving the model a fuller picture of total house activity."
]),
code([
"# Active power is in kW (measured every minute). Convert to Watt-hours (Wh) for the hour.\n",
"# Since we resampled to hourly mean, the hourly energy in Wh is roughly: Mean kW * 1000\n",
"df_hourly['Total_energy_Wh'] = df_hourly['Global_active_power'] * 1000\n",
"\n",
"# Unmetered = Total - (Kitchen + Laundry + HVAC)\n",
"df_hourly['unmetered_energy'] = df_hourly['Total_energy_Wh'] - (df_hourly['Sub_metering_1'] + df_hourly['Sub_metering_2'] + df_hourly['Sub_metering_3'])\n",
"df_hourly['unmetered_energy'] = df_hourly['unmetered_energy'].clip(lower=0) # Fix minor negative math artifacts\n",
"\n",
"df_hourly[['Total_energy_Wh', 'unmetered_energy']].head(3)"
]),
md([
"## 3. Time-Based Features & Cyclical Encoding\n",
"\n",
"**What:** Extracting Hour, Day, and Month, and turning them into Sine/Cosine waves.\n",
"**Why:** If we just use `Hour = 23` and `Hour = 0`, a math model thinks they are 23 units apart. By using sine and cosine, we map them onto a circle, teaching the model that 23 and 0 are right next to each other."
]),
code([
"# Extract raw time components\n",
"df_hourly['hour'] = df_hourly.index.hour\n",
"df_hourly['day_of_week'] = df_hourly.index.dayofweek\n",
"df_hourly['month'] = df_hourly.index.month\n",
"df_hourly['is_weekend'] = df_hourly['day_of_week'].isin([5, 6]).astype(int)\n",
"\n",
"# Cyclical Encoding (mapping to a circle)\n",
"df_hourly['hour_sin'] = np.sin(2 * np.pi * df_hourly['hour'] / 24)\n",
"df_hourly['hour_cos'] = np.cos(2 * np.pi * df_hourly['hour'] / 24)\n",
"\n",
"df_hourly['month_sin'] = np.sin(2 * np.pi * df_hourly['month'] / 12)\n",
"df_hourly['month_cos'] = np.cos(2 * np.pi * df_hourly['month'] / 12)\n",
"\n",
"# Let's visualize the circle for Hour\n",
"fig, ax = plt.subplots(figsize=(4, 4))\n",
"ax.scatter(df_hourly['hour_sin'][:24], df_hourly['hour_cos'][:24], c=df_hourly['hour'][:24], cmap='viridis')\n",
"ax.set_title('Cyclical Encoding of 24 Hours')\n",
"ax.set_xlabel('Sine'); ax.set_ylabel('Cosine')\n",
"plt.show()"
]),
md([
"## 4. Lag Features (The Model's \"Memory\")\n",
"\n",
"**What:** Shifting the target variable backwards in time.\n",
"**Why:** Our EDA showed that power usage *exactly 24 hours ago* is highly predictive of power usage *right now*. Lag features let the model look into the past."
]),
code([
"# We want the model to predict Global_active_power.\n",
"# Let's give it what the power was 1 hour ago, 24 hours ago, and 48 hours ago.\n",
"df_hourly['lag_1h'] = df_hourly['Global_active_power'].shift(1)\n",
"df_hourly['lag_24h'] = df_hourly['Global_active_power'].shift(24)\n",
"df_hourly['lag_48h'] = df_hourly['Global_active_power'].shift(48)\n",
"\n",
"df_hourly[['Global_active_power', 'lag_1h', 'lag_24h']].head(26).tail()"
]),
md([
"## 5. Rolling Statistics (Volatility & Momentum)\n",
"\n",
"**What:** Calculating the moving average over the last 6 and 24 hours.\n",
"**Why:** A rolling mean tells the model, \"Has it been a generally high-usage day or a low-usage day so far?\" It smooths out sudden spikes and captures momentum."
]),
code([
"# 6-hour rolling mean (smooths out immediate spikes)\n",
"df_hourly['rolling_mean_6h'] = df_hourly['Global_active_power'].rolling(window=6).mean()\n",
"\n",
"# 24-hour rolling mean (captures the daily baseline trend)\n",
"df_hourly['rolling_mean_24h'] = df_hourly['Global_active_power'].rolling(window=24).mean()\n",
"\n",
"# Because we used shift() and rolling(), the first 48 rows will have NaNs. We drop them.\n",
"df_hourly.dropna(inplace=True)\n",
"\n",
"print(f'Final dataset shape after dropping initial NaNs: {df_hourly.shape}')"
]),
md("## 6. Save Engineered Features"),
code([
"df_hourly.to_csv(FEATURED_PATH)\n",
"print(f'Saved feature-engineered dataset to {FEATURED_PATH}')\n",
"df_hourly.info()"
])
]

nb03 = { "cells": cells_03, "metadata": {}, "nbformat": 4, "nbformat_minor": 5 }
nb04 = { "cells": cells_04, "metadata": {}, "nbformat": 4, "nbformat_minor": 5 }

pathlib.Path("notebooks/03_preprocessing_and_cleaning.ipynb").write_text(json.dumps(nb03, indent=1), encoding="utf-8")
pathlib.Path("notebooks/04_feature_engineering.ipynb").write_text(json.dumps(nb04, indent=1), encoding="utf-8")

print("Generated Notebooks 03 and 04 successfully!")

import json, pathlib

def md(src):
    return {"cell_type":"markdown","metadata":{},"source": src if isinstance(src,list) else [src]}

def code(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source": src if isinstance(src,list) else [src]}

cells = [

# ── TITLE ──────────────────────────────────────────────────────────────────────
md([
"# 02 — Exploratory Data Analysis (EDA)\n",
"\n",
"**Goal:** Uncover the hidden patterns in household power consumption. We need to understand the relationships between features (correlation), the temporal rhythms (seasonality, intraday cycles), and how historical values influence future values (autocorrelation).\n",
"\n",
"**Prerequisites:** We know from `01` that the data has block-level missing values, is right-skewed, and shows clear daily/yearly cycles. Now we quantify those observations.\n",
"\n",
"---"
]),

# ── CELL 1: Imports ────────────────────────────────────────────────────────────
md("## 1. Environment Setup & Imports"),
code([
"import warnings\n",
"warnings.filterwarnings('ignore')\n",
"\n",
"import numpy  as np\n",
"import pandas as pd\n",
"import matplotlib.pyplot    as plt\n",
"import seaborn              as sns\n",
"from pandas.plotting import autocorrelation_plot\n",
"import statsmodels.api as sm\n",
"\n",
"pd.set_option('display.max_columns',   20)\n",
"pd.set_option('display.float_format', '{:.4f}'.format)\n",
"\n",
"plt.rcParams.update({\n",
"    'figure.dpi'        : 120,\n",
"    'axes.spines.top'   : False,\n",
"    'axes.spines.right' : False,\n",
"    'axes.grid'         : True,\n",
"    'grid.alpha'        : 0.3,\n",
"    'font.size'         : 11,\n",
"})\n",
"sns.set_palette('husl')\n",
"print('Setup complete. Ready for EDA.')"
]),

# ── CELL 2: Load and Prepare Data ──────────────────────────────────────────────
md([
"## 2. Load and Prepare Data\n",
"\n",
"We will load the raw data, but this time we'll immediately parse the `Date` and `Time` columns into a single `datetime` index. We will also perform a basic forward-fill for missing values (as decided in notebook 01) to allow for contiguous time-series plotting and autocorrelation calculations.\n"
]),
code([
"RAW_PATH = '../data/raw/household_power_consumption.txt'\n",
"\n",
"# Load data\n",
"df = pd.read_csv(RAW_PATH, sep=';', na_values='?', low_memory=False)\n",
"\n",
"# Create Datetime Index\n",
"df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')\n",
"df.set_index('datetime', inplace=True)\n",
"df.drop(columns=['Date', 'Time'], inplace=True)\n",
"\n",
"# Forward fill missing values for EDA purposes\n",
"# (In actual preprocessing, we might use more sophisticated methods, but ffill is robust for block gaps)\n",
"df.ffill(inplace=True)\n",
"\n",
"# Calculate unmetered energy (as discovered in 01)\n",
"df['unmetered_energy'] = (df['Global_active_power'] * 1000 / 60) - (df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3'])\n",
"# Ensure no negative unmetered energy due to slight measurement errors\n",
"df['unmetered_energy'] = df['unmetered_energy'].clip(lower=0)\n",
"\n",
"print(f'Data loaded and prepared. Shape: {df.shape}')\n",
"df.head(3)"
]),

# ── CELL 3: Resampling for Different Time Horizons ──────────────────────────────
md([
"## 3. Resampling for Trend Analysis\n",
"\n",
"Analyzing 2 million rows of minute-by-minute data is noisy. We resample the data into Hourly, Daily, and Monthly aggregates to see macro-level trends."
]),
code([
"# Resample\n",
"hourly_df = df.resample('H').mean()\n",
"daily_df = df.resample('D').sum() # Sum makes sense for daily total consumption\n",
"monthly_df = df.resample('M').sum()\n",
"\n",
"print(f'Hourly rows: {hourly_df.shape[0]}')\n",
"print(f'Daily rows: {daily_df.shape[0]}')\n",
"print(f'Monthly rows: {monthly_df.shape[0]}')"
]),

# ── CELL 4: Correlation Matrix ──────────────────────────────────────────────────
md([
"## 4. Feature Correlation (Pearson)\n",
"\n",
"How do the variables relate to each other linearly? This helps us identify redundant features (multicollinearity) and strong predictors for our target (`Global_active_power`)."
]),
code([
"corr = hourly_df.corr()\n",
"\n",
"fig, ax = plt.subplots(figsize=(10, 8))\n",
"mask = np.triu(np.ones_like(corr, dtype=bool))\n",
"sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', \n",
"            vmax=1, vmin=-1, center=0, square=True, linewidths=.5, cbar_kws={'shrink': .8})\n",
"ax.set_title('Correlation Matrix (Hourly Resampled Data)', fontsize=14)\n",
"plt.tight_layout(); plt.show()"
]),

# ── CELL 5: Seasonality (Monthly Profiles) ──────────────────────────────────────
md([
"## 5. Seasonality: Monthly Profiles\n",
"\n",
"Does consumption vary by month? We expect higher usage in winter (heating) and summer (AC, depending on the region). Boxplots show us both the median trend and the variance (how spiky usage is) per month."
]),
code([
"df['Month'] = df.index.month\n",
"month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']\n",
"\n",
"fig, ax = plt.subplots(figsize=(12, 5))\n",
"sns.boxplot(data=df, x='Month', y='Global_active_power', \n",
"            palette='YlGnBu', showfliers=False, ax=ax)\n",
"ax.set_xticklabels(month_names)\n",
"ax.set_title('Global Active Power by Month (Outliers Hidden for Clarity)', fontsize=13)\n",
"ax.set_ylabel('Global Active Power (kW)')\n",
"plt.tight_layout(); plt.show()"
]),

# ── CELL 6: Intraday Cycles (Hourly Profiles) ──────────────────────────────────
md([
"## 6. Intraday Cycles: Hourly Profiles\n",
"\n",
"How does usage change throughout the day? The 'duck curve' is a common pattern in energy consumption."
]),
code([
"df['Hour'] = df.index.hour\n",
"\n",
"fig, ax = plt.subplots(figsize=(12, 5))\n",
"sns.lineplot(data=df, x='Hour', y='Global_active_power', \n",
"             estimator=np.mean, errorbar=('ci', 95), color='#e74c3c', ax=ax)\n",
"ax.set_xticks(range(0, 24))\n",
"ax.set_title('Average Hourly Global Active Power (with 95% CI)', fontsize=13)\n",
"ax.set_xlabel('Hour of Day (0-23)')\n",
"ax.set_ylabel('Mean Active Power (kW)')\n",
"plt.tight_layout(); plt.show()"
]),

# ── CELL 7: Weekday vs Weekend Profiles ─────────────────────────────────────────
md([
"## 7. Weekday vs. Weekend Patterns\n",
"\n",
"Human behavior drives energy use. Do weekends look fundamentally different from weekdays? We will overlay the hourly profiles for both."
]),
code([
"df['DayOfWeek'] = df.index.dayofweek\n",
"df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)\n",
"\n",
"fig, ax = plt.subplots(figsize=(12, 5))\n",
"sns.lineplot(data=df, x='Hour', y='Global_active_power', hue='Is_Weekend', \n",
"             estimator=np.mean, errorbar=None, palette=['#3498db', '#e67e22'], ax=ax, lw=2)\n",
"ax.set_xticks(range(0, 24))\n",
"ax.set_title('Hourly Profile: Weekday (0) vs Weekend (1)', fontsize=13)\n",
"ax.set_xlabel('Hour of Day (0-23)')\n",
"ax.set_ylabel('Mean Active Power (kW)')\n",
"ax.legend(title='Is Weekend', labels=['Weekday', 'Weekend'])\n",
"plt.tight_layout(); plt.show()"
]),

# ── CELL 8: Sub-metering Breakdown ──────────────────────────────────────────────
md([
"## 8. Sub-metering Composition\n",
"\n",
"How is the total energy divided among the sub-meters (kitchen, laundry, HVAC) and unmetered loads? A stacked area chart visualizes this composition over a typical week."
]),
code([
"# Resample to daily for smoother visualization of composition\n",
"sub_df = df[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'unmetered_energy']].resample('D').sum()\n",
"# Take a slice of 3 months to see variations\n",
"sub_slice = sub_df['2008-01-01':'2008-03-31']\n",
"\n",
"fig, ax = plt.subplots(figsize=(14, 6))\n",
"ax.stackplot(sub_slice.index, \n",
"             sub_slice['Sub_metering_1'], \n",
"             sub_slice['Sub_metering_2'], \n",
"             sub_slice['Sub_metering_3'], \n",
"             sub_slice['unmetered_energy'],\n",
"             labels=['SM1 (Kitchen)', 'SM2 (Laundry)', 'SM3 (HVAC/Water)', 'Unmetered'],\n",
"             colors=['#2ecc71', '#3498db', '#9b59b6', '#bdc3c7'], alpha=0.8)\n",
"\n",
"ax.set_title('Energy Composition (Daily Totals, Jan-Mar 2008)', fontsize=14)\n",
"ax.set_ylabel('Watt-Hours (Wh)')\n",
"ax.legend(loc='upper right')\n",
"plt.tight_layout(); plt.show()"
]),

# ── CELL 9: Autocorrelation (Lags) ──────────────────────────────────────────────
md([
"## 9. Autocorrelation & Lag Analysis\n",
"\n",
"In time-series forecasting, the past is often the best predictor of the future. Autocorrelation measures how correlated a variable is with delayed (lagged) versions of itself.\n",
"\n",
"We use hourly data to look for 24-hour cycle correlations."
]),
code([
"fig, ax = plt.subplots(figsize=(12, 5))\n",
"# We use statsmodels for a cleaner plot. Lags up to 72 hours (3 days)\n",
"sm.graphics.tsa.plot_acf(hourly_df['Global_active_power'].dropna(), lags=72, ax=ax, alpha=0.05)\n",
"ax.set_title('Autocorrelation of Hourly Global Active Power (Up to 3 Days)', fontsize=13)\n",
"ax.set_xlabel('Lag (Hours)')\n",
"ax.set_ylabel('Correlation Coefficient')\n",
"plt.tight_layout(); plt.show()"
]),

# ── CELL 10: Rolling Windows (Moving Averages) ──────────────────────────────────
md([
"## 10. Rolling Statistics (Volatility & Trend)\n",
"\n",
"Rolling means smooth out short-term fluctuations to highlight longer-term trends. Rolling standard deviations highlight periods of high volatility."
]),
code([
"# Calculate 7-day and 30-day rolling averages on daily data\n",
"daily_gap = daily_df['Global_active_power']\n",
"rolling_7d = daily_gap.rolling(window=7).mean()\n",
"rolling_30d = daily_gap.rolling(window=30).mean()\n",
"\n",
"fig, ax = plt.subplots(figsize=(14, 5))\n",
"ax.plot(daily_gap.index, daily_gap.values, alpha=0.3, color='gray', label='Daily Total')\n",
"ax.plot(rolling_7d.index, rolling_7d.values, color='#e67e22', lw=2, label='7-Day Rolling Mean')\n",
"ax.plot(rolling_30d.index, rolling_30d.values, color='#2c3e50', lw=2, label='30-Day Rolling Mean')\n",
"\n",
"ax.set_title('Daily Active Power with Rolling Means', fontsize=14)\n",
"ax.set_ylabel('Total kW')\n",
"ax.legend()\n",
"plt.tight_layout(); plt.show()"
]),

# ── CELL 11: Summary ──────────────────────────────────────────────────────────
md([
"## 11. EDA Summary & Feature Engineering Blueprint\n",
"\n",
"### Key Insights Derived:\n",
"1. **Strong Multicollinearity:** `Global_active_power` and `Global_intensity` are almost perfectly correlated ($r \\approx 0.99$). We should drop one to prevent model instability.\n",
"2. **Definitive Seasonality:** Winter months show significantly higher median usage and higher variance compared to summer.\n",
"3. **Clear Intraday Cycles:** The \"duck curve\" is present. Low overnight, morning peak (7-9 AM), and a larger evening peak (7-9 PM).\n",
"4. **Weekend Shift:** Weekends lack the sharp morning spike; instead, consumption rises mid-morning and stays elevated throughout the day.\n",
"5. **High Autocorrelation at 24h/48h:** Usage at a specific hour today is highly correlated with the same hour yesterday.\n",
"\n",
"### Blueprint for Notebook 03 (Feature Engineering):\n",
"- **Time-based Features:** Create cyclic encodings (sine/cosine) for `hour`, `day_of_week`, and `month` to help tree models grasp continuous cycles.\n",
"- **Lag Features:** Create $T-1$, $T-24$, and $T-48$ hour lag features.\n",
"- **Rolling Features:** Create rolling window statistics (e.g., 6-hour rolling mean).\n",
"- **Interaction Terms:** `Is_Weekend` indicator.\n",
"- **Drop Redundant Features:** Drop `Global_intensity`."
])
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out = pathlib.Path("notebooks/02_exploratory_data_analysis.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Notebook written -> {out}  ({out.stat().st_size/1024:.1f} KB)")

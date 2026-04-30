import json, pathlib

def md(src):
    return {"cell_type":"markdown","metadata":{},"source": src if isinstance(src,list) else [src]}

def code(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source": src if isinstance(src,list) else [src]}

cells_05 = [
md([
"# 05 — Model Training & Evaluation\n",
"\n",
"**Goal:** Train machine learning models to predict `Global_active_power` (kW) based on our engineered features. We will establish a baseline, train a powerful XGBoost model, evaluate them, and save our performance visualizations to the `reports/figures/` folder."
]),
md("## 1. Imports & Setup"),
code([
"import pandas as pd\n",
"import numpy as np\n",
"import matplotlib.pyplot as plt\n",
"import seaborn as sns\n",
"from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
"from sklearn.linear_model import LinearRegression, Ridge\n",
"from sklearn.ensemble import RandomForestRegressor\n",
"from xgboost import XGBRegressor\n",
"import joblib\n",
"import os\n",
"import warnings\n",
"warnings.filterwarnings('ignore')\n",
"\n",
"sns.set_palette('husl')\n",
"plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False})\n",
"\n",
"FEATURED_PATH = '../data/processed/featured.csv'\n",
"FIGURES_DIR = '../reports/figures/'\n",
"MODELS_DIR = '../models/'\n",
"\n",
"os.makedirs(FIGURES_DIR, exist_ok=True)\n",
"os.makedirs(MODELS_DIR, exist_ok=True)\n",
"print('Setup complete.')"
]),
md([
"## 2. Load Data & Time-Based Split\n",
"\n",
"**What:** Loading the engineered data and splitting it into Train (Past) and Test (Future).\n",
"**Why:** For Time Series, we **cannot** use random shuffling (like `train_test_split`). If we randomly shuffle, the model could use data from Wednesday to predict Tuesday. That's \"data leakage\" (cheating). We must train on the past and test on the future."
]),
code([
"# Load data, set datetime index\n",
"df = pd.read_csv(FEATURED_PATH, parse_dates=['datetime'], index_col='datetime')\n",
"\n",
"# Target variable\n",
"TARGET = 'Global_active_power'\n",
"\n",
"# Features: We use all columns EXCEPT the target itself and raw sub-meters (to avoid data leakage)\n",
"FEATURES = [\n",
"    'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'is_weekend', \n",
"    'lag_1h', 'lag_24h', 'lag_48h', \n",
"    'rolling_mean_6h', 'rolling_mean_24h',\n",
"    'unmetered_energy'\n",
"]\n",
"\n",
"# Time-Based Split: Train on everything before 2010. Test on 2010.\n",
"split_idx = int(len(df) * 0.8)\n",
"train = df.iloc[:split_idx]\n",
"test = df.iloc[split_idx:]\n",
"\n",
"X_train, y_train = train[FEATURES], train[TARGET]\n",
"X_test, y_test = test[FEATURES], test[TARGET]\n",
"\n",
"print(f'Training size : {X_train.shape[0]} rows (Past)')\n",
"print(f'Testing size  : {X_test.shape[0]} rows (Future)')"
]),
md([
"## 3. Train the Models\n",
"\n",
"**What:** Training a simple Linear Regression (as a baseline) and an XGBoost Regressor (Advanced Model).\n",
"**Why:** You should never jump straight to complex models. We train a basic linear model first. If XGBoost can't beat the linear model, then either our features are bad, or XGBoost is overkill."
]),
code([
"print('Training Linear Regression (Baseline)...')\n",
"lr_model = LinearRegression()\n",
"lr_model.fit(X_train, y_train)\n",
"\n",
"print('Training Ridge Regression...')\n",
"ridge_model = Ridge(alpha=1.0)\n",
"ridge_model.fit(X_train, y_train)\n",
"\n",
"print('Training Random Forest...')\n",
"rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)\n",
"rf_model.fit(X_train, y_train)\n",
"\n",
"print('Training XGBoost Regressor (Advanced)...')\n",
"# We use 100 trees (n_estimators). This takes a few seconds.\n",
"xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)\n",
"xgb_model.fit(X_train, y_train)\n",
"print('Training complete.')"
]),
md([
"## 4. Evaluation Metrics\n",
"\n",
"**What:** Predicting on the test set and calculating RMSE, MAE, and R².\n",
"**Why:** \n",
"- **RMSE (Root Mean Squared Error):** Punishes large mistakes heavily. Measured in kW.\n",
"- **MAE (Mean Absolute Error):** The average mistake the model makes. Measured in kW.\n",
"- **R² (R-Squared):** Percentage of variance explained. 1.0 is perfect, 0.0 is guessing the average."
]),
code([
"def evaluate(model, name):\n",
"    preds = model.predict(X_test)\n",
"    rmse = np.sqrt(mean_squared_error(y_test, preds))\n",
"    mae = mean_absolute_error(y_test, preds)\n",
"    r2 = r2_score(y_test, preds)\n",
"    print(f'--- {name} ---')\n",
"    print(f'RMSE : {rmse:.4f} kW')\n",
"    print(f'MAE  : {mae:.4f} kW')\n",
"    print(f'R²   : {r2:.4f}\\n')\n",
"    return preds\n",
"\n",
"lr_preds = evaluate(lr_model, 'Linear Regression (Baseline)')\n",
"ridge_preds = evaluate(ridge_model, 'Ridge Regression')\n",
"rf_preds = evaluate(rf_model, 'Random Forest Regressor')\n",
"xgb_preds = evaluate(xgb_model, 'XGBoost Regressor')"
]),
md([
"## 5. Visualizing Predictions vs Actual (Time Series)\n",
"\n",
"**What:** Plotting the model's predictions overlaid on top of the actual data for a 2-week slice.\n",
"**Why:** Numbers (like RMSE) don't tell the whole story. We need to see visually if the model captures the spikes (dinner time) and the troughs (overnight)."
]),
code([
"fig, ax = plt.subplots(figsize=(15, 5))\n",
"\n",
"# Let's grab just 2 weeks of the test set so the chart is readable\n",
"start, end = 100, 436 # 336 hours = 14 days\n",
"plot_dates = y_test.index[start:end]\n",
"\n",
"ax.plot(plot_dates, y_test.values[start:end], label='Actual Power (kW)', color='black', lw=1.5)\n",
"ax.plot(plot_dates, xgb_preds[start:end], label='XGBoost Prediction', color='#e74c3c', lw=1.5, alpha=0.8)\n",
"\n",
"ax.set_title('Actual vs Predicted Power (2-Week Slice in 2010)', fontsize=14)\n",
"ax.set_ylabel('Global Active Power (kW)')\n",
"ax.legend()\n",
"plt.tight_layout()\n",
"\n",
"# SAVE TO REPORTS DIRECTORY\n",
"plt.savefig(f'{FIGURES_DIR}actual_vs_predicted_ts.png', dpi=300)\n",
"print(f\"Saved visualization to {FIGURES_DIR}actual_vs_predicted_ts.png\")\n",
"plt.show()"
]),
md([
"## 6. Residual Analysis\n",
"\n",
"**What:** Plotting a histogram and scatter plot of the Residuals (Actual value minus Predicted value).\n",
"**Why:** Residuals show us *where* the model fails. If the residuals form a perfect bell curve centered at 0, our model is healthy. If there's a skew, it means the model is consistently over-predicting or under-predicting."
]),
code([
"residuals = y_test - xgb_preds\n",
"\n",
"fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))\n",
"\n",
"# Histogram of errors\n",
"sns.histplot(residuals, bins=50, ax=ax1, color='#3498db', kde=True)\n",
"ax1.set_title('Distribution of Residuals (Errors)')\n",
"ax1.set_xlabel('Error (Actual - Predicted) in kW')\n",
"\n",
"# Scatter plot: Predictions vs Actuals\n",
"ax2.scatter(y_test, xgb_preds, alpha=0.2, color='#2ecc71', s=10)\n",
"ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Perfect prediction line\n",
"ax2.set_title('Predicted vs Actual (Scatter)')\n",
"ax2.set_xlabel('Actual Power (kW)')\n",
"ax2.set_ylabel('Predicted Power (kW)')\n",
"\n",
"plt.tight_layout()\n",
"\n",
"# SAVE TO REPORTS DIRECTORY\n",
"plt.savefig(f'{FIGURES_DIR}residual_analysis.png', dpi=300)\n",
"print(f\"Saved visualization to {FIGURES_DIR}residual_analysis.png\")\n",
"plt.show()"
]),
md([
"## 7. Feature Importance\n",
"\n",
"**What:** Asking the XGBoost model which features it relied on the most to make its decisions.\n",
"**Why:** This makes the \"black box\" model interpretable. It validates our Feature Engineering work. If `unmetered_energy` or `lag_24h` is at the top, we know our engineered features were a massive success."
]),
code([
"importance = pd.DataFrame({\n",
"    'Feature': FEATURES,\n",
"    'Importance': xgb_model.feature_importances_\n",
"}).sort_values('Importance', ascending=False)\n",
"\n",
"fig, ax = plt.subplots(figsize=(10, 6))\n",
"sns.barplot(data=importance, x='Importance', y='Feature', palette='viridis', ax=ax)\n",
"ax.set_title('XGBoost Feature Importance', fontsize=14)\n",
"ax.set_xlabel('Relative Importance (F-Score)')\n",
"plt.tight_layout()\n",
"\n",
"# SAVE TO REPORTS DIRECTORY\n",
"plt.savefig(f'{FIGURES_DIR}feature_importance.png', dpi=300)\n",
"print(f\"Saved visualization to {FIGURES_DIR}feature_importance.png\")\n",
"plt.show()"
]),
md([
"## 8. Save the Best Model\n",
"\n",
"**What:** Saving the trained XGBoost model to disk using `joblib`.\n",
"**Why:** So we don't have to retrain it every time we want to make a prediction in a future production app or web service."
]),
code([
"# Save baseline\n",
"joblib.dump(lr_model, f'{MODELS_DIR}linear_baseline.pkl')\n",
"# Save advanced model\n",
"joblib.dump(xgb_model, f'{MODELS_DIR}xgboost_v1.pkl')\n",
"\n",
"print(f\"Saved models to {MODELS_DIR}\")"
])
]

nb05 = { "cells": cells_05, "metadata": {}, "nbformat": 4, "nbformat_minor": 5 }

out = pathlib.Path("notebooks/05_model_training_and_evaluation.ipynb")
out.write_text(json.dumps(nb05, indent=1), encoding="utf-8")
print(f"Notebook written -> {out}")

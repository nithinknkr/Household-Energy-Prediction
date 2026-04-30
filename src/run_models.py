import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_palette('husl')
plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False})

FEATURED_PATH = 'data/processed/featured.csv'
FIGURES_DIR = 'reports/figures/'
MODELS_DIR = 'models/'
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

df = pd.read_csv(FEATURED_PATH, parse_dates=['datetime'], index_col='datetime')
TARGET = 'Global_active_power'
FEATURES = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'is_weekend', 
            'lag_1h', 'lag_24h', 'lag_48h', 'rolling_mean_6h', 'rolling_mean_24h', 'unmetered_energy']

split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]
X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Random Forest with controlled depth to avoid massive training time
rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)

def evaluate(model, name):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f'--- {name} ---')
    print(f'RMSE : {rmse:.4f} kW')
    print(f'MAE  : {mae:.4f} kW')
    print(f'R2   : {r2:.4f}\n')
    return preds

lr_preds = evaluate(lr_model, 'Linear Regression (Baseline)')
ridge_preds = evaluate(ridge_model, 'Ridge Regression')
rf_preds = evaluate(rf_model, 'Random Forest Regressor')
xgb_preds = evaluate(xgb_model, 'XGBoost Regressor')

# Plot 1
fig, ax = plt.subplots(figsize=(15, 5))
start, end = 100, 436
plot_dates = y_test.index[start:end]
ax.plot(plot_dates, y_test.values[start:end], label='Actual Power (kW)', color='black', lw=1.5)
ax.plot(plot_dates, xgb_preds[start:end], label='XGBoost Prediction', color='#e74c3c', lw=1.5, alpha=0.8)
ax.set_title('Actual vs Predicted Power (2-Week Slice in 2010)', fontsize=14)
ax.set_ylabel('Global Active Power (kW)')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}actual_vs_predicted_ts.png', dpi=300)
plt.close()

# Plot 2
residuals = y_test - xgb_preds
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(residuals, bins=50, ax=ax1, color='#3498db', kde=True)
ax1.set_title('Distribution of Residuals (Errors)')
ax1.set_xlabel('Error (Actual - Predicted) in kW')
ax2.scatter(y_test, xgb_preds, alpha=0.2, color='#2ecc71', s=10)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_title('Predicted vs Actual (Scatter)')
ax2.set_xlabel('Actual Power (kW)')
ax2.set_ylabel('Predicted Power (kW)')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}residual_analysis.png', dpi=300)
plt.close()

# Plot 3
importance = pd.DataFrame({'Feature': FEATURES, 'Importance': xgb_model.feature_importances_}).sort_values('Importance', ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=importance, x='Importance', y='Feature', palette='viridis', ax=ax)
ax.set_title('XGBoost Feature Importance', fontsize=14)
ax.set_xlabel('Relative Importance (F-Score)')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}feature_importance.png', dpi=300)
plt.close()

joblib.dump(lr_model, f'{MODELS_DIR}linear_baseline.pkl')
joblib.dump(ridge_model, f'{MODELS_DIR}ridge_regression.pkl')
joblib.dump(rf_model, f'{MODELS_DIR}random_forest_v1.pkl')
joblib.dump(xgb_model, f'{MODELS_DIR}xgboost_v1.pkl')
print('Script finished successfully. Images and models saved.')

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
FEATURED_PATH = 'data/processed/featured.csv'
MODELS_DIR = 'models/'

df = pd.read_csv(FEATURED_PATH, parse_dates=['datetime'], index_col='datetime')
TARGET = 'Global_active_power'
FEATURES = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'is_weekend', 
            'lag_1h', 'lag_24h', 'lag_48h', 'rolling_mean_6h', 'rolling_mean_24h', 'unmetered_energy']

split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]
X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

# TimeSeriesSplit is CRITICAL for Time Series data!
tscv = TimeSeriesSplit(n_splits=3)

# 1. Ridge Tuning
print("\n--- Tuning Ridge Regression ---")
ridge_param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
ridge_search = RandomizedSearchCV(Ridge(), ridge_param_grid, n_iter=4, cv=tscv, scoring='neg_mean_absolute_error', random_state=42)
ridge_search.fit(X_train, y_train)
best_ridge = ridge_search.best_estimator_
print(f"Best Ridge Params: {ridge_search.best_params_}")

# 2. Random Forest Tuning
print("\n--- Tuning Random Forest Regressor ---")
rf_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, 15]
}
rf_search = RandomizedSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), rf_param_grid, n_iter=4, cv=tscv, scoring='neg_mean_absolute_error', random_state=42)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
print(f"Best RF Params: {rf_search.best_params_}")

# 3. XGBoost Tuning
print("\n--- Tuning XGBoost Regressor ---")
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7]
}
xgb_search = RandomizedSearchCV(XGBRegressor(random_state=42), xgb_param_grid, n_iter=5, cv=tscv, scoring='neg_mean_absolute_error', random_state=42)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_
print(f"Best XGBoost Params: {xgb_search.best_params_}")

# Evaluate tuned models on the UNSEEN test set
print("\n=== FINAL TEST SET EVALUATION ===")
def evaluate(model, name):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name} -> MAE: {mae:.4f} kW | R2: {r2:.4f}")
    return mae

ridge_mae = evaluate(best_ridge, 'Tuned Ridge')
rf_mae = evaluate(best_rf, 'Tuned Random Forest')
xgb_mae = evaluate(best_xgb, 'Tuned XGBoost')

# Save the absolute best model
print("\nSaving the Best Model to models/best_model.pkl...")
# Based on previous results, XGBoost is likely the winner, but we do it dynamically
models_dict = {'Ridge': (best_ridge, ridge_mae), 'Random Forest': (best_rf, rf_mae), 'XGBoost': (best_xgb, xgb_mae)}
winner_name = min(models_dict, key=lambda k: models_dict[k][1])
winner_model = models_dict[winner_name][0]

joblib.dump(winner_model, f'{MODELS_DIR}best_model.pkl')
print(f"Winner: {winner_name}! Saved successfully.")

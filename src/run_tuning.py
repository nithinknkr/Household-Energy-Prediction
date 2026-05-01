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

tscv = TimeSeriesSplit(n_splits=3)

print("\n--- Tuning Ridge Regression ---")
ridge_param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
ridge_search = RandomizedSearchCV(Ridge(), ridge_param_grid, n_iter=4, cv=tscv, scoring='neg_mean_absolute_error', random_state=42)
ridge_search.fit(X_train, y_train)
best_ridge = ridge_search.best_estimator_

print("\n--- Tuning Random Forest Regressor ---")
rf_param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10, 15]}
rf_search = RandomizedSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), rf_param_grid, n_iter=4, cv=tscv, scoring='neg_mean_absolute_error', random_state=42)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

print("\n--- Tuning XGBoost Regressor ---")
xgb_param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7]}
xgb_search = RandomizedSearchCV(XGBRegressor(random_state=42), xgb_param_grid, n_iter=5, cv=tscv, scoring='neg_mean_absolute_error', random_state=42)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_

print("\n=== OVERFITTING VS UNDERFITTING ANALYSIS ===")
def check_generalization(model, name):
    train_preds = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, train_preds)
    train_r2 = r2_score(y_train, train_preds)
    
    test_preds = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)
    
    print(f"\n--- {name} ---")
    print(f"TRAIN -> MAE: {train_mae:.4f} kW | R2: {train_r2:.4f}")
    print(f"TEST  -> MAE: {test_mae:.4f} kW | R2: {test_r2:.4f}")
    
    if train_r2 > 0.90 and test_r2 < 0.60:
        print("Status: OVERFITTING (Memorized the past, failed the future)")
    elif train_r2 < 0.50 and test_r2 < 0.50:
        print("Status: UNDERFITTING (Too simple, failed to learn the pattern)")
    else:
        diff = test_mae - train_mae
        print(f"Status: GENERALIZED (Healthy. Test error is {diff:.4f} kW worse than Train)")
    return test_mae

ridge_mae = check_generalization(best_ridge, 'Tuned Ridge')
rf_mae = check_generalization(best_rf, 'Tuned Random Forest')
xgb_mae = check_generalization(best_xgb, 'Tuned XGBoost')

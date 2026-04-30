import json, pathlib

def md(src):
    return {"cell_type":"markdown","metadata":{},"source": src if isinstance(src,list) else [src]}

def code(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source": src if isinstance(src,list) else [src]}

cells_06 = [
md([
"# 06 — Hyperparameter Tuning\n",
"\n",
"**Goal:** Fine-tune our models to squeeze out the maximum possible performance without overfitting. We will use `TimeSeriesSplit` to ensure we don't accidentally leak future data into the past during cross-validation."
]),
md("## 1. Imports & Data Loading"),
code([
"import pandas as pd\n",
"from sklearn.metrics import mean_absolute_error, r2_score\n",
"from sklearn.linear_model import Ridge\n",
"from sklearn.ensemble import RandomForestRegressor\n",
"from xgboost import XGBRegressor\n",
"from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV\n",
"import joblib\n",
"import warnings\n",
"warnings.filterwarnings('ignore')\n",
"\n",
"FEATURED_PATH = '../data/processed/featured.csv'\n",
"MODELS_DIR = '../models/'\n",
"\n",
"df = pd.read_csv(FEATURED_PATH, parse_dates=['datetime'], index_col='datetime')\n",
"TARGET = 'Global_active_power'\n",
"FEATURES = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'is_weekend', \n",
"            'lag_1h', 'lag_24h', 'lag_48h', 'rolling_mean_6h', 'rolling_mean_24h', 'unmetered_energy']\n",
"\n",
"split_idx = int(len(df) * 0.8)\n",
"train = df.iloc[:split_idx]\n",
"test = df.iloc[split_idx:]\n",
"X_train, y_train = train[FEATURES], train[TARGET]\n",
"X_test, y_test = test[FEATURES], test[TARGET]\n",
"print(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}')"
]),
md([
"## 2. TimeSeriesSplit (Cross-Validation for the Future)\n",
"\n",
"**What:** Instead of standard K-Fold cross-validation (which shuffles data randomly), we use `TimeSeriesSplit`.\n",
"**Why:** If we shuffle time-series data, a model might train on December 2008 and test on January 2008. This is time travel. `TimeSeriesSplit` creates \"folds\" where the train set always strictly precedes the validation set."
]),
code([
"tscv = TimeSeriesSplit(n_splits=3)"
]),
md([
"## 3. Tuning Ridge Regression\n",
"\n",
"**What:** Testing different `alpha` penalties.\n",
"**Why:** To find the perfect balance between letting the model learn from features and punishing it for relying too heavily on any single feature."
]),
code([
"print('Tuning Ridge Regression...')\n",
"ridge_param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}\n",
"ridge_search = RandomizedSearchCV(Ridge(), ridge_param_grid, n_iter=4, cv=tscv, \n",
"                                  scoring='neg_mean_absolute_error', random_state=42)\n",
"ridge_search.fit(X_train, y_train)\n",
"best_ridge = ridge_search.best_estimator_\n",
"print(f'Best Ridge Params: {ridge_search.best_params_}')"
]),
md([
"## 4. Tuning Random Forest\n",
"\n",
"**What:** Adjusting the number of trees (`n_estimators`) and how deep they are allowed to grow (`max_depth`).\n",
"**Why:** A tree that is too deep memorizes the data (overfitting). A tree that is too shallow learns nothing (underfitting). We need the sweet spot."
]),
code([
"print('Tuning Random Forest...')\n",
"rf_param_grid = {\n",
"    'n_estimators': [50, 100],\n",
"    'max_depth': [5, 10, 15]\n",
"}\n",
"rf_search = RandomizedSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), \n",
"                               rf_param_grid, n_iter=4, cv=tscv, \n",
"                               scoring='neg_mean_absolute_error', random_state=42)\n",
"rf_search.fit(X_train, y_train)\n",
"best_rf = rf_search.best_estimator_\n",
"print(f'Best RF Params: {rf_search.best_params_}')"
]),
md([
"## 5. Tuning XGBoost\n",
"\n",
"**What:** Adjusting trees, depth, and `learning_rate`.\n",
"**Why:** In XGBoost, learning rate determines how aggressively each tree tries to fix the mistakes of the previous tree. A high learning rate is fast but reckless; a low learning rate is safe but requires hundreds of trees to finish the job."
]),
code([
"print('Tuning XGBoost...')\n",
"xgb_param_grid = {\n",
"    'n_estimators': [50, 100, 200],\n",
"    'learning_rate': [0.01, 0.05, 0.1],\n",
"    'max_depth': [3, 5, 7]\n",
"}\n",
"xgb_search = RandomizedSearchCV(XGBRegressor(random_state=42), xgb_param_grid, \n",
"                                n_iter=5, cv=tscv, scoring='neg_mean_absolute_error', random_state=42)\n",
"xgb_search.fit(X_train, y_train)\n",
"best_xgb = xgb_search.best_estimator_\n",
"print(f'Best XGBoost Params: {xgb_search.best_params_}')"
]),
md([
"## 6. Final Evaluation & Saving the Best Model\n",
"\n",
"**What:** Pitting all the finely-tuned models against each other on the unseen Test Set one last time.\n",
"**Why:** Cross-validation helps us tune, but the ultimate test is always a completely untouched, virgin dataset. The winner gets saved as `best_model.pkl` and goes to production!"
]),
code([
"def evaluate(model, name):\n",
"    preds = model.predict(X_test)\n",
"    mae = mean_absolute_error(y_test, preds)\n",
"    r2 = r2_score(y_test, preds)\n",
"    print(f'{name} -> MAE: {mae:.4f} kW | R²: {r2:.4f}')\n",
"    return mae\n",
"\n",
"ridge_mae = evaluate(best_ridge, 'Tuned Ridge')\n",
"rf_mae = evaluate(best_rf, 'Tuned Random Forest')\n",
"xgb_mae = evaluate(best_xgb, 'Tuned XGBoost')\n",
"\n",
"# Dynamically save the best\n",
"models_dict = {'Ridge': (best_ridge, ridge_mae), 'Random Forest': (best_rf, rf_mae), 'XGBoost': (best_xgb, xgb_mae)}\n",
"winner_name = min(models_dict, key=lambda k: models_dict[k][1])\n",
"winner_model = models_dict[winner_name][0]\n",
"\n",
"joblib.dump(winner_model, f'{MODELS_DIR}best_model.pkl')\n",
"print(f'\\nWinner: {winner_name}! Saved as best_model.pkl')"
])
]

nb06 = { "cells": cells_06, "metadata": {}, "nbformat": 4, "nbformat_minor": 5 }

out = pathlib.Path("notebooks/06_hyperparameter_tuning.ipynb")
out.write_text(json.dumps(nb06, indent=1), encoding="utf-8")
print(f"Notebook written -> {out}")

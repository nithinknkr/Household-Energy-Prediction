# Household Energy Prediction ⚡🏡

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)

A complete, end-to-end Machine Learning pipeline designed to predict household active power consumption using historical sensor data.

## 📌 Project Overview
This project processes over 2 million raw electricity readings to forecast energy demand. By engineering custom time-series features (lags, rolling statistics, cyclical encoding) and a domain-specific `unmetered_energy` feature, the champion **XGBoost** model predicts household grid spikes with high accuracy (R² ~ 0.79). 

These predictions allow grid operators to anticipate peak demand, balancing loads and preventing blackouts.

## 📂 Project Structure

```text
├── data/
│   ├── raw/             # Raw UCI dataset (household_power_consumption.txt)
│   └── processed/       # Cleaned and feature-engineered datasets
├── models/              # Serialized model artifacts (.pkl)
├── notebooks/           # Jupyter notebooks for Exploratory Data Analysis
├── reports/
│   └── figures/         # Evaluation plots (Actual vs Predicted, Feature Importance)
├── src/                 # Production-ready, OOP Python modules
│   ├── data_loader.py   # Ingests and formats raw txt data
│   ├── preprocessor.py  # Cleans missing values and parses timestamps
│   ├── feature_engineer.py # Creates time-series features and rolling metrics
│   ├── splitter.py      # Performs strict temporal train/test splits
│   ├── trainer.py       # Trains Ridge and XGBoost models
│   ├── evaluator.py     # Calculates metrics and generates plots
│   ├── pipeline.py      # Main orchestrator script
│   └── utils.py         # Standardized logging and timing decorators
├── config.py            # Centralized configuration (Paths, Hyperparams)
└── requirements.txt     # Python dependencies
```

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/nithinknkr/Household-Energy-Prediction.git
cd Household-Energy-Prediction
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the Dataset
Download the UCI "Individual household electric power consumption" dataset and place the unzipped `.txt` file into `data/raw/`.

### 3. Run the ML Pipeline
Execute the fully automated, object-oriented pipeline with a single command:
```bash
python src/pipeline.py
```
*This command will load the raw data, perform all cleaning and feature engineering, train the baseline and XGBoost models, save the best model to `models/best_model.pkl`, and output performance visualizations to `reports/figures/`.*

## 📊 Key Findings & Business Insights
- **Historical Inertia:** The strongest predictor of current energy consumption is usage from exactly 24 hours ago.
- **The "Blind" Load:** Approximately 53% of the household's energy footprint is unmetered. Accounting for this allows the model to predict massive usage spikes within a 22% error margin.
- **Overfitting Prevention:** By utilizing `TimeSeriesSplit` and tuning tree depth, the final model perfectly generalizes to unseen future data without memorizing the past.

## 🛠️ Built With
- **Pandas & NumPy** — Data manipulation and temporal feature engineering
- **Scikit-Learn** — Cross-validation (`TimeSeriesSplit`), randomized search, and baseline models
- **XGBoost** — The champion Gradient Boosting Regressor
- **Matplotlib & Seaborn** — Professional data visualization and residual analysis

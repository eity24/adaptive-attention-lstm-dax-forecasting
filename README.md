# Adaptive Attention-Based LSTM Forecasting (Thesis Project)

This project implements an adaptive deep learning framework for **financial time series forecasting** using **LSTM**, **attention mechanisms**, and **rolling-window retraining**.

The implementation is structured in a modular way to separate data processing, model building, and evaluation.

---

## Project Overview

The goal of this thesis is to improve forecasting performance for financial time series (DAX index) by combining:

- Long Short-Term Memory (LSTM) networks  
- Attention mechanisms  
- Rolling-window (adaptive) retraining  

The framework evaluates multiple models and compares their performance under non-stationary market conditions.

---

## Implemented Models

### 1. Baseline LSTM
- Standard LSTM model for log-return forecasting  
- Serves as a benchmark  

### 2. Attention-LSTM
- Extends baseline LSTM with an attention mechanism  
- Allows the model to focus on relevant time steps  

### 3. Rolling-Window LSTM
- Retrains the model sequentially on sliding windows  
- Improves adaptability to changing market conditions  

### 4. Adaptive Attention-LSTM (Final Model)
- Combines attention and rolling-window retraining  
- Showed modest numerical improvements in forecasting performance
  
 ### 5. Standard Return Baselines
- Naive persistence baseline  
  (forecasting tomorrow’s return using today’s return)
- Mean-return baseline  
  (forecasting using historical average return)

### 6. Feature Engineering Experiments
- Rolling volatility feature  
- RSI + moving average indicators  
- OHLC + return features

### 7. Statistical Validation
- Diebold–Mariano (DM) test  
- Used for statistical comparison of forecasting accuracy between models

---

## Data & Preprocessing

- Data source: **Yahoo Finance (DAX Index)**
- Period: **2010–2026**
- Steps:
  - Selection of closing prices  
  - Log-return computation  
  - Missing value removal  
  - Time-based train-test split (80/20)  
  - Standard scaling  
  - Sequence generation (window size = 20)  

---

## Evaluation

Models are evaluated using:

- RMSE (Root Mean Squared Error)  
- MAE (Mean Absolute Error)  
- MAPE (Mean Absolute Percentage Error)
- Directional Accuracy
- Diebold–Mariano statistical testing

Note: MAPE is less reliable due to near-zero log-return values.

---

## Additional Experiments

The repository also contains additional forecasting experiments and robustness checks:

### Baseline Comparisons
- Standard LSTM baseline
- Naive persistence baseline
- Mean-return baseline

### Feature Extensions
- Rolling volatility
- RSI + moving average indicators
- OHLC + return forecasting

### Statistical Testing
- Diebold–Mariano tests were conducted to compare forecasting performance statistically.

## Preliminary Findings

Preliminary findings indicate modest numerical improvements for the Adaptive Attention-LSTM and OHLC-based models compared to simpler benchmarks. However, statistical tests did not indicate statistically significant forecasting superiority.

## Project Structure

├─ src/  
│ ├─ attention_model.py  
│ ├─ data_loader.py  
│ ├─ evaluation.py  
│ ├─ models.py  
│ ├─ not_custom_attention.py  
│ ├─ plotting.py  
│ ├─ preprocessing.py  
│ ├─ results_table.py  
│ ├─ rolling_window.py  
│ ├─ sequence_builder.py  
│ └─ statistical_tests.py  

├─ main_data_prep.py  
├─ main_baseline_lstm.py  
├─ main_attention_lstm.py  
├─ main_rolling_baseline.py  
├─ main_rolling_attention.py  

├─ main_rolling_attention_volatility.py  
├─ main_rolling_attention_indicators.py  
├─ main_rolling_attention_ohlc.py  

├─ main_naive_baseline.py  
├─ main_dm_test.py  
├─ main_dm_test_mean_baseline.py  
├─ main_dm_test_ohlc.py  

├─ main_hyperparameter_tuning.py  
├─ main_model_comparison.py  
├─ main_residual_plot.py  
├─ main_results_table.py  
├─ main_train_test_plot.py  

├─ results/  
├─ requirements.txt  
└─ README.md

## How to Run

```bash
pip install -r requirements.txt

# run baseline model
python main_baseline_lstm.py

# run attention model
python main_attention_lstm.py

# run rolling baseline
python main_rolling_baseline.py

# run adaptive attention model (final)
python main_rolling_attention.py

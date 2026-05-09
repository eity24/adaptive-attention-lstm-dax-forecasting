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
- Provides the best overall performance  

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

Note: MAPE is less reliable due to near-zero log-return values.

---

## Project Structure
├─ src/
│ ├─ data_loader.py
│ ├─ preprocessing.py
│ ├─ sequence_builder.py
│ ├─ models.py
│ ├─ attention_model.py
│ ├─ rolling_window.py
│ └─ evaluation.py
│
├─ main_data_prep.py
├─ main_baseline_lstm.py
├─ main_attention_lstm.py
├─ main_rolling_baseline.py
├─ main_rolling_attention.py
│
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
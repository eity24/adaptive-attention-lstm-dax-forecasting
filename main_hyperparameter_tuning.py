import os
import numpy as np
import pandas as pd

from src.data_loader import download_dax_data, load_dax_data
from src.preprocessing import (
    keep_required_columns,
    compute_log_returns,
    scale_train_test
)
from src.attention_model import build_attention_lstm
from src.rolling_window import rolling_window_forecast
from src.evaluation import calculate_rmse, calculate_mae, calculate_mape


def main():
    os.makedirs("results", exist_ok=True)

    print("Step 1: Loading data...")
    download_dax_data()
    df = load_dax_data()

    print("Step 2: Preprocessing data...")
    df = keep_required_columns(df)
    df = compute_log_returns(df)

    series = df["log_return"].values

    print("Step 3: Scaling data...")
    train_part = series[:1000]
    test_part = series[1000:]

    train_scaled, test_scaled, scaler = scale_train_test(train_part, test_part)

    scaled_series = list(train_scaled) + list(test_scaled)
    scaled_series = np.array(scaled_series)

    epoch_values = [5, 10, 20, 30]

    results = []

    for epochs in epoch_values:
        print("\n====================================")
        print(f"Running Adaptive Attention-LSTM with epochs={epochs}")
        print("====================================")

        actuals, predictions = rolling_window_forecast(
            series=scaled_series,
            model_builder=build_attention_lstm,
            train_window=1000,
            sequence_length=20,
            epochs=epochs,
            batch_size=32,
            max_steps=50
        )

        rmse = calculate_rmse(actuals, predictions)
        mae = calculate_mae(actuals, predictions)
        mape = calculate_mape(actuals, predictions)

        results.append({
            "epochs": epochs,
            "rmse": rmse,
            "mae": mae,
            "mape": mape
        })

        print(f"Results for epochs={epochs}")
        print("RMSE:", rmse)
        print("MAE:", mae)
        print("MAPE:", mape)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "results/hyperparameter_epochs_tuning.csv",
        index=False
    )

    print("\n====================================")
    print("Hyperparameter tuning completed")
    print("====================================")
    print(results_df)

    print("\nSaved to: results/hyperparameter_epochs_tuning.csv")


if __name__ == "__main__":
    main()
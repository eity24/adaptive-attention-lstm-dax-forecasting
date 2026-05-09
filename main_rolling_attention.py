from src.data_loader import download_dax_data, load_dax_data
from src.preprocessing import (
    keep_required_columns,
    compute_log_returns,
    scale_train_test
)
from src.attention_model import build_attention_lstm
from src.rolling_window import rolling_window_forecast
from src.evaluation import calculate_rmse, calculate_mae, calculate_mape
import numpy as np


def main():
    print("Step 1: Downloading and loading data...")
    download_dax_data()
    df = load_dax_data()

    print("Step 2: Preprocessing data...")
    df = keep_required_columns(df)
    df = compute_log_returns(df)

    series = df["log_return"].values

    print("Step 3: Creating initial scaling...")
    train_part = series[:1000]
    test_part = series[1000:]

    train_scaled, test_scaled, scaler = scale_train_test(train_part, test_part)

    scaled_series = list(train_scaled) + list(test_scaled)
    scaled_series = np.array(scaled_series)

    print("Step 4: Running rolling-window Attention-LSTM (FINAL)...")

    actuals, predictions = rolling_window_forecast(
        series=scaled_series,
        model_builder=build_attention_lstm,
        train_window=1000,
        sequence_length=20,
        epochs=5,
        batch_size=32,
        max_steps=200   #
    )

    print("Step 5: Evaluating Adaptive Attention-LSTM...")

    rmse = calculate_rmse(actuals, predictions)
    mae = calculate_mae(actuals, predictions)
    mape = calculate_mape(actuals, predictions)

    print("\n====================================")
    print("FINAL: Adaptive Attention-LSTM Results (200 steps)")
    print("====================================")
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("MAPE:", mape)


if __name__ == "__main__":
    main()
from src.data_loader import download_dax_data, load_dax_data
from src.attention_model import build_attention_lstm
from src.rolling_window import rolling_window_forecast
from src.evaluation import (
    calculate_rmse,
    calculate_mae,
    calculate_mape,
    calculate_directional_accuracy
)
from src.plotting import plot_predictions, plot_predictions_zoom

from sklearn.preprocessing import StandardScaler
import numpy as np


def select_ohlc_columns(df):
    df = df.copy()

    selected = {}

    for col in df.columns:
        col_name = str(col).lower()

        if col_name == "date" or "date" in col_name:
            selected["Date"] = col

        elif "open" in col_name and "Open" not in selected:
            selected["Open"] = col

        elif "high" in col_name and "High" not in selected:
            selected["High"] = col

        elif "low" in col_name and "Low" not in selected:
            selected["Low"] = col

        elif "close" in col_name and "adj" not in col_name and "Close" not in selected:
            selected["Close"] = col

    needed = ["Date", "Open", "High", "Low", "Close"]

    missing = [
        name for name in needed
        if name not in selected
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    clean_df = df[
        [
            selected["Date"],
            selected["Open"],
            selected["High"],
            selected["Low"],
            selected["Close"]
        ]
    ].copy()

    clean_df.columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close"
    ]

    clean_df = clean_df.dropna().reset_index(drop=True)

    return clean_df

def main():
    print("Step 1: Downloading and loading data...")
    download_dax_data()
    df = load_dax_data()

    print("Step 2: Selecting OHLC columns...")
    df = select_ohlc_columns(df)

    print("Selected columns:")
    print(df.columns)

    print("Step 3: Computing log returns...")

    df["Close"] = df["Close"].astype(float)
    df["log_return"] = np.log(df["Close"]).diff()

    df = df.dropna().reset_index(drop=True)

    print("Step 4: Creating OHLC + return feature matrix...")

    features = df[
        [
            "Open",
            "High",
            "Low",
            "Close",
            "log_return"
        ]
    ].values

    print("Feature shape before scaling:", features.shape)

    print("Step 5: Scaling features...")

    train_part = features[:1000]
    test_part = features[1000:]

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(train_part)
    test_scaled = scaler.transform(test_part)

    scaled_features = np.vstack(
        [
            train_scaled,
            test_scaled
        ]
    )

    print("Feature shape after scaling:", scaled_features.shape)

    print("Step 6: Running Attention-LSTM with OHLC + return features...")

    actuals, predictions = rolling_window_forecast(
        series=scaled_features,
        model_builder=build_attention_lstm,
        train_window=1000,
        sequence_length=20,
        epochs=10,
        batch_size=32,
        max_steps=50,
        n_features=5,
        target_column=4
    )

    print("Step 7: Evaluating Attention-LSTM with OHLC + return features...")

    rmse = calculate_rmse(actuals, predictions)
    mae = calculate_mae(actuals, predictions)
    mape = calculate_mape(actuals, predictions)
    directional_accuracy = calculate_directional_accuracy(
        actuals,
        predictions
    )

    print("\n====================================")
    print("Attention-LSTM + OHLC + Return Results")
    print("====================================")
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("MAPE:", mape)
    print("Directional Accuracy:", directional_accuracy)

    np.save(
        "results/ohlc_actuals.npy",
        actuals
    )

    np.save(
        "results/ohlc_predictions.npy",
        predictions
    )

    print("OHLC model predictions saved!")

    print("Step 8: Plotting results...")

    plot_predictions(
        actuals,
        predictions,
        title="Attention-LSTM with OHLC + Return: Actual vs Predicted",
        save_path="results/ohlc_plot.png"
    )

    plot_predictions_zoom(
        actuals,
        predictions,
        title="Attention-LSTM with OHLC + Return (First 50 Steps)",
        n_points=50,
        save_path="results/ohlc_zoom_plot.png"
    )


if __name__ == "__main__":
    main()
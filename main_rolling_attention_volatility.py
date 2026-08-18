from src.data_loader import download_dax_data, load_dax_data
from src.preprocessing import (
    keep_required_columns,
    compute_log_returns,
    compute_rolling_volatility
)
from src.attention_model import build_attention_lstm
from src.rolling_window import rolling_window_forecast
from src.evaluation import (
    calculate_rmse,
    calculate_mae,
    calculate_mape,
    calculate_directional_accuracy
)
from src.plotting import (
    plot_predictions,
    plot_predictions_zoom
)

from sklearn.preprocessing import StandardScaler
import numpy as np


def main():
    print("Step 1: Downloading and loading data...")
    download_dax_data()
    df = load_dax_data()

    print("Step 2: Preprocessing data...")
    df = keep_required_columns(df)
    df = compute_log_returns(df)

    # NEW: rolling volatility feature
    df = compute_rolling_volatility(
        df,
        window=20
    )

    print("Step 3: Creating feature matrix...")

    # 2 features
    features = df[
        ["log_return", "rolling_volatility"]
    ].values

    print(
        "Feature shape before scaling:",
        features.shape
    )

    print("Step 4: Scaling features...")

    train_part = features[:1000]
    test_part = features[1000:]

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        train_part
    )

    test_scaled = scaler.transform(
        test_part
    )

    scaled_features = np.vstack([
        train_scaled,
        test_scaled
    ])

    print(
        "Feature shape after scaling:",
        scaled_features.shape
    )

    print(
        "Step 5: Running rolling-window "
        "Attention-LSTM with volatility feature..."
    )

    actuals, predictions = rolling_window_forecast(
        series=scaled_features,
        model_builder=build_attention_lstm,
        train_window=1000,
        sequence_length=20,
        epochs=10,
        batch_size=32,
        max_steps=200,   # first test run
        n_features=2,
        target_column=0
    )

    print(
        "Step 6: Evaluating Adaptive "
        "Attention-LSTM with volatility feature..."
    )

    rmse = calculate_rmse(
        actuals,
        predictions
    )

    mae = calculate_mae(
        actuals,
        predictions
    )

    mape = calculate_mape(
        actuals,
        predictions
    )

    directional_accuracy = (
        calculate_directional_accuracy(
            actuals,
            predictions
        )
    )

    print("\n====================================")
    print(
        "Adaptive Attention-LSTM "
        "+ Rolling Volatility Results"
    )
    print("====================================")
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("MAPE:", mape)
    print(
        "Directional Accuracy:",
        directional_accuracy
    )

    np.save(
        "results/volatility_adaptive_actuals.npy",
        actuals
    )

    np.save(
        "results/volatility_adaptive_predictions.npy",
        predictions
    )

    print(
        "Volatility model predictions saved!"
    )

    print("Step 7: Plotting results...")

    plot_predictions(
        actuals,
        predictions,
        title=(
            "Adaptive Attention-LSTM "
            "with Volatility: Actual vs Predicted"
        ),
        save_path=(
            "results/"
            "volatility_adaptive_plot.png"
        )
    )

    plot_predictions_zoom(
        actuals,
        predictions,
        title=(
            "Adaptive Attention-LSTM "
            "with Volatility "
            "(First 50 Steps)"
        ),
        n_points=50,
        save_path=(
            "results/"
            "volatility_adaptive_zoom_plot.png"
        )
    )


if __name__ == "__main__":
    main()
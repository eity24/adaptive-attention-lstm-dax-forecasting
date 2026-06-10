import os
import numpy as np
import pandas as pd

from src.data_loader import download_dax_data, load_dax_data
from src.preprocessing import (
    keep_required_columns,
    compute_log_returns,
    train_test_split_timewise,
    scale_train_test
)
from src.evaluation import (
    calculate_rmse,
    calculate_mae,
    calculate_mape,
    calculate_directional_accuracy
)


def main():
    os.makedirs("results", exist_ok=True)

    print("Step 1: Downloading and loading data...")
    download_dax_data()
    df = load_dax_data()

    print("Step 2: Preprocessing data...")
    df = keep_required_columns(df)
    df = compute_log_returns(df)

    series = df["log_return"].values

    print("Step 3: Train-test split...")
    train, test = train_test_split_timewise(series)

    print("Step 4: Scaling data...")
    train_scaled, test_scaled, scaler = scale_train_test(train, test)

    # Naive baseline needs same target structure as LSTM test sequences
    window_size = 20

    y_test = test_scaled[window_size:]

    print("Step 5: Creating Naive Baseline predictions...")

    # Baseline 1: Persistence model
    # Tomorrow return = today's return
    persistence_predictions = test_scaled[window_size - 1:-1]

    # Baseline 2: Mean return model
    # Tomorrow return = average training return
    mean_return_value = np.mean(train_scaled)
    mean_predictions = np.full_like(y_test, mean_return_value)

    print("Step 6: Evaluating Naive Baselines...")

    results = []

    for model_name, predictions in [
        ("Naive Persistence Baseline", persistence_predictions),
        ("Mean Return Baseline", mean_predictions)
    ]:
        rmse = calculate_rmse(y_test, predictions)
        mae = calculate_mae(y_test, predictions)
        mape = calculate_mape(y_test, predictions)
        directional_accuracy = calculate_directional_accuracy(
            y_test,
            predictions
        )

        results.append({
            "Model": model_name,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "Directional Accuracy": directional_accuracy
        })

        print("\n====================================")
        print(model_name)
        print("====================================")
        print("RMSE:", rmse)
        print("MAE:", mae)
        print("MAPE:", mape)
        print("Directional Accuracy:", directional_accuracy)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "results/naive_baseline_results.csv",
        index=False
    )

    np.save("results/naive_actuals.npy", y_test)
    np.save(
        "results/persistence_predictions.npy",
        persistence_predictions
    )
    np.save(
        "results/mean_return_predictions.npy",
        mean_predictions
    )

    print("\nNaive baseline results saved!")
    print("Saved to: results/naive_baseline_results.csv")


if __name__ == "__main__":
    main()
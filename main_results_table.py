import pandas as pd
from src.results_table import print_results_table


def main():

    results = [
        {
            "model": "Baseline LSTM",
            "rmse": 0.7275,
            "mae": 0.5346,
            "mape": 111.58
        },
        {
            "model": "Attention-LSTM",
            "rmse": 0.7269,
            "mae": 0.5342,
            "mape": 103.14
        },
        {
            "model": "Rolling LSTM (200)",
            "rmse": 0.7082,
            "mae": 0.5349,
            "mape": 127.20
        },
        {
            "model": "Adaptive Attention-LSTM",
            "rmse": 0.7069,
            "mae": 0.5346,
            "mape": 103.87
        }
    ]

    print_results_table(results)

    df = pd.DataFrame(results)

    df.to_csv(
        "results/model_comparison.csv",
        index=False
    )

    print("\nCSV saved successfully!")


if __name__ == "__main__":
    main()
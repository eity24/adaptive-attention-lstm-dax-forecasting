import numpy as np
from src.statistical_tests import diebold_mariano_test


def main():

    print("Loading predictions...")

    # Mean baseline
    y_true_baseline = np.load(
        "results/naive_actuals.npy"
    )

    mean_predictions = np.load(
        "results/mean_return_predictions.npy"
    )

    # Adaptive Attention-LSTM
    adaptive_actuals = np.load(
        "results/adaptive_actuals.npy"
    )

    adaptive_predictions = np.load(
        "results/adaptive_predictions.npy"
    )

    # Align lengths
    min_length = min(
        len(y_true_baseline),
        len(adaptive_actuals)
    )

    y_true = y_true_baseline[:min_length]

    mean_pred = mean_predictions[:min_length]

    adaptive_pred = adaptive_predictions[:min_length]

    print("Running Diebold-Mariano Test...")

    dm_stat, p_value = diebold_mariano_test(
        y_true,
        mean_pred,
        adaptive_pred
    )

    print("\n==============================")
    print("DM Test: Mean Baseline vs Adaptive Attention-LSTM")
    print("==============================")
    print("DM Statistic:", dm_stat)
    print("P-value:", p_value)

    if p_value < 0.05:
        print(
            "\nResult: Statistically "
            "significant difference."
        )
    else:
        print(
            "\nResult: No statistically "
            "significant difference."
        )


if __name__ == "__main__":
    main()
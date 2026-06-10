import numpy as np
from src.statistical_tests import diebold_mariano_test


def main():
    print("Loading saved predictions...")

    baseline_actuals = np.load("results/baseline_actuals.npy")
    baseline_predictions = np.load("results/baseline_predictions.npy")

    ohlc_actuals = np.load("results/ohlc_actuals.npy")
    ohlc_predictions = np.load("results/ohlc_predictions.npy")

    min_length = min(
        len(baseline_actuals),
        len(ohlc_actuals)
    )

    y_true = baseline_actuals[:min_length]
    baseline_pred = baseline_predictions[:min_length]
    ohlc_pred = ohlc_predictions[:min_length]

    print("Running Diebold-Mariano Test...")

    dm_stat, p_value = diebold_mariano_test(
        y_true,
        baseline_pred,
        ohlc_pred
    )

    print("\n===========================")
    print("DM Test: Baseline vs OHLC")
    print("===========================")
    print("DM Statistic:", dm_stat)
    print("P-value:", p_value)

    if p_value < 0.05:
        print("\nResult: Statistically significant difference.")
    else:
        print("\nResult: No statistically significant difference.")


if __name__ == "__main__":
    main()
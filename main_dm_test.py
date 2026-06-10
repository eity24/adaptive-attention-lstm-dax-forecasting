import numpy as np
from src.statistical_tests import diebold_mariano_test


def main():

    print("Loading saved predictions...")

    baseline_actuals = np.load(
        "results/baseline_actuals.npy"
    )

    baseline_predictions = np.load(
        "results/baseline_predictions.npy"
    )

    adaptive_actuals = np.load(
        "results/adaptive_actuals.npy"
    )

    adaptive_predictions = np.load(
        "results/adaptive_predictions.npy"
    )

    # same length
    min_length = min(
        len(baseline_actuals),
        len(adaptive_actuals)
    )

    y_true = baseline_actuals[:min_length]

    baseline_pred = baseline_predictions[:min_length]
    adaptive_pred = adaptive_predictions[:min_length]

    print("Running Diebold-Mariano Test...")

    dm_stat, p_value = diebold_mariano_test(
        y_true,
        baseline_pred,
        adaptive_pred
    )

    print("\n===========================")
    print("Diebold-Mariano Test Result")
    print("===========================")

    print("DM Statistic:", dm_stat)
    print("P-value:", p_value)

    if p_value < 0.05:
        print("\nResult: Statistically significant difference.")
    else:
        print("\nResult: No statistically significant difference.")


if __name__ == "__main__":
    main()
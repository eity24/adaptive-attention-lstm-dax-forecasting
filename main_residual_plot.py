import numpy as np
import matplotlib.pyplot as plt


def main():

    actuals = np.load("results/adaptive_actuals.npy")
    predictions = np.load("results/adaptive_predictions.npy")

    residuals = actuals - predictions

    plt.figure(figsize=(10, 6))

    plt.scatter(
        predictions,
        residuals,
        alpha=0.7
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.title("Residual Plot")
    plt.xlabel("Predicted Value")
    plt.ylabel("Residual Error")

    plt.savefig("results/residual_plot.png")
    plt.show()

    print("Residual plot saved!")


if __name__ == "__main__":
    main()
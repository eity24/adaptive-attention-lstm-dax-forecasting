import matplotlib.pyplot as plt


def main():

    models = [
        "Baseline LSTM",
        "Attention-LSTM",
        "Rolling LSTM",
        "Adaptive Attention"
    ]

    rmse = [
        0.7275,
        0.7269,
        0.7082,
        0.7069
    ]

    plt.figure(figsize=(10, 6))

    plt.bar(models, rmse)

    plt.title("Model Performance Comparison (RMSE)")
    plt.xlabel("Models")
    plt.ylabel("RMSE")

    plt.savefig("results/model_comparison.png")
    plt.show()

    print("Model comparison plot saved!")


if __name__ == "__main__":
    main()
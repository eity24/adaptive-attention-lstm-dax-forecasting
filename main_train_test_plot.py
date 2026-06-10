import matplotlib.pyplot as plt

from src.data_loader import download_dax_data, load_dax_data
from src.preprocessing import (
    keep_required_columns,
    compute_log_returns
)


def main():

    print("Loading data...")

    download_dax_data()
    df = load_dax_data()

    df = keep_required_columns(df)
    df = compute_log_returns(df)

    series = df["log_return"].values

    train = series[:1000]
    test = series[1000:]

    plt.figure(figsize=(12, 6))

    plt.plot(
        range(len(train)),
        train,
        label="Training Data"
    )

    plt.plot(
        range(len(train), len(train) + len(test)),
        test,
        label="Test Data"
    )

    plt.title("Train-Test Split of DAX Log Returns")
    plt.xlabel("Time")
    plt.ylabel("Log Return")
    plt.legend()

    plt.savefig("results/train_test_split.png")
    plt.show()

    print("Train-test plot saved!")


if __name__ == "__main__":
    main()
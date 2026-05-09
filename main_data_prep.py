from src.data_loader import download_dax_data, load_dax_data
from src.preprocessing import (
    keep_required_columns,
    compute_log_returns,
    train_test_split_timewise,
    scale_train_test
)
from src.sequence_builder import create_sequences


def main():
    print("Downloading data...")
    download_dax_data()

    df = load_dax_data()

    df = keep_required_columns(df)
    df = compute_log_returns(df)

    print(df.head())

    series = df["log_return"].values

    train, test = train_test_split_timewise(series)

    train_scaled, test_scaled, scaler = scale_train_test(train, test)

    X_train, y_train = create_sequences(train_scaled)
    X_test, y_test = create_sequences(test_scaled)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)


if __name__ == "__main__":
    main()
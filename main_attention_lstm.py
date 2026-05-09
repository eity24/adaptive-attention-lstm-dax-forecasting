from src.evaluation import calculate_rmse, calculate_mae, calculate_mape
from src.data_loader import download_dax_data, load_dax_data
from src.preprocessing import (
    keep_required_columns,
    compute_log_returns,
    train_test_split_timewise,
    scale_train_test
)
from src.sequence_builder import create_sequences
from src.attention_model import build_attention_lstm


def main():
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

    print("Step 5: Creating sequences...")
    X_train, y_train = create_sequences(train_scaled, window_size=20)
    X_test, y_test = create_sequences(test_scaled, window_size=20)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    print("Step 6: Building Attention-LSTM model...")
    model = build_attention_lstm(window_size=20)

    print(model.summary())

    print("Step 7: Training Attention-LSTM model...")
    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    print("Training completed.")

    print("Step 8: Making predictions...")
    y_pred = model.predict(X_test)
    y_pred = y_pred.flatten()

    print("Step 9: Evaluating model...")
    rmse = calculate_rmse(y_test, y_pred)
    mae = calculate_mae(y_test, y_pred)
    mape = calculate_mape(y_test, y_pred)

    print("Attention-LSTM Evaluation Results:")
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("MAPE:", mape)


if __name__ == "__main__":
    main()
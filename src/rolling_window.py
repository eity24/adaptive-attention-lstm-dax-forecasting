import numpy as np
from src.sequence_builder import create_sequences


def rolling_window_forecast(
    series,
    model_builder,
    train_window=1000,
    sequence_length=20,
    epochs=5,
    batch_size=32,
    max_steps=50
):
    predictions = []
    actuals = []

    total_steps = len(series) - train_window

    if max_steps is not None:
        total_steps = min(total_steps, max_steps)

    for step in range(total_steps):
        start_idx = step
        end_idx = start_idx + train_window

        train_data = series[start_idx:end_idx]
        actual_value = series[end_idx]

        X_train, y_train = create_sequences(train_data, window_size=sequence_length)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

        model = model_builder(window_size=sequence_length)

        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        last_sequence = train_data[-sequence_length:]
        X_test = last_sequence.reshape((1, sequence_length, 1))

        predicted_value = model.predict(X_test, verbose=0)[0][0]

        predictions.append(predicted_value)
        actuals.append(actual_value)

        print(f"Step {step + 1}/{total_steps} completed")

    return np.array(actuals), np.array(predictions)
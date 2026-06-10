import numpy as np


def create_sequences_multifeature(data, target, window_size=20):
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(target[i])

    return np.array(X), np.array(y)


def rolling_window_forecast(
    series,
    model_builder,
    train_window=1000,
    sequence_length=20,
    epochs=5,
    batch_size=32,
    max_steps=50,
    n_features=1,
    target_column=0
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

        actual_value = series[end_idx, target_column] if n_features > 1 else series[end_idx]

        if n_features > 1:
            target_data = train_data[:, target_column]
            X_train, y_train = create_sequences_multifeature(
                train_data,
                target_data,
                window_size=sequence_length
            )
        else:
            target_data = train_data
            X_train, y_train = create_sequences_multifeature(
                train_data.reshape(-1, 1),
                target_data,
                window_size=sequence_length
            )

        model = model_builder(
            window_size=sequence_length,
            n_features=n_features
        )

        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        last_sequence = train_data[-sequence_length:]

        if n_features == 1:
            X_test = last_sequence.reshape((1, sequence_length, 1))
        else:
            X_test = last_sequence.reshape((1, sequence_length, n_features))

        predicted_value = model.predict(X_test, verbose=0)[0][0]

        predictions.append(predicted_value)
        actuals.append(actual_value)

        print(f"Step {step + 1}/{total_steps} completed")

    return np.array(actuals), np.array(predictions)
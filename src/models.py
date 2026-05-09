from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_baseline_lstm(window_size=20):
    model = Sequential()

    model.add(LSTM(
        units=50,
        input_shape=(window_size, 1)
    ))

    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model
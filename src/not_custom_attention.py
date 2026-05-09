from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.layers import Attention


def build_attention_lstm(window_size=20):
    inputs = Input(shape=(window_size, 1))

    lstm_out = LSTM(50, return_sequences=True)(inputs)

    attention_out = Attention()([lstm_out, lstm_out])

    lstm_out2 = LSTM(50)(attention_out)

    dropout = Dropout(0.2)(lstm_out2)

    outputs = Dense(1)(dropout)

    model = Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model
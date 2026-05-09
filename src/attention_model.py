import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout


class SimpleAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SimpleAttention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True
        )

        self.b = self.add_weight(
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )

    def call(self, x):
        score = tf.tanh(tf.matmul(x, self.W) + self.b)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(attention_weights * x, axis=1)
        return context_vector


def build_attention_lstm(window_size=20):
    inputs = Input(shape=(window_size, 1))

    lstm_out = LSTM(50, return_sequences=True)(inputs)

    attention_out = SimpleAttention()(lstm_out)

    dropout = Dropout(0.2)(attention_out)

    outputs = Dense(1)(dropout)

    model = Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model
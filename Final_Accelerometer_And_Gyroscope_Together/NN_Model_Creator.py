import tensorflow as tf
from tensorflow import keras

class NN_Model_Creator:
    def create_dnn_model(self, unitsSize, activationFunction, lossFunction):

        if activationFunction.casefold() == "LeakyRelu".casefold():
            model = keras.Sequential()
            model.add(tf.keras.layers.Flatten(input_shape=(1, 49)))
            model.add(tf.keras.layers.Dense(unitsSize))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
            model.add(tf.keras.layers.Dense(64))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
            model.add(tf.keras.layers.Dense(8))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
            model.add(tf.keras.layers.Dense(6))

        else:
            model = keras.Sequential([
                tf.keras.layers.Flatten(input_shape=(1, 49)),
                tf.keras.layers.Dense(unitsSize, activation=activationFunction),
                tf.keras.layers.Dense(64, activation=activationFunction),
                tf.keras.layers.Dense(8, activation=activationFunction),
                tf.keras.layers.Dense(6)
            ])

        optimizer = tf.keras.optimizers.Adam()

        model.compile(loss=lossFunction,
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
        return model

    def create_cnn_model(self, filtersSize, kernelSize, activationFunction, lossFunction):

        if activationFunction.casefold() == "LeakyRelu".casefold():
            model = keras.Sequential()
            model.add(tf.keras.layers.Conv1D(filters=filtersSize, kernel_size=kernelSize, padding='same', activation='tanh'))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
            model.add(tf.keras.layers.Dense(64))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
            model.add(tf.keras.layers.Dense(8))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
            model.add(tf.keras.layers.Dense(6))

        else:
            model = keras.Sequential([
                tf.keras.layers.Conv1D(filters=filtersSize, kernel_size=kernelSize, padding='same', activation='tanh'),
                tf.keras.layers.Dense(64, activation=activationFunction),
                tf.keras.layers.Dense(8, activation=activationFunction),
                tf.keras.layers.Dense(6)
            ])

        optimizer = tf.keras.optimizers.Adam()

        model.compile(loss=lossFunction,
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
        return model

    def create_rnn_model(self, unitsSize, activationFunction, lossFunction):

        if activationFunction.casefold() == "LeakyRelu".casefold():
            model = keras.Sequential()
            model.add(tf.keras.layers.SimpleRNN(unitsSize))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
            model.add(tf.keras.layers.Dense(64))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
            model.add(tf.keras.layers.Dense(8))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
            model.add(tf.keras.layers.Dense(6))

        else:
            model = keras.Sequential([
                tf.keras.layers.SimpleRNN(unitsSize),
                tf.keras.layers.Dense(64, activation=activationFunction),
                tf.keras.layers.Dense(8, activation=activationFunction),
                tf.keras.layers.Dense(6)
            ])

        optimizer = tf.keras.optimizers.Adam()

        model.compile(loss=lossFunction,
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
        return model

    def create_lstm_model(self, unitsSize, activationFunction, lossFunction):

        if activationFunction.casefold() == "LeakyRelu".casefold():
            model = keras.Sequential()
            model.add(tf.keras.layers.LSTM(unitsSize))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
            model.add(tf.keras.layers.Dense(64))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
            model.add(tf.keras.layers.Dense(8))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
            model.add(tf.keras.layers.Dense(6))

        else:
            model = keras.Sequential([
                tf.keras.layers.LSTM(unitsSize),
                tf.keras.layers.Dense(64, activation=activationFunction),
                tf.keras.layers.Dense(8, activation=activationFunction),
                tf.keras.layers.Dense(6)
            ])

        optimizer = tf.keras.optimizers.Adam()

        model.compile(loss=lossFunction,
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
        return model

    def create_gru_model(self, unitsSize, activationFunction, lossFunction):

        if activationFunction.casefold() == "LeakyRelu".casefold():
            model = keras.Sequential()
            model.add(tf.keras.layers.GRU(unitsSize))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
            model.add(tf.keras.layers.Dense(64))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
            model.add(tf.keras.layers.Dense(8))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
            model.add(tf.keras.layers.Dense(6))

        else:
            model = keras.Sequential([
                tf.keras.layers.GRU(unitsSize),
                tf.keras.layers.Dense(64, activation=activationFunction),
                tf.keras.layers.Dense(8, activation=activationFunction),
                tf.keras.layers.Dense(6)
            ])

        optimizer = tf.keras.optimizers.Adam()

        model.compile(loss=lossFunction,
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
        return model
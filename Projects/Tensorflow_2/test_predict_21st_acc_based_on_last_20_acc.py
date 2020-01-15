import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from tools.data_controller import *

df = get_data_from_my_dataset()
print(df.head())

TRAIN_SPLIT = 5000

tf.random.set_seed(13)

# original_time_step = 0
# print(np.asfarray(df['simu_time'].head().to_numpy(), np.float))
# time_step = original_time_step - original_time_step[0]

uni_data = df['a1_x']
uni_data.index = df['simu_time']
print(uni_data.head())

uni_data.plot(subplots=True)
plt.show()

# Converting pandas.core.series.Series to numpy.ndarray because we have to normalize the data now
uni_data = uni_data.values

# Normalizing Data
# Because It is important to scale features before training a neural network.
# Standardization is a common way of doing this scaling by subtracting the mean
# and dividing by the standard deviation of each feature.
# You could also use a tf.keras.utils.normalize method that rescales the values into a range of [0,1].

# Note: The mean and standard deviation should only be computed using the training data.
uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
uni_train_std = uni_data[:TRAIN_SPLIT].std()

uni_data = (uni_data-uni_train_mean)/uni_train_std

univariate_past_history = 20
univariate_future_target = 0

x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)
x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)

print('Single window of past history')
print("Type and shape of x_train_uni : ", type(x_train_uni), x_train_uni.shape)
print(x_train_uni[0])
print('\n Target temperature to predict')
print("Type and shape of y_train_uni : ", type(y_train_uni), y_train_uni.shape)
print(y_train_uni[0])

show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')

def baseline(history):
  return np.mean(history)

show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,
           'Baseline Prediction Example')

## %%
# Recurrent neural network

BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
#train_univariate = train_univariate.cache().batch(BATCH_SIZE).repeat()
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')

# Let's make a sample prediction, to check the output of the model.
for x, y in val_univariate.take(1):
    print(simple_lstm_model.predict(x).shape)

EVALUATION_INTERVAL = 200
EPOCHS = 10

simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50)

for x, y in val_univariate.take(3):
    show_plot([x[0].numpy(), y[0].numpy(), simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')


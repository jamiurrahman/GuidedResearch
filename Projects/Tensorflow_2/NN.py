import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow import keras

tf.keras.backend.clear_session()

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from tools.data_controller import *

df = get_data_from_my_dataset()
print(df.head())

original_acc1_x_df = df["a1_x"]
original_position_x_t_df = df["positon_x_t"]
original_time_step_df = df["simu_time"]

original_acc1_x = np.asfarray(original_acc1_x_df.to_numpy(), np.float)
original_position_x_t = np.asfarray(original_position_x_t_df.to_numpy(), np.float)
original_time_step = np.asfarray(original_time_step_df.to_numpy(), np.float)
time_step = original_time_step - original_time_step[0]
delta_t = np.diff(original_time_step, axis=0)
delta_t = delta_t / 1000 # delta_t is in milli sec. ## Converting ms to sec

delta_s = abs(np.diff(original_position_x_t, axis=0))

# print(original_acc1_x[0:5])
# print(original_position_x_t[0:5])
# print(original_time_step[0:5])
# print(delta_t[0:5])


def plot(X, Y, Z, xlabel, ylabel, zlabel, color="b", marker="o"):
    limit = 6000
    #plt.plot(time_step[0:limit], original_position_x_t[0:limit], "g.-")
    #plt.show()

    #plt.plot(delta_t[0:limit], delta_s[0:limit], "b.")
    #plt.show()

    # Visualizing 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #ax.scatter(delta_t[0:limit], delta_s[0:limit], original_acc1_x[0:limit], c=color, marker=marker)
    ax.scatter(X[0:limit], Y[0:limit], Z[0:limit], c=color, marker=marker)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    plt.show()





tf.random.set_seed(13)

# Normalizing Data
# Because It is important to scale features before training a neural network.
# Standardization is a common way of doing this scaling by subtracting the mean
# and dividing by the standard deviation of each feature.
# You could also use a tf.keras.utils.normalize method that rescales the values into a range of [0,1].

def normalize(data, TRAIN_SPLIT = 5000):

    # Note: The mean and standard deviation should only be computed using the training data.
    data_train_mean = data[:TRAIN_SPLIT].mean()
    data_train_std = data[:TRAIN_SPLIT].std()

    return ((data - data_train_mean) / data_train_std)

original_acc1_x = np.expand_dims(original_acc1_x, axis=1) # Expanding one axis to column
delta_t = np.expand_dims(delta_t, axis=1) # Expanding one axis to column
delta_s = np.expand_dims(delta_s, axis=1) # Expanding one axis to column


TRAIN_SPLIT = 5000
VALIDATION_SPLIT = 500
normalized_acc1_x = normalize(original_acc1_x, TRAIN_SPLIT=TRAIN_SPLIT)
normalized_delta_t = normalize(delta_t, TRAIN_SPLIT=TRAIN_SPLIT)

#plot(delta_t, original_acc1_x, delta_s, "delta_t", "original_acc1_x", "delta_s")
#plot(normalized_delta_t, normalized_acc1_x, delta_s, "normalized_delta_t", "normalized_acc1_x", "delta_s")

def split(data, TRAIN_SPLIT = 5000, VALIDATION_SPLIT = 500):
    data_train = data[:TRAIN_SPLIT]
    data_validation = data[TRAIN_SPLIT:(TRAIN_SPLIT+VALIDATION_SPLIT)]
    data_test = data[(TRAIN_SPLIT+VALIDATION_SPLIT):None]

    return data_train, data_validation, data_validation

train_acc1_x, val_acc1_x, test_acc1_x = split(normalized_acc1_x)
train_delta_t, val_delta_t, test_delta_t = split(normalized_delta_t)
train_delta_s, val_delta_s, test_delta_s = split(delta_s)


def build_model():
  model = keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(1, 2)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

train_features = np.concatenate((train_acc1_x, train_delta_t), axis=1)
val_features = np.concatenate((val_acc1_x, val_delta_t), axis=1)

train_features = np.expand_dims(train_features, axis=1) # Expanding one axis to column
val_features = np.expand_dims(val_features, axis=1) # Expanding one axis to column

print("train_features info : ", type(train_features), np.shape(train_features))
print("train_delta_s info : ", type(train_delta_s), np.shape(train_delta_s))
print("val_features info : ", type(val_features), np.shape(val_features))
print("val_delta_s info : ", type(val_delta_s), np.shape(val_delta_s))

train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_delta_s))
val_dataset = tf.data.Dataset.from_tensor_slices((val_features, val_delta_s))

# Showing train_dataset
# for elem in train_dataset:
#     print(elem[0].numpy()) # Features
#     print(elem[1].numpy()) # Corresponding Labels


BATCH_SIZE = 8
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)

model = build_model()

history = model.fit(train_dataset, epochs=5)

#plt.semilogx(history.history["epoch"], history.history["loss"])
#plt.axis([1e-7, 1e-4, 0, 30])
#plt.show()

model.evaluate(val_dataset)

#plt.figure(figsize=(10, 6))
#plot_series(time_valid, x_valid[:, 3])
#plot_series(time_valid,  rnn_forecast[0:1000, :])
#plt.show()

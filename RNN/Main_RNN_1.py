import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

keras = tf.keras

import pickle
import os
cwd = os.getcwd() # getting current working directory

def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)

def window_dataset(series, window_size, batch_size=32,
                   shuffle_buffer=1000):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    #dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

def sequential_window_dataset(series, window_size):
    print("s : ")
    for s in series:
        print(s)

    #series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    #ds = ds.window(window_size + 1, shift=window_size, drop_remainder=True)
    #ds = ds.flat_map(lambda window: window.batch(window_size))

    print("win.numpy() : ")
    for win in ds:
        print(win.numpy())

    ds = ds.map(lambda window: (window[:-1], window[-1:]))
    ds = ds.batch(1).prefetch(1)

    print("ds type : ", type(ds))
    print("x and y.numpy() : ")
    for x, y in ds:
        #print("x and y type : ", type(x))
        print(x.numpy(), y.numpy())
    return ds
    #return ds.batch(1)

def createDataset(examples, labels):
    dataset = tf.data.Dataset.from_tensor_slices((examples, labels))
    #print("Dataset Info : ", dataset.element_spec)

    #print("x and y.numpy() : ")
    #for x, y in dataset:
    #    print(x.numpy(), y.numpy())

    return dataset.batch(100).prefetch(1)

def model_forecast(model, examples, labels):
    ds = tf.data.Dataset.from_tensor_slices((examples, labels))
    forecast = model.predict(ds)
    return forecast

class ResetStatesCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()

with open(cwd + '/Dataset/path_5/train.pickle', 'rb') as f:
    time_loaded, acc_4_data_features_loaded, gyro_4_data_features_loaded, ground_truth_1_labels_loaded = pickle.load(f)

#print(time_loaded[0:5])
time_loaded = time_loaded - (time_loaded[0]) # Starting time from 0

print("After Loading From Pickle:")
#print(time_loaded[0:5])
print("time_loaded shape : ", np.shape(time_loaded))
print("acc_4_data_features_loaded shape : ", np.shape(acc_4_data_features_loaded))
print("gyro_4_data_features_loaded shape : ", np.shape(gyro_4_data_features_loaded))
print("ground_truth_1_labels_loaded shape : ", np.shape(ground_truth_1_labels_loaded))

print("acc_4_data_features_loaded shape : ", np.shape(acc_4_data_features_loaded[:, 0]))

acc_4_x = acc_4_data_features_loaded[:, 0]
acc_4_y = acc_4_data_features_loaded[:, 1]
acc_4_z = acc_4_data_features_loaded[:, 2]
print("acc_4_x shape : ", np.shape(acc_4_x))

gyro_4_x = gyro_4_data_features_loaded[:, 0]
gyro_4_y = gyro_4_data_features_loaded[:, 1]
gyro_4_z = gyro_4_data_features_loaded[:, 2]
print("gyro_4_x shape : ", np.shape(gyro_4_x))

gt_positon_x = ground_truth_1_labels_loaded[:, 0]
gt_positon_y = ground_truth_1_labels_loaded[:, 1]
gt_positon_z = ground_truth_1_labels_loaded[:, 2]
gt_orientation_x = ground_truth_1_labels_loaded[:, 3]
gt_orientation_y = ground_truth_1_labels_loaded[:, 4]
gt_orientation_z = ground_truth_1_labels_loaded[:, 5]
print("gt_positon_x shape : ", np.shape(gt_positon_x))


# start = 0
# end = 1000
#
# plt.figure(figsize=(10, 6))
# plot_series(time_loaded, acc_4_data_features, start=start, end=end)
# plt.show()
#
# plt.figure(figsize=(10, 6))
# plot_series(time_loaded, acc_4_data_features_loaded[:, 1], end=end)
# plt.show()
#
# plt.figure(figsize=(10, 6))
# plot_series(time_loaded, acc_4_data_features_loaded[:, 2], end=end)
# plt.show()

split_time = 10000
split_time_end = int(split_time * 0.1) # For validation

time = time_loaded[0 : (split_time + split_time_end + split_time_end)]

print("time shape : ", np.shape(time))

series_acc_4_x = acc_4_x[0 : (split_time + split_time_end + split_time_end)]
series_acc_4_y = acc_4_y[0 : (split_time + split_time_end + split_time_end)]
series_acc_4_z = acc_4_z[0 : (split_time + split_time_end + split_time_end)]

series_gyro_4_x = gyro_4_x[0 : (split_time + split_time_end + split_time_end)]
series_gyro_4_y = gyro_4_y[0 : (split_time + split_time_end + split_time_end)]
series_gyro_4_z = gyro_4_z[0 : (split_time + split_time_end + split_time_end)]

series_gt_positon_x = gt_positon_x[0 : (split_time + split_time_end + split_time_end)]
series_gt_positon_y = gt_positon_y[0 : (split_time + split_time_end + split_time_end)]
series_gt_positon_z = gt_positon_z[0 : (split_time + split_time_end + split_time_end)]
series_gt_orientation_x = gt_orientation_x[0 : (split_time + split_time_end + split_time_end)]
series_gt_orientation_y = gt_orientation_y[0 : (split_time + split_time_end + split_time_end)]
series_gt_orientation_z = gt_orientation_z[0 : (split_time + split_time_end + split_time_end)]

print("series_acc_4_x shape : ", np.shape(series_acc_4_x))

# Expanding one dimention to concatenate later
# It also works
#series_acc_4_x = series_acc_4_x.reshape(-1, 1)
#series_gt_positon_x = series_gt_positon_x.reshape(-1, 1)

series_acc_4_x = series_acc_4_x[:, np.newaxis]
series_acc_4_y = series_acc_4_y[:, np.newaxis]
series_acc_4_z = series_acc_4_z[:, np.newaxis]
series_gt_positon_x = series_gt_positon_x[:, np.newaxis]

# Optional, doing only to align with series
time = time[:, np.newaxis]
print("time shape : ", np.shape(time))

print("series_acc_4_x shape : ", np.shape(series_acc_4_x))
print("series_gt_positon_x shape : ", np.shape(series_gt_positon_x))

series = np.concatenate([series_acc_4_x, series_acc_4_y, series_acc_4_z, series_gt_positon_x], axis=1) # column wise concatenation

print("series shape : ", np.shape(series))

# Need to concatenate all the features and labels
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:(split_time + split_time_end)]
x_valid = series[split_time:(split_time + split_time_end)]
time_test = time[(split_time + split_time_end):]
x_test = series[(split_time + split_time_end):]

#time_train = time_train[:, 0]
#x_train = x_train[:, 0]
#time_valid = time_valid[:, 0]
#x_valid = x_valid[:, 0]

print("time_train info:", type(time_train), time_train.shape)
print("x_train info:", type(x_train), x_train.shape)
print("time_valid info:", type(time_valid), time_valid.shape)
print("x_valid info:", type(x_valid), x_valid.shape)

x_train_dataset = createDataset(x_train[:, 0:3], x_train[:, 3])
x_valid_dataset = createDataset(x_valid[:, 0:3], x_valid[:, 3])
x_test_dataset = createDataset(x_test[:, 0:3], x_test[:, 3])

model = tf.keras.Sequential([
    keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
    tf.keras.layers.SimpleRNN(100, return_sequences=True),
    tf.keras.layers.SimpleRNN(100),
    tf.keras.layers.Dense(1, activation='relu'),
    keras.layers.Lambda(lambda x: x * 200.0)
])

# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

#lr_schedule = keras.callbacks.LearningRateScheduler(
#    lambda epoch: 1e-5 * 10**(epoch / 20))
optimizer = keras.optimizers.SGD(lr=1e-5, momentum=0.9)
model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

early_stopping = keras.callbacks.EarlyStopping(patience=50)
model_checkpoint = keras.callbacks.ModelCheckpoint(
    "my_checkpoint.h5", save_best_only=True)

history = model.fit(x_train_dataset, epochs=50, callbacks=[early_stopping, model_checkpoint])

#plt.semilogx(history.history["lr"], history.history["loss"])
#plt.axis([1e-7, 1e-4, 0, 30])
#plt.show()

# = keras.models.load_model("my_checkpoint.h5")

# For validation dataset
rnn_forecast = model.predict(tf.data.Dataset.from_tensor_slices((x_valid[:, 0:3])))

print("rnn_forecast.shape : ", rnn_forecast.shape)
print("rnn_forecast.type : ", type(rnn_forecast))

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid[:, 3])
plot_series(time_valid,  rnn_forecast[0:1000, :])
plt.show()

# For testing dataset
rnn_forecast = model.predict(tf.data.Dataset.from_tensor_slices((x_test[:, 0:3])))

plt.figure(figsize=(10, 6))
plot_series(time_test, x_test[:, 3])
plot_series(time_test,  rnn_forecast[0:1000, :])
plt.show()


# keras.backend.clear_session()
# tf.random.set_seed(42)
# np.random.seed(42)
#
# #tf.enable_eager_execution()
# print("tf.executing_eagerly() : ", tf.executing_eagerly())
#
# window_size = 10
# #train_set = window_dataset(x_train, window_size, batch_size=2)
# #valid_set = window_dataset(x_valid, window_size, batch_size=2)
#
# train_set = sequential_window_dataset(x_train, window_size)
# valid_set = sequential_window_dataset(x_valid, window_size)
#
# #print("train_set info:", type(train_set), train_set)
# print("train_set info:", train_set.element_spec)
#
# model = keras.models.Sequential([
#   keras.layers.Dense(1, input_shape=[1])
# ])
# optimizer = keras.optimizers.SGD(lr=1e-5, momentum=0.9)
# model.compile(loss=keras.losses.Huber(),
#               optimizer=optimizer,
#               metrics=["mae"])
# model.fit(train_set, epochs=10, validation_data=valid_set)












#plt.semilogx(history.history["lr"], history.history["loss"])
#plt.axis([1e-6, 1e-3, 0, 20])


# optimizer = keras.optimizers.Adam()
# model.compile(loss=keras.losses.Huber(),
#               optimizer=optimizer,
#               metrics=["mae"])
#
# history = model.fit(train_set, epochs=1)
#
# rnn_forecast = model.predict(series[np.newaxis, :, np.newaxis])
# rnn_forecast = rnn_forecast[0, split_time - 1:-1, 0]
#
#
#
# time_valid_reshaped = np.reshape(time_valid, (split_time_end,))
# x_valid_reshaped = np.reshape(x_valid, (split_time_end,))
# #rnn_forecast_reshaped = np.reshape(rnn_forecast, (split_time_end, 0))
#
# print("rnn_forecast.shape : ", rnn_forecast.shape)
# print("time_valid_reshaped.shape : ", time_valid_reshaped.shape)
# print("x_valid_reshaped.shape : ", x_valid_reshaped.shape)
#
# plt.figure(figsize=(10, 6))
# plot_series(time_valid_reshaped, x_valid_reshaped)
# plot_series(time_valid_reshaped, rnn_forecast)
#
# print("mean_absolute_error : ", keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy())


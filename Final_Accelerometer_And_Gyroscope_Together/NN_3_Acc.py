import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow import keras

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from  IPython import display
import pathlib
import shutil
import tempfile

tf.keras.backend.clear_session()

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from tools.data_controller import *

logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

df = get_data_from_my_dataset()
print(df.head())

original_acc_1_x_df = df["a_1_x"]
original_acc_2_x_df = df["a_2_x"]
original_acc_3_x_df = df["a_3_x"]
original_acc_4_x_df = df["a_4_x"]
original_acc_5_x_df = df["a_5_x"]
original_acc_6_x_df = df["a_6_x"]
original_acc_7_x_df = df["a_7_x"]
original_acc_8_x_df = df["a_8_x"]
original_position_x_t_df = df["gt_positon_x_t"]

original_acc_1_y_df = df["a_1_y"]
original_acc_2_y_df = df["a_2_y"]
original_acc_3_y_df = df["a_3_y"]
original_acc_4_y_df = df["a_4_y"]
original_acc_5_y_df = df["a_5_y"]
original_acc_6_y_df = df["a_6_y"]
original_acc_7_y_df = df["a_7_y"]
original_acc_8_y_df = df["a_8_y"]
original_position_y_t_df = df["gt_positon_y_t"]

original_acc_1_z_df = df["a_1_z"]
original_acc_2_z_df = df["a_2_z"]
original_acc_3_z_df = df["a_3_z"]
original_acc_4_z_df = df["a_4_z"]
original_acc_5_z_df = df["a_5_z"]
original_acc_6_z_df = df["a_6_z"]
original_acc_7_z_df = df["a_7_z"]
original_acc_8_z_df = df["a_8_z"]
original_position_z_t_df = df["gt_positon_z_t"]



original_time_step_df = df["simu_time"]



original_acc_1_x = np.asfarray(original_acc_1_x_df.to_numpy(), np.float)
original_acc_2_x = np.asfarray(original_acc_2_x_df.to_numpy(), np.float)
original_acc_3_x = np.asfarray(original_acc_3_x_df.to_numpy(), np.float)
original_acc_4_x = np.asfarray(original_acc_4_x_df.to_numpy(), np.float)
original_acc_5_x = np.asfarray(original_acc_5_x_df.to_numpy(), np.float)
original_acc_6_x = np.asfarray(original_acc_6_x_df.to_numpy(), np.float)
original_acc_7_x = np.asfarray(original_acc_7_x_df.to_numpy(), np.float)
original_acc_8_x = np.asfarray(original_acc_8_x_df.to_numpy(), np.float)
original_position_x_t = np.asfarray(original_position_x_t_df.to_numpy(), np.float)

original_acc_1_y = np.asfarray(original_acc_1_y_df.to_numpy(), np.float)
original_acc_2_y = np.asfarray(original_acc_2_y_df.to_numpy(), np.float)
original_acc_3_y = np.asfarray(original_acc_3_y_df.to_numpy(), np.float)
original_acc_4_y = np.asfarray(original_acc_4_y_df.to_numpy(), np.float)
original_acc_5_y = np.asfarray(original_acc_5_y_df.to_numpy(), np.float)
original_acc_6_y = np.asfarray(original_acc_6_y_df.to_numpy(), np.float)
original_acc_7_y = np.asfarray(original_acc_7_y_df.to_numpy(), np.float)
original_acc_8_y = np.asfarray(original_acc_8_y_df.to_numpy(), np.float)
original_position_y_t = np.asfarray(original_position_y_t_df.to_numpy(), np.float)

original_acc_1_z = np.asfarray(original_acc_1_z_df.to_numpy(), np.float)
original_acc_2_z = np.asfarray(original_acc_2_z_df.to_numpy(), np.float)
original_acc_3_z = np.asfarray(original_acc_3_z_df.to_numpy(), np.float)
original_acc_4_z = np.asfarray(original_acc_4_z_df.to_numpy(), np.float)
original_acc_5_z = np.asfarray(original_acc_5_z_df.to_numpy(), np.float)
original_acc_6_z = np.asfarray(original_acc_6_z_df.to_numpy(), np.float)
original_acc_7_z = np.asfarray(original_acc_7_z_df.to_numpy(), np.float)
original_acc_8_z = np.asfarray(original_acc_8_z_df.to_numpy(), np.float)
original_position_z_t = np.asfarray(original_position_z_t_df.to_numpy(), np.float)

original_time_step = np.asfarray(original_time_step_df.to_numpy(), np.float)
time_step = original_time_step - original_time_step[0]
delta_t = np.abs(np.diff(original_time_step, axis=0)) # delta_t should not be negative
delta_t = delta_t / 1000 # delta_t is in milli sec. ## Converting ms to sec

delta_s_x = np.abs(np.diff(original_position_x_t, axis=0)) # delta_s_x should not be negative
delta_s_y = np.abs(np.diff(original_position_y_t, axis=0)) # delta_s_y should not be negative
delta_s_z = np.abs(np.diff(original_position_z_t, axis=0)) # delta_s_z should not be negative

delta_s = ((delta_s_x**2) + (delta_s_y**2) + (delta_s_z**2))**0.5 # calculating total displacement

print(delta_s_x[0:5])
print(delta_s_y[0:5])
print(delta_s_z[0:5])
print(delta_s[0:5])

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
# We could also use a tf.keras.utils.normalize method that rescales the values into a range of [0,1].

def normalize(data, TRAIN_SIZE = 5000):

    # Note: The mean and standard deviation should only be computed using the training data.
    data_train_mean = data[:TRAIN_SIZE].mean()
    data_train_std = data[:TRAIN_SIZE].std()

    return ((data - data_train_mean) / data_train_std)

def denormalize(data, TRAIN_SPLIT = 5000):

    # Note: The mean and standard deviation should only be computed using the training data.
    data_train_mean = data[:TRAIN_SPLIT].mean()
    data_train_std = data[:TRAIN_SPLIT].std()

    return ((data * data_train_std) + data_train_mean)

original_acc_1_x = np.expand_dims(original_acc_1_x, axis=1) # Expanding one axis to column
original_acc_2_x = np.expand_dims(original_acc_2_x, axis=1) # Expanding one axis to column
original_acc_3_x = np.expand_dims(original_acc_3_x, axis=1) # Expanding one axis to column
original_acc_4_x = np.expand_dims(original_acc_4_x, axis=1) # Expanding one axis to column
original_acc_5_x = np.expand_dims(original_acc_5_x, axis=1) # Expanding one axis to column
original_acc_6_x = np.expand_dims(original_acc_6_x, axis=1) # Expanding one axis to column
original_acc_7_x = np.expand_dims(original_acc_7_x, axis=1) # Expanding one axis to column
original_acc_8_x = np.expand_dims(original_acc_8_x, axis=1) # Expanding one axis to column

original_acc_1_y = np.expand_dims(original_acc_1_y, axis=1) # Expanding one axis to column
original_acc_2_y = np.expand_dims(original_acc_2_y, axis=1) # Expanding one axis to column
original_acc_3_y = np.expand_dims(original_acc_3_y, axis=1) # Expanding one axis to column
original_acc_4_y = np.expand_dims(original_acc_4_y, axis=1) # Expanding one axis to column
original_acc_5_y = np.expand_dims(original_acc_5_y, axis=1) # Expanding one axis to column
original_acc_6_y = np.expand_dims(original_acc_6_y, axis=1) # Expanding one axis to column
original_acc_7_y = np.expand_dims(original_acc_7_y, axis=1) # Expanding one axis to column
original_acc_8_y = np.expand_dims(original_acc_8_y, axis=1) # Expanding one axis to column

original_acc_1_z = np.expand_dims(original_acc_1_z, axis=1) # Expanding one axis to column
original_acc_2_z = np.expand_dims(original_acc_2_z, axis=1) # Expanding one axis to column
original_acc_3_z = np.expand_dims(original_acc_3_z, axis=1) # Expanding one axis to column
original_acc_4_z = np.expand_dims(original_acc_4_z, axis=1) # Expanding one axis to column
original_acc_5_z = np.expand_dims(original_acc_5_z, axis=1) # Expanding one axis to column
original_acc_6_z = np.expand_dims(original_acc_6_z, axis=1) # Expanding one axis to column
original_acc_7_z = np.expand_dims(original_acc_7_z, axis=1) # Expanding one axis to column
original_acc_8_z = np.expand_dims(original_acc_8_z, axis=1) # Expanding one axis to column

delta_t = np.expand_dims(delta_t, axis=1) # Expanding one axis to column
delta_s = np.expand_dims(delta_s, axis=1) # Expanding one axis to column


TRAIN_SIZE = int(len(delta_s) * 0.8)#5000
VALIDATION_SIZE = int(len(delta_s) * 0.10)#500
TEST_SIZE = int(len(delta_s) - (TRAIN_SIZE + VALIDATION_SIZE))

normalized_acc_1_x = normalize(original_acc_1_x, TRAIN_SIZE=TRAIN_SIZE)
normalized_acc_2_x = normalize(original_acc_2_x, TRAIN_SIZE=TRAIN_SIZE)
normalized_acc_3_x = normalize(original_acc_3_x, TRAIN_SIZE=TRAIN_SIZE)
normalized_acc_4_x = normalize(original_acc_4_x, TRAIN_SIZE=TRAIN_SIZE)
normalized_acc_5_x = normalize(original_acc_5_x, TRAIN_SIZE=TRAIN_SIZE)
normalized_acc_6_x = normalize(original_acc_6_x, TRAIN_SIZE=TRAIN_SIZE)
normalized_acc_7_x = normalize(original_acc_7_x, TRAIN_SIZE=TRAIN_SIZE)
normalized_acc_8_x = normalize(original_acc_8_x, TRAIN_SIZE=TRAIN_SIZE)

normalized_acc_1_y = normalize(original_acc_1_y, TRAIN_SIZE=TRAIN_SIZE)
normalized_acc_2_y = normalize(original_acc_2_y, TRAIN_SIZE=TRAIN_SIZE)
normalized_acc_3_y = normalize(original_acc_3_y, TRAIN_SIZE=TRAIN_SIZE)
normalized_acc_4_y = normalize(original_acc_4_y, TRAIN_SIZE=TRAIN_SIZE)
normalized_acc_5_y = normalize(original_acc_5_y, TRAIN_SIZE=TRAIN_SIZE)
normalized_acc_6_y = normalize(original_acc_6_y, TRAIN_SIZE=TRAIN_SIZE)
normalized_acc_7_y = normalize(original_acc_7_y, TRAIN_SIZE=TRAIN_SIZE)
normalized_acc_8_y = normalize(original_acc_8_y, TRAIN_SIZE=TRAIN_SIZE)

normalized_acc_1_z = normalize(original_acc_1_z, TRAIN_SIZE=TRAIN_SIZE)
normalized_acc_2_z = normalize(original_acc_2_z, TRAIN_SIZE=TRAIN_SIZE)
normalized_acc_3_z = normalize(original_acc_3_z, TRAIN_SIZE=TRAIN_SIZE)
normalized_acc_4_z = normalize(original_acc_4_z, TRAIN_SIZE=TRAIN_SIZE)
normalized_acc_5_z = normalize(original_acc_5_z, TRAIN_SIZE=TRAIN_SIZE)
normalized_acc_6_z = normalize(original_acc_6_z, TRAIN_SIZE=TRAIN_SIZE)
normalized_acc_7_z = normalize(original_acc_7_z, TRAIN_SIZE=TRAIN_SIZE)
normalized_acc_8_z = normalize(original_acc_8_z, TRAIN_SIZE=TRAIN_SIZE)

normalized_delta_t = normalize(delta_t, TRAIN_SIZE=TRAIN_SIZE)

#plot(delta_t, original_acc1_x, delta_s, "delta_t", "original_acc1_x", "delta_s")
#plot(normalized_delta_t, normalized_acc1_x, delta_s, "normalized_delta_t", "normalized_acc1_x", "delta_s")

def split(data, TRAIN_SIZE = 5000, VALIDATION_SIZE = 500, TEST_SIZE = None):
    data_train = data[:TRAIN_SIZE]
    data_validation = data[TRAIN_SIZE:(TRAIN_SIZE + VALIDATION_SIZE)]
    data_test = data[(TRAIN_SIZE + VALIDATION_SIZE):(TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE)]

    return data_train, data_validation, data_test

train_acc_1_x, val_acc_1_x, test_acc_1_x = split(normalized_acc_1_x, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)
train_acc_2_x, val_acc_2_x, test_acc_2_x = split(normalized_acc_2_x, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)
train_acc_3_x, val_acc_3_x, test_acc_3_x = split(normalized_acc_3_x, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)
train_acc_4_x, val_acc_4_x, test_acc_4_x = split(normalized_acc_4_x, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)
train_acc_5_x, val_acc_5_x, test_acc_5_x = split(normalized_acc_5_x, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)
train_acc_6_x, val_acc_6_x, test_acc_6_x = split(normalized_acc_6_x, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)
train_acc_7_x, val_acc_7_x, test_acc_7_x = split(normalized_acc_7_x, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)
train_acc_8_x, val_acc_8_x, test_acc_8_x = split(normalized_acc_8_x, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)
train_delta_t, val_delta_t, test_delta_t = split(normalized_delta_t, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)

train_acc_1_y, val_acc_1_y, test_acc_1_y = split(normalized_acc_1_y, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)
train_acc_2_y, val_acc_2_y, test_acc_2_y = split(normalized_acc_2_y, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)
train_acc_3_y, val_acc_3_y, test_acc_3_y = split(normalized_acc_3_y, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)
train_acc_4_y, val_acc_4_y, test_acc_4_y = split(normalized_acc_4_y, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)
train_acc_5_y, val_acc_5_y, test_acc_5_y = split(normalized_acc_5_y, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)
train_acc_6_y, val_acc_6_y, test_acc_6_y = split(normalized_acc_6_y, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)
train_acc_7_y, val_acc_7_y, test_acc_7_y = split(normalized_acc_7_y, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)
train_acc_8_y, val_acc_8_y, test_acc_8_y = split(normalized_acc_8_y, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)
train_delta_t, val_delta_t, test_delta_t = split(normalized_delta_t, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)

train_acc_1_z, val_acc_1_z, test_acc_1_z = split(normalized_acc_1_z, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)
train_acc_2_z, val_acc_2_z, test_acc_2_z = split(normalized_acc_2_z, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)
train_acc_3_z, val_acc_3_z, test_acc_3_z = split(normalized_acc_3_z, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)
train_acc_4_z, val_acc_4_z, test_acc_4_z = split(normalized_acc_4_z, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)
train_acc_5_z, val_acc_5_z, test_acc_5_z = split(normalized_acc_5_z, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)
train_acc_6_z, val_acc_6_z, test_acc_6_z = split(normalized_acc_6_z, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)
train_acc_7_z, val_acc_7_z, test_acc_7_z = split(normalized_acc_7_z, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)
train_acc_8_z, val_acc_8_z, test_acc_8_z = split(normalized_acc_8_z, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)
train_delta_t, val_delta_t, test_delta_t = split(normalized_delta_t, TRAIN_SIZE = TRAIN_SIZE, VALIDATION_SIZE = VALIDATION_SIZE, TEST_SIZE = TEST_SIZE)

train_delta_s, val_delta_s, test_delta_s = split(delta_s, TRAIN_SIZE=TRAIN_SIZE, VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)


def build_model():
  model = keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(1, 25)),
    #tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(8, activation='tanh'),
    tf.keras.layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.Adam()

  model.compile(loss='mae',
                optimizer=optimizer,
                metrics=['mae', 'mse', 'accuracy'])
  return model

train_features = np.concatenate((train_acc_1_x, train_acc_2_x, train_acc_3_x, train_acc_4_x, train_acc_5_x, train_acc_6_x, train_acc_7_x, train_acc_8_x,
                                 train_acc_1_y, train_acc_2_y, train_acc_3_y, train_acc_4_y, train_acc_5_y, train_acc_6_y, train_acc_7_y, train_acc_8_y,
                                 train_acc_1_z, train_acc_2_z, train_acc_3_z, train_acc_4_z, train_acc_5_z, train_acc_6_z, train_acc_7_z, train_acc_8_z,
                                 train_delta_t), axis=1)
val_features = np.concatenate((val_acc_1_x, val_acc_2_x, val_acc_3_x, val_acc_4_x, val_acc_5_x, val_acc_6_x, val_acc_7_x, val_acc_8_x,
                               val_acc_1_y, val_acc_2_y, val_acc_3_y, val_acc_4_y, val_acc_5_y, val_acc_6_y, val_acc_7_y, val_acc_8_y,
                               val_acc_1_z, val_acc_2_z, val_acc_3_z, val_acc_4_z, val_acc_5_z, val_acc_6_z, val_acc_7_z, val_acc_8_z,
                               val_delta_t), axis=1)
test_features = np.concatenate((test_acc_1_x, test_acc_2_x, test_acc_3_x, test_acc_4_x, test_acc_5_x, test_acc_6_x, test_acc_7_x, test_acc_8_x,
                                test_acc_1_y, test_acc_2_y, test_acc_3_y, test_acc_4_y, test_acc_5_y, test_acc_6_y, test_acc_7_y, test_acc_8_y,
                                test_acc_1_z, test_acc_2_z, test_acc_3_z, test_acc_4_z, test_acc_5_z, test_acc_6_z, test_acc_7_z, test_acc_8_z,
                                test_delta_t), axis=1)

train_features = np.expand_dims(train_features, axis=1) # Expanding one axis to column
val_features = np.expand_dims(val_features, axis=1) # Expanding one axis to column
test_features = np.expand_dims(test_features, axis=1) # Expanding one axis to column

print("train_features info : ", type(train_features), np.shape(train_features))
print("train_delta_s info : ", type(train_delta_s), np.shape(train_delta_s))
print("val_features info : ", type(val_features), np.shape(val_features))
print("val_delta_s info : ", type(val_delta_s), np.shape(val_delta_s))

train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_delta_s))
val_dataset = tf.data.Dataset.from_tensor_slices((val_features, val_delta_s))
test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_delta_s))

# Showing train_dataset
# for elem in train_dataset:
#     print(elem[0].numpy()) # Features
#     print(elem[1].numpy()) # Corresponding Labels


BATCH_SIZE = 8
SHUFFLE_BUFFER_SIZE = 100
EPOCHS = 50

#train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model = build_model()

history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset)

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(EPOCHS)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
#plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Working with Validation dataset
val_predict_delta_s = model.predict(val_dataset)

plt.plot(val_delta_t, val_delta_s.flatten(), "b.", linestyle="None", label="true")
plt.plot(val_delta_t, val_predict_delta_s.flatten(), "r.", linestyle="None", label="prediction")
plt.ylabel('Delta S')
plt.xlabel('Delta Time')
plt.legend()
plt.title("Validation Dataset Result")
plt.show();

plt.plot(val_acc_1_x, val_delta_s.flatten(), "b.", linestyle="None", label="true")
plt.plot(val_acc_1_x, val_predict_delta_s.flatten(), "r.", linestyle="None", label="prediction")
plt.ylabel('Delta S')
plt.xlabel('Acceleration')
plt.legend()
plt.title("Validation Dataset Result")
plt.show();


# Working with test dataset
test_predict_delta_s = model.predict(test_dataset)

plt.plot(test_delta_t, test_delta_s.flatten(), "b.", linestyle="None", label="true")
plt.plot(test_delta_t, test_predict_delta_s.flatten(), "r.", linestyle="None", label="prediction")
plt.ylabel('Delta S')
plt.xlabel('Delta Time')
plt.legend()
plt.title("Test Dataset Result")
plt.show();

plt.plot(test_acc_1_x, test_delta_s.flatten(), "b.", linestyle="None", label="true")
plt.plot(test_acc_1_x, test_predict_delta_s.flatten(), "r.", linestyle="None", label="prediction")
plt.ylabel('Delta S')
plt.xlabel('Acceleration')
plt.legend()
plt.title("Test Dataset Result")
plt.show();

# Plotting test dataset in 3d
test_predict_delta_s = model.predict(test_dataset)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(test_delta_t, test_acc_1_x, test_delta_s.flatten(), color='b', cmap='Greens');
ax.scatter3D(test_delta_t, test_acc_1_x, test_predict_delta_s.flatten(), color='red', cmap='Greens');
ax.set_xlabel('Delta Time')
ax.set_ylabel('Acc')
ax.set_zlabel('Delta S')
plt.title("In X Axis")
plt.show();

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(test_delta_t, test_acc_1_y, test_delta_s.flatten(), color='b', cmap='Greens');
ax.scatter3D(test_delta_t, test_acc_1_y, test_predict_delta_s.flatten(), color='red', cmap='Greens');
ax.set_xlabel('Delta Time')
ax.set_ylabel('Acc')
ax.set_zlabel('Delta S')
plt.title("In Y Axis")
plt.show();

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(test_delta_t, test_acc_1_z, test_delta_s.flatten(), color='b', cmap='Greens');
ax.scatter3D(test_delta_t, test_acc_1_z, test_predict_delta_s.flatten(), color='red', cmap='Greens');
ax.set_xlabel('Delta Time')
ax.set_ylabel('Acc')
ax.set_zlabel('Delta S')
plt.title("In Z Axis")
plt.show();


#plt.figure(figsize=(8, 8))
#plt.subplot(1, 2, 1)
#plt.plot(epochs_range, acc, label='Training Accuracy')
#plt.plot(epochs_range, val_acc, label='Validation Accuracy')
#plt.legend(loc='lower right')
#plt.title('Training and Validation Accuracy')




#plotter = tfdocs.plots.HistoryPlotter(metric = 'accuracy')
#plotter.plot(history)

#plt.semilogx(history.history["epoch"], history.history["loss"])
#plt.axis([1e-7, 1e-4, 0, 30])
#plt.show()

#model.evaluate(val_dataset)

#plt.figure(figsize=(10, 6))
#plot_series(time_valid, x_valid[:, 3])
#plot_series(time_valid,  rnn_forecast[0:1000, :])
#plt.show()

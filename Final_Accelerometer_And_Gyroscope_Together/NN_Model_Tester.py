import pickle

from Data_PreProcessor import *
from Dataset_Creator import *
#from NN_Model_Creator import *
from Plotter import *
from Configuration import *
#from NN_Model_Saver import *

import tensorflow as tf
import tensorboard
import datetime
#import pandas as pd

import timeit

import logging

import glob

import os


# Configuring current configuration model and so on
# Configuration.configure()

# Working with Raw Data
data_preprocessor = Data_PreProcessor()
raw_data = data_preprocessor.get_data_for_all_sensors()
print(raw_data.head())

# Working with Dataset
train_dataset = tf.data.Dataset.from_generator
val_dataset = tf.data.Dataset.from_generator
test_dataset = tf.data.Dataset.from_generator

dataset_creator = Dataset_Creator()
dataset_creator.create_dataset_for_all_sensors(raw_data)

# Logging
logPath = "./logs_testing_result/" + Configuration.Current_Time + "/"
if not os.path.exists(logPath):
    os.makedirs(logPath)

# create logger with 'spam_application'
logger = logging.getLogger('my_application')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(os.path.join(logPath, "my_testing_result_log.log"))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)




# logger.info("\n***************************************************************************************\n")
# logger.info("Current Model: \n" + Configuration.Current_File_Name)
# logger.info("Current Model Info: \n" + Configuration.getInfo())

# start = timeit.default_timer()

train_dataset = dataset_creator.train_dataset
val_dataset = dataset_creator.val_dataset
test_dataset = dataset_creator.test_dataset

# train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
# train_dataset = train_dataset.batch(Configuration.Current_Batch_Size)
# val_dataset = val_dataset.batch(Configuration.Current_Batch_Size)
# test_dataset = test_dataset.batch(Configuration.Current_Batch_Size)
test_dataset = test_dataset.batch(512)



txtfiles = []
list_mae_x_position = []
list_mae_y_position = []
list_mae_z_position = []
list_mse_x_position = []
list_mse_y_position = []
list_mse_z_position = []
list_mae_x_orientation = []
list_mae_y_orientation = []
list_mae_z_orientation = []
list_mse_x_orientation = []
list_mse_y_orientation = []
list_mse_z_orientation = []
for file in glob.glob(".\\SavedModel\\20200413-190421\\*.h5"):
    # print(file)
#     txtfiles.append(file)
#
#
#
# for file in txtfiles:
#     print(file)
#     a, b, c, name = file.split("\\")
#     print(name)
#     current_file_serial = name.split("_")[0]
#     print(current_file_serial)
    current_loaded_model = tf.keras.models.load_model(file)

    # Working with test dataset
    test_predict_delta_s_and_alpha_in_x_y_and_z = current_loaded_model.predict(test_dataset)

    dataset_creator.test_delta_t = dataset_creator.denormalize(dataset_creator.test_delta_t,
                                                                         TRAIN_SPLIT=dataset_creator.TRAIN_SIZE)
    dataset_creator.test_delta_s_and_alpha_in_x_y_and_z = dataset_creator.denormalize(
        dataset_creator.test_delta_s_and_alpha_in_x_y_and_z, TRAIN_SPLIT=dataset_creator.TRAIN_SIZE)
    test_predict_delta_s_and_alpha_in_x_y_and_z = dataset_creator.denormalize(test_predict_delta_s_and_alpha_in_x_y_and_z,
                                                                      TRAIN_SPLIT=dataset_creator.TRAIN_SIZE)

    # print(test_predict_delta_s_in_x_y_and_z.shape)
    # print(len(test_predict_delta_s_in_x_y_and_z))
    # print(len(dataset_creator.test_delta_s_x_y_and_z))
    test_length = len(test_predict_delta_s_and_alpha_in_x_y_and_z)
    # print("test_length: ", test_length)

    # print(test_predict_delta_s_and_alpha_in_x_y_and_z.shape)
    # print(dataset_creator.test_delta_s_and_alpha_in_x_y_and_z.shape)

    a = dataset_creator.test_delta_s_and_alpha_in_x_y_and_z
    b = test_predict_delta_s_and_alpha_in_x_y_and_z
    a = np.reshape(a, (test_length, 6))
    b = np.reshape(b, (test_length, 6))

    subtract_result = np.absolute(a - b)

    mae_x_position = np.sum(subtract_result[:, 0]) / test_length
    mae_y_position = np.sum(subtract_result[:, 1]) / test_length
    mae_z_position = np.sum(subtract_result[:, 2]) / test_length

    mse_x_position = np.sum(np.square(subtract_result[:, 0])) / test_length
    mse_y_position = np.sum(np.square(subtract_result[:, 1])) / test_length
    mse_z_position = np.sum(np.square(subtract_result[:, 2])) / test_length

    mae_x_orientation = np.sum(subtract_result[:, 3]) / test_length
    mae_y_orientation = np.sum(subtract_result[:, 4]) / test_length
    mae_z_orientation = np.sum(subtract_result[:, 5]) / test_length

    mse_x_orientation = np.sum(np.square(subtract_result[:, 3])) / test_length
    mse_y_orientation = np.sum(np.square(subtract_result[:, 4])) / test_length
    mse_z_orientation = np.sum(np.square(subtract_result[:, 5])) / test_length

    # print("{},{},{},{},{},{},{},{},{},{},{},{},{}".format(file, mae_x, mae_y, mae_z, mse_x, mse_y, mse_z))
    logger.info("{},{},{},{},{},{},{},{},{},{},{},{},{}".format(file, mae_x_position, mae_y_position, mae_z_position, mse_x_position, mse_y_position, mse_z_position, mae_x_orientation, mae_y_orientation, mae_z_orientation, mse_x_orientation, mse_y_orientation, mse_z_orientation))

    list_mae_x_position.append(mae_x_position)
    list_mae_y_position.append(mae_y_position)
    list_mae_z_position.append(mae_z_position)
    list_mse_x_position.append(mse_x_position)
    list_mse_y_position.append(mse_y_position)
    list_mse_z_position.append(mse_z_position)

    list_mae_x_orientation.append(mae_x_orientation)
    list_mae_y_orientation.append(mae_y_orientation)
    list_mae_z_orientation.append(mae_z_orientation)
    list_mse_x_orientation.append(mse_x_orientation)
    list_mse_y_orientation.append(mse_y_orientation)
    list_mse_z_orientation.append(mse_z_orientation)


plt.clf()

# MAE
#epochs_range = range(EPOCHS)
#plt.plot(epochs_range, history.history[string])
#plt.plot(epochs_range, history.history['val_' + string])
plt.plot(list_mae_x_position, marker='o', color='cyan', linestyle='None')
plt.plot(list_mae_y_position, marker='^', color='magenta', linestyle='None')
plt.plot(list_mae_z_position, marker='*', color='blue', linestyle='None')
plt.plot(list_mae_x_orientation, marker='o', color='red', linestyle='None')
plt.plot(list_mae_y_orientation, marker='^', color='green', linestyle='None')
plt.plot(list_mae_z_orientation, marker='*', color='black', linestyle='None')
plt.xlabel("Model Version")
plt.ylabel("Mean Absolute Error (MAE)")
plt.legend(["MAE of Position in X axis", 'MAE of Position in Y axis', "MAE of Position in Z axis", "MAE of Orientation in X axis", 'MAE of Orientation in Y axis', "MAE of Orientation in Z axis"])
plt.savefig(".\\MAE_of_Position_and_Orientation_in_X_Y_Z.png")
plt.show()


# MSE
plt.plot(list_mse_x_position, marker='o', color='cyan', linestyle='None')
plt.plot(list_mse_y_position, marker='^', color='magenta', linestyle='None')
plt.plot(list_mse_z_position, marker='*', color='blue', linestyle='None')
plt.plot(list_mse_x_orientation, marker='o', color='red', linestyle='None')
plt.plot(list_mse_y_orientation, marker='^', color='green', linestyle='None')
plt.plot(list_mse_z_orientation, marker='*', color='black', linestyle='None')
plt.xlabel("Model Version")
plt.ylabel("Mean Squared Error (MSE)")
plt.legend(["MSE of Position in X axis", 'MSE of Position in Y axis', "MSE of Position in Z axis", "MSE of Orientation in X axis", 'MSE of Orientation in Y axis', "MSE of Orientation in Z axis"])
plt.savefig(".\\MSE_of_Position_and_Orientation_in_X_Y_Z.png")
plt.show()

# print(txtfiles)

# loaded_model = tf.keras.models.load_model(Configuration.ModelPath)
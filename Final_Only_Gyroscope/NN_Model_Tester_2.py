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



# Testing

import pandas as pd

csv_path = "./RawData/my_testing_result_seperated_gyro.csv"
df = pd.read_csv(csv_path)
print(df.head())

fileName = df["fileName"]
mae_x_orientation = df["mae_x_orientation"]
mae_y_orientation = df["mae_y_orientation"]
mae_z_orientation = df["mae_z_orientation"]

mse_x_orientation = df["mse_x_orientation"]
mse_y_orientation = df["mse_y_orientation"]
mse_z_orientation = df["mse_z_orientation"]

print(fileName.head())

# DNN Model Variables
mae_x_orientation_DNN = []
mae_y_orientation_DNN = []
mae_z_orientation_DNN = []

mse_x_orientation_DNN = []
mse_y_orientation_DNN = []
mse_z_orientation_DNN = []

# CNN Model Variables
mae_x_orientation_CNN = []
mae_y_orientation_CNN = []
mae_z_orientation_CNN = []

mse_x_orientation_CNN = []
mse_y_orientation_CNN = []
mse_z_orientation_CNN = []

# RNN Model Variables
mae_x_orientation_RNN = []
mae_y_orientation_RNN = []
mae_z_orientation_RNN = []

mse_x_orientation_RNN = []
mse_y_orientation_RNN = []
mse_z_orientation_RNN = []

# LSTM Model Variables
mae_x_orientation_LSTM = []
mae_y_orientation_LSTM = []
mae_z_orientation_LSTM = []

mse_x_orientation_LSTM = []
mse_y_orientation_LSTM = []
mse_z_orientation_LSTM = []

# GRU Model Variables
mae_x_orientation_GRU = []
mae_y_orientation_GRU = []
mae_z_orientation_GRU = []

mse_x_orientation_GRU = []
mse_y_orientation_GRU = []
mse_z_orientation_GRU = []

totalData = len(fileName)
for i in range(totalData):

    # DNN Model
    if "Loss_mae_DNN" in fileName[i]:
        mae_x_orientation_DNN.append(mae_x_orientation[i])
        mae_y_orientation_DNN.append(mae_y_orientation[i])
        mae_z_orientation_DNN.append(mae_z_orientation[i])

    elif "Loss_mse_DNN" in fileName[i]:
        mse_x_orientation_DNN.append(mse_x_orientation[i])
        mse_y_orientation_DNN.append(mse_y_orientation[i])
        mse_z_orientation_DNN.append(mse_z_orientation[i])

    # CNN Model
    elif "Loss_mae_CNN" in fileName[i]:
        mae_x_orientation_CNN.append(mae_x_orientation[i])
        mae_y_orientation_CNN.append(mae_y_orientation[i])
        mae_z_orientation_CNN.append(mae_z_orientation[i])

    elif "Loss_mse_CNN" in fileName[i]:
        mse_x_orientation_CNN.append(mse_x_orientation[i])
        mse_y_orientation_CNN.append(mse_y_orientation[i])
        mse_z_orientation_CNN.append(mse_z_orientation[i])

    # RNN Model
    elif "Loss_mae_RNN" in fileName[i]:
        mae_x_orientation_RNN.append(mae_x_orientation[i])
        mae_y_orientation_RNN.append(mae_y_orientation[i])
        mae_z_orientation_RNN.append(mae_z_orientation[i])

    elif "Loss_mse_RNN" in fileName[i]:
        mse_x_orientation_RNN.append(mse_x_orientation[i])
        mse_y_orientation_RNN.append(mse_y_orientation[i])
        mse_z_orientation_RNN.append(mse_z_orientation[i])

    # LSTM Model
    elif "Loss_mae_LSTM" in fileName[i]:
        mae_x_orientation_LSTM.append(mae_x_orientation[i])
        mae_y_orientation_LSTM.append(mae_y_orientation[i])
        mae_z_orientation_LSTM.append(mae_z_orientation[i])

    elif "Loss_mse_LSTM" in fileName[i]:
        mse_x_orientation_LSTM.append(mse_x_orientation[i])
        mse_y_orientation_LSTM.append(mse_y_orientation[i])
        mse_z_orientation_LSTM.append(mse_z_orientation[i])

    # GRU Model
    elif "Loss_mae_GRU" in fileName[i]:
        mae_x_orientation_GRU.append(mae_x_orientation[i])
        mae_y_orientation_GRU.append(mae_y_orientation[i])
        mae_z_orientation_GRU.append(mae_z_orientation[i])

    elif "Loss_mse_GRU" in fileName[i]:
        mse_x_orientation_GRU.append(mse_x_orientation[i])
        mse_y_orientation_GRU.append(mse_y_orientation[i])
        mse_z_orientation_GRU.append(mse_z_orientation[i])


plt.clf()

currentTime = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) # not needed
figurePath = ".\\SavedFigure\\MAE-Vs-MSE\\"

if not os.path.exists(figurePath):
    os.makedirs(figurePath)

# DNN Model

ymin = -0.001
ymax = 0.03

# MAE_And_MSE_of_Orientation_in_X_DNN
plt.title("DNN Model (Loss Function: MAE Vs MSE in X axis)")
plt.plot(mae_x_orientation_DNN, marker='o', color='cyan', linestyle='None')
plt.plot(mse_x_orientation_DNN, marker='o', color='red', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Orientation_in_X_DNN")
plt.show()

# MAE_And_MSE_of_Orientation_in_Y_DNN
plt.title("DNN Model (Loss Function: MAE Vs MSE in Y axis)")
plt.plot(mae_y_orientation_DNN, marker='^', color='magenta', linestyle='None')
plt.plot(mse_y_orientation_DNN, marker='^', color='green', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Orientation_in_Y_DNN")
plt.show()

# MAE_And_MSE_of_Orientation_in_Y_DNN
plt.title("DNN Model (Loss Function: MAE Vs MSE in Z axis)")
plt.plot(mae_z_orientation_DNN, marker='*', color='blue', linestyle='None')
plt.plot(mse_z_orientation_DNN, marker='*', color='black', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Orientation_in_Z_DNN")
plt.show()

# CNN Model

# MAE_And_MSE_of_Orientation_in_X_CNN
plt.title("CNN Model (Loss Function: MAE Vs MSE in X axis)")
plt.plot(mae_x_orientation_CNN, marker='o', color='cyan', linestyle='None')
plt.plot(mse_x_orientation_CNN, marker='o', color='red', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Orientation_in_X_CNN")
plt.show()

# MAE_And_MSE_of_Orientation_in_Y_CNN
plt.title("CNN Model (Loss Function: MAE Vs MSE in Y axis)")
plt.plot(mae_y_orientation_CNN, marker='^', color='magenta', linestyle='None')
plt.plot(mse_y_orientation_CNN, marker='^', color='green', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Orientation_in_Y_CNN")
plt.show()

# MAE_And_MSE_of_Orientation_in_Z_CNN
plt.title("CNN Model (Loss Function: MAE Vs MSE in Z axis)")
plt.plot(mae_z_orientation_CNN, marker='*', color='blue', linestyle='None')
plt.plot(mse_z_orientation_CNN, marker='*', color='black', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Orientation_in_Z_CNN")
plt.show()

# RNN Model

# MAE_And_MSE_of_Orientation_in_X_RNN
plt.title("RNN Model (Loss Function: MAE Vs MSE in X axis)")
plt.plot(mae_x_orientation_RNN, marker='o', color='cyan', linestyle='None')
plt.plot(mse_x_orientation_RNN, marker='o', color='red', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Orientation_in_X_RNN")
plt.show()

# MAE_And_MSE_of_Orientation_in_Y_RNN
plt.title("RNN Model (Loss Function: MAE Vs MSE in Y axis)")
plt.plot(mae_y_orientation_RNN, marker='^', color='magenta', linestyle='None')
plt.plot(mse_y_orientation_RNN, marker='^', color='green', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Orientation_in_Y_RNN")
plt.show()

# MAE_And_MSE_of_Orientation_in_Z_RNN
plt.title("RNN Model (Loss Function: MAE Vs MSE in Z axis)")
plt.plot(mae_z_orientation_RNN, marker='*', color='blue', linestyle='None')
plt.plot(mse_z_orientation_RNN, marker='*', color='black', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Orientation_in_Z_RNN")
plt.show()

# LSTM Model

# MAE_And_MSE_of_Orientation_in_X_LSTM
plt.title("LSTM Model (Loss Function: MAE Vs MSE in X axis)")
plt.plot(mae_x_orientation_LSTM, marker='o', color='cyan', linestyle='None')
plt.plot(mse_x_orientation_LSTM, marker='o', color='red', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Orientation_in_X_LSTM")
plt.show()

# MAE_And_MSE_of_Orientation_in_Y_LSTM
plt.title("LSTM Model (Loss Function: MAE Vs MSE in Y axis)")
plt.plot(mae_y_orientation_LSTM, marker='^', color='magenta', linestyle='None')
plt.plot(mse_y_orientation_LSTM, marker='^', color='green', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Orientation_in_Y_LSTM")
plt.show()

# MAE_And_MSE_of_Orientation_in_Z_LSTM
plt.title("LSTM Model (Loss Function: MAE Vs MSE in Z axis)")
plt.plot(mae_z_orientation_LSTM, marker='*', color='blue', linestyle='None')
plt.plot(mse_z_orientation_LSTM, marker='*', color='black', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Orientation_in_Z_LSTM")
plt.show()

# GRU Model

# MAE_And_MSE_of_Orientation_in_X_GRU
plt.title("GRU Model (Loss Function: MAE Vs MSE in X axis)")
plt.plot(mae_x_orientation_GRU, marker='o', color='cyan', linestyle='None')
plt.plot(mse_x_orientation_GRU, marker='o', color='red', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Orientation_in_X_GRU")
plt.show()

# MAE_And_MSE_of_Orientation_in_Y_GRU
plt.title("GRU Model (Loss Function: MAE Vs MSE in Y axis)")
plt.plot(mae_y_orientation_GRU, marker='^', color='magenta', linestyle='None')
plt.plot(mse_y_orientation_GRU, marker='^', color='green', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Orientation_in_Y_GRU")
plt.show()

# MAE_And_MSE_of_Orientation_in_Z_GRU
plt.title("GRU Model (Loss Function: MAE Vs MSE in Z axis)")
plt.plot(mae_z_orientation_GRU, marker='*', color='blue', linestyle='None')
plt.plot(mse_z_orientation_GRU, marker='*', color='black', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Orientation_in_Z_GRU")
plt.show()


'''
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
list_mae_x = []
list_mae_y = []
list_mae_z = []
list_mse_x = []
list_mse_y = []
list_mse_z = []
for file in glob.glob(".\\SavedModel\\20200408-231907\\*.h5"):
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
    test_predict_delta_s_in_x_y_and_z = current_loaded_model.predict(test_dataset)

    dataset_creator.test_delta_t = dataset_creator.denormalize(dataset_creator.test_delta_t,
                                                                         TRAIN_SPLIT=dataset_creator.TRAIN_SIZE)
    dataset_creator.test_delta_s_x_y_and_z = dataset_creator.denormalize(
        dataset_creator.test_delta_s_x_y_and_z, TRAIN_SPLIT=dataset_creator.TRAIN_SIZE)
    test_predict_delta_s_in_x_y_and_z = dataset_creator.denormalize(test_predict_delta_s_in_x_y_and_z,
                                                                      TRAIN_SPLIT=dataset_creator.TRAIN_SIZE)

    # print(test_predict_delta_s_in_x_y_and_z.shape)
    # print(len(test_predict_delta_s_in_x_y_and_z))
    # print(len(dataset_creator.test_delta_s_x_y_and_z))
    test_length = len(test_predict_delta_s_in_x_y_and_z)
    # print("test_length: ", test_length)
    mae_x = np.sum(np.absolute(dataset_creator.test_delta_s_x_y_and_z[0] - test_predict_delta_s_in_x_y_and_z[0])) / test_length
    mae_y = np.sum(np.absolute(dataset_creator.test_delta_s_x_y_and_z[1] - test_predict_delta_s_in_x_y_and_z[1])) / test_length
    mae_z = np.sum(np.absolute(dataset_creator.test_delta_s_x_y_and_z[2] - test_predict_delta_s_in_x_y_and_z[2])) / test_length

    mse_x = np.sum(
        np.square(dataset_creator.test_delta_s_x_y_and_z[0] - test_predict_delta_s_in_x_y_and_z[0])) / test_length
    mse_y = np.sum(
        np.square(dataset_creator.test_delta_s_x_y_and_z[1] - test_predict_delta_s_in_x_y_and_z[1])) / test_length
    mse_z = np.sum(
        np.square(dataset_creator.test_delta_s_x_y_and_z[2] - test_predict_delta_s_in_x_y_and_z[2])) / test_length
    # print("{},{},{},{},{},{},{}".format(file, mae_x, mae_y, mae_z, mse_x, mse_y, mse_z))
    logger.info("{},{},{},{},{},{},{}".format(file, mae_x, mae_y, mae_z, mse_x, mse_y, mse_z))

    list_mae_x.append(mae_x)
    list_mae_y.append(mae_y)
    list_mae_z.append(mae_z)
    list_mse_x.append(mse_x)
    list_mse_y.append(mse_y)
    list_mse_z.append(mse_z)


plt.clf()
#epochs_range = range(EPOCHS)
#plt.plot(epochs_range, history.history[string])
#plt.plot(epochs_range, history.history['val_' + string])
plt.plot(list_mae_x, marker='o', color='cyan', linestyle='None')
plt.plot(list_mae_y, marker='^', color='magenta', linestyle='None')
plt.plot(list_mae_z, marker='*', color='blue', linestyle='None')
plt.xlabel("Model Version")
plt.ylabel("Mean Absolute Error (MAE)")
plt.legend(["MAE of Orientation in X axis", 'MAE of Orientation in Y axis', "MAE of Orientation in Z axis"])
plt.savefig(".\\MAE_of_Orientation_in_X_Y_Z")
plt.show()

plt.plot(list_mse_x, marker='o', color='cyan', linestyle='None')
plt.plot(list_mse_y, marker='^', color='magenta', linestyle='None')
plt.plot(list_mse_z, marker='*', color='blue', linestyle='None')
plt.xlabel("Model Version")
plt.ylabel("Mean Squared Error (MSE)")
plt.legend(["MSE of Orientation in X axis", 'MSE of Orientation in Y axis', "MSE of Orientation in Z axis"])
plt.savefig(".\\MSE_of_Orientation_in_X_Y_Z")
plt.show()

# print(txtfiles)

# loaded_model = tf.keras.models.load_model(Configuration.ModelPath)

'''
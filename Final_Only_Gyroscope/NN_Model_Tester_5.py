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

# Units_32

# DNN Model Variables
mae_x_orientation_DNN_Units_32 = []
mae_y_orientation_DNN_Units_32 = []
mae_z_orientation_DNN_Units_32 = []

mse_x_orientation_DNN_Units_32 = []
mse_y_orientation_DNN_Units_32 = []
mse_z_orientation_DNN_Units_32 = []

# CNN Model Variables
mae_x_orientation_CNN_Units_32 = []
mae_y_orientation_CNN_Units_32 = []
mae_z_orientation_CNN_Units_32 = []

mse_x_orientation_CNN_Units_32 = []
mse_y_orientation_CNN_Units_32 = []
mse_z_orientation_CNN_Units_32 = []

# CNN Model Variables for Kernel
mae_x_orientation_CNN_Kernel_3 = []
mae_y_orientation_CNN_Kernel_3 = []
mae_z_orientation_CNN_Kernel_3 = []

mse_x_orientation_CNN_Kernel_3 = []
mse_y_orientation_CNN_Kernel_3 = []
mse_z_orientation_CNN_Kernel_3 = []

# RNN Model Variables
mae_x_orientation_RNN_Units_32 = []
mae_y_orientation_RNN_Units_32 = []
mae_z_orientation_RNN_Units_32 = []

mse_x_orientation_RNN_Units_32 = []
mse_y_orientation_RNN_Units_32 = []
mse_z_orientation_RNN_Units_32 = []

# LSTM Model Variables
mae_x_orientation_LSTM_Units_32 = []
mae_y_orientation_LSTM_Units_32 = []
mae_z_orientation_LSTM_Units_32 = []

mse_x_orientation_LSTM_Units_32 = []
mse_y_orientation_LSTM_Units_32 = []
mse_z_orientation_LSTM_Units_32 = []

# GRU Model Variables
mae_x_orientation_GRU_Units_32 = []
mae_y_orientation_GRU_Units_32 = []
mae_z_orientation_GRU_Units_32 = []

mse_x_orientation_GRU_Units_32 = []
mse_y_orientation_GRU_Units_32 = []
mse_z_orientation_GRU_Units_32 = []

# Units_64

# DNN Model Variables
mae_x_orientation_DNN_Units_64 = []
mae_y_orientation_DNN_Units_64 = []
mae_z_orientation_DNN_Units_64 = []

mse_x_orientation_DNN_Units_64 = []
mse_y_orientation_DNN_Units_64 = []
mse_z_orientation_DNN_Units_64 = []

# CNN Model Variables
mae_x_orientation_CNN_Units_64 = []
mae_y_orientation_CNN_Units_64 = []
mae_z_orientation_CNN_Units_64 = []

mse_x_orientation_CNN_Units_64 = []
mse_y_orientation_CNN_Units_64 = []
mse_z_orientation_CNN_Units_64 = []

# CNN Model Variables for Kernel
mae_x_orientation_CNN_Kernel_5 = []
mae_y_orientation_CNN_Kernel_5 = []
mae_z_orientation_CNN_Kernel_5 = []

mse_x_orientation_CNN_Kernel_5 = []
mse_y_orientation_CNN_Kernel_5 = []
mse_z_orientation_CNN_Kernel_5 = []

# RNN Model Variables
mae_x_orientation_RNN_Units_64 = []
mae_y_orientation_RNN_Units_64 = []
mae_z_orientation_RNN_Units_64 = []

mse_x_orientation_RNN_Units_64 = []
mse_y_orientation_RNN_Units_64 = []
mse_z_orientation_RNN_Units_64 = []

# LSTM Model Variables
mae_x_orientation_LSTM_Units_64 = []
mae_y_orientation_LSTM_Units_64 = []
mae_z_orientation_LSTM_Units_64 = []

mse_x_orientation_LSTM_Units_64 = []
mse_y_orientation_LSTM_Units_64 = []
mse_z_orientation_LSTM_Units_64 = []

# GRU Model Variables
mae_x_orientation_GRU_Units_64 = []
mae_y_orientation_GRU_Units_64 = []
mae_z_orientation_GRU_Units_64 = []

mse_x_orientation_GRU_Units_64 = []
mse_y_orientation_GRU_Units_64 = []
mse_z_orientation_GRU_Units_64 = []

# Units_128

# DNN Model Variables
mae_x_orientation_DNN_Units_128 = []
mae_y_orientation_DNN_Units_128 = []
mae_z_orientation_DNN_Units_128 = []

mse_x_orientation_DNN_Units_128 = []
mse_y_orientation_DNN_Units_128 = []
mse_z_orientation_DNN_Units_128 = []

# CNN Model Variables
mae_x_orientation_CNN_Units_128 = []
mae_y_orientation_CNN_Units_128 = []
mae_z_orientation_CNN_Units_128 = []

mse_x_orientation_CNN_Units_128 = []
mse_y_orientation_CNN_Units_128 = []
mse_z_orientation_CNN_Units_128 = []

# CNN Model Variables for Kernel
mae_x_orientation_CNN_Kernel_7 = []
mae_y_orientation_CNN_Kernel_7 = []
mae_z_orientation_CNN_Kernel_7 = []

mse_x_orientation_CNN_Kernel_7 = []
mse_y_orientation_CNN_Kernel_7 = []
mse_z_orientation_CNN_Kernel_7 = []

# RNN Model Variables
mae_x_orientation_RNN_Units_128 = []
mae_y_orientation_RNN_Units_128 = []
mae_z_orientation_RNN_Units_128 = []

mse_x_orientation_RNN_Units_128 = []
mse_y_orientation_RNN_Units_128 = []
mse_z_orientation_RNN_Units_128 = []

# LSTM Model Variables
mae_x_orientation_LSTM_Units_128 = []
mae_y_orientation_LSTM_Units_128 = []
mae_z_orientation_LSTM_Units_128 = []

mse_x_orientation_LSTM_Units_128 = []
mse_y_orientation_LSTM_Units_128 = []
mse_z_orientation_LSTM_Units_128 = []

# GRU Model Variables
mae_x_orientation_GRU_Units_128 = []
mae_y_orientation_GRU_Units_128 = []
mae_z_orientation_GRU_Units_128 = []

mse_x_orientation_GRU_Units_128 = []
mse_y_orientation_GRU_Units_128 = []
mse_z_orientation_GRU_Units_128 = []

# Units_256

# DNN Model Variables
mae_x_orientation_DNN_Units_256 = []
mae_y_orientation_DNN_Units_256 = []
mae_z_orientation_DNN_Units_256 = []

mse_x_orientation_DNN_Units_256 = []
mse_y_orientation_DNN_Units_256 = []
mse_z_orientation_DNN_Units_256 = []

# CNN Model Variables
mae_x_orientation_CNN_Units_256 = []
mae_y_orientation_CNN_Units_256 = []
mae_z_orientation_CNN_Units_256 = []

mse_x_orientation_CNN_Units_256 = []
mse_y_orientation_CNN_Units_256 = []
mse_z_orientation_CNN_Units_256 = []

# CNN Model Variables for Kernel
mae_x_orientation_CNN_Kernel_9 = []
mae_y_orientation_CNN_Kernel_9 = []
mae_z_orientation_CNN_Kernel_9 = []

mse_x_orientation_CNN_Kernel_9 = []
mse_y_orientation_CNN_Kernel_9 = []
mse_z_orientation_CNN_Kernel_9 = []

# RNN Model Variables
mae_x_orientation_RNN_Units_256 = []
mae_y_orientation_RNN_Units_256 = []
mae_z_orientation_RNN_Units_256 = []

mse_x_orientation_RNN_Units_256 = []
mse_y_orientation_RNN_Units_256 = []
mse_z_orientation_RNN_Units_256 = []

# LSTM Model Variables
mae_x_orientation_LSTM_Units_256 = []
mae_y_orientation_LSTM_Units_256 = []
mae_z_orientation_LSTM_Units_256 = []

mse_x_orientation_LSTM_Units_256 = []
mse_y_orientation_LSTM_Units_256 = []
mse_z_orientation_LSTM_Units_256 = []

# GRU Model Variables
mae_x_orientation_GRU_Units_256 = []
mae_y_orientation_GRU_Units_256 = []
mae_z_orientation_GRU_Units_256 = []

mse_x_orientation_GRU_Units_256 = []
mse_y_orientation_GRU_Units_256 = []
mse_z_orientation_GRU_Units_256 = []

# Units_512

# DNN Model Variables
mae_x_orientation_DNN_Units_512 = []
mae_y_orientation_DNN_Units_512 = []
mae_z_orientation_DNN_Units_512 = []

mse_x_orientation_DNN_Units_512 = []
mse_y_orientation_DNN_Units_512 = []
mse_z_orientation_DNN_Units_512 = []

# CNN Model Variables
mae_x_orientation_CNN_Units_512 = []
mae_y_orientation_CNN_Units_512 = []
mae_z_orientation_CNN_Units_512 = []

mse_x_orientation_CNN_Units_512 = []
mse_y_orientation_CNN_Units_512 = []
mse_z_orientation_CNN_Units_512 = []

# RNN Model Variables
mae_x_orientation_RNN_Units_512 = []
mae_y_orientation_RNN_Units_512 = []
mae_z_orientation_RNN_Units_512 = []

mse_x_orientation_RNN_Units_512 = []
mse_y_orientation_RNN_Units_512 = []
mse_z_orientation_RNN_Units_512 = []

# LSTM Model Variables
mae_x_orientation_LSTM_Units_512 = []
mae_y_orientation_LSTM_Units_512 = []
mae_z_orientation_LSTM_Units_512 = []

mse_x_orientation_LSTM_Units_512 = []
mse_y_orientation_LSTM_Units_512 = []
mse_z_orientation_LSTM_Units_512 = []

# GRU Model Variables
mae_x_orientation_GRU_Units_512 = []
mae_y_orientation_GRU_Units_512 = []
mae_z_orientation_GRU_Units_512 = []

mse_x_orientation_GRU_Units_512 = []
mse_y_orientation_GRU_Units_512 = []
mse_z_orientation_GRU_Units_512 = []

totalData = len(fileName)
for i in range(totalData):

    # Units_32

    # DNN Model
    if "Loss_mse_DNN" in fileName[i] and "Units_32" in fileName[i]:
        mae_x_orientation_DNN_Units_32.append(mae_x_orientation[i])
        mae_y_orientation_DNN_Units_32.append(mae_y_orientation[i])
        mae_z_orientation_DNN_Units_32.append(mae_z_orientation[i])

        mse_x_orientation_DNN_Units_32.append(mse_x_orientation[i])
        mse_y_orientation_DNN_Units_32.append(mse_y_orientation[i])
        mse_z_orientation_DNN_Units_32.append(mse_z_orientation[i])

    # CNN Model
    elif "Loss_mse_CNN" in fileName[i] and "Filter_32" in fileName[i]:
        mae_x_orientation_CNN_Units_32.append(mae_x_orientation[i])
        mae_y_orientation_CNN_Units_32.append(mae_y_orientation[i])
        mae_z_orientation_CNN_Units_32.append(mae_z_orientation[i])

        mse_x_orientation_CNN_Units_32.append(mse_x_orientation[i])
        mse_y_orientation_CNN_Units_32.append(mse_y_orientation[i])
        mse_z_orientation_CNN_Units_32.append(mse_z_orientation[i])

    # CNN Model for Kernel
    elif "Loss_mse_CNN" in fileName[i] and "Kernel_3" in fileName[i]:
        mae_x_orientation_CNN_Kernel_3.append(mae_x_orientation[i])
        mae_y_orientation_CNN_Kernel_3.append(mae_y_orientation[i])
        mae_z_orientation_CNN_Kernel_3.append(mae_z_orientation[i])

        mse_x_orientation_CNN_Kernel_3.append(mse_x_orientation[i])
        mse_y_orientation_CNN_Kernel_3.append(mse_y_orientation[i])
        mse_z_orientation_CNN_Kernel_3.append(mse_z_orientation[i])

    # RNN Model
    elif "Loss_mse_RNN" in fileName[i] and "Units_32" in fileName[i]:
        mae_x_orientation_RNN_Units_32.append(mae_x_orientation[i])
        mae_y_orientation_RNN_Units_32.append(mae_y_orientation[i])
        mae_z_orientation_RNN_Units_32.append(mae_z_orientation[i])

        mse_x_orientation_RNN_Units_32.append(mse_x_orientation[i])
        mse_y_orientation_RNN_Units_32.append(mse_y_orientation[i])
        mse_z_orientation_RNN_Units_32.append(mse_z_orientation[i])

    # LSTM Model
    elif "Loss_mse_LSTM" in fileName[i] and "Units_32" in fileName[i]:
        mae_x_orientation_LSTM_Units_32.append(mae_x_orientation[i])
        mae_y_orientation_LSTM_Units_32.append(mae_y_orientation[i])
        mae_z_orientation_LSTM_Units_32.append(mae_z_orientation[i])

        mse_x_orientation_LSTM_Units_32.append(mse_x_orientation[i])
        mse_y_orientation_LSTM_Units_32.append(mse_y_orientation[i])
        mse_z_orientation_LSTM_Units_32.append(mse_z_orientation[i])

    # GRU Model
    elif "Loss_mse_GRU" in fileName[i] and "Units_32" in fileName[i]:
        mae_x_orientation_GRU_Units_32.append(mae_x_orientation[i])
        mae_y_orientation_GRU_Units_32.append(mae_y_orientation[i])
        mae_z_orientation_GRU_Units_32.append(mae_z_orientation[i])

        mse_x_orientation_GRU_Units_32.append(mse_x_orientation[i])
        mse_y_orientation_GRU_Units_32.append(mse_y_orientation[i])
        mse_z_orientation_GRU_Units_32.append(mse_z_orientation[i])

    # Units_64

    # DNN Model
    elif "Loss_mse_DNN" in fileName[i] and "Units_64" in fileName[i]:
        mae_x_orientation_DNN_Units_64.append(mae_x_orientation[i])
        mae_y_orientation_DNN_Units_64.append(mae_y_orientation[i])
        mae_z_orientation_DNN_Units_64.append(mae_z_orientation[i])

        mse_x_orientation_DNN_Units_64.append(mse_x_orientation[i])
        mse_y_orientation_DNN_Units_64.append(mse_y_orientation[i])
        mse_z_orientation_DNN_Units_64.append(mse_z_orientation[i])

    # CNN Model
    elif "Loss_mse_CNN" in fileName[i] and "Filter_64" in fileName[i]:
        mae_x_orientation_CNN_Units_64.append(mae_x_orientation[i])
        mae_y_orientation_CNN_Units_64.append(mae_y_orientation[i])
        mae_z_orientation_CNN_Units_64.append(mae_z_orientation[i])

        mse_x_orientation_CNN_Units_64.append(mse_x_orientation[i])
        mse_y_orientation_CNN_Units_64.append(mse_y_orientation[i])
        mse_z_orientation_CNN_Units_64.append(mse_z_orientation[i])

    # CNN Model for Kernel
    elif "Loss_mse_CNN" in fileName[i] and "Kernel_5" in fileName[i]:
        mae_x_orientation_CNN_Kernel_5.append(mae_x_orientation[i])
        mae_y_orientation_CNN_Kernel_5.append(mae_y_orientation[i])
        mae_z_orientation_CNN_Kernel_5.append(mae_z_orientation[i])

        mse_x_orientation_CNN_Kernel_5.append(mse_x_orientation[i])
        mse_y_orientation_CNN_Kernel_5.append(mse_y_orientation[i])
        mse_z_orientation_CNN_Kernel_5.append(mse_z_orientation[i])

    # RNN Model
    elif "Loss_mse_RNN" in fileName[i] and "Units_64" in fileName[i]:
        mae_x_orientation_RNN_Units_64.append(mae_x_orientation[i])
        mae_y_orientation_RNN_Units_64.append(mae_y_orientation[i])
        mae_z_orientation_RNN_Units_64.append(mae_z_orientation[i])

        mse_x_orientation_RNN_Units_64.append(mse_x_orientation[i])
        mse_y_orientation_RNN_Units_64.append(mse_y_orientation[i])
        mse_z_orientation_RNN_Units_64.append(mse_z_orientation[i])

    # LSTM Model
    elif "Loss_mse_LSTM" in fileName[i] and "Units_64" in fileName[i]:
        mae_x_orientation_LSTM_Units_64.append(mae_x_orientation[i])
        mae_y_orientation_LSTM_Units_64.append(mae_y_orientation[i])
        mae_z_orientation_LSTM_Units_64.append(mae_z_orientation[i])

        mse_x_orientation_LSTM_Units_64.append(mse_x_orientation[i])
        mse_y_orientation_LSTM_Units_64.append(mse_y_orientation[i])
        mse_z_orientation_LSTM_Units_64.append(mse_z_orientation[i])

    # GRU Model
    elif "Loss_mse_GRU" in fileName[i] and "Units_64" in fileName[i]:
        mae_x_orientation_GRU_Units_64.append(mae_x_orientation[i])
        mae_y_orientation_GRU_Units_64.append(mae_y_orientation[i])
        mae_z_orientation_GRU_Units_64.append(mae_z_orientation[i])

        mse_x_orientation_GRU_Units_64.append(mse_x_orientation[i])
        mse_y_orientation_GRU_Units_64.append(mse_y_orientation[i])
        mse_z_orientation_GRU_Units_64.append(mse_z_orientation[i])

    # Units_128

    # DNN Model
    elif "Loss_mse_DNN" in fileName[i] and "Units_128" in fileName[i]:
        mae_x_orientation_DNN_Units_128.append(mae_x_orientation[i])
        mae_y_orientation_DNN_Units_128.append(mae_y_orientation[i])
        mae_z_orientation_DNN_Units_128.append(mae_z_orientation[i])

        mse_x_orientation_DNN_Units_128.append(mse_x_orientation[i])
        mse_y_orientation_DNN_Units_128.append(mse_y_orientation[i])
        mse_z_orientation_DNN_Units_128.append(mse_z_orientation[i])

    # CNN Model
    elif "Loss_mse_CNN" in fileName[i] and "Filter_128" in fileName[i]:
        mae_x_orientation_CNN_Units_128.append(mae_x_orientation[i])
        mae_y_orientation_CNN_Units_128.append(mae_y_orientation[i])
        mae_z_orientation_CNN_Units_128.append(mae_z_orientation[i])

        mse_x_orientation_CNN_Units_128.append(mse_x_orientation[i])
        mse_y_orientation_CNN_Units_128.append(mse_y_orientation[i])
        mse_z_orientation_CNN_Units_128.append(mse_z_orientation[i])

    # CNN Model for Kernel
    elif "Loss_mse_CNN" in fileName[i] and "Kernel_7" in fileName[i]:
        mae_x_orientation_CNN_Kernel_7.append(mae_x_orientation[i])
        mae_y_orientation_CNN_Kernel_7.append(mae_y_orientation[i])
        mae_z_orientation_CNN_Kernel_7.append(mae_z_orientation[i])

        mse_x_orientation_CNN_Kernel_7.append(mse_x_orientation[i])
        mse_y_orientation_CNN_Kernel_7.append(mse_y_orientation[i])
        mse_z_orientation_CNN_Kernel_7.append(mse_z_orientation[i])

    # RNN Model
    elif "Loss_mse_RNN" in fileName[i] and "Units_128" in fileName[i]:
        mae_x_orientation_RNN_Units_128.append(mae_x_orientation[i])
        mae_y_orientation_RNN_Units_128.append(mae_y_orientation[i])
        mae_z_orientation_RNN_Units_128.append(mae_z_orientation[i])

        mse_x_orientation_RNN_Units_128.append(mse_x_orientation[i])
        mse_y_orientation_RNN_Units_128.append(mse_y_orientation[i])
        mse_z_orientation_RNN_Units_128.append(mse_z_orientation[i])

    # LSTM Model
    elif "Loss_mse_LSTM" in fileName[i] and "Units_128" in fileName[i]:
        mae_x_orientation_LSTM_Units_128.append(mae_x_orientation[i])
        mae_y_orientation_LSTM_Units_128.append(mae_y_orientation[i])
        mae_z_orientation_LSTM_Units_128.append(mae_z_orientation[i])

        mse_x_orientation_LSTM_Units_128.append(mse_x_orientation[i])
        mse_y_orientation_LSTM_Units_128.append(mse_y_orientation[i])
        mse_z_orientation_LSTM_Units_128.append(mse_z_orientation[i])

    # GRU Model
    elif "Loss_mse_GRU" in fileName[i] and "Units_128" in fileName[i]:
        mae_x_orientation_GRU_Units_128.append(mae_x_orientation[i])
        mae_y_orientation_GRU_Units_128.append(mae_y_orientation[i])
        mae_z_orientation_GRU_Units_128.append(mae_z_orientation[i])

        mse_x_orientation_GRU_Units_128.append(mse_x_orientation[i])
        mse_y_orientation_GRU_Units_128.append(mse_y_orientation[i])
        mse_z_orientation_GRU_Units_128.append(mse_z_orientation[i])

    # Units_256

    # DNN Model
    elif "Loss_mse_DNN" in fileName[i] and "Units_256" in fileName[i]:
        mae_x_orientation_DNN_Units_256.append(mae_x_orientation[i])
        mae_y_orientation_DNN_Units_256.append(mae_y_orientation[i])
        mae_z_orientation_DNN_Units_256.append(mae_z_orientation[i])

        mse_x_orientation_DNN_Units_256.append(mse_x_orientation[i])
        mse_y_orientation_DNN_Units_256.append(mse_y_orientation[i])
        mse_z_orientation_DNN_Units_256.append(mse_z_orientation[i])

    # CNN Model
    elif "Loss_mse_CNN" in fileName[i] and "Filter_256" in fileName[i]:
        mae_x_orientation_CNN_Units_256.append(mae_x_orientation[i])
        mae_y_orientation_CNN_Units_256.append(mae_y_orientation[i])
        mae_z_orientation_CNN_Units_256.append(mae_z_orientation[i])

        mse_x_orientation_CNN_Units_256.append(mse_x_orientation[i])
        mse_y_orientation_CNN_Units_256.append(mse_y_orientation[i])
        mse_z_orientation_CNN_Units_256.append(mse_z_orientation[i])

    # CNN Model for Kernel
    elif "Loss_mse_CNN" in fileName[i] and "Kernel_9" in fileName[i]:
        mae_x_orientation_CNN_Kernel_9.append(mae_x_orientation[i])
        mae_y_orientation_CNN_Kernel_9.append(mae_y_orientation[i])
        mae_z_orientation_CNN_Kernel_9.append(mae_z_orientation[i])

        mse_x_orientation_CNN_Kernel_9.append(mse_x_orientation[i])
        mse_y_orientation_CNN_Kernel_9.append(mse_y_orientation[i])
        mse_z_orientation_CNN_Kernel_9.append(mse_z_orientation[i])

    # RNN Model
    elif "Loss_mse_RNN" in fileName[i] and "Units_256" in fileName[i]:
        mae_x_orientation_RNN_Units_256.append(mae_x_orientation[i])
        mae_y_orientation_RNN_Units_256.append(mae_y_orientation[i])
        mae_z_orientation_RNN_Units_256.append(mae_z_orientation[i])

        mse_x_orientation_RNN_Units_256.append(mse_x_orientation[i])
        mse_y_orientation_RNN_Units_256.append(mse_y_orientation[i])
        mse_z_orientation_RNN_Units_256.append(mse_z_orientation[i])

    # LSTM Model
    elif "Loss_mse_LSTM" in fileName[i] and "Units_256" in fileName[i]:
        mae_x_orientation_LSTM_Units_256.append(mae_x_orientation[i])
        mae_y_orientation_LSTM_Units_256.append(mae_y_orientation[i])
        mae_z_orientation_LSTM_Units_256.append(mae_z_orientation[i])

        mse_x_orientation_LSTM_Units_256.append(mse_x_orientation[i])
        mse_y_orientation_LSTM_Units_256.append(mse_y_orientation[i])
        mse_z_orientation_LSTM_Units_256.append(mse_z_orientation[i])

    # GRU Model
    elif "Loss_mse_GRU" in fileName[i] and "Units_256" in fileName[i]:
        mae_x_orientation_GRU_Units_256.append(mae_x_orientation[i])
        mae_y_orientation_GRU_Units_256.append(mae_y_orientation[i])
        mae_z_orientation_GRU_Units_256.append(mae_z_orientation[i])

        mse_x_orientation_GRU_Units_256.append(mse_x_orientation[i])
        mse_y_orientation_GRU_Units_256.append(mse_y_orientation[i])
        mse_z_orientation_GRU_Units_256.append(mse_z_orientation[i])

    # Units_512

    # DNN Model
    elif "Loss_mse_DNN" in fileName[i] and "Units_512" in fileName[i]:
        mae_x_orientation_DNN_Units_512.append(mae_x_orientation[i])
        mae_y_orientation_DNN_Units_512.append(mae_y_orientation[i])
        mae_z_orientation_DNN_Units_512.append(mae_z_orientation[i])

        mse_x_orientation_DNN_Units_512.append(mse_x_orientation[i])
        mse_y_orientation_DNN_Units_512.append(mse_y_orientation[i])
        mse_z_orientation_DNN_Units_512.append(mse_z_orientation[i])

    # CNN Model
    elif "Loss_mse_CNN" in fileName[i] and "Filter_512" in fileName[i]:
        mae_x_orientation_CNN_Units_512.append(mae_x_orientation[i])
        mae_y_orientation_CNN_Units_512.append(mae_y_orientation[i])
        mae_z_orientation_CNN_Units_512.append(mae_z_orientation[i])

        mse_x_orientation_CNN_Units_512.append(mse_x_orientation[i])
        mse_y_orientation_CNN_Units_512.append(mse_y_orientation[i])
        mse_z_orientation_CNN_Units_512.append(mse_z_orientation[i])

    # RNN Model
    elif "Loss_mse_RNN" in fileName[i] and "Units_512" in fileName[i]:
        mae_x_orientation_RNN_Units_512.append(mae_x_orientation[i])
        mae_y_orientation_RNN_Units_512.append(mae_y_orientation[i])
        mae_z_orientation_RNN_Units_512.append(mae_z_orientation[i])

        mse_x_orientation_RNN_Units_512.append(mse_x_orientation[i])
        mse_y_orientation_RNN_Units_512.append(mse_y_orientation[i])
        mse_z_orientation_RNN_Units_512.append(mse_z_orientation[i])

    # LSTM Model
    elif "Loss_mse_LSTM" in fileName[i] and "Units_512" in fileName[i]:
        mae_x_orientation_LSTM_Units_512.append(mae_x_orientation[i])
        mae_y_orientation_LSTM_Units_512.append(mae_y_orientation[i])
        mae_z_orientation_LSTM_Units_512.append(mae_z_orientation[i])

        mse_x_orientation_LSTM_Units_512.append(mse_x_orientation[i])
        mse_y_orientation_LSTM_Units_512.append(mse_y_orientation[i])
        mse_z_orientation_LSTM_Units_512.append(mse_z_orientation[i])

    # GRU Model
    elif "Loss_mse_GRU" in fileName[i] and "Units_512" in fileName[i]:
        mae_x_orientation_GRU_Units_512.append(mae_x_orientation[i])
        mae_y_orientation_GRU_Units_512.append(mae_y_orientation[i])
        mae_z_orientation_GRU_Units_512.append(mae_z_orientation[i])

        mse_x_orientation_GRU_Units_512.append(mse_x_orientation[i])
        mse_y_orientation_GRU_Units_512.append(mse_y_orientation[i])
        mse_z_orientation_GRU_Units_512.append(mse_z_orientation[i])




plt.clf()

currentTime = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) # not needed
figurePath = ".\\SavedFigure\\UnitSizes-32-64-128-256-512\\MSE_of_Orientation_in_Testing_Data\\"

if not os.path.exists(figurePath):
    os.makedirs(figurePath)

# Unit_Sizes

# DNN Model

ymin = -0.001
ymax = 0.015

# UnitSizes_32_64_128_256_512_of_Orientation_in_X_DNN
plt.title("DNN Model in X axis")
plt.plot(mse_x_orientation_DNN_Units_32, marker='o', color='blue', linestyle='None')
plt.plot(mse_x_orientation_DNN_Units_64, marker='o', color='green', linestyle='None')
plt.plot(mse_x_orientation_DNN_Units_128, marker='o', color='black', linestyle='None')
plt.plot(mse_x_orientation_DNN_Units_256, marker='o', color='magenta', linestyle='None')
plt.plot(mse_x_orientation_DNN_Units_512, marker='o', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_X_DNN")
plt.show()

# UnitSizes_32_64_128_256_512_of_Orientation_in_Y_DNN
plt.title("DNN Model in Y axis")
plt.plot(mse_y_orientation_DNN_Units_32, marker='^', color='blue', linestyle='None')
plt.plot(mse_y_orientation_DNN_Units_64, marker='^', color='green', linestyle='None')
plt.plot(mse_y_orientation_DNN_Units_128, marker='^', color='black', linestyle='None')
plt.plot(mse_y_orientation_DNN_Units_256, marker='^', color='magenta', linestyle='None')
plt.plot(mse_y_orientation_DNN_Units_512, marker='^', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_Y_DNN")
plt.show()

# UnitSizes_32_64_128_256_512_of_Orientation_in_Z_DNN
plt.title("DNN Model in Z axis")
plt.plot(mse_z_orientation_DNN_Units_32, marker='*', color='blue', linestyle='None')
plt.plot(mse_z_orientation_DNN_Units_64, marker='*', color='green', linestyle='None')
plt.plot(mse_z_orientation_DNN_Units_128, marker='*', color='black', linestyle='None')
plt.plot(mse_z_orientation_DNN_Units_256, marker='*', color='magenta', linestyle='None')
plt.plot(mse_z_orientation_DNN_Units_512, marker='*', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_Z_DNN")
plt.show()

# CNN Model

# UnitSizes_32_64_128_256_512_of_Orientation_in_X_CNN
plt.title("CNN Model in X axis")
plt.plot(mse_x_orientation_CNN_Units_32, marker='o', color='blue', linestyle='None')
plt.plot(mse_x_orientation_CNN_Units_64, marker='o', color='green', linestyle='None')
plt.plot(mse_x_orientation_CNN_Units_128, marker='o', color='black', linestyle='None')
plt.plot(mse_x_orientation_CNN_Units_256, marker='o', color='magenta', linestyle='None')
plt.plot(mse_x_orientation_CNN_Units_512, marker='o', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_X_CNN")
plt.show()

# UnitSizes_32_64_128_256_512_of_Orientation_in_Y_CNN
plt.title("CNN Model in Y axis")
plt.plot(mse_y_orientation_CNN_Units_32, marker='^', color='blue', linestyle='None')
plt.plot(mse_y_orientation_CNN_Units_64, marker='^', color='green', linestyle='None')
plt.plot(mse_y_orientation_CNN_Units_128, marker='^', color='black', linestyle='None')
plt.plot(mse_y_orientation_CNN_Units_256, marker='^', color='magenta', linestyle='None')
plt.plot(mse_y_orientation_CNN_Units_512, marker='^', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_Y_CNN")
plt.show()

# UnitSizes_32_64_128_256_512_of_Orientation_in_Z_CNN
plt.title("CNN Model in Z axis")
plt.plot(mse_z_orientation_CNN_Units_32, marker='*', color='blue', linestyle='None')
plt.plot(mse_z_orientation_CNN_Units_64, marker='*', color='green', linestyle='None')
plt.plot(mse_z_orientation_CNN_Units_128, marker='*', color='black', linestyle='None')
plt.plot(mse_z_orientation_CNN_Units_256, marker='*', color='magenta', linestyle='None')
plt.plot(mse_z_orientation_CNN_Units_512, marker='*', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_Z_CNN")
plt.show()

# CNN Model for Kernel

# KernelSizes_3_5_7_9_of_Orientation_in_X_CNN
plt.title("CNN Model in X axis")
plt.plot(mse_x_orientation_CNN_Kernel_3, marker='o', color='blue', linestyle='None')
plt.plot(mse_x_orientation_CNN_Kernel_5, marker='o', color='green', linestyle='None')
plt.plot(mse_x_orientation_CNN_Kernel_7, marker='o', color='black', linestyle='None')
plt.plot(mse_x_orientation_CNN_Kernel_9, marker='o', color='magenta', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Kernel Size: 3", "Kernel Size: 5", "Kernel Size: 7", "Kernel Size: 9"])
plt.savefig(figurePath + "KernelSizes_3_5_7_9_of_Orientation_in_X_CNN")
plt.show()

# KernelSizes_3_5_7_9_of_Orientation_in_Y_CNN
plt.title("CNN Model in Y axis")
plt.plot(mse_y_orientation_CNN_Kernel_3, marker='o', color='blue', linestyle='None')
plt.plot(mse_y_orientation_CNN_Kernel_5, marker='o', color='green', linestyle='None')
plt.plot(mse_y_orientation_CNN_Kernel_7, marker='o', color='black', linestyle='None')
plt.plot(mse_y_orientation_CNN_Kernel_9, marker='o', color='magenta', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Kernel Size: 3", "Kernel Size: 5", "Kernel Size: 7", "Kernel Size: 9"])
plt.savefig(figurePath + "KernelSizes_3_5_7_9_of_Orientation_in_Y_CNN")
plt.show()

# KernelSizes_3_5_7_9_of_Orientation_in_Z_CNN
plt.title("CNN Model in Z axis")
plt.plot(mse_z_orientation_CNN_Kernel_3, marker='o', color='blue', linestyle='None')
plt.plot(mse_z_orientation_CNN_Kernel_5, marker='o', color='green', linestyle='None')
plt.plot(mse_z_orientation_CNN_Kernel_7, marker='o', color='black', linestyle='None')
plt.plot(mse_z_orientation_CNN_Kernel_9, marker='o', color='magenta', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Kernel Size: 3", "Kernel Size: 5", "Kernel Size: 7", "Kernel Size: 9"])
plt.savefig(figurePath + "KernelSizes_3_5_7_9_of_Orientation_in_Z_CNN")
plt.show()

# RNN Model

# UnitSizes_32_64_128_256_512_of_Orientation_in_X_RNN
plt.title("RNN Model in X axis")
plt.plot(mse_x_orientation_RNN_Units_32, marker='o', color='blue', linestyle='None')
plt.plot(mse_x_orientation_RNN_Units_64, marker='o', color='green', linestyle='None')
plt.plot(mse_x_orientation_RNN_Units_128, marker='o', color='black', linestyle='None')
plt.plot(mse_x_orientation_RNN_Units_256, marker='o', color='magenta', linestyle='None')
plt.plot(mse_x_orientation_RNN_Units_512, marker='o', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_X_RNN")
plt.show()

# UnitSizes_32_64_128_256_512_of_Orientation_in_Y_RNN
plt.title("RNN Model in Y axis")
plt.plot(mse_y_orientation_RNN_Units_32, marker='^', color='blue', linestyle='None')
plt.plot(mse_y_orientation_RNN_Units_64, marker='^', color='green', linestyle='None')
plt.plot(mse_y_orientation_RNN_Units_128, marker='^', color='black', linestyle='None')
plt.plot(mse_y_orientation_RNN_Units_256, marker='^', color='magenta', linestyle='None')
plt.plot(mse_y_orientation_RNN_Units_512, marker='^', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_Y_RNN")
plt.show()

# UnitSizes_32_64_128_256_512_of_Orientation_in_Z_RNN
plt.title("RNN Model in Z axis")
plt.plot(mse_z_orientation_RNN_Units_32, marker='*', color='blue', linestyle='None')
plt.plot(mse_z_orientation_RNN_Units_64, marker='*', color='green', linestyle='None')
plt.plot(mse_z_orientation_RNN_Units_128, marker='*', color='black', linestyle='None')
plt.plot(mse_z_orientation_RNN_Units_256, marker='*', color='magenta', linestyle='None')
plt.plot(mse_z_orientation_RNN_Units_512, marker='*', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_Z_RNN")
plt.show()

# LSTM Model

# UnitSizes_32_64_128_256_512_of_Orientation_in_X_LSTM
plt.title("LSTM Model in X axis")
plt.plot(mse_x_orientation_LSTM_Units_32, marker='o', color='blue', linestyle='None')
plt.plot(mse_x_orientation_LSTM_Units_64, marker='o', color='green', linestyle='None')
plt.plot(mse_x_orientation_LSTM_Units_128, marker='o', color='black', linestyle='None')
plt.plot(mse_x_orientation_LSTM_Units_256, marker='o', color='magenta', linestyle='None')
plt.plot(mse_x_orientation_LSTM_Units_512, marker='o', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_X_LSTM")
plt.show()

# UnitSizes_32_64_128_256_512_of_Orientation_in_Y_LSTM
plt.title("LSTM Model in Y axis")
plt.plot(mse_y_orientation_LSTM_Units_32, marker='^', color='blue', linestyle='None')
plt.plot(mse_y_orientation_LSTM_Units_64, marker='^', color='green', linestyle='None')
plt.plot(mse_y_orientation_LSTM_Units_128, marker='^', color='black', linestyle='None')
plt.plot(mse_y_orientation_LSTM_Units_256, marker='^', color='magenta', linestyle='None')
plt.plot(mse_y_orientation_LSTM_Units_512, marker='^', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_Y_LSTM")
plt.show()

# UnitSizes_32_64_128_256_512_of_Orientation_in_Z_LSTM
plt.title("LSTM Model in Z axis")
plt.plot(mse_z_orientation_LSTM_Units_32, marker='*', color='blue', linestyle='None')
plt.plot(mse_z_orientation_LSTM_Units_64, marker='*', color='green', linestyle='None')
plt.plot(mse_z_orientation_LSTM_Units_128, marker='*', color='black', linestyle='None')
plt.plot(mse_z_orientation_LSTM_Units_256, marker='*', color='magenta', linestyle='None')
plt.plot(mse_z_orientation_LSTM_Units_512, marker='*', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_Z_LSTM")
plt.show()

# GRU Model

# UnitSizes_32_64_128_256_512_of_Orientation_in_X_GRU
plt.title("GRU Model in X axis")
plt.plot(mse_x_orientation_GRU_Units_32, marker='o', color='blue', linestyle='None')
plt.plot(mse_x_orientation_GRU_Units_64, marker='o', color='green', linestyle='None')
plt.plot(mse_x_orientation_GRU_Units_128, marker='o', color='black', linestyle='None')
plt.plot(mse_x_orientation_GRU_Units_256, marker='o', color='magenta', linestyle='None')
plt.plot(mse_x_orientation_GRU_Units_512, marker='o', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_X_GRU")
plt.show()

# UnitSizes_32_64_128_256_512_of_Orientation_in_Y_GRU
plt.title("GRU Model in Y axis")
plt.plot(mse_y_orientation_GRU_Units_32, marker='^', color='blue', linestyle='None')
plt.plot(mse_y_orientation_GRU_Units_64, marker='^', color='green', linestyle='None')
plt.plot(mse_y_orientation_GRU_Units_128, marker='^', color='black', linestyle='None')
plt.plot(mse_y_orientation_GRU_Units_256, marker='^', color='magenta', linestyle='None')
plt.plot(mse_y_orientation_GRU_Units_512, marker='^', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_Y_GRU")
plt.show()

# UnitSizes_32_64_128_256_512_of_Orientation_in_Z_GRU
plt.title("GRU Model in Z axis")
plt.plot(mse_z_orientation_GRU_Units_32, marker='*', color='blue', linestyle='None')
plt.plot(mse_z_orientation_GRU_Units_64, marker='*', color='green', linestyle='None')
plt.plot(mse_z_orientation_GRU_Units_128, marker='*', color='black', linestyle='None')
plt.plot(mse_z_orientation_GRU_Units_256, marker='*', color='magenta', linestyle='None')
plt.plot(mse_z_orientation_GRU_Units_512, marker='*', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_Z_GRU")
plt.show()



###########################################################################################################

currentTime = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) # not needed
figurePath = ".\\SavedFigure\\UnitSizes-32-64-128-256-512\\MAE_of_Orientation_in_Testing_Data\\"

if not os.path.exists(figurePath):
    os.makedirs(figurePath)

# Unit_Sizes

# DNN Model

ymin = -0.001
ymax = 0.015

# UnitSizes_32_64_128_256_512_of_Orientation_in_X_DNN
plt.title("DNN Model in X axis")
plt.plot(mae_x_orientation_DNN_Units_32, marker='o', color='blue', linestyle='None')
plt.plot(mae_x_orientation_DNN_Units_64, marker='o', color='green', linestyle='None')
plt.plot(mae_x_orientation_DNN_Units_128, marker='o', color='black', linestyle='None')
plt.plot(mae_x_orientation_DNN_Units_256, marker='o', color='magenta', linestyle='None')
plt.plot(mae_x_orientation_DNN_Units_512, marker='o', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_X_DNN")
plt.show()

# UnitSizes_32_64_128_256_512_of_Orientation_in_Y_DNN
plt.title("DNN Model in Y axis")
plt.plot(mae_y_orientation_DNN_Units_32, marker='^', color='blue', linestyle='None')
plt.plot(mae_y_orientation_DNN_Units_64, marker='^', color='green', linestyle='None')
plt.plot(mae_y_orientation_DNN_Units_128, marker='^', color='black', linestyle='None')
plt.plot(mae_y_orientation_DNN_Units_256, marker='^', color='magenta', linestyle='None')
plt.plot(mae_y_orientation_DNN_Units_512, marker='^', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_Y_DNN")
plt.show()

# UnitSizes_32_64_128_256_512_of_Orientation_in_Z_DNN
plt.title("DNN Model in Z axis")
plt.plot(mae_z_orientation_DNN_Units_32, marker='*', color='blue', linestyle='None')
plt.plot(mae_z_orientation_DNN_Units_64, marker='*', color='green', linestyle='None')
plt.plot(mae_z_orientation_DNN_Units_128, marker='*', color='black', linestyle='None')
plt.plot(mae_z_orientation_DNN_Units_256, marker='*', color='magenta', linestyle='None')
plt.plot(mae_z_orientation_DNN_Units_512, marker='*', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_Z_DNN")
plt.show()

# CNN Model

# UnitSizes_32_64_128_256_512_of_Orientation_in_X_CNN
plt.title("CNN Model in X axis")
plt.plot(mae_x_orientation_CNN_Units_32, marker='o', color='blue', linestyle='None')
plt.plot(mae_x_orientation_CNN_Units_64, marker='o', color='green', linestyle='None')
plt.plot(mae_x_orientation_CNN_Units_128, marker='o', color='black', linestyle='None')
plt.plot(mae_x_orientation_CNN_Units_256, marker='o', color='magenta', linestyle='None')
plt.plot(mae_x_orientation_CNN_Units_512, marker='o', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_X_CNN")
plt.show()

# UnitSizes_32_64_128_256_512_of_Orientation_in_Y_CNN
plt.title("CNN Model in Y axis")
plt.plot(mae_y_orientation_CNN_Units_32, marker='^', color='blue', linestyle='None')
plt.plot(mae_y_orientation_CNN_Units_64, marker='^', color='green', linestyle='None')
plt.plot(mae_y_orientation_CNN_Units_128, marker='^', color='black', linestyle='None')
plt.plot(mae_y_orientation_CNN_Units_256, marker='^', color='magenta', linestyle='None')
plt.plot(mae_y_orientation_CNN_Units_512, marker='^', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_Y_CNN")
plt.show()

# UnitSizes_32_64_128_256_512_of_Orientation_in_Z_CNN
plt.title("CNN Model in Z axis")
plt.plot(mae_z_orientation_CNN_Units_32, marker='*', color='blue', linestyle='None')
plt.plot(mae_z_orientation_CNN_Units_64, marker='*', color='green', linestyle='None')
plt.plot(mae_z_orientation_CNN_Units_128, marker='*', color='black', linestyle='None')
plt.plot(mae_z_orientation_CNN_Units_256, marker='*', color='magenta', linestyle='None')
plt.plot(mae_z_orientation_CNN_Units_512, marker='*', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_Z_CNN")
plt.show()

# CNN Model for Kernel

# KernelSizes_3_5_7_9_of_Orientation_in_X_CNN
plt.title("CNN Model in X axis")
plt.plot(mae_x_orientation_CNN_Kernel_3, marker='o', color='blue', linestyle='None')
plt.plot(mae_x_orientation_CNN_Kernel_5, marker='o', color='green', linestyle='None')
plt.plot(mae_x_orientation_CNN_Kernel_7, marker='o', color='black', linestyle='None')
plt.plot(mae_x_orientation_CNN_Kernel_9, marker='o', color='magenta', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Kernel Size: 3", "Kernel Size: 5", "Kernel Size: 7", "Kernel Size: 9"])
plt.savefig(figurePath + "KernelSizes_3_5_7_9_of_Orientation_in_X_CNN")
plt.show()

# KernelSizes_3_5_7_9_of_Orientation_in_Y_CNN
plt.title("CNN Model in Y axis")
plt.plot(mae_y_orientation_CNN_Kernel_3, marker='o', color='blue', linestyle='None')
plt.plot(mae_y_orientation_CNN_Kernel_5, marker='o', color='green', linestyle='None')
plt.plot(mae_y_orientation_CNN_Kernel_7, marker='o', color='black', linestyle='None')
plt.plot(mae_y_orientation_CNN_Kernel_9, marker='o', color='magenta', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Kernel Size: 3", "Kernel Size: 5", "Kernel Size: 7", "Kernel Size: 9"])
plt.savefig(figurePath + "KernelSizes_3_5_7_9_of_Orientation_in_Y_CNN")
plt.show()

# KernelSizes_3_5_7_9_of_Orientation_in_Z_CNN
plt.title("CNN Model in Z axis")
plt.plot(mae_z_orientation_CNN_Kernel_3, marker='o', color='blue', linestyle='None')
plt.plot(mae_z_orientation_CNN_Kernel_5, marker='o', color='green', linestyle='None')
plt.plot(mae_z_orientation_CNN_Kernel_7, marker='o', color='black', linestyle='None')
plt.plot(mae_z_orientation_CNN_Kernel_9, marker='o', color='magenta', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Kernel Size: 3", "Kernel Size: 5", "Kernel Size: 7", "Kernel Size: 9"])
plt.savefig(figurePath + "KernelSizes_3_5_7_9_of_Orientation_in_Z_CNN")
plt.show()

# RNN Model

# UnitSizes_32_64_128_256_512_of_Orientation_in_X_RNN
plt.title("RNN Model in X axis")
plt.plot(mae_x_orientation_RNN_Units_32, marker='o', color='blue', linestyle='None')
plt.plot(mae_x_orientation_RNN_Units_64, marker='o', color='green', linestyle='None')
plt.plot(mae_x_orientation_RNN_Units_128, marker='o', color='black', linestyle='None')
plt.plot(mae_x_orientation_RNN_Units_256, marker='o', color='magenta', linestyle='None')
plt.plot(mae_x_orientation_RNN_Units_512, marker='o', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_X_RNN")
plt.show()

# UnitSizes_32_64_128_256_512_of_Orientation_in_Y_RNN
plt.title("RNN Model in Y axis")
plt.plot(mae_y_orientation_RNN_Units_32, marker='^', color='blue', linestyle='None')
plt.plot(mae_y_orientation_RNN_Units_64, marker='^', color='green', linestyle='None')
plt.plot(mae_y_orientation_RNN_Units_128, marker='^', color='black', linestyle='None')
plt.plot(mae_y_orientation_RNN_Units_256, marker='^', color='magenta', linestyle='None')
plt.plot(mae_y_orientation_RNN_Units_512, marker='^', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_Y_RNN")
plt.show()

# UnitSizes_32_64_128_256_512_of_Orientation_in_Z_RNN
plt.title("RNN Model in Z axis")
plt.plot(mae_z_orientation_RNN_Units_32, marker='*', color='blue', linestyle='None')
plt.plot(mae_z_orientation_RNN_Units_64, marker='*', color='green', linestyle='None')
plt.plot(mae_z_orientation_RNN_Units_128, marker='*', color='black', linestyle='None')
plt.plot(mae_z_orientation_RNN_Units_256, marker='*', color='magenta', linestyle='None')
plt.plot(mae_z_orientation_RNN_Units_512, marker='*', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_Z_RNN")
plt.show()

# LSTM Model

# UnitSizes_32_64_128_256_512_of_Orientation_in_X_LSTM
plt.title("LSTM Model in X axis")
plt.plot(mae_x_orientation_LSTM_Units_32, marker='o', color='blue', linestyle='None')
plt.plot(mae_x_orientation_LSTM_Units_64, marker='o', color='green', linestyle='None')
plt.plot(mae_x_orientation_LSTM_Units_128, marker='o', color='black', linestyle='None')
plt.plot(mae_x_orientation_LSTM_Units_256, marker='o', color='magenta', linestyle='None')
plt.plot(mae_x_orientation_LSTM_Units_512, marker='o', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_X_LSTM")
plt.show()

# UnitSizes_32_64_128_256_512_of_Orientation_in_Y_LSTM
plt.title("LSTM Model in Y axis")
plt.plot(mae_y_orientation_LSTM_Units_32, marker='^', color='blue', linestyle='None')
plt.plot(mae_y_orientation_LSTM_Units_64, marker='^', color='green', linestyle='None')
plt.plot(mae_y_orientation_LSTM_Units_128, marker='^', color='black', linestyle='None')
plt.plot(mae_y_orientation_LSTM_Units_256, marker='^', color='magenta', linestyle='None')
plt.plot(mae_y_orientation_LSTM_Units_512, marker='^', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_Y_LSTM")
plt.show()

# UnitSizes_32_64_128_256_512_of_Orientation_in_Z_LSTM
plt.title("LSTM Model in Z axis")
plt.plot(mae_z_orientation_LSTM_Units_32, marker='*', color='blue', linestyle='None')
plt.plot(mae_z_orientation_LSTM_Units_64, marker='*', color='green', linestyle='None')
plt.plot(mae_z_orientation_LSTM_Units_128, marker='*', color='black', linestyle='None')
plt.plot(mae_z_orientation_LSTM_Units_256, marker='*', color='magenta', linestyle='None')
plt.plot(mae_z_orientation_LSTM_Units_512, marker='*', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_Z_LSTM")
plt.show()

# GRU Model

# UnitSizes_32_64_128_256_512_of_Orientation_in_X_GRU
plt.title("GRU Model in X axis")
plt.plot(mae_x_orientation_GRU_Units_32, marker='o', color='blue', linestyle='None')
plt.plot(mae_x_orientation_GRU_Units_64, marker='o', color='green', linestyle='None')
plt.plot(mae_x_orientation_GRU_Units_128, marker='o', color='black', linestyle='None')
plt.plot(mae_x_orientation_GRU_Units_256, marker='o', color='magenta', linestyle='None')
plt.plot(mae_x_orientation_GRU_Units_512, marker='o', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_X_GRU")
plt.show()

# UnitSizes_32_64_128_256_512_of_Orientation_in_Y_GRU
plt.title("GRU Model in Y axis")
plt.plot(mae_y_orientation_GRU_Units_32, marker='^', color='blue', linestyle='None')
plt.plot(mae_y_orientation_GRU_Units_64, marker='^', color='green', linestyle='None')
plt.plot(mae_y_orientation_GRU_Units_128, marker='^', color='black', linestyle='None')
plt.plot(mae_y_orientation_GRU_Units_256, marker='^', color='magenta', linestyle='None')
plt.plot(mae_y_orientation_GRU_Units_512, marker='^', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_Y_GRU")
plt.show()

# UnitSizes_32_64_128_256_512_of_Orientation_in_Z_GRU
plt.title("GRU Model in Z axis")
plt.plot(mae_z_orientation_GRU_Units_32, marker='*', color='blue', linestyle='None')
plt.plot(mae_z_orientation_GRU_Units_64, marker='*', color='green', linestyle='None')
plt.plot(mae_z_orientation_GRU_Units_128, marker='*', color='black', linestyle='None')
plt.plot(mae_z_orientation_GRU_Units_256, marker='*', color='magenta', linestyle='None')
plt.plot(mae_z_orientation_GRU_Units_512, marker='*', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Unit Size: 32", "Unit Size: 64", "Unit Size: 128", "Unit Size: 256", "Unit Size: 512"])
plt.savefig(figurePath + "UnitSizes_32_64_128_256_512_of_Orientation_in_Z_GRU")
plt.show()
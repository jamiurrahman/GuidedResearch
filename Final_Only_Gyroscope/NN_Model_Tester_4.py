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

# Batch_8

# DNN Model Variables
mae_x_orientation_DNN_Batch_8 = []
mae_y_orientation_DNN_Batch_8 = []
mae_z_orientation_DNN_Batch_8 = []

mse_x_orientation_DNN_Batch_8 = []
mse_y_orientation_DNN_Batch_8 = []
mse_z_orientation_DNN_Batch_8 = []

# CNN Model Variables
mae_x_orientation_CNN_Batch_8 = []
mae_y_orientation_CNN_Batch_8 = []
mae_z_orientation_CNN_Batch_8 = []

mse_x_orientation_CNN_Batch_8 = []
mse_y_orientation_CNN_Batch_8 = []
mse_z_orientation_CNN_Batch_8 = []

# RNN Model Variables
mae_x_orientation_RNN_Batch_8 = []
mae_y_orientation_RNN_Batch_8 = []
mae_z_orientation_RNN_Batch_8 = []

mse_x_orientation_RNN_Batch_8 = []
mse_y_orientation_RNN_Batch_8 = []
mse_z_orientation_RNN_Batch_8 = []

# LSTM Model Variables
mae_x_orientation_LSTM_Batch_8 = []
mae_y_orientation_LSTM_Batch_8 = []
mae_z_orientation_LSTM_Batch_8 = []

mse_x_orientation_LSTM_Batch_8 = []
mse_y_orientation_LSTM_Batch_8 = []
mse_z_orientation_LSTM_Batch_8 = []

# GRU Model Variables
mae_x_orientation_GRU_Batch_8 = []
mae_y_orientation_GRU_Batch_8 = []
mae_z_orientation_GRU_Batch_8 = []

mse_x_orientation_GRU_Batch_8 = []
mse_y_orientation_GRU_Batch_8 = []
mse_z_orientation_GRU_Batch_8 = []

# Batch_16

# DNN Model Variables
mae_x_orientation_DNN_Batch_16 = []
mae_y_orientation_DNN_Batch_16 = []
mae_z_orientation_DNN_Batch_16 = []

mse_x_orientation_DNN_Batch_16 = []
mse_y_orientation_DNN_Batch_16 = []
mse_z_orientation_DNN_Batch_16 = []

# CNN Model Variables
mae_x_orientation_CNN_Batch_16 = []
mae_y_orientation_CNN_Batch_16 = []
mae_z_orientation_CNN_Batch_16 = []

mse_x_orientation_CNN_Batch_16 = []
mse_y_orientation_CNN_Batch_16 = []
mse_z_orientation_CNN_Batch_16 = []

# RNN Model Variables
mae_x_orientation_RNN_Batch_16 = []
mae_y_orientation_RNN_Batch_16 = []
mae_z_orientation_RNN_Batch_16 = []

mse_x_orientation_RNN_Batch_16 = []
mse_y_orientation_RNN_Batch_16 = []
mse_z_orientation_RNN_Batch_16 = []

# LSTM Model Variables
mae_x_orientation_LSTM_Batch_16 = []
mae_y_orientation_LSTM_Batch_16 = []
mae_z_orientation_LSTM_Batch_16 = []

mse_x_orientation_LSTM_Batch_16 = []
mse_y_orientation_LSTM_Batch_16 = []
mse_z_orientation_LSTM_Batch_16 = []

# GRU Model Variables
mae_x_orientation_GRU_Batch_16 = []
mae_y_orientation_GRU_Batch_16 = []
mae_z_orientation_GRU_Batch_16 = []

mse_x_orientation_GRU_Batch_16 = []
mse_y_orientation_GRU_Batch_16 = []
mse_z_orientation_GRU_Batch_16 = []

# Batch_32

# DNN Model Variables
mae_x_orientation_DNN_Batch_32 = []
mae_y_orientation_DNN_Batch_32 = []
mae_z_orientation_DNN_Batch_32 = []

mse_x_orientation_DNN_Batch_32 = []
mse_y_orientation_DNN_Batch_32 = []
mse_z_orientation_DNN_Batch_32 = []

# CNN Model Variables
mae_x_orientation_CNN_Batch_32 = []
mae_y_orientation_CNN_Batch_32 = []
mae_z_orientation_CNN_Batch_32 = []

mse_x_orientation_CNN_Batch_32 = []
mse_y_orientation_CNN_Batch_32 = []
mse_z_orientation_CNN_Batch_32 = []

# RNN Model Variables
mae_x_orientation_RNN_Batch_32 = []
mae_y_orientation_RNN_Batch_32 = []
mae_z_orientation_RNN_Batch_32 = []

mse_x_orientation_RNN_Batch_32 = []
mse_y_orientation_RNN_Batch_32 = []
mse_z_orientation_RNN_Batch_32 = []

# LSTM Model Variables
mae_x_orientation_LSTM_Batch_32 = []
mae_y_orientation_LSTM_Batch_32 = []
mae_z_orientation_LSTM_Batch_32 = []

mse_x_orientation_LSTM_Batch_32 = []
mse_y_orientation_LSTM_Batch_32 = []
mse_z_orientation_LSTM_Batch_32 = []

# GRU Model Variables
mae_x_orientation_GRU_Batch_32 = []
mae_y_orientation_GRU_Batch_32 = []
mae_z_orientation_GRU_Batch_32 = []

mse_x_orientation_GRU_Batch_32 = []
mse_y_orientation_GRU_Batch_32 = []
mse_z_orientation_GRU_Batch_32 = []

# Batch_64

# DNN Model Variables
mae_x_orientation_DNN_Batch_64 = []
mae_y_orientation_DNN_Batch_64 = []
mae_z_orientation_DNN_Batch_64 = []

mse_x_orientation_DNN_Batch_64 = []
mse_y_orientation_DNN_Batch_64 = []
mse_z_orientation_DNN_Batch_64 = []

# CNN Model Variables
mae_x_orientation_CNN_Batch_64 = []
mae_y_orientation_CNN_Batch_64 = []
mae_z_orientation_CNN_Batch_64 = []

mse_x_orientation_CNN_Batch_64 = []
mse_y_orientation_CNN_Batch_64 = []
mse_z_orientation_CNN_Batch_64 = []

# RNN Model Variables
mae_x_orientation_RNN_Batch_64 = []
mae_y_orientation_RNN_Batch_64 = []
mae_z_orientation_RNN_Batch_64 = []

mse_x_orientation_RNN_Batch_64 = []
mse_y_orientation_RNN_Batch_64 = []
mse_z_orientation_RNN_Batch_64 = []

# LSTM Model Variables
mae_x_orientation_LSTM_Batch_64 = []
mae_y_orientation_LSTM_Batch_64 = []
mae_z_orientation_LSTM_Batch_64 = []

mse_x_orientation_LSTM_Batch_64 = []
mse_y_orientation_LSTM_Batch_64 = []
mse_z_orientation_LSTM_Batch_64 = []

# GRU Model Variables
mae_x_orientation_GRU_Batch_64 = []
mae_y_orientation_GRU_Batch_64 = []
mae_z_orientation_GRU_Batch_64 = []

mse_x_orientation_GRU_Batch_64 = []
mse_y_orientation_GRU_Batch_64 = []
mse_z_orientation_GRU_Batch_64 = []

# Batch_128

# DNN Model Variables
mae_x_orientation_DNN_Batch_128 = []
mae_y_orientation_DNN_Batch_128 = []
mae_z_orientation_DNN_Batch_128 = []

mse_x_orientation_DNN_Batch_128 = []
mse_y_orientation_DNN_Batch_128 = []
mse_z_orientation_DNN_Batch_128 = []

# CNN Model Variables
mae_x_orientation_CNN_Batch_128 = []
mae_y_orientation_CNN_Batch_128 = []
mae_z_orientation_CNN_Batch_128 = []

mse_x_orientation_CNN_Batch_128 = []
mse_y_orientation_CNN_Batch_128 = []
mse_z_orientation_CNN_Batch_128 = []

# RNN Model Variables
mae_x_orientation_RNN_Batch_128 = []
mae_y_orientation_RNN_Batch_128 = []
mae_z_orientation_RNN_Batch_128 = []

mse_x_orientation_RNN_Batch_128 = []
mse_y_orientation_RNN_Batch_128 = []
mse_z_orientation_RNN_Batch_128 = []

# LSTM Model Variables
mae_x_orientation_LSTM_Batch_128 = []
mae_y_orientation_LSTM_Batch_128 = []
mae_z_orientation_LSTM_Batch_128 = []

mse_x_orientation_LSTM_Batch_128 = []
mse_y_orientation_LSTM_Batch_128 = []
mse_z_orientation_LSTM_Batch_128 = []

# GRU Model Variables
mae_x_orientation_GRU_Batch_128 = []
mae_y_orientation_GRU_Batch_128 = []
mae_z_orientation_GRU_Batch_128 = []

mse_x_orientation_GRU_Batch_128 = []
mse_y_orientation_GRU_Batch_128 = []
mse_z_orientation_GRU_Batch_128 = []

# Batch_256

# DNN Model Variables
mae_x_orientation_DNN_Batch_256 = []
mae_y_orientation_DNN_Batch_256 = []
mae_z_orientation_DNN_Batch_256 = []

mse_x_orientation_DNN_Batch_256 = []
mse_y_orientation_DNN_Batch_256 = []
mse_z_orientation_DNN_Batch_256 = []

# CNN Model Variables
mae_x_orientation_CNN_Batch_256 = []
mae_y_orientation_CNN_Batch_256 = []
mae_z_orientation_CNN_Batch_256 = []

mse_x_orientation_CNN_Batch_256 = []
mse_y_orientation_CNN_Batch_256 = []
mse_z_orientation_CNN_Batch_256 = []

# RNN Model Variables
mae_x_orientation_RNN_Batch_256 = []
mae_y_orientation_RNN_Batch_256 = []
mae_z_orientation_RNN_Batch_256 = []

mse_x_orientation_RNN_Batch_256 = []
mse_y_orientation_RNN_Batch_256 = []
mse_z_orientation_RNN_Batch_256 = []

# LSTM Model Variables
mae_x_orientation_LSTM_Batch_256 = []
mae_y_orientation_LSTM_Batch_256 = []
mae_z_orientation_LSTM_Batch_256 = []

mse_x_orientation_LSTM_Batch_256 = []
mse_y_orientation_LSTM_Batch_256 = []
mse_z_orientation_LSTM_Batch_256 = []

# GRU Model Variables
mae_x_orientation_GRU_Batch_256 = []
mae_y_orientation_GRU_Batch_256 = []
mae_z_orientation_GRU_Batch_256 = []

mse_x_orientation_GRU_Batch_256 = []
mse_y_orientation_GRU_Batch_256 = []
mse_z_orientation_GRU_Batch_256 = []

# Batch_512

# DNN Model Variables
mae_x_orientation_DNN_Batch_512 = []
mae_y_orientation_DNN_Batch_512 = []
mae_z_orientation_DNN_Batch_512 = []

mse_x_orientation_DNN_Batch_512 = []
mse_y_orientation_DNN_Batch_512 = []
mse_z_orientation_DNN_Batch_512 = []

# CNN Model Variables
mae_x_orientation_CNN_Batch_512 = []
mae_y_orientation_CNN_Batch_512 = []
mae_z_orientation_CNN_Batch_512 = []

mse_x_orientation_CNN_Batch_512 = []
mse_y_orientation_CNN_Batch_512 = []
mse_z_orientation_CNN_Batch_512 = []

# RNN Model Variables
mae_x_orientation_RNN_Batch_512 = []
mae_y_orientation_RNN_Batch_512 = []
mae_z_orientation_RNN_Batch_512 = []

mse_x_orientation_RNN_Batch_512 = []
mse_y_orientation_RNN_Batch_512 = []
mse_z_orientation_RNN_Batch_512 = []

# LSTM Model Variables
mae_x_orientation_LSTM_Batch_512 = []
mae_y_orientation_LSTM_Batch_512 = []
mae_z_orientation_LSTM_Batch_512 = []

mse_x_orientation_LSTM_Batch_512 = []
mse_y_orientation_LSTM_Batch_512 = []
mse_z_orientation_LSTM_Batch_512 = []

# GRU Model Variables
mae_x_orientation_GRU_Batch_512 = []
mae_y_orientation_GRU_Batch_512 = []
mae_z_orientation_GRU_Batch_512 = []

mse_x_orientation_GRU_Batch_512 = []
mse_y_orientation_GRU_Batch_512 = []
mse_z_orientation_GRU_Batch_512 = []

totalData = len(fileName)
for i in range(totalData):

    # Batch_8

    # DNN Model
    if "Loss_mse_DNN" in fileName[i] and "Batch_8" in fileName[i]:
        mae_x_orientation_DNN_Batch_8.append(mae_x_orientation[i])
        mae_y_orientation_DNN_Batch_8.append(mae_y_orientation[i])
        mae_z_orientation_DNN_Batch_8.append(mae_z_orientation[i])

        mse_x_orientation_DNN_Batch_8.append(mse_x_orientation[i])
        mse_y_orientation_DNN_Batch_8.append(mse_y_orientation[i])
        mse_z_orientation_DNN_Batch_8.append(mse_z_orientation[i])

    # CNN Model
    elif "Loss_mse_CNN" in fileName[i] and "Batch_8" in fileName[i]:
        mae_x_orientation_CNN_Batch_8.append(mae_x_orientation[i])
        mae_y_orientation_CNN_Batch_8.append(mae_y_orientation[i])
        mae_z_orientation_CNN_Batch_8.append(mae_z_orientation[i])

        mse_x_orientation_CNN_Batch_8.append(mse_x_orientation[i])
        mse_y_orientation_CNN_Batch_8.append(mse_y_orientation[i])
        mse_z_orientation_CNN_Batch_8.append(mse_z_orientation[i])

    # RNN Model
    elif "Loss_mse_RNN" in fileName[i] and "Batch_8" in fileName[i]:
        mae_x_orientation_RNN_Batch_8.append(mae_x_orientation[i])
        mae_y_orientation_RNN_Batch_8.append(mae_y_orientation[i])
        mae_z_orientation_RNN_Batch_8.append(mae_z_orientation[i])

        mse_x_orientation_RNN_Batch_8.append(mse_x_orientation[i])
        mse_y_orientation_RNN_Batch_8.append(mse_y_orientation[i])
        mse_z_orientation_RNN_Batch_8.append(mse_z_orientation[i])

    # LSTM Model
    elif "Loss_mse_LSTM" in fileName[i] and "Batch_8" in fileName[i]:
        mae_x_orientation_LSTM_Batch_8.append(mae_x_orientation[i])
        mae_y_orientation_LSTM_Batch_8.append(mae_y_orientation[i])
        mae_z_orientation_LSTM_Batch_8.append(mae_z_orientation[i])

        mse_x_orientation_LSTM_Batch_8.append(mse_x_orientation[i])
        mse_y_orientation_LSTM_Batch_8.append(mse_y_orientation[i])
        mse_z_orientation_LSTM_Batch_8.append(mse_z_orientation[i])

    # GRU Model
    elif "Loss_mse_GRU" in fileName[i] and "Batch_8" in fileName[i]:
        mae_x_orientation_GRU_Batch_8.append(mae_x_orientation[i])
        mae_y_orientation_GRU_Batch_8.append(mae_y_orientation[i])
        mae_z_orientation_GRU_Batch_8.append(mae_z_orientation[i])

        mse_x_orientation_GRU_Batch_8.append(mse_x_orientation[i])
        mse_y_orientation_GRU_Batch_8.append(mse_y_orientation[i])
        mse_z_orientation_GRU_Batch_8.append(mse_z_orientation[i])

    # Batch_16

    # DNN Model
    if "Loss_mse_DNN" in fileName[i] and "Batch_16" in fileName[i]:
        mae_x_orientation_DNN_Batch_16.append(mae_x_orientation[i])
        mae_y_orientation_DNN_Batch_16.append(mae_y_orientation[i])
        mae_z_orientation_DNN_Batch_16.append(mae_z_orientation[i])

        mse_x_orientation_DNN_Batch_16.append(mse_x_orientation[i])
        mse_y_orientation_DNN_Batch_16.append(mse_y_orientation[i])
        mse_z_orientation_DNN_Batch_16.append(mse_z_orientation[i])

    # CNN Model
    elif "Loss_mse_CNN" in fileName[i] and "Batch_16" in fileName[i]:
        mae_x_orientation_CNN_Batch_16.append(mae_x_orientation[i])
        mae_y_orientation_CNN_Batch_16.append(mae_y_orientation[i])
        mae_z_orientation_CNN_Batch_16.append(mae_z_orientation[i])

        mse_x_orientation_CNN_Batch_16.append(mse_x_orientation[i])
        mse_y_orientation_CNN_Batch_16.append(mse_y_orientation[i])
        mse_z_orientation_CNN_Batch_16.append(mse_z_orientation[i])

    # RNN Model
    elif "Loss_mse_RNN" in fileName[i] and "Batch_16" in fileName[i]:
        mae_x_orientation_RNN_Batch_16.append(mae_x_orientation[i])
        mae_y_orientation_RNN_Batch_16.append(mae_y_orientation[i])
        mae_z_orientation_RNN_Batch_16.append(mae_z_orientation[i])

        mse_x_orientation_RNN_Batch_16.append(mse_x_orientation[i])
        mse_y_orientation_RNN_Batch_16.append(mse_y_orientation[i])
        mse_z_orientation_RNN_Batch_16.append(mse_z_orientation[i])

    # LSTM Model
    elif "Loss_mse_LSTM" in fileName[i] and "Batch_16" in fileName[i]:
        mae_x_orientation_LSTM_Batch_16.append(mae_x_orientation[i])
        mae_y_orientation_LSTM_Batch_16.append(mae_y_orientation[i])
        mae_z_orientation_LSTM_Batch_16.append(mae_z_orientation[i])

        mse_x_orientation_LSTM_Batch_16.append(mse_x_orientation[i])
        mse_y_orientation_LSTM_Batch_16.append(mse_y_orientation[i])
        mse_z_orientation_LSTM_Batch_16.append(mse_z_orientation[i])

    # GRU Model
    elif "Loss_mse_GRU" in fileName[i] and "Batch_16" in fileName[i]:
        mae_x_orientation_GRU_Batch_16.append(mae_x_orientation[i])
        mae_y_orientation_GRU_Batch_16.append(mae_y_orientation[i])
        mae_z_orientation_GRU_Batch_16.append(mae_z_orientation[i])

        mse_x_orientation_GRU_Batch_16.append(mse_x_orientation[i])
        mse_y_orientation_GRU_Batch_16.append(mse_y_orientation[i])
        mse_z_orientation_GRU_Batch_16.append(mse_z_orientation[i])

    # Batch_32

    # DNN Model
    if "Loss_mse_DNN" in fileName[i] and "Batch_32" in fileName[i]:
        mae_x_orientation_DNN_Batch_32.append(mae_x_orientation[i])
        mae_y_orientation_DNN_Batch_32.append(mae_y_orientation[i])
        mae_z_orientation_DNN_Batch_32.append(mae_z_orientation[i])

        mse_x_orientation_DNN_Batch_32.append(mse_x_orientation[i])
        mse_y_orientation_DNN_Batch_32.append(mse_y_orientation[i])
        mse_z_orientation_DNN_Batch_32.append(mse_z_orientation[i])

    # CNN Model
    elif "Loss_mse_CNN" in fileName[i] and "Batch_32" in fileName[i]:
        mae_x_orientation_CNN_Batch_32.append(mae_x_orientation[i])
        mae_y_orientation_CNN_Batch_32.append(mae_y_orientation[i])
        mae_z_orientation_CNN_Batch_32.append(mae_z_orientation[i])

        mse_x_orientation_CNN_Batch_32.append(mse_x_orientation[i])
        mse_y_orientation_CNN_Batch_32.append(mse_y_orientation[i])
        mse_z_orientation_CNN_Batch_32.append(mse_z_orientation[i])

    # RNN Model
    elif "Loss_mse_RNN" in fileName[i] and "Batch_32" in fileName[i]:
        mae_x_orientation_RNN_Batch_32.append(mae_x_orientation[i])
        mae_y_orientation_RNN_Batch_32.append(mae_y_orientation[i])
        mae_z_orientation_RNN_Batch_32.append(mae_z_orientation[i])

        mse_x_orientation_RNN_Batch_32.append(mse_x_orientation[i])
        mse_y_orientation_RNN_Batch_32.append(mse_y_orientation[i])
        mse_z_orientation_RNN_Batch_32.append(mse_z_orientation[i])

    # LSTM Model
    elif "Loss_mse_LSTM" in fileName[i] and "Batch_32" in fileName[i]:
        mae_x_orientation_LSTM_Batch_32.append(mae_x_orientation[i])
        mae_y_orientation_LSTM_Batch_32.append(mae_y_orientation[i])
        mae_z_orientation_LSTM_Batch_32.append(mae_z_orientation[i])

        mse_x_orientation_LSTM_Batch_32.append(mse_x_orientation[i])
        mse_y_orientation_LSTM_Batch_32.append(mse_y_orientation[i])
        mse_z_orientation_LSTM_Batch_32.append(mse_z_orientation[i])

    # GRU Model
    elif "Loss_mse_GRU" in fileName[i] and "Batch_32" in fileName[i]:
        mae_x_orientation_GRU_Batch_32.append(mae_x_orientation[i])
        mae_y_orientation_GRU_Batch_32.append(mae_y_orientation[i])
        mae_z_orientation_GRU_Batch_32.append(mae_z_orientation[i])

        mse_x_orientation_GRU_Batch_32.append(mse_x_orientation[i])
        mse_y_orientation_GRU_Batch_32.append(mse_y_orientation[i])
        mse_z_orientation_GRU_Batch_32.append(mse_z_orientation[i])

    # Batch_64

    # DNN Model
    if "Loss_mse_DNN" in fileName[i] and "Batch_64" in fileName[i]:
        mae_x_orientation_DNN_Batch_64.append(mae_x_orientation[i])
        mae_y_orientation_DNN_Batch_64.append(mae_y_orientation[i])
        mae_z_orientation_DNN_Batch_64.append(mae_z_orientation[i])

        mse_x_orientation_DNN_Batch_64.append(mse_x_orientation[i])
        mse_y_orientation_DNN_Batch_64.append(mse_y_orientation[i])
        mse_z_orientation_DNN_Batch_64.append(mse_z_orientation[i])

    # CNN Model
    elif "Loss_mse_CNN" in fileName[i] and "Batch_64" in fileName[i]:
        mae_x_orientation_CNN_Batch_64.append(mae_x_orientation[i])
        mae_y_orientation_CNN_Batch_64.append(mae_y_orientation[i])
        mae_z_orientation_CNN_Batch_64.append(mae_z_orientation[i])

        mse_x_orientation_CNN_Batch_64.append(mse_x_orientation[i])
        mse_y_orientation_CNN_Batch_64.append(mse_y_orientation[i])
        mse_z_orientation_CNN_Batch_64.append(mse_z_orientation[i])

    # RNN Model
    elif "Loss_mse_RNN" in fileName[i] and "Batch_64" in fileName[i]:
        mae_x_orientation_RNN_Batch_64.append(mae_x_orientation[i])
        mae_y_orientation_RNN_Batch_64.append(mae_y_orientation[i])
        mae_z_orientation_RNN_Batch_64.append(mae_z_orientation[i])

        mse_x_orientation_RNN_Batch_64.append(mse_x_orientation[i])
        mse_y_orientation_RNN_Batch_64.append(mse_y_orientation[i])
        mse_z_orientation_RNN_Batch_64.append(mse_z_orientation[i])

    # LSTM Model
    elif "Loss_mse_LSTM" in fileName[i] and "Batch_64" in fileName[i]:
        mae_x_orientation_LSTM_Batch_64.append(mae_x_orientation[i])
        mae_y_orientation_LSTM_Batch_64.append(mae_y_orientation[i])
        mae_z_orientation_LSTM_Batch_64.append(mae_z_orientation[i])

        mse_x_orientation_LSTM_Batch_64.append(mse_x_orientation[i])
        mse_y_orientation_LSTM_Batch_64.append(mse_y_orientation[i])
        mse_z_orientation_LSTM_Batch_64.append(mse_z_orientation[i])

    # GRU Model
    elif "Loss_mse_GRU" in fileName[i] and "Batch_64" in fileName[i]:
        mae_x_orientation_GRU_Batch_64.append(mae_x_orientation[i])
        mae_y_orientation_GRU_Batch_64.append(mae_y_orientation[i])
        mae_z_orientation_GRU_Batch_64.append(mae_z_orientation[i])

        mse_x_orientation_GRU_Batch_64.append(mse_x_orientation[i])
        mse_y_orientation_GRU_Batch_64.append(mse_y_orientation[i])
        mse_z_orientation_GRU_Batch_64.append(mse_z_orientation[i])

    # Batch_128

    # DNN Model
    if "Loss_mse_DNN" in fileName[i] and "Batch_128" in fileName[i]:
        mae_x_orientation_DNN_Batch_128.append(mae_x_orientation[i])
        mae_y_orientation_DNN_Batch_128.append(mae_y_orientation[i])
        mae_z_orientation_DNN_Batch_128.append(mae_z_orientation[i])

        mse_x_orientation_DNN_Batch_128.append(mse_x_orientation[i])
        mse_y_orientation_DNN_Batch_128.append(mse_y_orientation[i])
        mse_z_orientation_DNN_Batch_128.append(mse_z_orientation[i])

    # CNN Model
    elif "Loss_mse_CNN" in fileName[i] and "Batch_128" in fileName[i]:
        mae_x_orientation_CNN_Batch_128.append(mae_x_orientation[i])
        mae_y_orientation_CNN_Batch_128.append(mae_y_orientation[i])
        mae_z_orientation_CNN_Batch_128.append(mae_z_orientation[i])

        mse_x_orientation_CNN_Batch_128.append(mse_x_orientation[i])
        mse_y_orientation_CNN_Batch_128.append(mse_y_orientation[i])
        mse_z_orientation_CNN_Batch_128.append(mse_z_orientation[i])

    # RNN Model
    elif "Loss_mse_RNN" in fileName[i] and "Batch_128" in fileName[i]:
        mae_x_orientation_RNN_Batch_128.append(mae_x_orientation[i])
        mae_y_orientation_RNN_Batch_128.append(mae_y_orientation[i])
        mae_z_orientation_RNN_Batch_128.append(mae_z_orientation[i])

        mse_x_orientation_RNN_Batch_128.append(mse_x_orientation[i])
        mse_y_orientation_RNN_Batch_128.append(mse_y_orientation[i])
        mse_z_orientation_RNN_Batch_128.append(mse_z_orientation[i])

    # LSTM Model
    elif "Loss_mse_LSTM" in fileName[i] and "Batch_128" in fileName[i]:
        mae_x_orientation_LSTM_Batch_128.append(mae_x_orientation[i])
        mae_y_orientation_LSTM_Batch_128.append(mae_y_orientation[i])
        mae_z_orientation_LSTM_Batch_128.append(mae_z_orientation[i])

        mse_x_orientation_LSTM_Batch_128.append(mse_x_orientation[i])
        mse_y_orientation_LSTM_Batch_128.append(mse_y_orientation[i])
        mse_z_orientation_LSTM_Batch_128.append(mse_z_orientation[i])

    # GRU Model
    elif "Loss_mse_GRU" in fileName[i] and "Batch_128" in fileName[i]:
        mae_x_orientation_GRU_Batch_128.append(mae_x_orientation[i])
        mae_y_orientation_GRU_Batch_128.append(mae_y_orientation[i])
        mae_z_orientation_GRU_Batch_128.append(mae_z_orientation[i])

        mse_x_orientation_GRU_Batch_128.append(mse_x_orientation[i])
        mse_y_orientation_GRU_Batch_128.append(mse_y_orientation[i])
        mse_z_orientation_GRU_Batch_128.append(mse_z_orientation[i])

    # Batch_256

    # DNN Model
    if "Loss_mse_DNN" in fileName[i] and "Batch_256" in fileName[i]:
        mae_x_orientation_DNN_Batch_256.append(mae_x_orientation[i])
        mae_y_orientation_DNN_Batch_256.append(mae_y_orientation[i])
        mae_z_orientation_DNN_Batch_256.append(mae_z_orientation[i])

        mse_x_orientation_DNN_Batch_256.append(mse_x_orientation[i])
        mse_y_orientation_DNN_Batch_256.append(mse_y_orientation[i])
        mse_z_orientation_DNN_Batch_256.append(mse_z_orientation[i])

    # CNN Model
    elif "Loss_mse_CNN" in fileName[i] and "Batch_256" in fileName[i]:
        mae_x_orientation_CNN_Batch_256.append(mae_x_orientation[i])
        mae_y_orientation_CNN_Batch_256.append(mae_y_orientation[i])
        mae_z_orientation_CNN_Batch_256.append(mae_z_orientation[i])

        mse_x_orientation_CNN_Batch_256.append(mse_x_orientation[i])
        mse_y_orientation_CNN_Batch_256.append(mse_y_orientation[i])
        mse_z_orientation_CNN_Batch_256.append(mse_z_orientation[i])

    # RNN Model
    elif "Loss_mse_RNN" in fileName[i] and "Batch_256" in fileName[i]:
        mae_x_orientation_RNN_Batch_256.append(mae_x_orientation[i])
        mae_y_orientation_RNN_Batch_256.append(mae_y_orientation[i])
        mae_z_orientation_RNN_Batch_256.append(mae_z_orientation[i])

        mse_x_orientation_RNN_Batch_256.append(mse_x_orientation[i])
        mse_y_orientation_RNN_Batch_256.append(mse_y_orientation[i])
        mse_z_orientation_RNN_Batch_256.append(mse_z_orientation[i])

    # LSTM Model
    elif "Loss_mse_LSTM" in fileName[i] and "Batch_256" in fileName[i]:
        mae_x_orientation_LSTM_Batch_256.append(mae_x_orientation[i])
        mae_y_orientation_LSTM_Batch_256.append(mae_y_orientation[i])
        mae_z_orientation_LSTM_Batch_256.append(mae_z_orientation[i])

        mse_x_orientation_LSTM_Batch_256.append(mse_x_orientation[i])
        mse_y_orientation_LSTM_Batch_256.append(mse_y_orientation[i])
        mse_z_orientation_LSTM_Batch_256.append(mse_z_orientation[i])

    # GRU Model
    elif "Loss_mse_GRU" in fileName[i] and "Batch_256" in fileName[i]:
        mae_x_orientation_GRU_Batch_256.append(mae_x_orientation[i])
        mae_y_orientation_GRU_Batch_256.append(mae_y_orientation[i])
        mae_z_orientation_GRU_Batch_256.append(mae_z_orientation[i])

        mse_x_orientation_GRU_Batch_256.append(mse_x_orientation[i])
        mse_y_orientation_GRU_Batch_256.append(mse_y_orientation[i])
        mse_z_orientation_GRU_Batch_256.append(mse_z_orientation[i])

    # Batch_512

    # DNN Model
    if "Loss_mse_DNN" in fileName[i] and "Batch_512" in fileName[i]:
        mae_x_orientation_DNN_Batch_512.append(mae_x_orientation[i])
        mae_y_orientation_DNN_Batch_512.append(mae_y_orientation[i])
        mae_z_orientation_DNN_Batch_512.append(mae_z_orientation[i])

        mse_x_orientation_DNN_Batch_512.append(mse_x_orientation[i])
        mse_y_orientation_DNN_Batch_512.append(mse_y_orientation[i])
        mse_z_orientation_DNN_Batch_512.append(mse_z_orientation[i])

    # CNN Model
    elif "Loss_mse_CNN" in fileName[i] and "Batch_512" in fileName[i]:
        mae_x_orientation_CNN_Batch_512.append(mae_x_orientation[i])
        mae_y_orientation_CNN_Batch_512.append(mae_y_orientation[i])
        mae_z_orientation_CNN_Batch_512.append(mae_z_orientation[i])

        mse_x_orientation_CNN_Batch_512.append(mse_x_orientation[i])
        mse_y_orientation_CNN_Batch_512.append(mse_y_orientation[i])
        mse_z_orientation_CNN_Batch_512.append(mse_z_orientation[i])

    # RNN Model
    elif "Loss_mse_RNN" in fileName[i] and "Batch_512" in fileName[i]:
        mae_x_orientation_RNN_Batch_512.append(mae_x_orientation[i])
        mae_y_orientation_RNN_Batch_512.append(mae_y_orientation[i])
        mae_z_orientation_RNN_Batch_512.append(mae_z_orientation[i])

        mse_x_orientation_RNN_Batch_512.append(mse_x_orientation[i])
        mse_y_orientation_RNN_Batch_512.append(mse_y_orientation[i])
        mse_z_orientation_RNN_Batch_512.append(mse_z_orientation[i])

    # LSTM Model
    elif "Loss_mse_LSTM" in fileName[i] and "Batch_512" in fileName[i]:
        mae_x_orientation_LSTM_Batch_512.append(mae_x_orientation[i])
        mae_y_orientation_LSTM_Batch_512.append(mae_y_orientation[i])
        mae_z_orientation_LSTM_Batch_512.append(mae_z_orientation[i])

        mse_x_orientation_LSTM_Batch_512.append(mse_x_orientation[i])
        mse_y_orientation_LSTM_Batch_512.append(mse_y_orientation[i])
        mse_z_orientation_LSTM_Batch_512.append(mse_z_orientation[i])

    # GRU Model
    elif "Loss_mse_GRU" in fileName[i] and "Batch_512" in fileName[i]:
        mae_x_orientation_GRU_Batch_512.append(mae_x_orientation[i])
        mae_y_orientation_GRU_Batch_512.append(mae_y_orientation[i])
        mae_z_orientation_GRU_Batch_512.append(mae_z_orientation[i])

        mse_x_orientation_GRU_Batch_512.append(mse_x_orientation[i])
        mse_y_orientation_GRU_Batch_512.append(mse_y_orientation[i])
        mse_z_orientation_GRU_Batch_512.append(mse_z_orientation[i])


plt.clf()

currentTime = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) # not needed
figurePath = ".\\SavedFigure\\BatchSize-8-16-32-64-128-256\\MSE_of_Orientation_in_Testing_Data\\"

if not os.path.exists(figurePath):
    os.makedirs(figurePath)

# print("mse_x_orientation_DNN_Batch_8: ", mse_x_orientation_DNN_Batch_8)
# print("mse_x_orientation_DNN_Batch_16: ", mse_x_orientation_DNN_Batch_16)
# print("mse_x_orientation_DNN_Batch_32: ", mse_x_orientation_DNN_Batch_32)
# print("mse_x_orientation_DNN_Batch_64: ", mse_x_orientation_DNN_Batch_64)
# print("mse_x_orientation_DNN_Batch_128: ", mse_x_orientation_DNN_Batch_128)
# print("mse_x_orientation_DNN_Batch_256: ", mse_x_orientation_DNN_Batch_256)
# print("mse_x_orientation_DNN_Batch_512: ", mse_x_orientation_DNN_Batch_512)

# Batch_Size

# DNN Model

ymin = -0.001
ymax = 0.015

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_X_DNN
plt.title("DNN Model in X axis")
plt.plot(mse_x_orientation_DNN_Batch_8, marker='o', color='cyan', linestyle='None')
plt.plot(mse_x_orientation_DNN_Batch_16, marker='o', color='red', linestyle='None')
plt.plot(mse_x_orientation_DNN_Batch_32, marker='o', color='blue', linestyle='None')
plt.plot(mse_x_orientation_DNN_Batch_64, marker='o', color='green', linestyle='None')
plt.plot(mse_x_orientation_DNN_Batch_128, marker='o', color='black', linestyle='None')
plt.plot(mse_x_orientation_DNN_Batch_256, marker='o', color='magenta', linestyle='None')
plt.plot(mse_x_orientation_DNN_Batch_512, marker='o', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_X_DNN")
plt.show()

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Y_DNN
plt.title("DNN Model in Y axis")
plt.plot(mse_y_orientation_DNN_Batch_8, marker='^', color='cyan', linestyle='None')
plt.plot(mse_y_orientation_DNN_Batch_16, marker='^', color='red', linestyle='None')
plt.plot(mse_y_orientation_DNN_Batch_32, marker='^', color='blue', linestyle='None')
plt.plot(mse_y_orientation_DNN_Batch_64, marker='^', color='green', linestyle='None')
plt.plot(mse_y_orientation_DNN_Batch_128, marker='^', color='black', linestyle='None')
plt.plot(mse_y_orientation_DNN_Batch_256, marker='^', color='magenta', linestyle='None')
plt.plot(mse_y_orientation_DNN_Batch_512, marker='^', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Y_DNN")
plt.show()

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Z_DNN
plt.title("DNN Model in Z axis")
plt.plot(mse_z_orientation_DNN_Batch_8, marker='*', color='cyan', linestyle='None')
plt.plot(mse_z_orientation_DNN_Batch_16, marker='*', color='red', linestyle='None')
plt.plot(mse_z_orientation_DNN_Batch_32, marker='*', color='blue', linestyle='None')
plt.plot(mse_z_orientation_DNN_Batch_64, marker='*', color='green', linestyle='None')
plt.plot(mse_z_orientation_DNN_Batch_128, marker='*', color='black', linestyle='None')
plt.plot(mse_z_orientation_DNN_Batch_256, marker='*', color='magenta', linestyle='None')
plt.plot(mse_z_orientation_DNN_Batch_512, marker='*', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Z_DNN")
plt.show()

# CNN Model

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_X_CNN
plt.title("CNN Model in X axis")
plt.plot(mse_x_orientation_CNN_Batch_8, marker='o', color='cyan', linestyle='None')
plt.plot(mse_x_orientation_CNN_Batch_16, marker='o', color='red', linestyle='None')
plt.plot(mse_x_orientation_CNN_Batch_32, marker='o', color='blue', linestyle='None')
plt.plot(mse_x_orientation_CNN_Batch_64, marker='o', color='green', linestyle='None')
plt.plot(mse_x_orientation_CNN_Batch_128, marker='o', color='black', linestyle='None')
plt.plot(mse_x_orientation_CNN_Batch_256, marker='o', color='magenta', linestyle='None')
plt.plot(mse_x_orientation_CNN_Batch_512, marker='o', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_X_CNN")
plt.show()

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Y_CNN
plt.title("CNN Model in Y axis")
plt.plot(mse_y_orientation_CNN_Batch_8, marker='^', color='cyan', linestyle='None')
plt.plot(mse_y_orientation_CNN_Batch_16, marker='^', color='red', linestyle='None')
plt.plot(mse_y_orientation_CNN_Batch_32, marker='^', color='blue', linestyle='None')
plt.plot(mse_y_orientation_CNN_Batch_64, marker='^', color='green', linestyle='None')
plt.plot(mse_y_orientation_CNN_Batch_128, marker='^', color='black', linestyle='None')
plt.plot(mse_y_orientation_CNN_Batch_256, marker='^', color='magenta', linestyle='None')
plt.plot(mse_y_orientation_CNN_Batch_512, marker='^', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Y_CNN")
plt.show()

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Z_CNN
plt.title("CNN Model in Z axis")
plt.plot(mse_z_orientation_CNN_Batch_8, marker='*', color='cyan', linestyle='None')
plt.plot(mse_z_orientation_CNN_Batch_16, marker='*', color='red', linestyle='None')
plt.plot(mse_z_orientation_CNN_Batch_32, marker='*', color='blue', linestyle='None')
plt.plot(mse_z_orientation_CNN_Batch_64, marker='*', color='green', linestyle='None')
plt.plot(mse_z_orientation_CNN_Batch_128, marker='*', color='black', linestyle='None')
plt.plot(mse_z_orientation_CNN_Batch_256, marker='*', color='magenta', linestyle='None')
plt.plot(mse_z_orientation_CNN_Batch_512, marker='*', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Z_CNN")
plt.show()

# RNN Model

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_X_RNN
plt.title("RNN Model in X axis")
plt.plot(mse_x_orientation_RNN_Batch_8, marker='o', color='cyan', linestyle='None')
plt.plot(mse_x_orientation_RNN_Batch_16, marker='o', color='red', linestyle='None')
plt.plot(mse_x_orientation_RNN_Batch_32, marker='o', color='blue', linestyle='None')
plt.plot(mse_x_orientation_RNN_Batch_64, marker='o', color='green', linestyle='None')
plt.plot(mse_x_orientation_RNN_Batch_128, marker='o', color='black', linestyle='None')
plt.plot(mse_x_orientation_RNN_Batch_256, marker='o', color='magenta', linestyle='None')
plt.plot(mse_x_orientation_RNN_Batch_512, marker='o', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_X_RNN")
plt.show()

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Y_RNN
plt.title("RNN Model in Y axis")
plt.plot(mse_y_orientation_RNN_Batch_8, marker='^', color='cyan', linestyle='None')
plt.plot(mse_y_orientation_RNN_Batch_16, marker='^', color='red', linestyle='None')
plt.plot(mse_y_orientation_RNN_Batch_32, marker='^', color='blue', linestyle='None')
plt.plot(mse_y_orientation_RNN_Batch_64, marker='^', color='green', linestyle='None')
plt.plot(mse_y_orientation_RNN_Batch_128, marker='^', color='black', linestyle='None')
plt.plot(mse_y_orientation_RNN_Batch_256, marker='^', color='magenta', linestyle='None')
plt.plot(mse_y_orientation_RNN_Batch_512, marker='^', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Y_RNN")
plt.show()

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Z_RNN
plt.title("RNN Model in Z axis")
plt.plot(mse_z_orientation_RNN_Batch_8, marker='*', color='cyan', linestyle='None')
plt.plot(mse_z_orientation_RNN_Batch_16, marker='*', color='red', linestyle='None')
plt.plot(mse_z_orientation_RNN_Batch_32, marker='*', color='blue', linestyle='None')
plt.plot(mse_z_orientation_RNN_Batch_64, marker='*', color='green', linestyle='None')
plt.plot(mse_z_orientation_RNN_Batch_128, marker='*', color='black', linestyle='None')
plt.plot(mse_z_orientation_RNN_Batch_256, marker='*', color='magenta', linestyle='None')
plt.plot(mse_z_orientation_RNN_Batch_512, marker='*', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Z_RNN")
plt.show()

# LSTM Model

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_X_LSTM
plt.title("LSTM Model in X axis")
plt.plot(mse_x_orientation_LSTM_Batch_8, marker='o', color='cyan', linestyle='None')
plt.plot(mse_x_orientation_LSTM_Batch_16, marker='o', color='red', linestyle='None')
plt.plot(mse_x_orientation_LSTM_Batch_32, marker='o', color='blue', linestyle='None')
plt.plot(mse_x_orientation_LSTM_Batch_64, marker='o', color='green', linestyle='None')
plt.plot(mse_x_orientation_LSTM_Batch_128, marker='o', color='black', linestyle='None')
plt.plot(mse_x_orientation_LSTM_Batch_256, marker='o', color='magenta', linestyle='None')
plt.plot(mse_x_orientation_LSTM_Batch_512, marker='o', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_X_LSTM")
plt.show()

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Y_LSTM
plt.title("LSTM Model in Y axis")
plt.plot(mse_y_orientation_LSTM_Batch_8, marker='^', color='cyan', linestyle='None')
plt.plot(mse_y_orientation_LSTM_Batch_16, marker='^', color='red', linestyle='None')
plt.plot(mse_y_orientation_LSTM_Batch_32, marker='^', color='blue', linestyle='None')
plt.plot(mse_y_orientation_LSTM_Batch_64, marker='^', color='green', linestyle='None')
plt.plot(mse_y_orientation_LSTM_Batch_128, marker='^', color='black', linestyle='None')
plt.plot(mse_y_orientation_LSTM_Batch_256, marker='^', color='magenta', linestyle='None')
plt.plot(mse_y_orientation_LSTM_Batch_512, marker='^', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Y_LSTM")
plt.show()

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Z_LSTM
plt.title("LSTM Model in Z axis")
plt.plot(mse_z_orientation_LSTM_Batch_8, marker='*', color='cyan', linestyle='None')
plt.plot(mse_z_orientation_LSTM_Batch_16, marker='*', color='red', linestyle='None')
plt.plot(mse_z_orientation_LSTM_Batch_32, marker='*', color='blue', linestyle='None')
plt.plot(mse_z_orientation_LSTM_Batch_64, marker='*', color='green', linestyle='None')
plt.plot(mse_z_orientation_LSTM_Batch_128, marker='*', color='black', linestyle='None')
plt.plot(mse_z_orientation_LSTM_Batch_256, marker='*', color='magenta', linestyle='None')
plt.plot(mse_z_orientation_LSTM_Batch_512, marker='*', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Z_LSTM")
plt.show()

# GRU Model

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_X_GRU
plt.title("GRU Model in X axis")
plt.plot(mse_x_orientation_GRU_Batch_8, marker='o', color='cyan', linestyle='None')
plt.plot(mse_x_orientation_GRU_Batch_16, marker='o', color='red', linestyle='None')
plt.plot(mse_x_orientation_GRU_Batch_32, marker='o', color='blue', linestyle='None')
plt.plot(mse_x_orientation_GRU_Batch_64, marker='o', color='green', linestyle='None')
plt.plot(mse_x_orientation_GRU_Batch_128, marker='o', color='black', linestyle='None')
plt.plot(mse_x_orientation_GRU_Batch_256, marker='o', color='magenta', linestyle='None')
plt.plot(mse_x_orientation_GRU_Batch_512, marker='o', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_X_GRU")
plt.show()

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Y_GRU
plt.title("GRU Model in Y axis")
plt.plot(mse_y_orientation_GRU_Batch_8, marker='^', color='cyan', linestyle='None')
plt.plot(mse_y_orientation_GRU_Batch_16, marker='^', color='red', linestyle='None')
plt.plot(mse_y_orientation_GRU_Batch_32, marker='^', color='blue', linestyle='None')
plt.plot(mse_y_orientation_GRU_Batch_64, marker='^', color='green', linestyle='None')
plt.plot(mse_y_orientation_GRU_Batch_128, marker='^', color='black', linestyle='None')
plt.plot(mse_y_orientation_GRU_Batch_256, marker='^', color='magenta', linestyle='None')
plt.plot(mse_y_orientation_GRU_Batch_512, marker='^', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Y_GRU")
plt.show()

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Z_GRU
plt.title("GRU Model in Z axis")
plt.plot(mse_z_orientation_GRU_Batch_8, marker='*', color='cyan', linestyle='None')
plt.plot(mse_z_orientation_GRU_Batch_16, marker='*', color='red', linestyle='None')
plt.plot(mse_z_orientation_GRU_Batch_32, marker='*', color='blue', linestyle='None')
plt.plot(mse_z_orientation_GRU_Batch_64, marker='*', color='green', linestyle='None')
plt.plot(mse_z_orientation_GRU_Batch_128, marker='*', color='black', linestyle='None')
plt.plot(mse_z_orientation_GRU_Batch_256, marker='*', color='magenta', linestyle='None')
plt.plot(mse_z_orientation_GRU_Batch_512, marker='*', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Z_GRU")
plt.show()




###############################################################################################################

currentTime = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) # not needed
figurePath = ".\\SavedFigure\\BatchSize-8-16-32-64-128-256\\MAE_of_Orientation_in_Testing_Data\\"

if not os.path.exists(figurePath):
    os.makedirs(figurePath)

# Batch_Size

# DNN Model

ymin = -0.001
ymax = 0.015

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_X_DNN
plt.title("DNN Model in X axis")
plt.plot(mae_x_orientation_DNN_Batch_8, marker='o', color='cyan', linestyle='None')
plt.plot(mae_x_orientation_DNN_Batch_16, marker='o', color='red', linestyle='None')
plt.plot(mae_x_orientation_DNN_Batch_32, marker='o', color='blue', linestyle='None')
plt.plot(mae_x_orientation_DNN_Batch_64, marker='o', color='green', linestyle='None')
plt.plot(mae_x_orientation_DNN_Batch_128, marker='o', color='black', linestyle='None')
plt.plot(mae_x_orientation_DNN_Batch_256, marker='o', color='magenta', linestyle='None')
plt.plot(mae_x_orientation_DNN_Batch_512, marker='o', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_X_DNN")
plt.show()

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Y_DNN
plt.title("DNN Model in Y axis")
plt.plot(mae_y_orientation_DNN_Batch_8, marker='^', color='cyan', linestyle='None')
plt.plot(mae_y_orientation_DNN_Batch_16, marker='^', color='red', linestyle='None')
plt.plot(mae_y_orientation_DNN_Batch_32, marker='^', color='blue', linestyle='None')
plt.plot(mae_y_orientation_DNN_Batch_64, marker='^', color='green', linestyle='None')
plt.plot(mae_y_orientation_DNN_Batch_128, marker='^', color='black', linestyle='None')
plt.plot(mae_y_orientation_DNN_Batch_256, marker='^', color='magenta', linestyle='None')
plt.plot(mae_y_orientation_DNN_Batch_512, marker='^', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Y_DNN")
plt.show()

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Z_DNN
plt.title("DNN Model in Z axis")
plt.plot(mae_z_orientation_DNN_Batch_8, marker='*', color='cyan', linestyle='None')
plt.plot(mae_z_orientation_DNN_Batch_16, marker='*', color='red', linestyle='None')
plt.plot(mae_z_orientation_DNN_Batch_32, marker='*', color='blue', linestyle='None')
plt.plot(mae_z_orientation_DNN_Batch_64, marker='*', color='green', linestyle='None')
plt.plot(mae_z_orientation_DNN_Batch_128, marker='*', color='black', linestyle='None')
plt.plot(mae_z_orientation_DNN_Batch_256, marker='*', color='magenta', linestyle='None')
plt.plot(mae_z_orientation_DNN_Batch_512, marker='*', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Z_DNN")
plt.show()

# CNN Model

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_X_CNN
plt.title("CNN Model in X axis")
plt.plot(mae_x_orientation_CNN_Batch_8, marker='o', color='cyan', linestyle='None')
plt.plot(mae_x_orientation_CNN_Batch_16, marker='o', color='red', linestyle='None')
plt.plot(mae_x_orientation_CNN_Batch_32, marker='o', color='blue', linestyle='None')
plt.plot(mae_x_orientation_CNN_Batch_64, marker='o', color='green', linestyle='None')
plt.plot(mae_x_orientation_CNN_Batch_128, marker='o', color='black', linestyle='None')
plt.plot(mae_x_orientation_CNN_Batch_256, marker='o', color='magenta', linestyle='None')
plt.plot(mae_x_orientation_CNN_Batch_512, marker='o', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_X_CNN")
plt.show()

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Y_CNN
plt.title("CNN Model in Y axis")
plt.plot(mae_y_orientation_CNN_Batch_8, marker='^', color='cyan', linestyle='None')
plt.plot(mae_y_orientation_CNN_Batch_16, marker='^', color='red', linestyle='None')
plt.plot(mae_y_orientation_CNN_Batch_32, marker='^', color='blue', linestyle='None')
plt.plot(mae_y_orientation_CNN_Batch_64, marker='^', color='green', linestyle='None')
plt.plot(mae_y_orientation_CNN_Batch_128, marker='^', color='black', linestyle='None')
plt.plot(mae_y_orientation_CNN_Batch_256, marker='^', color='magenta', linestyle='None')
plt.plot(mae_y_orientation_CNN_Batch_512, marker='^', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Y_CNN")
plt.show()

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Z_CNN
plt.title("CNN Model in Z axis")
plt.plot(mae_z_orientation_CNN_Batch_8, marker='*', color='cyan', linestyle='None')
plt.plot(mae_z_orientation_CNN_Batch_16, marker='*', color='red', linestyle='None')
plt.plot(mae_z_orientation_CNN_Batch_32, marker='*', color='blue', linestyle='None')
plt.plot(mae_z_orientation_CNN_Batch_64, marker='*', color='green', linestyle='None')
plt.plot(mae_z_orientation_CNN_Batch_128, marker='*', color='black', linestyle='None')
plt.plot(mae_z_orientation_CNN_Batch_256, marker='*', color='magenta', linestyle='None')
plt.plot(mae_z_orientation_CNN_Batch_512, marker='*', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Z_CNN")
plt.show()

# RNN Model

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_X_RNN
plt.title("RNN Model in X axis")
plt.plot(mae_x_orientation_RNN_Batch_8, marker='o', color='cyan', linestyle='None')
plt.plot(mae_x_orientation_RNN_Batch_16, marker='o', color='red', linestyle='None')
plt.plot(mae_x_orientation_RNN_Batch_32, marker='o', color='blue', linestyle='None')
plt.plot(mae_x_orientation_RNN_Batch_64, marker='o', color='green', linestyle='None')
plt.plot(mae_x_orientation_RNN_Batch_128, marker='o', color='black', linestyle='None')
plt.plot(mae_x_orientation_RNN_Batch_256, marker='o', color='magenta', linestyle='None')
plt.plot(mae_x_orientation_RNN_Batch_512, marker='o', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_X_RNN")
plt.show()

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Y_RNN
plt.title("RNN Model in Y axis")
plt.plot(mae_y_orientation_RNN_Batch_8, marker='^', color='cyan', linestyle='None')
plt.plot(mae_y_orientation_RNN_Batch_16, marker='^', color='red', linestyle='None')
plt.plot(mae_y_orientation_RNN_Batch_32, marker='^', color='blue', linestyle='None')
plt.plot(mae_y_orientation_RNN_Batch_64, marker='^', color='green', linestyle='None')
plt.plot(mae_y_orientation_RNN_Batch_128, marker='^', color='black', linestyle='None')
plt.plot(mae_y_orientation_RNN_Batch_256, marker='^', color='magenta', linestyle='None')
plt.plot(mae_y_orientation_RNN_Batch_512, marker='^', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Y_RNN")
plt.show()

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Z_RNN
plt.title("RNN Model in Z axis")
plt.plot(mae_z_orientation_RNN_Batch_8, marker='*', color='cyan', linestyle='None')
plt.plot(mae_z_orientation_RNN_Batch_16, marker='*', color='red', linestyle='None')
plt.plot(mae_z_orientation_RNN_Batch_32, marker='*', color='blue', linestyle='None')
plt.plot(mae_z_orientation_RNN_Batch_64, marker='*', color='green', linestyle='None')
plt.plot(mae_z_orientation_RNN_Batch_128, marker='*', color='black', linestyle='None')
plt.plot(mae_z_orientation_RNN_Batch_256, marker='*', color='magenta', linestyle='None')
plt.plot(mae_z_orientation_RNN_Batch_512, marker='*', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Z_RNN")
plt.show()

# LSTM Model

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_X_LSTM
plt.title("LSTM Model in X axis")
plt.plot(mae_x_orientation_LSTM_Batch_8, marker='o', color='cyan', linestyle='None')
plt.plot(mae_x_orientation_LSTM_Batch_16, marker='o', color='red', linestyle='None')
plt.plot(mae_x_orientation_LSTM_Batch_32, marker='o', color='blue', linestyle='None')
plt.plot(mae_x_orientation_LSTM_Batch_64, marker='o', color='green', linestyle='None')
plt.plot(mae_x_orientation_LSTM_Batch_128, marker='o', color='black', linestyle='None')
plt.plot(mae_x_orientation_LSTM_Batch_256, marker='o', color='magenta', linestyle='None')
plt.plot(mae_x_orientation_LSTM_Batch_512, marker='o', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_X_LSTM")
plt.show()

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Y_LSTM
plt.title("LSTM Model in Y axis")
plt.plot(mae_y_orientation_LSTM_Batch_8, marker='^', color='cyan', linestyle='None')
plt.plot(mae_y_orientation_LSTM_Batch_16, marker='^', color='red', linestyle='None')
plt.plot(mae_y_orientation_LSTM_Batch_32, marker='^', color='blue', linestyle='None')
plt.plot(mae_y_orientation_LSTM_Batch_64, marker='^', color='green', linestyle='None')
plt.plot(mae_y_orientation_LSTM_Batch_128, marker='^', color='black', linestyle='None')
plt.plot(mae_y_orientation_LSTM_Batch_256, marker='^', color='magenta', linestyle='None')
plt.plot(mae_y_orientation_LSTM_Batch_512, marker='^', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Y_LSTM")
plt.show()

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Z_LSTM
plt.title("LSTM Model in Z axis")
plt.plot(mae_z_orientation_LSTM_Batch_8, marker='*', color='cyan', linestyle='None')
plt.plot(mae_z_orientation_LSTM_Batch_16, marker='*', color='red', linestyle='None')
plt.plot(mae_z_orientation_LSTM_Batch_32, marker='*', color='blue', linestyle='None')
plt.plot(mae_z_orientation_LSTM_Batch_64, marker='*', color='green', linestyle='None')
plt.plot(mae_z_orientation_LSTM_Batch_128, marker='*', color='black', linestyle='None')
plt.plot(mae_z_orientation_LSTM_Batch_256, marker='*', color='magenta', linestyle='None')
plt.plot(mae_z_orientation_LSTM_Batch_512, marker='*', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Z_LSTM")
plt.show()

# GRU Model

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_X_GRU
plt.title("GRU Model in X axis")
plt.plot(mae_x_orientation_GRU_Batch_8, marker='o', color='cyan', linestyle='None')
plt.plot(mae_x_orientation_GRU_Batch_16, marker='o', color='red', linestyle='None')
plt.plot(mae_x_orientation_GRU_Batch_32, marker='o', color='blue', linestyle='None')
plt.plot(mae_x_orientation_GRU_Batch_64, marker='o', color='green', linestyle='None')
plt.plot(mae_x_orientation_GRU_Batch_128, marker='o', color='black', linestyle='None')
plt.plot(mae_x_orientation_GRU_Batch_256, marker='o', color='magenta', linestyle='None')
plt.plot(mae_x_orientation_GRU_Batch_512, marker='o', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_X_GRU")
plt.show()

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Y_GRU
plt.title("GRU Model in Y axis")
plt.plot(mae_y_orientation_GRU_Batch_8, marker='^', color='cyan', linestyle='None')
plt.plot(mae_y_orientation_GRU_Batch_16, marker='^', color='red', linestyle='None')
plt.plot(mae_y_orientation_GRU_Batch_32, marker='^', color='blue', linestyle='None')
plt.plot(mae_y_orientation_GRU_Batch_64, marker='^', color='green', linestyle='None')
plt.plot(mae_y_orientation_GRU_Batch_128, marker='^', color='black', linestyle='None')
plt.plot(mae_y_orientation_GRU_Batch_256, marker='^', color='magenta', linestyle='None')
plt.plot(mae_y_orientation_GRU_Batch_512, marker='^', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Y_GRU")
plt.show()

# BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Z_GRU
plt.title("GRU Model in Z axis")
plt.plot(mae_z_orientation_GRU_Batch_8, marker='*', color='cyan', linestyle='None')
plt.plot(mae_z_orientation_GRU_Batch_16, marker='*', color='red', linestyle='None')
plt.plot(mae_z_orientation_GRU_Batch_32, marker='*', color='blue', linestyle='None')
plt.plot(mae_z_orientation_GRU_Batch_64, marker='*', color='green', linestyle='None')
plt.plot(mae_z_orientation_GRU_Batch_128, marker='*', color='black', linestyle='None')
plt.plot(mae_z_orientation_GRU_Batch_256, marker='*', color='magenta', linestyle='None')
plt.plot(mae_z_orientation_GRU_Batch_512, marker='*', color='yellow', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Batch Size: 8", "Batch Size: 16", "Batch Size: 32", "Batch Size: 64", "Batch Size: 128", "Batch Size: 256", "Batch Size: 512"])
plt.savefig(figurePath + "BatchSize_8_16_32_64_128_256_512_of_Orientation_in_Z_GRU")
plt.show()
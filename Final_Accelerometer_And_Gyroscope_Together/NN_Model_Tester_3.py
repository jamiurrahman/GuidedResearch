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

csv_path = "./RawData/my_testing_result_together.csv"
df = pd.read_csv(csv_path)
print(df.head())

fileName = df["fileName"]
mae_x_position = df["mae_x_position"]
mae_y_position = df["mae_y_position"]
mae_z_position = df["mae_z_position"]

mse_x_position = df["mse_x_position"]
mse_y_position = df["mse_y_position"]
mse_z_position = df["mse_z_position"]

print(fileName.head())

# Ac_Func_tanh

# DNN Model Variables
mae_x_position_DNN_Ac_Func_tanh = []
mae_y_position_DNN_Ac_Func_tanh = []
mae_z_position_DNN_Ac_Func_tanh = []

mse_x_position_DNN_Ac_Func_tanh = []
mse_y_position_DNN_Ac_Func_tanh = []
mse_z_position_DNN_Ac_Func_tanh = []

# CNN Model Variables
mae_x_position_CNN_Ac_Func_tanh = []
mae_y_position_CNN_Ac_Func_tanh = []
mae_z_position_CNN_Ac_Func_tanh = []

mse_x_position_CNN_Ac_Func_tanh = []
mse_y_position_CNN_Ac_Func_tanh = []
mse_z_position_CNN_Ac_Func_tanh = []

# RNN Model Variables
mae_x_position_RNN_Ac_Func_tanh = []
mae_y_position_RNN_Ac_Func_tanh = []
mae_z_position_RNN_Ac_Func_tanh = []

mse_x_position_RNN_Ac_Func_tanh = []
mse_y_position_RNN_Ac_Func_tanh = []
mse_z_position_RNN_Ac_Func_tanh = []

# LSTM Model Variables
mae_x_position_LSTM_Ac_Func_tanh = []
mae_y_position_LSTM_Ac_Func_tanh = []
mae_z_position_LSTM_Ac_Func_tanh = []

mse_x_position_LSTM_Ac_Func_tanh = []
mse_y_position_LSTM_Ac_Func_tanh = []
mse_z_position_LSTM_Ac_Func_tanh = []

# GRU Model Variables
mae_x_position_GRU_Ac_Func_tanh = []
mae_y_position_GRU_Ac_Func_tanh = []
mae_z_position_GRU_Ac_Func_tanh = []

mse_x_position_GRU_Ac_Func_tanh = []
mse_y_position_GRU_Ac_Func_tanh = []
mse_z_position_GRU_Ac_Func_tanh = []

# Ac_Func_relu

# DNN Model Variables
mae_x_position_DNN_Ac_Func_relu = []
mae_y_position_DNN_Ac_Func_relu = []
mae_z_position_DNN_Ac_Func_relu = []

mse_x_position_DNN_Ac_Func_relu = []
mse_y_position_DNN_Ac_Func_relu = []
mse_z_position_DNN_Ac_Func_relu = []

# CNN Model Variables
mae_x_position_CNN_Ac_Func_relu = []
mae_y_position_CNN_Ac_Func_relu = []
mae_z_position_CNN_Ac_Func_relu = []

mse_x_position_CNN_Ac_Func_relu = []
mse_y_position_CNN_Ac_Func_relu = []
mse_z_position_CNN_Ac_Func_relu = []

# RNN Model Variables
mae_x_position_RNN_Ac_Func_relu = []
mae_y_position_RNN_Ac_Func_relu = []
mae_z_position_RNN_Ac_Func_relu = []

mse_x_position_RNN_Ac_Func_relu = []
mse_y_position_RNN_Ac_Func_relu = []
mse_z_position_RNN_Ac_Func_relu = []

# LSTM Model Variables
mae_x_position_LSTM_Ac_Func_relu = []
mae_y_position_LSTM_Ac_Func_relu = []
mae_z_position_LSTM_Ac_Func_relu = []

mse_x_position_LSTM_Ac_Func_relu = []
mse_y_position_LSTM_Ac_Func_relu = []
mse_z_position_LSTM_Ac_Func_relu = []

# GRU Model Variables
mae_x_position_GRU_Ac_Func_relu = []
mae_y_position_GRU_Ac_Func_relu = []
mae_z_position_GRU_Ac_Func_relu = []

mse_x_position_GRU_Ac_Func_relu = []
mse_y_position_GRU_Ac_Func_relu = []
mse_z_position_GRU_Ac_Func_relu = []

# Ac_Func_LeakyRelu

# DNN Model Variables
mae_x_position_DNN_Ac_Func_LeakyRelu = []
mae_y_position_DNN_Ac_Func_LeakyRelu = []
mae_z_position_DNN_Ac_Func_LeakyRelu = []

mse_x_position_DNN_Ac_Func_LeakyRelu = []
mse_y_position_DNN_Ac_Func_LeakyRelu = []
mse_z_position_DNN_Ac_Func_LeakyRelu = []

# CNN Model Variables
mae_x_position_CNN_Ac_Func_LeakyRelu = []
mae_y_position_CNN_Ac_Func_LeakyRelu = []
mae_z_position_CNN_Ac_Func_LeakyRelu = []

mse_x_position_CNN_Ac_Func_LeakyRelu = []
mse_y_position_CNN_Ac_Func_LeakyRelu = []
mse_z_position_CNN_Ac_Func_LeakyRelu = []

# RNN Model Variables
mae_x_position_RNN_Ac_Func_LeakyRelu = []
mae_y_position_RNN_Ac_Func_LeakyRelu = []
mae_z_position_RNN_Ac_Func_LeakyRelu = []

mse_x_position_RNN_Ac_Func_LeakyRelu = []
mse_y_position_RNN_Ac_Func_LeakyRelu = []
mse_z_position_RNN_Ac_Func_LeakyRelu = []

# LSTM Model Variables
mae_x_position_LSTM_Ac_Func_LeakyRelu = []
mae_y_position_LSTM_Ac_Func_LeakyRelu = []
mae_z_position_LSTM_Ac_Func_LeakyRelu = []

mse_x_position_LSTM_Ac_Func_LeakyRelu = []
mse_y_position_LSTM_Ac_Func_LeakyRelu = []
mse_z_position_LSTM_Ac_Func_LeakyRelu = []

# GRU Model Variables
mae_x_position_GRU_Ac_Func_LeakyRelu = []
mae_y_position_GRU_Ac_Func_LeakyRelu = []
mae_z_position_GRU_Ac_Func_LeakyRelu = []

mse_x_position_GRU_Ac_Func_LeakyRelu = []
mse_y_position_GRU_Ac_Func_LeakyRelu = []
mse_z_position_GRU_Ac_Func_LeakyRelu = []

totalData = len(fileName)
for i in range(totalData):

    # Ac_Func_relu

    # DNN Model
    if "Loss_mse_DNN" in fileName[i] and "Ac_Func_tanh" in fileName[i]:
        mae_x_position_DNN_Ac_Func_tanh.append(mae_x_position[i])
        mae_y_position_DNN_Ac_Func_tanh.append(mae_y_position[i])
        mae_z_position_DNN_Ac_Func_tanh.append(mae_z_position[i])

        mse_x_position_DNN_Ac_Func_tanh.append(mse_x_position[i])
        mse_y_position_DNN_Ac_Func_tanh.append(mse_y_position[i])
        mse_z_position_DNN_Ac_Func_tanh.append(mse_z_position[i])

    # CNN Model
    elif "Loss_mse_CNN" in fileName[i] and "Ac_Func_tanh" in fileName[i]:
        mae_x_position_CNN_Ac_Func_tanh.append(mae_x_position[i])
        mae_y_position_CNN_Ac_Func_tanh.append(mae_y_position[i])
        mae_z_position_CNN_Ac_Func_tanh.append(mae_z_position[i])

        mse_x_position_CNN_Ac_Func_tanh.append(mse_x_position[i])
        mse_y_position_CNN_Ac_Func_tanh.append(mse_y_position[i])
        mse_z_position_CNN_Ac_Func_tanh.append(mse_z_position[i])

    # RNN Model
    elif "Loss_mse_RNN" in fileName[i] and "Ac_Func_tanh" in fileName[i]:
        mae_x_position_RNN_Ac_Func_tanh.append(mae_x_position[i])
        mae_y_position_RNN_Ac_Func_tanh.append(mae_y_position[i])
        mae_z_position_RNN_Ac_Func_tanh.append(mae_z_position[i])

        mse_x_position_RNN_Ac_Func_tanh.append(mse_x_position[i])
        mse_y_position_RNN_Ac_Func_tanh.append(mse_y_position[i])
        mse_z_position_RNN_Ac_Func_tanh.append(mse_z_position[i])

    # LSTM Model
    elif "Loss_mse_LSTM" in fileName[i] and "Ac_Func_tanh" in fileName[i]:
        mae_x_position_LSTM_Ac_Func_tanh.append(mae_x_position[i])
        mae_y_position_LSTM_Ac_Func_tanh.append(mae_y_position[i])
        mae_z_position_LSTM_Ac_Func_tanh.append(mae_z_position[i])

        mse_x_position_LSTM_Ac_Func_tanh.append(mse_x_position[i])
        mse_y_position_LSTM_Ac_Func_tanh.append(mse_y_position[i])
        mse_z_position_LSTM_Ac_Func_tanh.append(mse_z_position[i])

    # GRU Model
    elif "Loss_mse_GRU" in fileName[i] and "Ac_Func_tanh" in fileName[i]:
        mae_x_position_GRU_Ac_Func_tanh.append(mae_x_position[i])
        mae_y_position_GRU_Ac_Func_tanh.append(mae_y_position[i])
        mae_z_position_GRU_Ac_Func_tanh.append(mae_z_position[i])

        mse_x_position_GRU_Ac_Func_tanh.append(mse_x_position[i])
        mse_y_position_GRU_Ac_Func_tanh.append(mse_y_position[i])
        mse_z_position_GRU_Ac_Func_tanh.append(mse_z_position[i])

    # Ac_Func_relu

    # DNN Model
    if "Loss_mse_DNN" in fileName[i] and "Ac_Func_relu" in fileName[i]:
        mae_x_position_DNN_Ac_Func_relu.append(mae_x_position[i])
        mae_y_position_DNN_Ac_Func_relu.append(mae_y_position[i])
        mae_z_position_DNN_Ac_Func_relu.append(mae_z_position[i])

        mse_x_position_DNN_Ac_Func_relu.append(mse_x_position[i])
        mse_y_position_DNN_Ac_Func_relu.append(mse_y_position[i])
        mse_z_position_DNN_Ac_Func_relu.append(mse_z_position[i])

    # CNN Model
    elif "Loss_mse_CNN" in fileName[i] and "Ac_Func_relu" in fileName[i]:
        mae_x_position_CNN_Ac_Func_relu.append(mae_x_position[i])
        mae_y_position_CNN_Ac_Func_relu.append(mae_y_position[i])
        mae_z_position_CNN_Ac_Func_relu.append(mae_z_position[i])

        mse_x_position_CNN_Ac_Func_relu.append(mse_x_position[i])
        mse_y_position_CNN_Ac_Func_relu.append(mse_y_position[i])
        mse_z_position_CNN_Ac_Func_relu.append(mse_z_position[i])

    # RNN Model
    elif "Loss_mse_RNN" in fileName[i] and "Ac_Func_relu" in fileName[i]:
        mae_x_position_RNN_Ac_Func_relu.append(mae_x_position[i])
        mae_y_position_RNN_Ac_Func_relu.append(mae_y_position[i])
        mae_z_position_RNN_Ac_Func_relu.append(mae_z_position[i])

        mse_x_position_RNN_Ac_Func_relu.append(mse_x_position[i])
        mse_y_position_RNN_Ac_Func_relu.append(mse_y_position[i])
        mse_z_position_RNN_Ac_Func_relu.append(mse_z_position[i])

    # LSTM Model
    elif "Loss_mse_LSTM" in fileName[i] and "Ac_Func_relu" in fileName[i]:
        mae_x_position_LSTM_Ac_Func_relu.append(mae_x_position[i])
        mae_y_position_LSTM_Ac_Func_relu.append(mae_y_position[i])
        mae_z_position_LSTM_Ac_Func_relu.append(mae_z_position[i])

        mse_x_position_LSTM_Ac_Func_relu.append(mse_x_position[i])
        mse_y_position_LSTM_Ac_Func_relu.append(mse_y_position[i])
        mse_z_position_LSTM_Ac_Func_relu.append(mse_z_position[i])

    # GRU Model
    elif "Loss_mse_GRU" in fileName[i] and "Ac_Func_relu" in fileName[i]:
        mae_x_position_GRU_Ac_Func_relu.append(mae_x_position[i])
        mae_y_position_GRU_Ac_Func_relu.append(mae_y_position[i])
        mae_z_position_GRU_Ac_Func_relu.append(mae_z_position[i])

        mse_x_position_GRU_Ac_Func_relu.append(mse_x_position[i])
        mse_y_position_GRU_Ac_Func_relu.append(mse_y_position[i])
        mse_z_position_GRU_Ac_Func_relu.append(mse_z_position[i])

    # Ac_Func_LeakyRelu

    # DNN Model
    if "Loss_mse_DNN" in fileName[i] and "Ac_Func_LeakyRelu" in fileName[i]:
        mae_x_position_DNN_Ac_Func_LeakyRelu.append(mae_x_position[i])
        mae_y_position_DNN_Ac_Func_LeakyRelu.append(mae_y_position[i])
        mae_z_position_DNN_Ac_Func_LeakyRelu.append(mae_z_position[i])

        mse_x_position_DNN_Ac_Func_LeakyRelu.append(mse_x_position[i])
        mse_y_position_DNN_Ac_Func_LeakyRelu.append(mse_y_position[i])
        mse_z_position_DNN_Ac_Func_LeakyRelu.append(mse_z_position[i])

    # CNN Model
    elif "Loss_mse_CNN" in fileName[i] and "Ac_Func_LeakyRelu" in fileName[i]:
        mae_x_position_CNN_Ac_Func_LeakyRelu.append(mae_x_position[i])
        mae_y_position_CNN_Ac_Func_LeakyRelu.append(mae_y_position[i])
        mae_z_position_CNN_Ac_Func_LeakyRelu.append(mae_z_position[i])

        mse_x_position_CNN_Ac_Func_LeakyRelu.append(mse_x_position[i])
        mse_y_position_CNN_Ac_Func_LeakyRelu.append(mse_y_position[i])
        mse_z_position_CNN_Ac_Func_LeakyRelu.append(mse_z_position[i])

    # RNN Model
    elif "Loss_mse_RNN" in fileName[i] and "Ac_Func_LeakyRelu" in fileName[i]:
        mae_x_position_RNN_Ac_Func_LeakyRelu.append(mae_x_position[i])
        mae_y_position_RNN_Ac_Func_LeakyRelu.append(mae_y_position[i])
        mae_z_position_RNN_Ac_Func_LeakyRelu.append(mae_z_position[i])

        mse_x_position_RNN_Ac_Func_LeakyRelu.append(mse_x_position[i])
        mse_y_position_RNN_Ac_Func_LeakyRelu.append(mse_y_position[i])
        mse_z_position_RNN_Ac_Func_LeakyRelu.append(mse_z_position[i])

    # LSTM Model
    elif "Loss_mse_LSTM" in fileName[i] and "Ac_Func_LeakyRelu" in fileName[i]:
        mae_x_position_LSTM_Ac_Func_LeakyRelu.append(mae_x_position[i])
        mae_y_position_LSTM_Ac_Func_LeakyRelu.append(mae_y_position[i])
        mae_z_position_LSTM_Ac_Func_LeakyRelu.append(mae_z_position[i])

        mse_x_position_LSTM_Ac_Func_LeakyRelu.append(mse_x_position[i])
        mse_y_position_LSTM_Ac_Func_LeakyRelu.append(mse_y_position[i])
        mse_z_position_LSTM_Ac_Func_LeakyRelu.append(mse_z_position[i])

    # GRU Model
    elif "Loss_mse_GRU" in fileName[i] and "Ac_Func_LeakyRelu" in fileName[i]:
        mae_x_position_GRU_Ac_Func_LeakyRelu.append(mae_x_position[i])
        mae_y_position_GRU_Ac_Func_LeakyRelu.append(mae_y_position[i])
        mae_z_position_GRU_Ac_Func_LeakyRelu.append(mae_z_position[i])

        mse_x_position_GRU_Ac_Func_LeakyRelu.append(mse_x_position[i])
        mse_y_position_GRU_Ac_Func_LeakyRelu.append(mse_y_position[i])
        mse_z_position_GRU_Ac_Func_LeakyRelu.append(mse_z_position[i])


mae_x_orientation = df["mae_x_orientation"]
mae_y_orientation = df["mae_y_orientation"]
mae_z_orientation = df["mae_z_orientation"]

mse_x_orientation = df["mse_x_orientation"]
mse_y_orientation = df["mse_y_orientation"]
mse_z_orientation = df["mse_z_orientation"]

print(fileName.head())

# Ac_Func_tanh

# DNN Model Variables
mae_x_orientation_DNN_Ac_Func_tanh = []
mae_y_orientation_DNN_Ac_Func_tanh = []
mae_z_orientation_DNN_Ac_Func_tanh = []

mse_x_orientation_DNN_Ac_Func_tanh = []
mse_y_orientation_DNN_Ac_Func_tanh = []
mse_z_orientation_DNN_Ac_Func_tanh = []

# CNN Model Variables
mae_x_orientation_CNN_Ac_Func_tanh = []
mae_y_orientation_CNN_Ac_Func_tanh = []
mae_z_orientation_CNN_Ac_Func_tanh = []

mse_x_orientation_CNN_Ac_Func_tanh = []
mse_y_orientation_CNN_Ac_Func_tanh = []
mse_z_orientation_CNN_Ac_Func_tanh = []

# RNN Model Variables
mae_x_orientation_RNN_Ac_Func_tanh = []
mae_y_orientation_RNN_Ac_Func_tanh = []
mae_z_orientation_RNN_Ac_Func_tanh = []

mse_x_orientation_RNN_Ac_Func_tanh = []
mse_y_orientation_RNN_Ac_Func_tanh = []
mse_z_orientation_RNN_Ac_Func_tanh = []

# LSTM Model Variables
mae_x_orientation_LSTM_Ac_Func_tanh = []
mae_y_orientation_LSTM_Ac_Func_tanh = []
mae_z_orientation_LSTM_Ac_Func_tanh = []

mse_x_orientation_LSTM_Ac_Func_tanh = []
mse_y_orientation_LSTM_Ac_Func_tanh = []
mse_z_orientation_LSTM_Ac_Func_tanh = []

# GRU Model Variables
mae_x_orientation_GRU_Ac_Func_tanh = []
mae_y_orientation_GRU_Ac_Func_tanh = []
mae_z_orientation_GRU_Ac_Func_tanh = []

mse_x_orientation_GRU_Ac_Func_tanh = []
mse_y_orientation_GRU_Ac_Func_tanh = []
mse_z_orientation_GRU_Ac_Func_tanh = []

# Ac_Func_relu

# DNN Model Variables
mae_x_orientation_DNN_Ac_Func_relu = []
mae_y_orientation_DNN_Ac_Func_relu = []
mae_z_orientation_DNN_Ac_Func_relu = []

mse_x_orientation_DNN_Ac_Func_relu = []
mse_y_orientation_DNN_Ac_Func_relu = []
mse_z_orientation_DNN_Ac_Func_relu = []

# CNN Model Variables
mae_x_orientation_CNN_Ac_Func_relu = []
mae_y_orientation_CNN_Ac_Func_relu = []
mae_z_orientation_CNN_Ac_Func_relu = []

mse_x_orientation_CNN_Ac_Func_relu = []
mse_y_orientation_CNN_Ac_Func_relu = []
mse_z_orientation_CNN_Ac_Func_relu = []

# RNN Model Variables
mae_x_orientation_RNN_Ac_Func_relu = []
mae_y_orientation_RNN_Ac_Func_relu = []
mae_z_orientation_RNN_Ac_Func_relu = []

mse_x_orientation_RNN_Ac_Func_relu = []
mse_y_orientation_RNN_Ac_Func_relu = []
mse_z_orientation_RNN_Ac_Func_relu = []

# LSTM Model Variables
mae_x_orientation_LSTM_Ac_Func_relu = []
mae_y_orientation_LSTM_Ac_Func_relu = []
mae_z_orientation_LSTM_Ac_Func_relu = []

mse_x_orientation_LSTM_Ac_Func_relu = []
mse_y_orientation_LSTM_Ac_Func_relu = []
mse_z_orientation_LSTM_Ac_Func_relu = []

# GRU Model Variables
mae_x_orientation_GRU_Ac_Func_relu = []
mae_y_orientation_GRU_Ac_Func_relu = []
mae_z_orientation_GRU_Ac_Func_relu = []

mse_x_orientation_GRU_Ac_Func_relu = []
mse_y_orientation_GRU_Ac_Func_relu = []
mse_z_orientation_GRU_Ac_Func_relu = []

# Ac_Func_LeakyRelu

# DNN Model Variables
mae_x_orientation_DNN_Ac_Func_LeakyRelu = []
mae_y_orientation_DNN_Ac_Func_LeakyRelu = []
mae_z_orientation_DNN_Ac_Func_LeakyRelu = []

mse_x_orientation_DNN_Ac_Func_LeakyRelu = []
mse_y_orientation_DNN_Ac_Func_LeakyRelu = []
mse_z_orientation_DNN_Ac_Func_LeakyRelu = []

# CNN Model Variables
mae_x_orientation_CNN_Ac_Func_LeakyRelu = []
mae_y_orientation_CNN_Ac_Func_LeakyRelu = []
mae_z_orientation_CNN_Ac_Func_LeakyRelu = []

mse_x_orientation_CNN_Ac_Func_LeakyRelu = []
mse_y_orientation_CNN_Ac_Func_LeakyRelu = []
mse_z_orientation_CNN_Ac_Func_LeakyRelu = []

# RNN Model Variables
mae_x_orientation_RNN_Ac_Func_LeakyRelu = []
mae_y_orientation_RNN_Ac_Func_LeakyRelu = []
mae_z_orientation_RNN_Ac_Func_LeakyRelu = []

mse_x_orientation_RNN_Ac_Func_LeakyRelu = []
mse_y_orientation_RNN_Ac_Func_LeakyRelu = []
mse_z_orientation_RNN_Ac_Func_LeakyRelu = []

# LSTM Model Variables
mae_x_orientation_LSTM_Ac_Func_LeakyRelu = []
mae_y_orientation_LSTM_Ac_Func_LeakyRelu = []
mae_z_orientation_LSTM_Ac_Func_LeakyRelu = []

mse_x_orientation_LSTM_Ac_Func_LeakyRelu = []
mse_y_orientation_LSTM_Ac_Func_LeakyRelu = []
mse_z_orientation_LSTM_Ac_Func_LeakyRelu = []

# GRU Model Variables
mae_x_orientation_GRU_Ac_Func_LeakyRelu = []
mae_y_orientation_GRU_Ac_Func_LeakyRelu = []
mae_z_orientation_GRU_Ac_Func_LeakyRelu = []

mse_x_orientation_GRU_Ac_Func_LeakyRelu = []
mse_y_orientation_GRU_Ac_Func_LeakyRelu = []
mse_z_orientation_GRU_Ac_Func_LeakyRelu = []

totalData = len(fileName)
for i in range(totalData):

    # Ac_Func_relu

    # DNN Model
    if "Loss_mse_DNN" in fileName[i] and "Ac_Func_tanh" in fileName[i]:
        mae_x_orientation_DNN_Ac_Func_tanh.append(mae_x_orientation[i])
        mae_y_orientation_DNN_Ac_Func_tanh.append(mae_y_orientation[i])
        mae_z_orientation_DNN_Ac_Func_tanh.append(mae_z_orientation[i])

        mse_x_orientation_DNN_Ac_Func_tanh.append(mse_x_orientation[i])
        mse_y_orientation_DNN_Ac_Func_tanh.append(mse_y_orientation[i])
        mse_z_orientation_DNN_Ac_Func_tanh.append(mse_z_orientation[i])

    # CNN Model
    elif "Loss_mse_CNN" in fileName[i] and "Ac_Func_tanh" in fileName[i]:
        mae_x_orientation_CNN_Ac_Func_tanh.append(mae_x_orientation[i])
        mae_y_orientation_CNN_Ac_Func_tanh.append(mae_y_orientation[i])
        mae_z_orientation_CNN_Ac_Func_tanh.append(mae_z_orientation[i])

        mse_x_orientation_CNN_Ac_Func_tanh.append(mse_x_orientation[i])
        mse_y_orientation_CNN_Ac_Func_tanh.append(mse_y_orientation[i])
        mse_z_orientation_CNN_Ac_Func_tanh.append(mse_z_orientation[i])

    # RNN Model
    elif "Loss_mse_RNN" in fileName[i] and "Ac_Func_tanh" in fileName[i]:
        mae_x_orientation_RNN_Ac_Func_tanh.append(mae_x_orientation[i])
        mae_y_orientation_RNN_Ac_Func_tanh.append(mae_y_orientation[i])
        mae_z_orientation_RNN_Ac_Func_tanh.append(mae_z_orientation[i])

        mse_x_orientation_RNN_Ac_Func_tanh.append(mse_x_orientation[i])
        mse_y_orientation_RNN_Ac_Func_tanh.append(mse_y_orientation[i])
        mse_z_orientation_RNN_Ac_Func_tanh.append(mse_z_orientation[i])

    # LSTM Model
    elif "Loss_mse_LSTM" in fileName[i] and "Ac_Func_tanh" in fileName[i]:
        mae_x_orientation_LSTM_Ac_Func_tanh.append(mae_x_orientation[i])
        mae_y_orientation_LSTM_Ac_Func_tanh.append(mae_y_orientation[i])
        mae_z_orientation_LSTM_Ac_Func_tanh.append(mae_z_orientation[i])

        mse_x_orientation_LSTM_Ac_Func_tanh.append(mse_x_orientation[i])
        mse_y_orientation_LSTM_Ac_Func_tanh.append(mse_y_orientation[i])
        mse_z_orientation_LSTM_Ac_Func_tanh.append(mse_z_orientation[i])

    # GRU Model
    elif "Loss_mse_GRU" in fileName[i] and "Ac_Func_tanh" in fileName[i]:
        mae_x_orientation_GRU_Ac_Func_tanh.append(mae_x_orientation[i])
        mae_y_orientation_GRU_Ac_Func_tanh.append(mae_y_orientation[i])
        mae_z_orientation_GRU_Ac_Func_tanh.append(mae_z_orientation[i])

        mse_x_orientation_GRU_Ac_Func_tanh.append(mse_x_orientation[i])
        mse_y_orientation_GRU_Ac_Func_tanh.append(mse_y_orientation[i])
        mse_z_orientation_GRU_Ac_Func_tanh.append(mse_z_orientation[i])

    # Ac_Func_relu

    # DNN Model
    if "Loss_mse_DNN" in fileName[i] and "Ac_Func_relu" in fileName[i]:
        mae_x_orientation_DNN_Ac_Func_relu.append(mae_x_orientation[i])
        mae_y_orientation_DNN_Ac_Func_relu.append(mae_y_orientation[i])
        mae_z_orientation_DNN_Ac_Func_relu.append(mae_z_orientation[i])

        mse_x_orientation_DNN_Ac_Func_relu.append(mse_x_orientation[i])
        mse_y_orientation_DNN_Ac_Func_relu.append(mse_y_orientation[i])
        mse_z_orientation_DNN_Ac_Func_relu.append(mse_z_orientation[i])

    # CNN Model
    elif "Loss_mse_CNN" in fileName[i] and "Ac_Func_relu" in fileName[i]:
        mae_x_orientation_CNN_Ac_Func_relu.append(mae_x_orientation[i])
        mae_y_orientation_CNN_Ac_Func_relu.append(mae_y_orientation[i])
        mae_z_orientation_CNN_Ac_Func_relu.append(mae_z_orientation[i])

        mse_x_orientation_CNN_Ac_Func_relu.append(mse_x_orientation[i])
        mse_y_orientation_CNN_Ac_Func_relu.append(mse_y_orientation[i])
        mse_z_orientation_CNN_Ac_Func_relu.append(mse_z_orientation[i])

    # RNN Model
    elif "Loss_mse_RNN" in fileName[i] and "Ac_Func_relu" in fileName[i]:
        mae_x_orientation_RNN_Ac_Func_relu.append(mae_x_orientation[i])
        mae_y_orientation_RNN_Ac_Func_relu.append(mae_y_orientation[i])
        mae_z_orientation_RNN_Ac_Func_relu.append(mae_z_orientation[i])

        mse_x_orientation_RNN_Ac_Func_relu.append(mse_x_orientation[i])
        mse_y_orientation_RNN_Ac_Func_relu.append(mse_y_orientation[i])
        mse_z_orientation_RNN_Ac_Func_relu.append(mse_z_orientation[i])

    # LSTM Model
    elif "Loss_mse_LSTM" in fileName[i] and "Ac_Func_relu" in fileName[i]:
        mae_x_orientation_LSTM_Ac_Func_relu.append(mae_x_orientation[i])
        mae_y_orientation_LSTM_Ac_Func_relu.append(mae_y_orientation[i])
        mae_z_orientation_LSTM_Ac_Func_relu.append(mae_z_orientation[i])

        mse_x_orientation_LSTM_Ac_Func_relu.append(mse_x_orientation[i])
        mse_y_orientation_LSTM_Ac_Func_relu.append(mse_y_orientation[i])
        mse_z_orientation_LSTM_Ac_Func_relu.append(mse_z_orientation[i])

    # GRU Model
    elif "Loss_mse_GRU" in fileName[i] and "Ac_Func_relu" in fileName[i]:
        mae_x_orientation_GRU_Ac_Func_relu.append(mae_x_orientation[i])
        mae_y_orientation_GRU_Ac_Func_relu.append(mae_y_orientation[i])
        mae_z_orientation_GRU_Ac_Func_relu.append(mae_z_orientation[i])

        mse_x_orientation_GRU_Ac_Func_relu.append(mse_x_orientation[i])
        mse_y_orientation_GRU_Ac_Func_relu.append(mse_y_orientation[i])
        mse_z_orientation_GRU_Ac_Func_relu.append(mse_z_orientation[i])

    # Ac_Func_LeakyRelu

    # DNN Model
    if "Loss_mse_DNN" in fileName[i] and "Ac_Func_LeakyRelu" in fileName[i]:
        mae_x_orientation_DNN_Ac_Func_LeakyRelu.append(mae_x_orientation[i])
        mae_y_orientation_DNN_Ac_Func_LeakyRelu.append(mae_y_orientation[i])
        mae_z_orientation_DNN_Ac_Func_LeakyRelu.append(mae_z_orientation[i])

        mse_x_orientation_DNN_Ac_Func_LeakyRelu.append(mse_x_orientation[i])
        mse_y_orientation_DNN_Ac_Func_LeakyRelu.append(mse_y_orientation[i])
        mse_z_orientation_DNN_Ac_Func_LeakyRelu.append(mse_z_orientation[i])

    # CNN Model
    elif "Loss_mse_CNN" in fileName[i] and "Ac_Func_LeakyRelu" in fileName[i]:
        mae_x_orientation_CNN_Ac_Func_LeakyRelu.append(mae_x_orientation[i])
        mae_y_orientation_CNN_Ac_Func_LeakyRelu.append(mae_y_orientation[i])
        mae_z_orientation_CNN_Ac_Func_LeakyRelu.append(mae_z_orientation[i])

        mse_x_orientation_CNN_Ac_Func_LeakyRelu.append(mse_x_orientation[i])
        mse_y_orientation_CNN_Ac_Func_LeakyRelu.append(mse_y_orientation[i])
        mse_z_orientation_CNN_Ac_Func_LeakyRelu.append(mse_z_orientation[i])

    # RNN Model
    elif "Loss_mse_RNN" in fileName[i] and "Ac_Func_LeakyRelu" in fileName[i]:
        mae_x_orientation_RNN_Ac_Func_LeakyRelu.append(mae_x_orientation[i])
        mae_y_orientation_RNN_Ac_Func_LeakyRelu.append(mae_y_orientation[i])
        mae_z_orientation_RNN_Ac_Func_LeakyRelu.append(mae_z_orientation[i])

        mse_x_orientation_RNN_Ac_Func_LeakyRelu.append(mse_x_orientation[i])
        mse_y_orientation_RNN_Ac_Func_LeakyRelu.append(mse_y_orientation[i])
        mse_z_orientation_RNN_Ac_Func_LeakyRelu.append(mse_z_orientation[i])

    # LSTM Model
    elif "Loss_mse_LSTM" in fileName[i] and "Ac_Func_LeakyRelu" in fileName[i]:
        mae_x_orientation_LSTM_Ac_Func_LeakyRelu.append(mae_x_orientation[i])
        mae_y_orientation_LSTM_Ac_Func_LeakyRelu.append(mae_y_orientation[i])
        mae_z_orientation_LSTM_Ac_Func_LeakyRelu.append(mae_z_orientation[i])

        mse_x_orientation_LSTM_Ac_Func_LeakyRelu.append(mse_x_orientation[i])
        mse_y_orientation_LSTM_Ac_Func_LeakyRelu.append(mse_y_orientation[i])
        mse_z_orientation_LSTM_Ac_Func_LeakyRelu.append(mse_z_orientation[i])

    # GRU Model
    elif "Loss_mse_GRU" in fileName[i] and "Ac_Func_LeakyRelu" in fileName[i]:
        mae_x_orientation_GRU_Ac_Func_LeakyRelu.append(mae_x_orientation[i])
        mae_y_orientation_GRU_Ac_Func_LeakyRelu.append(mae_y_orientation[i])
        mae_z_orientation_GRU_Ac_Func_LeakyRelu.append(mae_z_orientation[i])

        mse_x_orientation_GRU_Ac_Func_LeakyRelu.append(mse_x_orientation[i])
        mse_y_orientation_GRU_Ac_Func_LeakyRelu.append(mse_y_orientation[i])
        mse_z_orientation_GRU_Ac_Func_LeakyRelu.append(mse_z_orientation[i])

plt.clf()

currentTime = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) # not needed
figurePath = ".\\SavedFigure\\tanh-Vs-relu-Vs-LeakyRelu\\MSE_of_Position_in_Testing_Data\\"

if not os.path.exists(figurePath):
    os.makedirs(figurePath)

# Ac_Func

# DNN Model

ymin = -0.001
ymax = 0.03

# tanh_relu_And_LeakyRelu_of_Position_in_X_DNN
plt.title("DNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in X axis)")
plt.plot(mse_x_position_DNN_Ac_Func_tanh, marker='o', color='cyan', linestyle='None')
plt.plot(mse_x_position_DNN_Ac_Func_relu, marker='o', color='red', linestyle='None')
plt.plot(mse_x_position_DNN_Ac_Func_LeakyRelu, marker='o', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_X_DNN")
plt.show()

# tanh_relu_And_LeakyRelu_of_Position_in_Y_DNN
plt.title("DNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in Y axis)")
plt.plot(mse_y_position_DNN_Ac_Func_tanh, marker='^', color='cyan', linestyle='None')
plt.plot(mse_y_position_DNN_Ac_Func_relu, marker='^', color='red', linestyle='None')
plt.plot(mse_y_position_DNN_Ac_Func_LeakyRelu, marker='^', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_Y_DNN")
plt.show()

# tanh_relu_And_LeakyRelu_of_Position_in_Z_DNN
plt.title("DNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in Z axis)")
plt.plot(mse_z_position_DNN_Ac_Func_tanh, marker='*', color='cyan', linestyle='None')
plt.plot(mse_z_position_DNN_Ac_Func_relu, marker='*', color='red', linestyle='None')
plt.plot(mse_z_position_DNN_Ac_Func_LeakyRelu, marker='*', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_Z_DNN")
plt.show()

# CNN Model

# tanh_relu_And_LeakyRelu_of_Position_in_X_CNN
plt.title("CNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in X axis)")
plt.plot(mse_x_position_CNN_Ac_Func_tanh, marker='o', color='cyan', linestyle='None')
plt.plot(mse_x_position_CNN_Ac_Func_relu, marker='o', color='red', linestyle='None')
plt.plot(mse_x_position_CNN_Ac_Func_LeakyRelu, marker='o', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_X_CNN")
plt.show()

# tanh_relu_And_LeakyRelu_of_Position_in_Y_CNN
plt.title("CNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in Y axis)")
plt.plot(mse_y_position_CNN_Ac_Func_tanh, marker='^', color='cyan', linestyle='None')
plt.plot(mse_y_position_CNN_Ac_Func_relu, marker='^', color='red', linestyle='None')
plt.plot(mse_y_position_CNN_Ac_Func_LeakyRelu, marker='^', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_Y_CNN")
plt.show()

# tanh_relu_And_LeakyRelu_of_Position_in_Z_CNN
plt.title("CNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in Z axis)")
plt.plot(mse_z_position_CNN_Ac_Func_tanh, marker='*', color='cyan', linestyle='None')
plt.plot(mse_z_position_CNN_Ac_Func_relu, marker='*', color='red', linestyle='None')
plt.plot(mse_z_position_CNN_Ac_Func_LeakyRelu, marker='*', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_Z_CNN")
plt.show()

# RNN Model

# tanh_relu_And_LeakyRelu_of_Position_in_X_RNN
plt.title("RNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in X axis)")
plt.plot(mse_x_position_RNN_Ac_Func_tanh, marker='o', color='cyan', linestyle='None')
plt.plot(mse_x_position_RNN_Ac_Func_relu, marker='o', color='red', linestyle='None')
plt.plot(mse_x_position_RNN_Ac_Func_LeakyRelu, marker='o', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_X_RNN")
plt.show()

# tanh_relu_And_LeakyRelu_of_Position_in_Y_RNN
plt.title("RNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in Y axis)")
plt.plot(mse_y_position_RNN_Ac_Func_tanh, marker='^', color='cyan', linestyle='None')
plt.plot(mse_y_position_RNN_Ac_Func_relu, marker='^', color='red', linestyle='None')
plt.plot(mse_y_position_RNN_Ac_Func_LeakyRelu, marker='^', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_Y_RNN")
plt.show()

# tanh_relu_And_LeakyRelu_of_Position_in_Z_RNN
plt.title("RNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in Z axis)")
plt.plot(mse_z_position_RNN_Ac_Func_tanh, marker='*', color='cyan', linestyle='None')
plt.plot(mse_z_position_RNN_Ac_Func_relu, marker='*', color='red', linestyle='None')
plt.plot(mse_z_position_RNN_Ac_Func_LeakyRelu, marker='*', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_Z_RNN")
plt.show()

# LSTM Model

# tanh_relu_And_LeakyRelu_of_Position_in_X_LSTM
plt.title("LSTM Model (Activation Function: tanh Vs relu Vs LeakyRelu in X axis)")
plt.plot(mse_x_position_LSTM_Ac_Func_tanh, marker='o', color='cyan', linestyle='None')
plt.plot(mse_x_position_LSTM_Ac_Func_relu, marker='o', color='red', linestyle='None')
plt.plot(mse_x_position_LSTM_Ac_Func_LeakyRelu, marker='o', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_X_LSTM")
plt.show()

# tanh_relu_And_LeakyRelu_of_Position_in_Y_LSTM
plt.title("LSTM Model (Activation Function: tanh Vs relu Vs LeakyRelu in Y axis)")
plt.plot(mse_y_position_LSTM_Ac_Func_tanh, marker='^', color='cyan', linestyle='None')
plt.plot(mse_y_position_LSTM_Ac_Func_relu, marker='^', color='red', linestyle='None')
plt.plot(mse_y_position_LSTM_Ac_Func_LeakyRelu, marker='^', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_Y_LSTM")
plt.show()

# tanh_relu_And_LeakyRelu_of_Position_in_Z_LSTM
plt.title("LSTM Model (Activation Function: tanh Vs relu Vs LeakyRelu in Z axis)")
plt.plot(mse_z_position_LSTM_Ac_Func_tanh, marker='*', color='cyan', linestyle='None')
plt.plot(mse_z_position_LSTM_Ac_Func_relu, marker='*', color='red', linestyle='None')
plt.plot(mse_z_position_LSTM_Ac_Func_LeakyRelu, marker='*', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_Z_LSTM")
plt.show()

# GRU Model

# tanh_relu_And_LeakyRelu_of_Position_in_X_GRU
plt.title("GRU Model (Activation Function: tanh Vs relu Vs LeakyRelu in X axis)")
plt.plot(mse_x_position_GRU_Ac_Func_tanh, marker='o', color='cyan', linestyle='None')
plt.plot(mse_x_position_GRU_Ac_Func_relu, marker='o', color='red', linestyle='None')
plt.plot(mse_x_position_GRU_Ac_Func_LeakyRelu, marker='o', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_X_GRU")
plt.show()

# tanh_relu_And_LeakyRelu_of_Position_in_Y_GRU
plt.title("GRU Model (Activation Function: tanh Vs relu Vs LeakyRelu in Y axis)")
plt.plot(mse_y_position_GRU_Ac_Func_tanh, marker='^', color='cyan', linestyle='None')
plt.plot(mse_y_position_GRU_Ac_Func_relu, marker='^', color='red', linestyle='None')
plt.plot(mse_y_position_GRU_Ac_Func_LeakyRelu, marker='^', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_Y_GRU")
plt.show()

# tanh_relu_And_LeakyRelu_of_Position_in_Z_GRU
plt.title("GRU Model (Activation Function: tanh Vs relu Vs LeakyRelu in Z axis)")
plt.plot(mse_z_position_GRU_Ac_Func_tanh, marker='*', color='cyan', linestyle='None')
plt.plot(mse_z_position_GRU_Ac_Func_relu, marker='*', color='red', linestyle='None')
plt.plot(mse_z_position_GRU_Ac_Func_LeakyRelu, marker='*', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_Z_GRU")
plt.show()



# Ac_Func

print("mse_x_orientation_DNN_Ac_Func_tanh: ", mse_x_orientation_DNN_Ac_Func_tanh)
print("mse_x_orientation_DNN_Ac_Func_relu: ", mse_x_orientation_DNN_Ac_Func_relu)
print("mse_x_orientation_DNN_Ac_Func_LeakyRelu: ", mse_x_orientation_DNN_Ac_Func_LeakyRelu)

# DNN Model

ymin = -0.001
ymax = 0.015

# tanh_relu_And_LeakyRelu_of_Orientation_in_X_DNN
plt.title("DNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in X axis)")
plt.plot(mse_x_orientation_DNN_Ac_Func_tanh, marker='o', color='cyan', linestyle='None')
plt.plot(mse_x_orientation_DNN_Ac_Func_relu, marker='o', color='red', linestyle='None')
plt.plot(mse_x_orientation_DNN_Ac_Func_LeakyRelu, marker='o', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_X_DNN")
plt.show()

# tanh_relu_And_LeakyRelu_of_Orientation_in_Y_DNN
plt.title("DNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in Y axis)")
plt.plot(mse_y_orientation_DNN_Ac_Func_tanh, marker='^', color='cyan', linestyle='None')
plt.plot(mse_y_orientation_DNN_Ac_Func_relu, marker='^', color='red', linestyle='None')
plt.plot(mse_y_orientation_DNN_Ac_Func_LeakyRelu, marker='^', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_Y_DNN")
plt.show()

# tanh_relu_And_LeakyRelu_of_Orientation_in_Z_DNN
plt.title("DNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in Z axis)")
plt.plot(mse_z_orientation_DNN_Ac_Func_tanh, marker='*', color='cyan', linestyle='None')
plt.plot(mse_z_orientation_DNN_Ac_Func_relu, marker='*', color='red', linestyle='None')
plt.plot(mse_z_orientation_DNN_Ac_Func_LeakyRelu, marker='*', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_Z_DNN")
plt.show()

# CNN Model

# tanh_relu_And_LeakyRelu_of_Orientation_in_X_CNN
plt.title("CNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in X axis)")
plt.plot(mse_x_orientation_CNN_Ac_Func_tanh, marker='o', color='cyan', linestyle='None')
plt.plot(mse_x_orientation_CNN_Ac_Func_relu, marker='o', color='red', linestyle='None')
plt.plot(mse_x_orientation_CNN_Ac_Func_LeakyRelu, marker='o', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_X_CNN")
plt.show()

# tanh_relu_And_LeakyRelu_of_Orientation_in_Y_CNN
plt.title("CNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in Y axis)")
plt.plot(mse_y_orientation_CNN_Ac_Func_tanh, marker='^', color='cyan', linestyle='None')
plt.plot(mse_y_orientation_CNN_Ac_Func_relu, marker='^', color='red', linestyle='None')
plt.plot(mse_y_orientation_CNN_Ac_Func_LeakyRelu, marker='^', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_Y_CNN")
plt.show()

# tanh_relu_And_LeakyRelu_of_Orientation_in_Z_CNN
plt.title("CNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in Z axis)")
plt.plot(mse_z_orientation_CNN_Ac_Func_tanh, marker='*', color='cyan', linestyle='None')
plt.plot(mse_z_orientation_CNN_Ac_Func_relu, marker='*', color='red', linestyle='None')
plt.plot(mse_z_orientation_CNN_Ac_Func_LeakyRelu, marker='*', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_Z_CNN")
plt.show()

# RNN Model

# tanh_relu_And_LeakyRelu_of_Orientation_in_X_RNN
plt.title("RNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in X axis)")
plt.plot(mse_x_orientation_RNN_Ac_Func_tanh, marker='o', color='cyan', linestyle='None')
plt.plot(mse_x_orientation_RNN_Ac_Func_relu, marker='o', color='red', linestyle='None')
plt.plot(mse_x_orientation_RNN_Ac_Func_LeakyRelu, marker='o', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_X_RNN")
plt.show()

# tanh_relu_And_LeakyRelu_of_Orientation_in_Y_RNN
plt.title("RNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in Y axis)")
plt.plot(mse_y_orientation_RNN_Ac_Func_tanh, marker='^', color='cyan', linestyle='None')
plt.plot(mse_y_orientation_RNN_Ac_Func_relu, marker='^', color='red', linestyle='None')
plt.plot(mse_y_orientation_RNN_Ac_Func_LeakyRelu, marker='^', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_Y_RNN")
plt.show()

# tanh_relu_And_LeakyRelu_of_Orientation_in_Z_RNN
plt.title("RNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in Z axis)")
plt.plot(mse_z_orientation_RNN_Ac_Func_tanh, marker='*', color='cyan', linestyle='None')
plt.plot(mse_z_orientation_RNN_Ac_Func_relu, marker='*', color='red', linestyle='None')
plt.plot(mse_z_orientation_RNN_Ac_Func_LeakyRelu, marker='*', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_Z_RNN")
plt.show()

# LSTM Model

# tanh_relu_And_LeakyRelu_of_Orientation_in_X_LSTM
plt.title("LSTM Model (Activation Function: tanh Vs relu Vs LeakyRelu in X axis)")
plt.plot(mse_x_orientation_LSTM_Ac_Func_tanh, marker='o', color='cyan', linestyle='None')
plt.plot(mse_x_orientation_LSTM_Ac_Func_relu, marker='o', color='red', linestyle='None')
plt.plot(mse_x_orientation_LSTM_Ac_Func_LeakyRelu, marker='o', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_X_LSTM")
plt.show()

# tanh_relu_And_LeakyRelu_of_Orientation_in_Y_LSTM
plt.title("LSTM Model (Activation Function: tanh Vs relu Vs LeakyRelu in Y axis)")
plt.plot(mse_y_orientation_LSTM_Ac_Func_tanh, marker='^', color='cyan', linestyle='None')
plt.plot(mse_y_orientation_LSTM_Ac_Func_relu, marker='^', color='red', linestyle='None')
plt.plot(mse_y_orientation_LSTM_Ac_Func_LeakyRelu, marker='^', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_Y_LSTM")
plt.show()

# tanh_relu_And_LeakyRelu_of_Orientation_in_Z_LSTM
plt.title("LSTM Model (Activation Function: tanh Vs relu Vs LeakyRelu in Z axis)")
plt.plot(mse_z_orientation_LSTM_Ac_Func_tanh, marker='*', color='cyan', linestyle='None')
plt.plot(mse_z_orientation_LSTM_Ac_Func_relu, marker='*', color='red', linestyle='None')
plt.plot(mse_z_orientation_LSTM_Ac_Func_LeakyRelu, marker='*', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_Z_LSTM")
plt.show()

# GRU Model

# tanh_relu_And_LeakyRelu_of_Orientation_in_X_GRU
plt.title("GRU Model (Activation Function: tanh Vs relu Vs LeakyRelu in X axis)")
plt.plot(mse_x_orientation_GRU_Ac_Func_tanh, marker='o', color='cyan', linestyle='None')
plt.plot(mse_x_orientation_GRU_Ac_Func_relu, marker='o', color='red', linestyle='None')
plt.plot(mse_x_orientation_GRU_Ac_Func_LeakyRelu, marker='o', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_X_GRU")
plt.show()

# tanh_relu_And_LeakyRelu_of_Orientation_in_Y_GRU
plt.title("GRU Model (Activation Function: tanh Vs relu Vs LeakyRelu in Y axis)")
plt.plot(mse_y_orientation_GRU_Ac_Func_tanh, marker='^', color='cyan', linestyle='None')
plt.plot(mse_y_orientation_GRU_Ac_Func_relu, marker='^', color='red', linestyle='None')
plt.plot(mse_y_orientation_GRU_Ac_Func_LeakyRelu, marker='^', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_Y_GRU")
plt.show()

# tanh_relu_And_LeakyRelu_of_Orientation_in_Z_GRU
plt.title("GRU Model (Activation Function: tanh Vs relu Vs LeakyRelu in Z axis)")
plt.plot(mse_z_orientation_GRU_Ac_Func_tanh, marker='*', color='cyan', linestyle='None')
plt.plot(mse_z_orientation_GRU_Ac_Func_relu, marker='*', color='red', linestyle='None')
plt.plot(mse_z_orientation_GRU_Ac_Func_LeakyRelu, marker='*', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MSE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_Z_GRU")
plt.show()




##############################################################################################################

currentTime = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) # not needed
figurePath = ".\\SavedFigure\\tanh-Vs-relu-Vs-LeakyRelu\\MAE_of_Position_in_Testing_Data\\"

if not os.path.exists(figurePath):
    os.makedirs(figurePath)

# Ac_Func

# DNN Model

ymin = -0.001
ymax = 0.03

# tanh_relu_And_LeakyRelu_of_Position_in_X_DNN
plt.title("DNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in X axis)")
plt.plot(mae_x_position_DNN_Ac_Func_tanh, marker='o', color='cyan', linestyle='None')
plt.plot(mae_x_position_DNN_Ac_Func_relu, marker='o', color='red', linestyle='None')
plt.plot(mae_x_position_DNN_Ac_Func_LeakyRelu, marker='o', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_X_DNN")
plt.show()

# tanh_relu_And_LeakyRelu_of_Position_in_Y_DNN
plt.title("DNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in Y axis)")
plt.plot(mae_y_position_DNN_Ac_Func_tanh, marker='^', color='cyan', linestyle='None')
plt.plot(mae_y_position_DNN_Ac_Func_relu, marker='^', color='red', linestyle='None')
plt.plot(mae_y_position_DNN_Ac_Func_LeakyRelu, marker='^', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_Y_DNN")
plt.show()

# tanh_relu_And_LeakyRelu_of_Position_in_Z_DNN
plt.title("DNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in Z axis)")
plt.plot(mae_z_position_DNN_Ac_Func_tanh, marker='*', color='cyan', linestyle='None')
plt.plot(mae_z_position_DNN_Ac_Func_relu, marker='*', color='red', linestyle='None')
plt.plot(mae_z_position_DNN_Ac_Func_LeakyRelu, marker='*', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_Z_DNN")
plt.show()

# CNN Model

# tanh_relu_And_LeakyRelu_of_Position_in_X_CNN
plt.title("CNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in X axis)")
plt.plot(mae_x_position_CNN_Ac_Func_tanh, marker='o', color='cyan', linestyle='None')
plt.plot(mae_x_position_CNN_Ac_Func_relu, marker='o', color='red', linestyle='None')
plt.plot(mae_x_position_CNN_Ac_Func_LeakyRelu, marker='o', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_X_CNN")
plt.show()

# tanh_relu_And_LeakyRelu_of_Position_in_Y_CNN
plt.title("CNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in Y axis)")
plt.plot(mae_y_position_CNN_Ac_Func_tanh, marker='^', color='cyan', linestyle='None')
plt.plot(mae_y_position_CNN_Ac_Func_relu, marker='^', color='red', linestyle='None')
plt.plot(mae_y_position_CNN_Ac_Func_LeakyRelu, marker='^', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_Y_CNN")
plt.show()

# tanh_relu_And_LeakyRelu_of_Position_in_Z_CNN
plt.title("CNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in Z axis)")
plt.plot(mae_z_position_CNN_Ac_Func_tanh, marker='*', color='cyan', linestyle='None')
plt.plot(mae_z_position_CNN_Ac_Func_relu, marker='*', color='red', linestyle='None')
plt.plot(mae_z_position_CNN_Ac_Func_LeakyRelu, marker='*', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_Z_CNN")
plt.show()

# RNN Model

# tanh_relu_And_LeakyRelu_of_Position_in_X_RNN
plt.title("RNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in X axis)")
plt.plot(mae_x_position_RNN_Ac_Func_tanh, marker='o', color='cyan', linestyle='None')
plt.plot(mae_x_position_RNN_Ac_Func_relu, marker='o', color='red', linestyle='None')
plt.plot(mae_x_position_RNN_Ac_Func_LeakyRelu, marker='o', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_X_RNN")
plt.show()

# tanh_relu_And_LeakyRelu_of_Position_in_Y_RNN
plt.title("RNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in Y axis)")
plt.plot(mae_y_position_RNN_Ac_Func_tanh, marker='^', color='cyan', linestyle='None')
plt.plot(mae_y_position_RNN_Ac_Func_relu, marker='^', color='red', linestyle='None')
plt.plot(mae_y_position_RNN_Ac_Func_LeakyRelu, marker='^', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_Y_RNN")
plt.show()

# tanh_relu_And_LeakyRelu_of_Position_in_Z_RNN
plt.title("RNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in Z axis)")
plt.plot(mae_z_position_RNN_Ac_Func_tanh, marker='*', color='cyan', linestyle='None')
plt.plot(mae_z_position_RNN_Ac_Func_relu, marker='*', color='red', linestyle='None')
plt.plot(mae_z_position_RNN_Ac_Func_LeakyRelu, marker='*', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_Z_RNN")
plt.show()

# LSTM Model

# tanh_relu_And_LeakyRelu_of_Position_in_X_LSTM
plt.title("LSTM Model (Activation Function: tanh Vs relu Vs LeakyRelu in X axis)")
plt.plot(mae_x_position_LSTM_Ac_Func_tanh, marker='o', color='cyan', linestyle='None')
plt.plot(mae_x_position_LSTM_Ac_Func_relu, marker='o', color='red', linestyle='None')
plt.plot(mae_x_position_LSTM_Ac_Func_LeakyRelu, marker='o', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_X_LSTM")
plt.show()

# tanh_relu_And_LeakyRelu_of_Position_in_Y_LSTM
plt.title("LSTM Model (Activation Function: tanh Vs relu Vs LeakyRelu in Y axis)")
plt.plot(mae_y_position_LSTM_Ac_Func_tanh, marker='^', color='cyan', linestyle='None')
plt.plot(mae_y_position_LSTM_Ac_Func_relu, marker='^', color='red', linestyle='None')
plt.plot(mae_y_position_LSTM_Ac_Func_LeakyRelu, marker='^', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_Y_LSTM")
plt.show()

# tanh_relu_And_LeakyRelu_of_Position_in_Z_LSTM
plt.title("LSTM Model (Activation Function: tanh Vs relu Vs LeakyRelu in Z axis)")
plt.plot(mae_z_position_LSTM_Ac_Func_tanh, marker='*', color='cyan', linestyle='None')
plt.plot(mae_z_position_LSTM_Ac_Func_relu, marker='*', color='red', linestyle='None')
plt.plot(mae_z_position_LSTM_Ac_Func_LeakyRelu, marker='*', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_Z_LSTM")
plt.show()

# GRU Model

# tanh_relu_And_LeakyRelu_of_Position_in_X_GRU
plt.title("GRU Model (Activation Function: tanh Vs relu Vs LeakyRelu in X axis)")
plt.plot(mae_x_position_GRU_Ac_Func_tanh, marker='o', color='cyan', linestyle='None')
plt.plot(mae_x_position_GRU_Ac_Func_relu, marker='o', color='red', linestyle='None')
plt.plot(mae_x_position_GRU_Ac_Func_LeakyRelu, marker='o', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_X_GRU")
plt.show()

# tanh_relu_And_LeakyRelu_of_Position_in_Y_GRU
plt.title("GRU Model (Activation Function: tanh Vs relu Vs LeakyRelu in Y axis)")
plt.plot(mae_y_position_GRU_Ac_Func_tanh, marker='^', color='cyan', linestyle='None')
plt.plot(mae_y_position_GRU_Ac_Func_relu, marker='^', color='red', linestyle='None')
plt.plot(mae_y_position_GRU_Ac_Func_LeakyRelu, marker='^', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_Y_GRU")
plt.show()

# tanh_relu_And_LeakyRelu_of_Position_in_Z_GRU
plt.title("GRU Model (Activation Function: tanh Vs relu Vs LeakyRelu in Z axis)")
plt.plot(mae_z_position_GRU_Ac_Func_tanh, marker='*', color='cyan', linestyle='None')
plt.plot(mae_z_position_GRU_Ac_Func_relu, marker='*', color='red', linestyle='None')
plt.plot(mae_z_position_GRU_Ac_Func_LeakyRelu, marker='*', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Position_in_Z_GRU")
plt.show()



# Ac_Func

print("mae_x_orientation_DNN_Ac_Func_tanh: ", mae_x_orientation_DNN_Ac_Func_tanh)
print("mae_x_orientation_DNN_Ac_Func_relu: ", mae_x_orientation_DNN_Ac_Func_relu)
print("mae_x_orientation_DNN_Ac_Func_LeakyRelu: ", mae_x_orientation_DNN_Ac_Func_LeakyRelu)

# DNN Model

ymin = -0.001
ymax = 0.015

# tanh_relu_And_LeakyRelu_of_Orientation_in_X_DNN
plt.title("DNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in X axis)")
plt.plot(mae_x_orientation_DNN_Ac_Func_tanh, marker='o', color='cyan', linestyle='None')
plt.plot(mae_x_orientation_DNN_Ac_Func_relu, marker='o', color='red', linestyle='None')
plt.plot(mae_x_orientation_DNN_Ac_Func_LeakyRelu, marker='o', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_X_DNN")
plt.show()

# tanh_relu_And_LeakyRelu_of_Orientation_in_Y_DNN
plt.title("DNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in Y axis)")
plt.plot(mae_y_orientation_DNN_Ac_Func_tanh, marker='^', color='cyan', linestyle='None')
plt.plot(mae_y_orientation_DNN_Ac_Func_relu, marker='^', color='red', linestyle='None')
plt.plot(mae_y_orientation_DNN_Ac_Func_LeakyRelu, marker='^', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_Y_DNN")
plt.show()

# tanh_relu_And_LeakyRelu_of_Orientation_in_Z_DNN
plt.title("DNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in Z axis)")
plt.plot(mae_z_orientation_DNN_Ac_Func_tanh, marker='*', color='cyan', linestyle='None')
plt.plot(mae_z_orientation_DNN_Ac_Func_relu, marker='*', color='red', linestyle='None')
plt.plot(mae_z_orientation_DNN_Ac_Func_LeakyRelu, marker='*', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_Z_DNN")
plt.show()

# CNN Model

# tanh_relu_And_LeakyRelu_of_Orientation_in_X_CNN
plt.title("CNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in X axis)")
plt.plot(mae_x_orientation_CNN_Ac_Func_tanh, marker='o', color='cyan', linestyle='None')
plt.plot(mae_x_orientation_CNN_Ac_Func_relu, marker='o', color='red', linestyle='None')
plt.plot(mae_x_orientation_CNN_Ac_Func_LeakyRelu, marker='o', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_X_CNN")
plt.show()

# tanh_relu_And_LeakyRelu_of_Orientation_in_Y_CNN
plt.title("CNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in Y axis)")
plt.plot(mae_y_orientation_CNN_Ac_Func_tanh, marker='^', color='cyan', linestyle='None')
plt.plot(mae_y_orientation_CNN_Ac_Func_relu, marker='^', color='red', linestyle='None')
plt.plot(mae_y_orientation_CNN_Ac_Func_LeakyRelu, marker='^', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_Y_CNN")
plt.show()

# tanh_relu_And_LeakyRelu_of_Orientation_in_Z_CNN
plt.title("CNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in Z axis)")
plt.plot(mae_z_orientation_CNN_Ac_Func_tanh, marker='*', color='cyan', linestyle='None')
plt.plot(mae_z_orientation_CNN_Ac_Func_relu, marker='*', color='red', linestyle='None')
plt.plot(mae_z_orientation_CNN_Ac_Func_LeakyRelu, marker='*', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_Z_CNN")
plt.show()

# RNN Model

# tanh_relu_And_LeakyRelu_of_Orientation_in_X_RNN
plt.title("RNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in X axis)")
plt.plot(mae_x_orientation_RNN_Ac_Func_tanh, marker='o', color='cyan', linestyle='None')
plt.plot(mae_x_orientation_RNN_Ac_Func_relu, marker='o', color='red', linestyle='None')
plt.plot(mae_x_orientation_RNN_Ac_Func_LeakyRelu, marker='o', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_X_RNN")
plt.show()

# tanh_relu_And_LeakyRelu_of_Orientation_in_Y_RNN
plt.title("RNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in Y axis)")
plt.plot(mae_y_orientation_RNN_Ac_Func_tanh, marker='^', color='cyan', linestyle='None')
plt.plot(mae_y_orientation_RNN_Ac_Func_relu, marker='^', color='red', linestyle='None')
plt.plot(mae_y_orientation_RNN_Ac_Func_LeakyRelu, marker='^', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_Y_RNN")
plt.show()

# tanh_relu_And_LeakyRelu_of_Orientation_in_Z_RNN
plt.title("RNN Model (Activation Function: tanh Vs relu Vs LeakyRelu in Z axis)")
plt.plot(mae_z_orientation_RNN_Ac_Func_tanh, marker='*', color='cyan', linestyle='None')
plt.plot(mae_z_orientation_RNN_Ac_Func_relu, marker='*', color='red', linestyle='None')
plt.plot(mae_z_orientation_RNN_Ac_Func_LeakyRelu, marker='*', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_Z_RNN")
plt.show()

# LSTM Model

# tanh_relu_And_LeakyRelu_of_Orientation_in_X_LSTM
plt.title("LSTM Model (Activation Function: tanh Vs relu Vs LeakyRelu in X axis)")
plt.plot(mae_x_orientation_LSTM_Ac_Func_tanh, marker='o', color='cyan', linestyle='None')
plt.plot(mae_x_orientation_LSTM_Ac_Func_relu, marker='o', color='red', linestyle='None')
plt.plot(mae_x_orientation_LSTM_Ac_Func_LeakyRelu, marker='o', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_X_LSTM")
plt.show()

# tanh_relu_And_LeakyRelu_of_Orientation_in_Y_LSTM
plt.title("LSTM Model (Activation Function: tanh Vs relu Vs LeakyRelu in Y axis)")
plt.plot(mae_y_orientation_LSTM_Ac_Func_tanh, marker='^', color='cyan', linestyle='None')
plt.plot(mae_y_orientation_LSTM_Ac_Func_relu, marker='^', color='red', linestyle='None')
plt.plot(mae_y_orientation_LSTM_Ac_Func_LeakyRelu, marker='^', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_Y_LSTM")
plt.show()

# tanh_relu_And_LeakyRelu_of_Orientation_in_Z_LSTM
plt.title("LSTM Model (Activation Function: tanh Vs relu Vs LeakyRelu in Z axis)")
plt.plot(mae_z_orientation_LSTM_Ac_Func_tanh, marker='*', color='cyan', linestyle='None')
plt.plot(mae_z_orientation_LSTM_Ac_Func_relu, marker='*', color='red', linestyle='None')
plt.plot(mae_z_orientation_LSTM_Ac_Func_LeakyRelu, marker='*', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_Z_LSTM")
plt.show()

# GRU Model

# tanh_relu_And_LeakyRelu_of_Orientation_in_X_GRU
plt.title("GRU Model (Activation Function: tanh Vs relu Vs LeakyRelu in X axis)")
plt.plot(mae_x_orientation_GRU_Ac_Func_tanh, marker='o', color='cyan', linestyle='None')
plt.plot(mae_x_orientation_GRU_Ac_Func_relu, marker='o', color='red', linestyle='None')
plt.plot(mae_x_orientation_GRU_Ac_Func_LeakyRelu, marker='o', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_X_GRU")
plt.show()

# tanh_relu_And_LeakyRelu_of_Orientation_in_Y_GRU
plt.title("GRU Model (Activation Function: tanh Vs relu Vs LeakyRelu in Y axis)")
plt.plot(mae_y_orientation_GRU_Ac_Func_tanh, marker='^', color='cyan', linestyle='None')
plt.plot(mae_y_orientation_GRU_Ac_Func_relu, marker='^', color='red', linestyle='None')
plt.plot(mae_y_orientation_GRU_Ac_Func_LeakyRelu, marker='^', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_Y_GRU")
plt.show()

# tanh_relu_And_LeakyRelu_of_Orientation_in_Z_GRU
plt.title("GRU Model (Activation Function: tanh Vs relu Vs LeakyRelu in Z axis)")
plt.plot(mae_z_orientation_GRU_Ac_Func_tanh, marker='*', color='cyan', linestyle='None')
plt.plot(mae_z_orientation_GRU_Ac_Func_relu, marker='*', color='red', linestyle='None')
plt.plot(mae_z_orientation_GRU_Ac_Func_LeakyRelu, marker='*', color='blue', linestyle='None')
axes = plt.gca()
axes.set_ylim([ymin,ymax])
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Activation Function: tanh", "Activation Function: relu", "Activation Function: LeakyRelu"])
plt.savefig(figurePath + "tanh_relu_And_LeakyRelu_of_Orientation_in_Z_GRU")
plt.show()

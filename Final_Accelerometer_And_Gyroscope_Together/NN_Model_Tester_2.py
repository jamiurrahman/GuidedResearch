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

# DNN Model Variables
mae_x_position_DNN = []
mae_y_position_DNN = []
mae_z_position_DNN = []

mse_x_position_DNN = []
mse_y_position_DNN = []
mse_z_position_DNN = []

# CNN Model Variables
mae_x_position_CNN = []
mae_y_position_CNN = []
mae_z_position_CNN = []

mse_x_position_CNN = []
mse_y_position_CNN = []
mse_z_position_CNN = []

# RNN Model Variables
mae_x_position_RNN = []
mae_y_position_RNN = []
mae_z_position_RNN = []

mse_x_position_RNN = []
mse_y_position_RNN = []
mse_z_position_RNN = []

# LSTM Model Variables
mae_x_position_LSTM = []
mae_y_position_LSTM = []
mae_z_position_LSTM = []

mse_x_position_LSTM = []
mse_y_position_LSTM = []
mse_z_position_LSTM = []

# GRU Model Variables
mae_x_position_GRU = []
mae_y_position_GRU = []
mae_z_position_GRU = []

mse_x_position_GRU = []
mse_y_position_GRU = []
mse_z_position_GRU = []

totalData = len(fileName)
for i in range(totalData):

    # DNN Model
    if "Loss_mae_DNN" in fileName[i]:
        mae_x_position_DNN.append(mae_x_position[i])
        mae_y_position_DNN.append(mae_y_position[i])
        mae_z_position_DNN.append(mae_z_position[i])

    elif "Loss_mse_DNN" in fileName[i]:
        mse_x_position_DNN.append(mse_x_position[i])
        mse_y_position_DNN.append(mse_y_position[i])
        mse_z_position_DNN.append(mse_z_position[i])

    # CNN Model
    elif "Loss_mae_CNN" in fileName[i]:
        mae_x_position_CNN.append(mae_x_position[i])
        mae_y_position_CNN.append(mae_y_position[i])
        mae_z_position_CNN.append(mae_z_position[i])

    elif "Loss_mse_CNN" in fileName[i]:
        mse_x_position_CNN.append(mse_x_position[i])
        mse_y_position_CNN.append(mse_y_position[i])
        mse_z_position_CNN.append(mse_z_position[i])

    # RNN Model
    elif "Loss_mae_RNN" in fileName[i]:
        mae_x_position_RNN.append(mae_x_position[i])
        mae_y_position_RNN.append(mae_y_position[i])
        mae_z_position_RNN.append(mae_z_position[i])

    elif "Loss_mse_RNN" in fileName[i]:
        mse_x_position_RNN.append(mse_x_position[i])
        mse_y_position_RNN.append(mse_y_position[i])
        mse_z_position_RNN.append(mse_z_position[i])

    # LSTM Model
    elif "Loss_mae_LSTM" in fileName[i]:
        mae_x_position_LSTM.append(mae_x_position[i])
        mae_y_position_LSTM.append(mae_y_position[i])
        mae_z_position_LSTM.append(mae_z_position[i])

    elif "Loss_mse_LSTM" in fileName[i]:
        mse_x_position_LSTM.append(mse_x_position[i])
        mse_y_position_LSTM.append(mse_y_position[i])
        mse_z_position_LSTM.append(mse_z_position[i])

    # GRU Model
    elif "Loss_mae_GRU" in fileName[i]:
        mae_x_position_GRU.append(mae_x_position[i])
        mae_y_position_GRU.append(mae_y_position[i])
        mae_z_position_GRU.append(mae_z_position[i])

    elif "Loss_mse_GRU" in fileName[i]:
        mse_x_position_GRU.append(mse_x_position[i])
        mse_y_position_GRU.append(mse_y_position[i])
        mse_z_position_GRU.append(mse_z_position[i])


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

# MAE_And_MSE_of_Position_in_X_DNN
plt.title("DNN Model (Loss Function: MAE Vs MSE in X axis)")
plt.plot(mae_x_position_DNN, marker='o', color='cyan', linestyle='None')
plt.plot(mse_x_position_DNN, marker='o', color='red', linestyle='None')
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Position_in_X_DNN")
plt.show()

# MAE_And_MSE_of_Position_in_Y_DNN
plt.title("DNN Model (Loss Function: MAE Vs MSE in Y axis)")
plt.plot(mae_y_position_DNN, marker='^', color='magenta', linestyle='None')
plt.plot(mse_y_position_DNN, marker='^', color='green', linestyle='None')
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Position_in_Y_DNN")
plt.show()

# MAE_And_MSE_of_Position_in_Y_DNN
plt.title("DNN Model (Loss Function: MAE Vs MSE in Z axis)")
plt.plot(mae_z_position_DNN, marker='*', color='blue', linestyle='None')
plt.plot(mse_z_position_DNN, marker='*', color='black', linestyle='None')
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Position_in_Z_DNN")
plt.show()

# CNN Model

# MAE_And_MSE_of_Position_in_X_CNN
plt.title("CNN Model (Loss Function: MAE Vs MSE in X axis)")
plt.plot(mae_x_position_CNN, marker='o', color='cyan', linestyle='None')
plt.plot(mse_x_position_CNN, marker='o', color='red', linestyle='None')
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Position_in_X_CNN")
plt.show()

# MAE_And_MSE_of_Position_in_Y_CNN
plt.title("CNN Model (Loss Function: MAE Vs MSE in Y axis)")
plt.plot(mae_y_position_CNN, marker='^', color='magenta', linestyle='None')
plt.plot(mse_y_position_CNN, marker='^', color='green', linestyle='None')
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Position_in_Y_CNN")
plt.show()

# MAE_And_MSE_of_Position_in_Z_CNN
plt.title("CNN Model (Loss Function: MAE Vs MSE in Z axis)")
plt.plot(mae_z_position_CNN, marker='*', color='blue', linestyle='None')
plt.plot(mse_z_position_CNN, marker='*', color='black', linestyle='None')
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Position_in_Z_CNN")
plt.show()

# RNN Model

# MAE_And_MSE_of_Position_in_X_RNN
plt.title("RNN Model (Loss Function: MAE Vs MSE in X axis)")
plt.plot(mae_x_position_RNN, marker='o', color='cyan', linestyle='None')
plt.plot(mse_x_position_RNN, marker='o', color='red', linestyle='None')
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Position_in_X_RNN")
plt.show()

# MAE_And_MSE_of_Position_in_Y_RNN
plt.title("RNN Model (Loss Function: MAE Vs MSE in Y axis)")
plt.plot(mae_y_position_RNN, marker='^', color='magenta', linestyle='None')
plt.plot(mse_y_position_RNN, marker='^', color='green', linestyle='None')
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Position_in_Y_RNN")
plt.show()

# MAE_And_MSE_of_Position_in_Z_RNN
plt.title("RNN Model (Loss Function: MAE Vs MSE in Z axis)")
plt.plot(mae_z_position_RNN, marker='*', color='blue', linestyle='None')
plt.plot(mse_z_position_RNN, marker='*', color='black', linestyle='None')
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Position_in_Z_RNN")
plt.show()

# LSTM Model

# MAE_And_MSE_of_Position_in_X_LSTM
plt.title("LSTM Model (Loss Function: MAE Vs MSE in X axis)")
plt.plot(mae_x_position_LSTM, marker='o', color='cyan', linestyle='None')
plt.plot(mse_x_position_LSTM, marker='o', color='red', linestyle='None')
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Position_in_X_LSTM")
plt.show()

# MAE_And_MSE_of_Position_in_Y_LSTM
plt.title("LSTM Model (Loss Function: MAE Vs MSE in Y axis)")
plt.plot(mae_y_position_LSTM, marker='^', color='magenta', linestyle='None')
plt.plot(mse_y_position_LSTM, marker='^', color='green', linestyle='None')
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Position_in_Y_LSTM")
plt.show()

# MAE_And_MSE_of_Position_in_Z_LSTM
plt.title("LSTM Model (Loss Function: MAE Vs MSE in Z axis)")
plt.plot(mae_z_position_LSTM, marker='*', color='blue', linestyle='None')
plt.plot(mse_z_position_LSTM, marker='*', color='black', linestyle='None')
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Position_in_Z_LSTM")
plt.show()

# GRU Model

# MAE_And_MSE_of_Position_in_X_GRU
plt.title("GRU Model (Loss Function: MAE Vs MSE in X axis)")
plt.plot(mae_x_position_GRU, marker='o', color='cyan', linestyle='None')
plt.plot(mse_x_position_GRU, marker='o', color='red', linestyle='None')
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Position_in_X_GRU")
plt.show()

# MAE_And_MSE_of_Position_in_Y_GRU
plt.title("GRU Model (Loss Function: MAE Vs MSE in Y axis)")
plt.plot(mae_y_position_GRU, marker='^', color='magenta', linestyle='None')
plt.plot(mse_y_position_GRU, marker='^', color='green', linestyle='None')
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Position_in_Y_GRU")
plt.show()

# MAE_And_MSE_of_Position_in_Z_GRU
plt.title("GRU Model (Loss Function: MAE Vs MSE in Z axis)")
plt.plot(mae_z_position_GRU, marker='*', color='blue', linestyle='None')
plt.plot(mse_z_position_GRU, marker='*', color='black', linestyle='None')
plt.xlabel("Model Version")
plt.ylabel("MAE of Position in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Position_in_Z_GRU")
plt.show()



# DNN Model

# MAE_And_MSE_of_Orientation_in_X_DNN
plt.title("DNN Model (Loss Function: MAE Vs MSE in X axis)")
plt.plot(mae_x_orientation_DNN, marker='o', color='cyan', linestyle='None')
plt.plot(mse_x_orientation_DNN, marker='o', color='red', linestyle='None')
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Orientation_in_X_DNN")
plt.show()

# MAE_And_MSE_of_Orientation_in_Y_DNN
plt.title("DNN Model (Loss Function: MAE Vs MSE in Y axis)")
plt.plot(mae_y_orientation_DNN, marker='^', color='magenta', linestyle='None')
plt.plot(mse_y_orientation_DNN, marker='^', color='green', linestyle='None')
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Orientation_in_Y_DNN")
plt.show()

# MAE_And_MSE_of_Orientation_in_Y_DNN
plt.title("DNN Model (Loss Function: MAE Vs MSE in Z axis)")
plt.plot(mae_z_orientation_DNN, marker='*', color='blue', linestyle='None')
plt.plot(mse_z_orientation_DNN, marker='*', color='black', linestyle='None')
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
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Orientation_in_X_CNN")
plt.show()

# MAE_And_MSE_of_Orientation_in_Y_CNN
plt.title("CNN Model (Loss Function: MAE Vs MSE in Y axis)")
plt.plot(mae_y_orientation_CNN, marker='^', color='magenta', linestyle='None')
plt.plot(mse_y_orientation_CNN, marker='^', color='green', linestyle='None')
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Orientation_in_Y_CNN")
plt.show()

# MAE_And_MSE_of_Orientation_in_Z_CNN
plt.title("CNN Model (Loss Function: MAE Vs MSE in Z axis)")
plt.plot(mae_z_orientation_CNN, marker='*', color='blue', linestyle='None')
plt.plot(mse_z_orientation_CNN, marker='*', color='black', linestyle='None')
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
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Orientation_in_X_RNN")
plt.show()

# MAE_And_MSE_of_Orientation_in_Y_RNN
plt.title("RNN Model (Loss Function: MAE Vs MSE in Y axis)")
plt.plot(mae_y_orientation_RNN, marker='^', color='magenta', linestyle='None')
plt.plot(mse_y_orientation_RNN, marker='^', color='green', linestyle='None')
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Orientation_in_Y_RNN")
plt.show()

# MAE_And_MSE_of_Orientation_in_Z_RNN
plt.title("RNN Model (Loss Function: MAE Vs MSE in Z axis)")
plt.plot(mae_z_orientation_RNN, marker='*', color='blue', linestyle='None')
plt.plot(mse_z_orientation_RNN, marker='*', color='black', linestyle='None')
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
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Orientation_in_X_LSTM")
plt.show()

# MAE_And_MSE_of_Orientation_in_Y_LSTM
plt.title("LSTM Model (Loss Function: MAE Vs MSE in Y axis)")
plt.plot(mae_y_orientation_LSTM, marker='^', color='magenta', linestyle='None')
plt.plot(mse_y_orientation_LSTM, marker='^', color='green', linestyle='None')
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Orientation_in_Y_LSTM")
plt.show()

# MAE_And_MSE_of_Orientation_in_Z_LSTM
plt.title("LSTM Model (Loss Function: MAE Vs MSE in Z axis)")
plt.plot(mae_z_orientation_LSTM, marker='*', color='blue', linestyle='None')
plt.plot(mse_z_orientation_LSTM, marker='*', color='black', linestyle='None')
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
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Orientation_in_X_GRU")
plt.show()

# MAE_And_MSE_of_Orientation_in_Y_GRU
plt.title("GRU Model (Loss Function: MAE Vs MSE in Y axis)")
plt.plot(mae_y_orientation_GRU, marker='^', color='magenta', linestyle='None')
plt.plot(mse_y_orientation_GRU, marker='^', color='green', linestyle='None')
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Orientation_in_Y_GRU")
plt.show()

# MAE_And_MSE_of_Orientation_in_Z_GRU
plt.title("GRU Model (Loss Function: MAE Vs MSE in Z axis)")
plt.plot(mae_z_orientation_GRU, marker='*', color='blue', linestyle='None')
plt.plot(mse_z_orientation_GRU, marker='*', color='black', linestyle='None')
plt.xlabel("Model Version")
plt.ylabel("MAE of Orientation in Testing Data")
plt.legend(["Loss Function: MAE", "Loss Function: MSE"])
plt.savefig(figurePath + "MAE_And_MSE_of_Orientation_in_Z_GRU")
plt.show()


from ConfigParser import *
from Configuration import *
from MyDeepLearningPipeline import *
import datetime

# From Controller
# Start
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

# Configuring current configuration model and so on
#
# Configuration.configure()

#class CurrentConfigChecker:
parser = ConfigParser()
parser.parse("Config.xml")

print(parser.printAllValues())

Configuration.Current_Save_Model = parser.Save_Model
Configuration.Current_Save_History = parser.Save_History
Configuration.Current_Save_TensorBoard = parser.Save_TensorBoard
Configuration.Current_Save_Figure = parser.Save_Figure

counter = 840
Configuration.Current_Time = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

myDeepLearningPipeline = MyDeepLearningPipeline()
myDeepLearningPipeline.preLoadDataset()

for Activation_Function in parser.Activation_Functions:
    # LoadData(Activation_Function) --> Create dataframe
    Configuration.Current_Activation_Function = Activation_Function

    for Batch_Size in parser.Batch_Sizes:

        Configuration.Current_Batch_Size = int(Batch_Size)

        for Epoch in parser.Epochs:

            Configuration.Current_Epoch = int(Epoch)

            for Loss in parser.Losses:

                Configuration.Current_Loss = Loss

                # DNN
                for DNN_Units_Size in parser.DNN_Units_Sizes:
                    counter += 1

                    Configuration.Current_Active_Model = Configuration.Model.DNN
                    Configuration.Current_DNN_Units_Size = int(DNN_Units_Size)

                    Configuration.setCurrentFileName(counter)
                    Configuration.printCurrentFileName(counter)

                    Configuration.configure()

                    myDeepLearningPipeline.runTraining()

                # CNN
                for CNN_Filter_Size in parser.CNN_Filter_Sizes:

                    Configuration.Current_Active_Model = Configuration.Model.CNN
                    Configuration.Current_CNN_Filter_Size = int(CNN_Filter_Size)

                    for CNN_Kernel_Size in parser.CNN_Kernel_Sizes:
                        counter += 1

                        Configuration.Current_CNN_Kernel_Size = int(CNN_Kernel_Size)

                        Configuration.setCurrentFileName(counter)
                        Configuration.printCurrentFileName(counter)

                        Configuration.configure()

                        myDeepLearningPipeline.runTraining()

                # RNN
                for RNN_Units_Size in parser.RNN_Units_Sizes:
                    counter += 1

                    Configuration.Current_Active_Model = Configuration.Model.RNN
                    Configuration.Current_RNN_Units_Size = int(RNN_Units_Size)

                    Configuration.setCurrentFileName(counter)
                    Configuration.printCurrentFileName(counter)

                    Configuration.configure()

                    myDeepLearningPipeline.runTraining()

                # LSTM
                for LSTM_Units_Size in parser.LSTM_Units_Sizes:
                    counter += 1

                    Configuration.Current_Active_Model = Configuration.Model.LSTM
                    Configuration.Current_LSTM_Units_Size = int(LSTM_Units_Size)

                    Configuration.setCurrentFileName(counter)
                    Configuration.printCurrentFileName(counter)

                    Configuration.configure()

                    myDeepLearningPipeline.runTraining()

                # GRU
                for GRU_Units_Size in parser.GRU_Units_Sizes:
                    counter += 1

                    Configuration.Current_Active_Model = Configuration.Model.GRU
                    Configuration.Current_GRU_Units_Size = int(GRU_Units_Size)

                    Configuration.setCurrentFileName(counter)
                    Configuration.printCurrentFileName(counter)

                    Configuration.configure()

                    myDeepLearningPipeline.runTraining()

print("Total Number of Models: ", counter)
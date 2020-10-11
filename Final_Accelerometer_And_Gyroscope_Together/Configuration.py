from NN_Model_Creator import *
from enum import Enum

import numpy as np

class Configuration:
    # Settting values from parser
    Current_Save_Model = 0
    Current_Save_History = 0
    Current_Save_TensorBoard = 0
    Current_Save_Figure = 0

    Current_Activation_Function = ""
    Current_Batch_Size = 0
    Current_Epoch = 0
    Current_Loss = ""

    Current_DNN_Units_Size = 0
    Current_CNN_Filter_Size = 0
    Current_CNN_Kernel_Size = 0
    Current_RNN_Units_Size = 0
    Current_LSTM_Units_Size = 0
    Current_GRU_Units_Size = 0

    class Model(Enum):
        DNN = 1
        CNN = 2
        RNN = 3
        LSTM = 4
        GRU = 5

    Current_Active_Model = Model(1)
    Current_File_Name = ""
    Current_Time = ""

    # Configuration
    # DNN = 0
    # CNN = 0
    # RNN = 0
    # LSTM = 0
    # GRU = 0

    #SAVE_MODEL = 0
    #SAVE_HISTORY = 0

    #BATCH_SIZE = 8
    #SHUFFLE_BUFFER_SIZE = 100
    #EPOCHS = 200

    HistoryPath = str
    ModelPath = str
    TensorBoardPath = str
    FigurePath = str
    #nn_model_creator = NN_Model_Creator()
    model = tf.keras.Sequential()
    #model = tf.python.keras.engine.sequential.Sequential()

    @classmethod
    def printAllValues(cls):

        print("Current_Active_Model : ", cls.Current_Active_Model.name)

        print("Current_Save_Model : ", cls.Current_Save_Model)
        print("Current_Save_History : ", cls.Current_Save_History)
        print("Current_Save_TensorBoard : ", cls.Current_Save_TensorBoard)
        print("Current_Save_Figure : ", cls.Current_Save_Figure)

        print("Current_Activation_Function : ", cls.Current_Activation_Function)
        print("Current_Batch_Size : ", cls.Current_Batch_Size)
        print("Current_Epoch : ", cls.Current_Epoch)
        print("Current_Loss : ", cls.Current_Loss)

        if cls.Current_Active_Model == cls.Model.DNN:
            print("Current_DNN_Units_Size : ", cls.Current_DNN_Units_Size)

        elif cls.Current_Active_Model == cls.Model.CNN:
            print("Current_CNN_Filter_Size : ", cls.Current_CNN_Filter_Size)
            print("Current_CNN_Kernel_Size : ", cls.Current_CNN_Kernel_Size)

        elif cls.Current_Active_Model == cls.Model.RNN:
            print("Current_RNN_Units_Size : ", cls.Current_RNN_Units_Size)

        elif cls.Current_Active_Model == cls.Model.LSTM:
            print("Current_LSTM_Units_Size : ", cls.Current_LSTM_Units_Size)

        elif cls.Current_Active_Model == cls.Model.GRU:
            print("Current_GRU_Units_Size : ", cls.Current_GRU_Units_Size)

        else:
            print("Check ENUM --> There is something wrong with enum implementation")

    @classmethod
    def getInfo(cls):

        info = "Current_Active_Model : " + cls.Current_Active_Model.name + "\n"

        info += "Current_Save_Model : " + cls.Current_Save_Model + "\n"
        info += "Current_Save_History : " + cls.Current_Save_History + "\n"
        info += "Current_Save_TensorBoard : " + cls.Current_Save_TensorBoard + "\n"
        info += "Current_Save_Figure : " + cls.Current_Save_Figure + "\n"

        info += "Current_Activation_Function : " + cls.Current_Activation_Function + "\n"
        info += "Current_Batch_Size : " + str(cls.Current_Batch_Size) + "\n"
        info += "Current_Epoch : " + str(cls.Current_Epoch) + "\n"
        info += "Current_Loss : " + cls.Current_Loss + "\n"

        if cls.Current_Active_Model == cls.Model.DNN:
            info += "Current_DNN_Units_Size : " + str(cls.Current_DNN_Units_Size) + "\n"

        elif cls.Current_Active_Model == cls.Model.CNN:
            info += "Current_CNN_Filter_Size : " + str(cls.Current_CNN_Filter_Size) + "\n"
            info += "Current_CNN_Kernel_Size : " + str(cls.Current_CNN_Kernel_Size) + "\n"

        elif cls.Current_Active_Model == cls.Model.RNN:
            info += "Current_RNN_Units_Size : " + str(cls.Current_RNN_Units_Size) + "\n"

        elif cls.Current_Active_Model == cls.Model.LSTM:
            info += "Current_LSTM_Units_Size : " + str(cls.Current_LSTM_Units_Size) + "\n"

        elif cls.Current_Active_Model == cls.Model.GRU:
            info += "Current_GRU_Units_Size : " + str(cls.Current_GRU_Units_Size) + "\n"

        else:
            info += "Check ENUM --> There is something wrong with enum implementation" + "\n"

        return info




    @classmethod
    def printCurrentFileName(cls, counter):
        print("Current_File_Name", cls.Current_File_Name)

    # Not Needed
    # @classmethod
    # def getCurrentFileName(cls, counter):
    #     return cls.Current_File_Name

    @classmethod
    def setCurrentFileName(cls, counter):
        name = str(counter) \
               + "_Ac_Func_" + str(cls.Current_Activation_Function) \
               + "_Batch_" + str(cls.Current_Batch_Size) + "_Epochs_" + str(cls.Current_Epoch) \
               + "_Loss_" + cls.Current_Loss + "_" + cls.Current_Active_Model.name

        if cls.Current_Active_Model == cls.Model.DNN:
            name = name + "_Units_" + str(cls.Current_DNN_Units_Size)

        elif cls.Current_Active_Model == cls.Model.CNN:
            name = name + "_Filter_" + str(cls.Current_CNN_Filter_Size) \
                        + "_Kernel_" + str(cls.Current_CNN_Kernel_Size)

        elif cls.Current_Active_Model == cls.Model.RNN:
            name = name + "_Units_" + str(cls.Current_RNN_Units_Size)

        elif cls.Current_Active_Model == cls.Model.LSTM:
            name = name + "_Units_" + str(cls.Current_LSTM_Units_Size)

        elif cls.Current_Active_Model == cls.Model.GRU:
            name = name + "_Units_" + str(cls.Current_GRU_Units_Size)

        else:
            print("Check ENUM --> There is something wrong with enum implementation")

        cls.Current_File_Name = name



    @classmethod
    def configure(cls):

        nn_model_creator = NN_Model_Creator()

        if cls.Current_Active_Model == cls.Model.DNN:
            cls.model = nn_model_creator.create_dnn_model(cls.Current_DNN_Units_Size,
                                                              cls.Current_Activation_Function,
                                                              cls.Current_Loss)


        elif cls.Current_Active_Model == cls.Model.CNN:
            cls.model = nn_model_creator.create_cnn_model(cls.Current_CNN_Filter_Size,
                                                              cls.Current_CNN_Kernel_Size,
                                                              cls.Current_Activation_Function,
                                                              cls.Current_Loss)

        elif cls.Current_Active_Model == cls.Model.RNN:
            cls.model = nn_model_creator.create_rnn_model(cls.Current_RNN_Units_Size,
                                                              cls.Current_Activation_Function,
                                                              cls.Current_Loss)

        elif cls.Current_Active_Model == cls.Model.LSTM:
            cls.model = nn_model_creator.create_lstm_model(cls.Current_LSTM_Units_Size,
                                                               cls.Current_Activation_Function,
                                                               cls.Current_Loss)

        elif cls.Current_Active_Model == cls.Model.GRU:
            cls.model = nn_model_creator.create_gru_model(cls.Current_GRU_Units_Size,
                                                              cls.Current_Activation_Function,
                                                              cls.Current_Loss)

        else:
            print("Check ENUM --> There is something wrong with enum implementation")

        cls.HistoryPath = "./SavedHistory/" + cls.Current_Time + "/"
                          #+ "/" + cls.Current_File_Name
        cls.ModelPath = "./SavedModel/" + cls.Current_Time + "/"
                          #+ "/" + cls.Current_File_Name + ".h5"
        cls.TensorBoardPath = ".\\SavedTensorBoard\\" + cls.Current_Time \
                          + "\\" + cls.Current_File_Name + "\\"
        cls.FigurePath = "./SavedFigure/" + cls.Current_Time + "/"
                          #+ "/" + cls.Current_File_Name + "/"

    # def set_historyName(self, name):
    #     self.historyName = name
    #
    # def get_historyName(self):
    #     return self.historyName
    #
    # def set_modelName(self, name):
    #     self.modelName = name
    #
    # def get_modelName(self):
    #     return self.modelName


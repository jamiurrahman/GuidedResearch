import xml.etree.ElementTree as ET

class ConfigParser:
    def parse(self, name):

        myTree = ET.parse(name)
        myRoot = myTree.getroot()

        self.Save_Model = myRoot.find("Save_Model").text
        self.Save_History = myRoot.find("Save_History").text
        self.Save_TensorBoard = myRoot.find("Save_TensorBoard").text
        self.Save_Figure = myRoot.find("Save_Figure").text

        self.Activation_Functions = myRoot.find("Activation_Functions").text.split(" ")
        self.Batch_Sizes = myRoot.find("Batch_Sizes").text.split(" ")
        self.Epochs = myRoot.find("Epochs").text.split(" ")
        self.Losses = myRoot.find("Loss").text.split(" ")

        # DNN
        self.DNN_Units_Sizes = myRoot.find("DNN").find("Units_Sizes").text.split(" ")

        # CNN
        self.CNN_Filter_Sizes = myRoot.find("CNN").find("Filter_Sizes").text.split(" ")
        self.CNN_Kernel_Sizes = myRoot.find("CNN").find("Kernel_Sizes").text.split(" ")

        self.RNN_Units_Sizes = myRoot.find("RNN").find("Units_Sizes").text.split(" ")

        self.LSTM_Units_Sizes = myRoot.find("LSTM").find("Units_Sizes").text.split(" ")

        self.GRU_Units_Sizes = myRoot.find("GRU").find("Units_Sizes").text.split(" ")

    def printAllValues(self):

        print(self.Save_Model)
        print(self.Save_History)
        print(self.Save_TensorBoard)

        print(self.Activation_Functions)
        print(self.Batch_Sizes)
        print(self.Epochs)
        print(self.Losses)

        print(self.DNN_Units_Sizes)

        print(self.CNN_Filter_Sizes)
        print(self.CNN_Kernel_Sizes)

        print(self.RNN_Units_Sizes)
        print(self.LSTM_Units_Sizes)
        print(self.GRU_Units_Sizes)

    def getFileName(self):
        fileName = "_Activation_Function_" + self.Activation_Functions \
               + "_Batch_Size_" + self.Batch_Sizes + "_Epochs_" + self.Epochs \
               + "_CNN_Filter_Size_" + self.CNN_Filter_Sizes + "_CNN_Kernel_Size_" + self.CNN_Kernel_Sizes
        return fileName
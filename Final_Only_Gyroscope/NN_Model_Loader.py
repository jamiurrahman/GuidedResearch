import pickle

from Configuration import *

import tensorflow as tf
import json
import matplotlib.pyplot as plt

Configuration.configure()

loaded_model = tf.keras.models.load_model(Configuration.ModelPath)

#loaded_model.summary()

'''
with open("./SavedHistory/my_history_NN.json", "r") as read_file:
    print(type(read_file))
    history = json.load(read_file)

# print(type(history))
# print(history)
#
# print(history["loss"])
# print(history["mae"])
#
# print(history["loss"]["0"])
# print(history["mae"]["1"])

#print("Loaded_History:", (history["loss"])[0])
'''
with open(Configuration.HistoryPath, 'rb') as file_pi:
    history = pickle.load(file_pi)


def plot_graphs(history, string):
    #epochs_range = range(EPOCHS)
    #plt.plot(epochs_range, history.history[string])
    #plt.plot(epochs_range, history.history['val_' + string])

    #print(history[string])
    #print(history['val_' + string])

    plt.plot(history[string])
    plt.plot(history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    #plt.ylim([0,.001])
    #plt.xlim([200, EPOCHS])
    plt.show()

#plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
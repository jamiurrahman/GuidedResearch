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
train_dataset = dataset_creator.train_dataset
val_dataset = dataset_creator.val_dataset
test_dataset = dataset_creator.test_dataset



# train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
train_dataset = train_dataset.batch(Configuration.BATCH_SIZE)
val_dataset = val_dataset.batch(Configuration.BATCH_SIZE)
test_dataset = test_dataset.batch(Configuration.BATCH_SIZE)

'''
# Working with Model
nn_model_creator = NN_Model_Creator()


if Configuration.DNN:
    model = nn_model_creator.create_dnn_model()
    Configuration.HistoryPath = "./SavedHistory/my_history_DNN"
    Configuration.ModelPath = "./SavedModel/my_model_DNN.h5"

elif Configuration.CNN:
    model = nn_model_creator.create_cnn_model()
    Configuration.HistoryPath = "./SavedHistory/my_history_CNN"
    Configuration.ModelPath = "./SavedModel/my_model_CNN.h5"

elif Configuration.CNN:
    model = nn_model_creator.create_lstm_model()
    Configuration.HistoryPath = "./SavedHistory/my_history_LSTM"
    Configuration.ModelPath = "./SavedModel/my_model_LSTM.h5"

elif Configuration.GRU:
    model = nn_model_creator.create_gru_model()
    Configuration.HistoryPath = "./SavedHistory/my_history_GRU"
    Configuration.ModelPath = "./SavedModel/my_model_GRU.h5"

print("Model Type", type(model))
'''
#Configuration.model.summary()

'''
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    #print("Result:", logs.get('mae'))
    if(logs.get('mae')>0.6):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
'''
log_dir = ".\\temp_logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = Configuration.model.fit(train_dataset, epochs=Configuration.EPOCHS, callbacks=[tensorboard_callback], validation_data=val_dataset)


# Saving Model and History to file

# convert the history.history dict to a pandas DataFrame:
#hist_df = pd.DataFrame(history.history)

#with open('./SavedHistory/my_history_NN.json', 'w') as file:
#    hist_df.to_json(file)

#NN_Model_Saver.save()
if Configuration.SAVE_HISTORY:
    with open(Configuration.HistoryPath, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

if Configuration.SAVE_MODEL:
    Configuration.model.save(Configuration.ModelPath)

def plot_graphs(history, string):
    #epochs_range = range(EPOCHS)
    #plt.plot(epochs_range, history.history[string])
    #plt.plot(epochs_range, history.history['val_' + string])
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel("Loss (mae)")
    plt.legend(["Train", 'Validation'])
    #plt.ylim([0,.001])
    #plt.xlim([200, EPOCHS])
    plt.show()

#plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

Configuration.model.summary()

'''
# Working with test dataset
test_predict_delta_s_x_y_and_z = Configuration.model.predict(test_dataset)

dataset_creator.test_delta_t = dataset_creator.denormalize(dataset_creator.test_delta_t)
dataset_creator.test_delta_s_x_y_and_z = dataset_creator.denormalize(dataset_creator.test_delta_s_x_y_and_z)
test_predict_delta_s_x_y_and_z = dataset_creator.denormalize(test_predict_delta_s_x_y_and_z)

plt.plot(dataset_creator.test_delta_t, dataset_creator.test_delta_s_x_y_and_z[:, 0], "b.", linestyle="None", label="true")
plt.plot(dataset_creator.test_delta_t, test_predict_delta_s_x_y_and_z[:, 0], "r.", linestyle="None", label="prediction")
plt.ylabel('Delta Displacement (m)')
plt.xlabel('Delta Time (ms)')
plt.legend()
plt.title("Test Dataset Result in x-axis")
plt.show();

plt.plot(dataset_creator.test_delta_t, dataset_creator.test_delta_s_x_y_and_z[:, 1], "b.", linestyle="None", label="true")
plt.plot(dataset_creator.test_delta_t, test_predict_delta_s_x_y_and_z[:, 1], "r.", linestyle="None", label="prediction")
plt.ylabel('Delta Displacement (m)')
plt.xlabel('Delta Time (ms)')
plt.legend()
plt.title("Test Dataset Result in y-axis")
plt.show();

plt.plot(dataset_creator.test_delta_t, dataset_creator.test_delta_s_x_y_and_z[:, 2], "b.", linestyle="None", label="true")
plt.plot(dataset_creator.test_delta_t, test_predict_delta_s_x_y_and_z[:, 2], "r.", linestyle="None", label="prediction")
plt.ylabel('Delta Displacement (m)')
plt.xlabel('Delta Time (ms)')
plt.legend()
plt.title("Test Dataset Result in z-axis")
plt.show();

print(dataset_creator.test_delta_s_x_y_and_z.shape)
print(test_predict_delta_s_x_y_and_z.shape)
mae = sum(abs(dataset_creator.test_delta_s_x_y_and_z - test_predict_delta_s_x_y_and_z)) / len(test_predict_delta_s_x_y_and_z)
print("Total MAE Error: ", mae)
'''


'''
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

# Working with Validation dataset
val_predict_delta_s = model.predict(val_dataset)

# Working with Plotter
epochs_range = range(EPOCHS)

plotter = Plotter()




#plotter.plot(epochs_range, loss, val_loss, "Training and Validation Loss", "Epochs", "Loss", "Training Loss", "Validation Loss", linestyle='-', marker="o")
#plotter.plot(dataset_creator.val_delta_t, dataset_creator.val_delta_s.flatten(), val_predict_delta_s.flatten(), "Validation Dataset Result", "Delta Time", "Delta Time", "true", "prediction")

plt.plot(val_delta_t, val_delta_s.flatten(), "b.", linestyle="None", label="true")
plt.plot(val_delta_t, val_predict_delta_s.flatten(), "r.", linestyle="None", label="prediction")
plt.ylabel('Delta S')
plt.xlabel('Delta Time')
plt.legend()
plt.title("Validation Dataset Result")
plt.show();

plt.plot(val_acc_1_x, val_delta_s.flatten(), "b.", linestyle="None", label="true")
plt.plot(val_acc_1_x, val_predict_delta_s.flatten(), "r.", linestyle="None", label="prediction")
plt.ylabel('Delta S')
plt.xlabel('Acceleration')
plt.legend()
plt.title("Validation Dataset Result")
plt.show();


# Working with test dataset
test_predict_delta_s = model.predict(test_dataset)

plt.plot(test_delta_t, test_delta_s.flatten(), "b.", linestyle="None", label="true")
plt.plot(test_delta_t, test_predict_delta_s.flatten(), "r.", linestyle="None", label="prediction")
plt.ylabel('Delta S')
plt.xlabel('Delta Time')
plt.legend()
plt.title("Test Dataset Result")
plt.show();

plt.plot(test_acc_1_x, test_delta_s.flatten(), "b.", linestyle="None", label="true")
plt.plot(test_acc_1_x, test_predict_delta_s.flatten(), "r.", linestyle="None", label="prediction")
plt.ylabel('Delta S')
plt.xlabel('Acceleration')
plt.legend()
plt.title("Test Dataset Result")
plt.show();

# Plotting test dataset in 3d
test_predict_delta_s = model.predict(test_dataset)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(test_delta_t, test_acc_1_x, test_delta_s.flatten(), color='b', cmap='Greens');
ax.scatter3D(test_delta_t, test_acc_1_x, test_predict_delta_s.flatten(), color='red', cmap='Greens');
ax.set_xlabel('Delta Time')
ax.set_ylabel('Acc')
ax.set_zlabel('Delta S')
plt.title("In X Axis")
plt.show();

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(test_delta_t, test_acc_1_y, test_delta_s.flatten(), color='b', cmap='Greens');
ax.scatter3D(test_delta_t, test_acc_1_y, test_predict_delta_s.flatten(), color='red', cmap='Greens');
ax.set_xlabel('Delta Time')
ax.set_ylabel('Acc')
ax.set_zlabel('Delta S')
plt.title("In Y Axis")
plt.show();

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(test_delta_t, test_acc_1_z, test_delta_s.flatten(), color='b', cmap='Greens');
ax.scatter3D(test_delta_t, test_acc_1_z, test_predict_delta_s.flatten(), color='red', cmap='Greens');
ax.set_xlabel('Delta Time')
ax.set_ylabel('Acc')
ax.set_zlabel('Delta S')
plt.title("In Z Axis")
plt.show();
'''
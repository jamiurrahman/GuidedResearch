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

import os

class MyDeepLearningPipeline:

    def preLoadDataset(self):
        # Configuring current configuration model and so on
        # Configuration.configure()

        # Working with Raw Data
        data_preprocessor = Data_PreProcessor()
        raw_data = data_preprocessor.get_data_for_all_sensors()
        print(raw_data.head())

        # Working with Dataset
        self.train_dataset = tf.data.Dataset.from_generator
        self.val_dataset = tf.data.Dataset.from_generator
        self.test_dataset = tf.data.Dataset.from_generator

        self.dataset_creator = Dataset_Creator()
        self.dataset_creator.create_dataset_for_all_sensors(raw_data)

        # Logging
        logPath = "./logs/" + Configuration.Current_Time + "/"
        if not os.path.exists(logPath):
            os.makedirs(logPath)

        # create logger with 'spam_application'
        self.logger = logging.getLogger('my_application')
        self.logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(os.path.join(logPath, "my_log.log"))
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)

    def runTraining(self):



        self.logger.info("\n***************************************************************************************\n")
        self.logger.info("Current Model: \n" + Configuration.Current_File_Name)
        self.logger.info("Current Model Info: \n" + Configuration.getInfo())

        start = timeit.default_timer()

        self.train_dataset = self.dataset_creator.train_dataset
        self.val_dataset = self.dataset_creator.val_dataset
        self.test_dataset = self.dataset_creator.test_dataset

        # train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        self.train_dataset = self.train_dataset.batch(Configuration.Current_Batch_Size)
        self.val_dataset = self.val_dataset.batch(Configuration.Current_Batch_Size)
        self.test_dataset = self.test_dataset.batch(Configuration.Current_Batch_Size)

        # if Configuration.Current_Save_TensorBoard:
        #     # tensorBoard_dir = ".\\temp_logs\\fit\\" + Configuration.Current_File_Name
        #     tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=Configuration.TensorBoardPath, histogram_freq=1)
        #
        #     history = Configuration.model.fit(self.train_dataset, epochs=Configuration.Current_Epoch, callbacks=[tensorboard_callback], validation_data=self.val_dataset)
        #
        # else:
        history = Configuration.model.fit(self.train_dataset, epochs=Configuration.Current_Epoch, validation_data=self.val_dataset)

        stop = timeit.default_timer()
        #print("For {} time needed: {}".format(Configuration.Current_File_Name, (stop - start)))
        self.logger.info("Time needed: {}".format((stop - start)))


        # Saving Model and History to file

        # convert the history.history dict to a pandas DataFrame:
        #hist_df = pd.DataFrame(history.history)

        #with open('./SavedHistory/my_history_NN.json', 'w') as file:
        #    hist_df.to_json(file)

        #NN_Model_Saver.save()
        if Configuration.Current_Save_History:
            if not os.path.exists(Configuration.HistoryPath):
                os.makedirs(Configuration.HistoryPath)

            with open(os.path.join(Configuration.HistoryPath, Configuration.Current_File_Name), 'wb') as file_pi:
                pickle.dump(history.history, file_pi)

        if not os.path.exists(Configuration.ModelPath):
            os.makedirs(Configuration.ModelPath)

        if Configuration.Current_Save_Model:
            Configuration.model.save(os.path.join(Configuration.ModelPath, Configuration.Current_File_Name + ".h5"))

        def plot_graphs(history, string):
            plt.clf()
            #epochs_range = range(EPOCHS)
            #plt.plot(epochs_range, history.history[string])
            #plt.plot(epochs_range, history.history['val_' + string])
            plt.plot(history.history[string])
            plt.plot(history.history['val_' + string])
            plt.xlabel("Epochs")
            plt.ylabel("Loss (" + Configuration.Current_Loss + ")")
            plt.legend(["Train", 'Validation'])
            #plt.ylim([0,.001])
            #plt.xlim([200, EPOCHS])

            #plt.show()
            if Configuration.Current_Save_Figure:
                if not os.path.exists(Configuration.FigurePath):
                    os.makedirs(Configuration.FigurePath)

                plt.savefig(os.path.join(Configuration.FigurePath, Configuration.Current_File_Name + ".png"))

            else:
                plt.show()

        #plot_graphs(history, 'accuracy')
        plot_graphs(history, 'loss')


        Configuration.model.summary(print_fn=self.logger.info)
        #self.logger.info("History: \n", history.history['loss'])
        #self.logger.info("History: \n", history.history['val_loss'])


        # Working with test dataset
        test_predict_delta_alpha_x_y_and_z = Configuration.model.predict(self.test_dataset)
        
        self.dataset_creator.test_delta_t = self.dataset_creator.denormalize(self.dataset_creator.test_delta_t)
        self.dataset_creator.test_delta_alpha_x_y_and_z = self.dataset_creator.denormalize(self.dataset_creator.test_delta_alpha_x_y_and_z)
        test_predict_delta_alpha_x_y_and_z = self.dataset_creator.denormalize(test_predict_delta_alpha_x_y_and_z)

        '''
        plt.plot(self.dataset_creator.test_delta_t, self.dataset_creator.test_delta_s_x_y_and_z[:, 0], "b.", linestyle="None", label="true")
        plt.plot(self.dataset_creator.test_delta_t, test_predict_delta_s_x_y_and_z[:, 0], "r.", linestyle="None", label="prediction")
        plt.ylabel('Delta Displacement (m)')
        plt.xlabel('Delta Time (ms)')
        plt.legend()
        plt.title("Test Dataset Result in x-axis")
        plt.show();
        
        plt.plot(self.dataset_creator.test_delta_t, self.dataset_creator.test_delta_s_x_y_and_z[:, 1], "b.", linestyle="None", label="true")
        plt.plot(self.dataset_creator.test_delta_t, test_predict_delta_s_x_y_and_z[:, 1], "r.", linestyle="None", label="prediction")
        plt.ylabel('Delta Displacement (m)')
        plt.xlabel('Delta Time (ms)')
        plt.legend()
        plt.title("Test Dataset Result in y-axis")
        plt.show();
        
        plt.plot(self.dataset_creator.test_delta_t, self.dataset_creator.test_delta_s_x_y_and_z[:, 2], "b.", linestyle="None", label="true")
        plt.plot(self.dataset_creator.test_delta_t, test_predict_delta_s_x_y_and_z[:, 2], "r.", linestyle="None", label="prediction")
        plt.ylabel('Delta Displacement (m)')
        plt.xlabel('Delta Time (ms)')
        plt.legend()
        plt.title("Test Dataset Result in z-axis")
        plt.show();
        '''

        # print(self.dataset_creator.test_delta_s_x_y_and_z.shape)
        # print(test_predict_delta_s_x_y_and_z.shape)
        mae = sum(abs(self.dataset_creator.test_delta_alpha_x_y_and_z - test_predict_delta_alpha_x_y_and_z)) / len(test_predict_delta_alpha_x_y_and_z)
        #print("Total MAE Error: ", mae)
        self.logger.info("Total MAE Error: {}".format(mae))



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
import numpy as np
import tensorflow as tf




class Dataset_Creator:
    #train_dataset = tf.data.Dataset.from_generator
    #val_dataset = tf.data.Dataset.from_generator
    #test_dataset = tf.data.Dataset.from_generator

    #val_delta_t = []
    #val_delta_alpha = []

    # Normalizing Data
    # Because It is important to scale features before training a neural network.
    # Standardization is a common way of doing this scaling by subtracting the mean
    # and dividing by the standard deviation of each feature.
    # We could also use a tf.keras.utils.normalize method that rescales the values into a range of [0,1].

    def normalize(self, data, TRAIN_SIZE=5000):
        # Note: The mean and standard deviation should only be computed using the training data.
        self.data_train_mean = data[:TRAIN_SIZE].mean()
        self.data_train_std = data[:TRAIN_SIZE].std()

        return ((data - self.data_train_mean) / self.data_train_std)

    def denormalize(self, data, TRAIN_SPLIT=5000):
        # Note: The mean and standard deviation should only be computed using the training data.
        data_train_mean = data[:TRAIN_SPLIT].mean()
        data_train_std = data[:TRAIN_SPLIT].std()

        return ((data * data_train_std) + data_train_mean)

    # def denormalize(self, data):
    #     # Note: The mean and standard deviation should only be computed using the training data.
    #     # data_train_mean = data[:TRAIN_SPLIT].mean()
    #     # data_train_std = data[:TRAIN_SPLIT].std()
    #
    #     return ((data * self.data_train_std) + self.data_train_mean)

    def create_dataset_for_all_sensors(self, df):
        original_gyro_1_x_df = df["w_1_x"]
        original_gyro_2_x_df = df["w_2_x"]
        original_gyro_3_x_df = df["w_3_x"]
        original_gyro_4_x_df = df["w_4_x"]
        original_gyro_5_x_df = df["w_5_x"]
        original_gyro_6_x_df = df["w_6_x"]
        original_gyro_7_x_df = df["w_7_x"]
        original_gyro_8_x_df = df["w_8_x"]
        original_orientation_x_t_df = df["gt_orientation_x_t"]

        original_gyro_1_y_df = df["w_1_y"]
        original_gyro_2_y_df = df["w_2_y"]
        original_gyro_3_y_df = df["w_3_y"]
        original_gyro_4_y_df = df["w_4_y"]
        original_gyro_5_y_df = df["w_5_y"]
        original_gyro_6_y_df = df["w_6_y"]
        original_gyro_7_y_df = df["w_7_y"]
        original_gyro_8_y_df = df["w_8_y"]
        original_orientation_y_t_df = df["gt_orientation_y_t"]

        original_gyro_1_z_df = df["w_1_z"]
        original_gyro_2_z_df = df["w_2_z"]
        original_gyro_3_z_df = df["w_3_z"]
        original_gyro_4_z_df = df["w_4_z"]
        original_gyro_5_z_df = df["w_5_z"]
        original_gyro_6_z_df = df["w_6_z"]
        original_gyro_7_z_df = df["w_7_z"]
        original_gyro_8_z_df = df["w_8_z"]
        original_orientation_z_t_df = df["gt_orientation_z_t"]

        original_time_step_df = df["simu_time"]

        original_gyro_1_x = np.asfarray(original_gyro_1_x_df.to_numpy(), np.float)
        original_gyro_2_x = np.asfarray(original_gyro_2_x_df.to_numpy(), np.float)
        original_gyro_3_x = np.asfarray(original_gyro_3_x_df.to_numpy(), np.float)
        original_gyro_4_x = np.asfarray(original_gyro_4_x_df.to_numpy(), np.float)
        original_gyro_5_x = np.asfarray(original_gyro_5_x_df.to_numpy(), np.float)
        original_gyro_6_x = np.asfarray(original_gyro_6_x_df.to_numpy(), np.float)
        original_gyro_7_x = np.asfarray(original_gyro_7_x_df.to_numpy(), np.float)
        original_gyro_8_x = np.asfarray(original_gyro_8_x_df.to_numpy(), np.float)
        original_orientation_x_t = np.asfarray(original_orientation_x_t_df.to_numpy(), np.float)

        original_gyro_1_y = np.asfarray(original_gyro_1_y_df.to_numpy(), np.float)
        original_gyro_2_y = np.asfarray(original_gyro_2_y_df.to_numpy(), np.float)
        original_gyro_3_y = np.asfarray(original_gyro_3_y_df.to_numpy(), np.float)
        original_gyro_4_y = np.asfarray(original_gyro_4_y_df.to_numpy(), np.float)
        original_gyro_5_y = np.asfarray(original_gyro_5_y_df.to_numpy(), np.float)
        original_gyro_6_y = np.asfarray(original_gyro_6_y_df.to_numpy(), np.float)
        original_gyro_7_y = np.asfarray(original_gyro_7_y_df.to_numpy(), np.float)
        original_gyro_8_y = np.asfarray(original_gyro_8_y_df.to_numpy(), np.float)
        original_orientation_y_t = np.asfarray(original_orientation_y_t_df.to_numpy(), np.float)

        original_gyro_1_z = np.asfarray(original_gyro_1_z_df.to_numpy(), np.float)
        original_gyro_2_z = np.asfarray(original_gyro_2_z_df.to_numpy(), np.float)
        original_gyro_3_z = np.asfarray(original_gyro_3_z_df.to_numpy(), np.float)
        original_gyro_4_z = np.asfarray(original_gyro_4_z_df.to_numpy(), np.float)
        original_gyro_5_z = np.asfarray(original_gyro_5_z_df.to_numpy(), np.float)
        original_gyro_6_z = np.asfarray(original_gyro_6_z_df.to_numpy(), np.float)
        original_gyro_7_z = np.asfarray(original_gyro_7_z_df.to_numpy(), np.float)
        original_gyro_8_z = np.asfarray(original_gyro_8_z_df.to_numpy(), np.float)
        original_orientation_z_t = np.asfarray(original_orientation_z_t_df.to_numpy(), np.float)

        original_time_step = np.asfarray(original_time_step_df.to_numpy(), np.float)
        time_step = original_time_step - original_time_step[0]
        delta_t = np.abs(np.diff(original_time_step, axis=0))  # delta_t should not be negative
        #delta_t = delta_t / 1000  # delta_t is in milli sec. ## Converting ms to sec

        delta_alpha_x = (np.diff(original_orientation_x_t, axis=0))
        delta_alpha_y = (np.diff(original_orientation_y_t, axis=0))
        delta_alpha_z = (np.diff(original_orientation_z_t, axis=0))

        #delta_alpha = ((delta_alpha_x ** 2) + (delta_alpha_y ** 2) + (delta_alpha_z ** 2)) ** 0.5  # calculating total displacement

        print(delta_alpha_x[0:5])
        print(delta_alpha_y[0:5])
        print(delta_alpha_z[0:5])
        #print(delta_alpha[0:5])

        original_gyro_1_x = np.expand_dims(original_gyro_1_x, axis=1)  # Expanding one axis to column
        original_gyro_2_x = np.expand_dims(original_gyro_2_x, axis=1)  # Expanding one axis to column
        original_gyro_3_x = np.expand_dims(original_gyro_3_x, axis=1)  # Expanding one axis to column
        original_gyro_4_x = np.expand_dims(original_gyro_4_x, axis=1)  # Expanding one axis to column
        original_gyro_5_x = np.expand_dims(original_gyro_5_x, axis=1)  # Expanding one axis to column
        original_gyro_6_x = np.expand_dims(original_gyro_6_x, axis=1)  # Expanding one axis to column
        original_gyro_7_x = np.expand_dims(original_gyro_7_x, axis=1)  # Expanding one axis to column
        original_gyro_8_x = np.expand_dims(original_gyro_8_x, axis=1)  # Expanding one axis to column

        original_gyro_1_y = np.expand_dims(original_gyro_1_y, axis=1)  # Expanding one axis to column
        original_gyro_2_y = np.expand_dims(original_gyro_2_y, axis=1)  # Expanding one axis to column
        original_gyro_3_y = np.expand_dims(original_gyro_3_y, axis=1)  # Expanding one axis to column
        original_gyro_4_y = np.expand_dims(original_gyro_4_y, axis=1)  # Expanding one axis to column
        original_gyro_5_y = np.expand_dims(original_gyro_5_y, axis=1)  # Expanding one axis to column
        original_gyro_6_y = np.expand_dims(original_gyro_6_y, axis=1)  # Expanding one axis to column
        original_gyro_7_y = np.expand_dims(original_gyro_7_y, axis=1)  # Expanding one axis to column
        original_gyro_8_y = np.expand_dims(original_gyro_8_y, axis=1)  # Expanding one axis to column

        original_gyro_1_z = np.expand_dims(original_gyro_1_z, axis=1)  # Expanding one axis to column
        original_gyro_2_z = np.expand_dims(original_gyro_2_z, axis=1)  # Expanding one axis to column
        original_gyro_3_z = np.expand_dims(original_gyro_3_z, axis=1)  # Expanding one axis to column
        original_gyro_4_z = np.expand_dims(original_gyro_4_z, axis=1)  # Expanding one axis to column
        original_gyro_5_z = np.expand_dims(original_gyro_5_z, axis=1)  # Expanding one axis to column
        original_gyro_6_z = np.expand_dims(original_gyro_6_z, axis=1)  # Expanding one axis to column
        original_gyro_7_z = np.expand_dims(original_gyro_7_z, axis=1)  # Expanding one axis to column
        original_gyro_8_z = np.expand_dims(original_gyro_8_z, axis=1)  # Expanding one axis to column

        delta_t = np.expand_dims(delta_t, axis=1)  # Expanding one axis to column
        # delta_alpha = np.expand_dims(delta_alpha, axis=1)  # Expanding one axis to column
        delta_alpha_x = np.expand_dims(delta_alpha_x, axis=1)  # Expanding one axis to column
        delta_alpha_y = np.expand_dims(delta_alpha_y, axis=1)  # Expanding one axis to column
        delta_alpha_z = np.expand_dims(delta_alpha_z, axis=1)  # Expanding one axis to column

        self.TRAIN_SIZE = int(len(delta_t) * 0.6)  # 5000
        VALIDATION_SIZE = int(len(delta_t) * 0.20)  # 500
        TEST_SIZE = int(len(delta_t) - (self.TRAIN_SIZE + VALIDATION_SIZE))

        normalized_gyro_1_x = self.normalize(original_gyro_1_x, TRAIN_SIZE=self.TRAIN_SIZE)
        normalized_gyro_2_x = self.normalize(original_gyro_2_x, TRAIN_SIZE=self.TRAIN_SIZE)
        normalized_gyro_3_x = self.normalize(original_gyro_3_x, TRAIN_SIZE=self.TRAIN_SIZE)
        normalized_gyro_4_x = self.normalize(original_gyro_4_x, TRAIN_SIZE=self.TRAIN_SIZE)
        normalized_gyro_5_x = self.normalize(original_gyro_5_x, TRAIN_SIZE=self.TRAIN_SIZE)
        normalized_gyro_6_x = self.normalize(original_gyro_6_x, TRAIN_SIZE=self.TRAIN_SIZE)
        normalized_gyro_7_x = self.normalize(original_gyro_7_x, TRAIN_SIZE=self.TRAIN_SIZE)
        normalized_gyro_8_x = self.normalize(original_gyro_8_x, TRAIN_SIZE=self.TRAIN_SIZE)

        normalized_gyro_1_y = self.normalize(original_gyro_1_y, TRAIN_SIZE=self.TRAIN_SIZE)
        normalized_gyro_2_y = self.normalize(original_gyro_2_y, TRAIN_SIZE=self.TRAIN_SIZE)
        normalized_gyro_3_y = self.normalize(original_gyro_3_y, TRAIN_SIZE=self.TRAIN_SIZE)
        normalized_gyro_4_y = self.normalize(original_gyro_4_y, TRAIN_SIZE=self.TRAIN_SIZE)
        normalized_gyro_5_y = self.normalize(original_gyro_5_y, TRAIN_SIZE=self.TRAIN_SIZE)
        normalized_gyro_6_y = self.normalize(original_gyro_6_y, TRAIN_SIZE=self.TRAIN_SIZE)
        normalized_gyro_7_y = self.normalize(original_gyro_7_y, TRAIN_SIZE=self.TRAIN_SIZE)
        normalized_gyro_8_y = self.normalize(original_gyro_8_y, TRAIN_SIZE=self.TRAIN_SIZE)

        normalized_gyro_1_z = self.normalize(original_gyro_1_z, TRAIN_SIZE=self.TRAIN_SIZE)
        normalized_gyro_2_z = self.normalize(original_gyro_2_z, TRAIN_SIZE=self.TRAIN_SIZE)
        normalized_gyro_3_z = self.normalize(original_gyro_3_z, TRAIN_SIZE=self.TRAIN_SIZE)
        normalized_gyro_4_z = self.normalize(original_gyro_4_z, TRAIN_SIZE=self.TRAIN_SIZE)
        normalized_gyro_5_z = self.normalize(original_gyro_5_z, TRAIN_SIZE=self.TRAIN_SIZE)
        normalized_gyro_6_z = self.normalize(original_gyro_6_z, TRAIN_SIZE=self.TRAIN_SIZE)
        normalized_gyro_7_z = self.normalize(original_gyro_7_z, TRAIN_SIZE=self.TRAIN_SIZE)
        normalized_gyro_8_z = self.normalize(original_gyro_8_z, TRAIN_SIZE=self.TRAIN_SIZE)

        normalized_delta_t = self.normalize(delta_t, TRAIN_SIZE=self.TRAIN_SIZE)

        # plot(delta_t, original_acc1_x, delta_alpha, "delta_t", "original_acc1_x", "delta_alpha")
        # plot(normalized_delta_t, normalized_acc1_x, delta_alpha, "normalized_delta_t", "normalized_acc1_x", "delta_alpha")

        def split(data, TRAIN_SIZE=5000, VALIDATION_SIZE=500, TEST_SIZE=None):
            data_train = data[:TRAIN_SIZE]
            data_validation = data[TRAIN_SIZE:(TRAIN_SIZE + VALIDATION_SIZE)]
            data_test = data[(TRAIN_SIZE + VALIDATION_SIZE):(TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE)]

            return data_train, data_validation, data_test

        train_gyro_1_x, val_gyro_1_x, test_gyro_1_x = split(normalized_gyro_1_x, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)
        train_gyro_2_x, val_gyro_2_x, test_gyro_2_x = split(normalized_gyro_2_x, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)
        train_gyro_3_x, val_gyro_3_x, test_gyro_3_x = split(normalized_gyro_3_x, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)
        train_gyro_4_x, val_gyro_4_x, test_gyro_4_x = split(normalized_gyro_4_x, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)
        train_gyro_5_x, val_gyro_5_x, test_gyro_5_x = split(normalized_gyro_5_x, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)
        train_gyro_6_x, val_gyro_6_x, test_gyro_6_x = split(normalized_gyro_6_x, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)
        train_gyro_7_x, val_gyro_7_x, test_gyro_7_x = split(normalized_gyro_7_x, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)
        train_gyro_8_x, val_gyro_8_x, test_gyro_8_x = split(normalized_gyro_8_x, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)
        train_delta_t, val_delta_t, test_delta_t = split(normalized_delta_t, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)

        train_gyro_1_y, val_gyro_1_y, test_gyro_1_y = split(normalized_gyro_1_y, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)
        train_gyro_2_y, val_gyro_2_y, test_gyro_2_y = split(normalized_gyro_2_y, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)
        train_gyro_3_y, val_gyro_3_y, test_gyro_3_y = split(normalized_gyro_3_y, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)
        train_gyro_4_y, val_gyro_4_y, test_gyro_4_y = split(normalized_gyro_4_y, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)
        train_gyro_5_y, val_gyro_5_y, test_gyro_5_y = split(normalized_gyro_5_y, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)
        train_gyro_6_y, val_gyro_6_y, test_gyro_6_y = split(normalized_gyro_6_y, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)
        train_gyro_7_y, val_gyro_7_y, test_gyro_7_y = split(normalized_gyro_7_y, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)
        train_gyro_8_y, val_gyro_8_y, test_gyro_8_y = split(normalized_gyro_8_y, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)
        train_delta_t, val_delta_t, test_delta_t = split(normalized_delta_t, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)

        train_gyro_1_z, val_gyro_1_z, test_gyro_1_z = split(normalized_gyro_1_z, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)
        train_gyro_2_z, val_gyro_2_z, test_gyro_2_z = split(normalized_gyro_2_z, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)
        train_gyro_3_z, val_gyro_3_z, test_gyro_3_z = split(normalized_gyro_3_z, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)
        train_gyro_4_z, val_gyro_4_z, test_gyro_4_z = split(normalized_gyro_4_z, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)
        train_gyro_5_z, val_gyro_5_z, test_gyro_5_z = split(normalized_gyro_5_z, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)
        train_gyro_6_z, val_gyro_6_z, test_gyro_6_z = split(normalized_gyro_6_z, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)
        train_gyro_7_z, val_gyro_7_z, test_gyro_7_z = split(normalized_gyro_7_z, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)
        train_gyro_8_z, val_gyro_8_z, test_gyro_8_z = split(normalized_gyro_8_z, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)
        train_delta_t, val_delta_t, self.test_delta_t = split(normalized_delta_t, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)

        train_delta_alpha_x, val_delta_alpha_x, self.test_delta_alpha_x = split(delta_alpha_x, TRAIN_SIZE=self.TRAIN_SIZE,
                                                         VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)
        train_delta_alpha_y, val_delta_alpha_y, self.test_delta_alpha_y = split(delta_alpha_y, TRAIN_SIZE=self.TRAIN_SIZE,
                                                              VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)
        train_delta_alpha_z, val_delta_alpha_z, self.test_delta_alpha_z = split(delta_alpha_z, TRAIN_SIZE=self.TRAIN_SIZE,
                                                              VALIDATION_SIZE=VALIDATION_SIZE, TEST_SIZE=TEST_SIZE)

        train_features = np.concatenate((train_gyro_1_x, train_gyro_2_x, train_gyro_3_x, train_gyro_4_x, train_gyro_5_x,
                                         train_gyro_6_x, train_gyro_7_x, train_gyro_8_x,
                                         train_gyro_1_y, train_gyro_2_y, train_gyro_3_y, train_gyro_4_y, train_gyro_5_y,
                                         train_gyro_6_y, train_gyro_7_y, train_gyro_8_y,
                                         train_gyro_1_z, train_gyro_2_z, train_gyro_3_z, train_gyro_4_z, train_gyro_5_z,
                                         train_gyro_6_z, train_gyro_7_z, train_gyro_8_z,
                                         train_delta_t), axis=1)
        val_features = np.concatenate((val_gyro_1_x, val_gyro_2_x, val_gyro_3_x, val_gyro_4_x, val_gyro_5_x, val_gyro_6_x, val_gyro_7_x, val_gyro_8_x,
                                        val_gyro_1_y, val_gyro_2_y, val_gyro_3_y, val_gyro_4_y, val_gyro_5_y, val_gyro_6_y, val_gyro_7_y, val_gyro_8_y,
                                        val_gyro_1_z, val_gyro_2_z, val_gyro_3_z, val_gyro_4_z, val_gyro_5_z, val_gyro_6_z, val_gyro_7_z, val_gyro_8_z,
                                        val_delta_t), axis=1)
        test_features = np.concatenate((test_gyro_1_x, test_gyro_2_x, test_gyro_3_x, test_gyro_4_x, test_gyro_5_x,
                                        test_gyro_6_x, test_gyro_7_x, test_gyro_8_x,
                                        test_gyro_1_y, test_gyro_2_y, test_gyro_3_y, test_gyro_4_y, test_gyro_5_y,
                                        test_gyro_6_y, test_gyro_7_y, test_gyro_8_y,
                                        test_gyro_1_z, test_gyro_2_z, test_gyro_3_z, test_gyro_4_z, test_gyro_5_z,
                                        test_gyro_6_z, test_gyro_7_z, test_gyro_8_z,
                                        self.test_delta_t), axis=1)

        train_features = np.expand_dims(train_features, axis=1)  # Expanding one axis to column
        val_features = np.expand_dims(val_features, axis=1)  # Expanding one axis to column
        test_features = np.expand_dims(test_features, axis=1)  # Expanding one axis to column

        print("train_features info : ", type(train_features), np.shape(train_features))
        #print("train_delta_alpha info : ", type(train_delta_alpha), np.shape(train_delta_alpha))
        print("train_delta_alpha_x info : ", type(train_delta_alpha_x), np.shape(train_delta_alpha_x))
        print("train_delta_alpha_y info : ", type(train_delta_alpha_y), np.shape(train_delta_alpha_y))
        print("train_delta_alpha_z info : ", type(train_delta_alpha_z), np.shape(train_delta_alpha_z))
        print("val_features info : ", type(val_features), np.shape(val_features))
        #print("val_delta_alpha info : ", type(val_delta_alpha), np.shape(val_delta_alpha))
        print("val_delta_alpha_x info : ", type(val_delta_alpha_x), np.shape(val_delta_alpha_x))
        print("val_delta_alpha_y info : ", type(val_delta_alpha_y), np.shape(val_delta_alpha_y))
        print("val_delta_alpha_z info : ", type(val_delta_alpha_z), np.shape(val_delta_alpha_z))

        # self.train_dataset = tf.data.Dataset.from_tensor_slices(
        #     (train_features, train_delta_alpha_x))
        # self.val_dataset = tf.data.Dataset.from_tensor_slices(
        #     (val_features, val_delta_alpha_x))
        # self.test_dataset = tf.data.Dataset.from_tensor_slices(
        #     (test_features, test_delta_alpha_x))

        train_delta_alpha_x_y_and_z = np.concatenate((train_delta_alpha_x, train_delta_alpha_y, train_delta_alpha_z), axis=1)
        val_delta_alpha_x_y_and_z = np.concatenate((val_delta_alpha_x, val_delta_alpha_y, val_delta_alpha_z), axis=1)
        self.test_delta_alpha_x_y_and_z = np.concatenate((self.test_delta_alpha_x, self.test_delta_alpha_y, self.test_delta_alpha_z), axis=1)

        train_dataset_features = tf.data.Dataset.from_tensor_slices(train_features)
        train_dataset_labels = tf.data.Dataset.from_tensor_slices(train_delta_alpha_x_y_and_z)

        val_dataset_features = tf.data.Dataset.from_tensor_slices(val_features)
        val_dataset_labels = tf.data.Dataset.from_tensor_slices(val_delta_alpha_x_y_and_z)

        test_dataset_features = tf.data.Dataset.from_tensor_slices(test_features)
        test_dataset_labels = tf.data.Dataset.from_tensor_slices(self.test_delta_alpha_x_y_and_z)

        self.train_dataset = tf.data.Dataset.zip((train_dataset_features, train_dataset_labels))
        self.val_dataset = tf.data.Dataset.zip((val_dataset_features, val_dataset_labels))
        self.test_dataset = tf.data.Dataset.zip((test_dataset_features, test_dataset_labels))





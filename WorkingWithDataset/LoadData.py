import numpy as np
import pickle
import os
cwd = os.getcwd() # getting current working directory

# import timeit
#
# start_raw_data = timeit.default_timer()
#
# time = np.loadtxt(cwd + '/Dataset/path_5/acc_4.txt',comments='#', delimiter=' ', skiprows=1, usecols=range(0, 1))
# acc_4_data_features = np.loadtxt(cwd + '/Dataset/path_5/acc_4.txt',comments='#', delimiter=' ', skiprows=1, usecols=range(1, 4))
# gyro_4_data_features = np.loadtxt(cwd + '/Dataset/path_5/gyro_4.txt',comments='#', delimiter=' ', skiprows=1, usecols=range(1, 4))
# ground_truth_1_labels = np.loadtxt(cwd + '/Dataset/path_5/ground_truth.txt', comments='#', delimiter=' ', skiprows=1, usecols=range(1, 7))
#
# stop_raw_data = timeit.default_timer()
#
# print("Raw Data Info:")
# print(np.shape(time))
# print(np.shape(acc_4_data_features))
# print(np.shape(gyro_4_data_features))
# print(np.shape(ground_truth_1_labels))
#
# print((time[0:5]))
# print((acc_4_data_features[0:5]))
# print((gyro_4_data_features[0:5]))
# print((ground_truth_1_labels[0:5]))
#
# #print(np.array(time[0:10]))
#
# start_dumping_data = timeit.default_timer()
#
# with open(cwd + "/Dataset/path_5/train.pickle", 'wb') as f:
#     pickle.dump([time, acc_4_data_features, gyro_4_data_features, ground_truth_1_labels], f)
#
# stop_dumping_data = timeit.default_timer()
#
# start_loading_data = timeit.default_timer()

with open(cwd + '/Dataset/path_5/train.pickle', 'rb') as f:
    time_loaded, acc_4_data_features_loaded, gyro_4_data_features_loaded, ground_truth_1_labels_loaded = pickle.load(f)

# stop_loading_data = timeit.default_timer()

print("After Loading From Pickle - Data Info:")
print(np.shape(time_loaded))
print(np.shape(acc_4_data_features_loaded))
print(np.shape(gyro_4_data_features_loaded))
print(np.shape(ground_truth_1_labels_loaded))

# print((time_loaded[0:5]))
# print((acc_4_data_features_loaded[0:5]))
# print((acc_4_data_features_loaded[0:5, :]))
# print((gyro_4_data_features_loaded[0:5]))
# print((ground_truth_1_labels_loaded[0:5]))

# print("Time for reading raw data from txt file: ", (start_raw_data - stop_raw_data))
# print("Time for dumping raw data into pickle file : ", (start_dumping_data - stop_dumping_data))
# print("Time for loading raw data from pickle file: ", (start_loading_data - stop_loading_data))

# Calculating acceleration:
acc_4_data_features_calculated = np.sqrt(np.sum(np.square(acc_4_data_features_loaded), axis=1))

# Calculating gyro:
gyro_4_data_features_calculated = np.sqrt(np.sum(np.square(gyro_4_data_features_loaded), axis=1))

### draw the scatterplot, with color-coded points
import matplotlib.pyplot as plt

color_train = "g"
color_test = "r"
color_prediction = "black"

label_train = "train-acceleration"
label_test = "test-acceleration"
label_prediction = "prediction"

label_x = "time"
label_y = "acceleration"

acc_4_x_data_features = np.array(acc_4_data_features_calculated[:10000:10])
time_calculated = np.array(time_loaded[:10000:10] - time_loaded[0])  # Starting time from zero

# acc_4_x_data_features.reshape(acc_4_x_data_features.shape[0],-1)
# time_loaded.reshape(time_loaded.shape[0],-1)

# print(np.shape(acc_4_x_data_features))
# print(np.shape(time_calculated))

# Reshaping for sklearn library
acc_4_data_features_reshaped = np.reshape(acc_4_x_data_features, (1000, 1))
time_reshaped = np.reshape(time_calculated, (1000, 1))

# print(np.shape(acc_4_x_data_features))
# print(np.shape(time_calculated))

# print(acc_4_data_features_reshaped[:5])
# print(time_reshaped)


# Separating Training and Test Data

### training-testing split needed in regression, just like classification
from sklearn.model_selection import train_test_split

features = time_reshaped
targets = acc_4_data_features_reshaped
feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=0.1,
                                                                          random_state=42)

print("shape of feature train : ", np.shape(feature_train))
print("shape of feature train : ", np.shape(target_train))

# Linear Regression
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(feature_train, target_train)

# printing some information from my fitted regression line
# y = mx + c
print("reg slope (m) : ", reg.coef_)
print("reg offset (c) : ", reg.intercept_)

# working with score (r-squared score)
# r-squared score for training data is not useful
print("r-squared score for training data: ", reg.score(feature_train, target_train))

# r-squared score for testing data is very useful
print("r-squared score for testing data: ", reg.score(feature_test, target_test))

for feature, target in zip(feature_train, target_train):
    plt.scatter(feature, target, color="g")

for feature, target in zip(feature_test, target_test):
    plt.scatter(feature, target, color="r")

### labels for the legend
plt.scatter(feature_train, target_train, color=color_train, label=label_train)
plt.scatter(feature_test, target_test, color=color_test, label=label_test)
plt.plot(feature_test, reg.predict(target_test), color=color_prediction, label=label_prediction)

plt.xlabel(label_x)
plt.ylabel(label_y)
plt.legend()
plt.show()

# plt.savefig(cwd + "/Dataset/path_5/accelerationVStimeGraph.png")







### identify and remove the most outlier-y points
from outlier_cleaner import outlierCleaner

cleaned_data = []
try:
    predictions = reg.predict(feature_train)
    cleaned_data = outlierCleaner(predictions, feature_train, target_train)
except NameError:
    print("my regression object doesn't exist, or isn't name reg")
    print("can't make predictions to use in identifying outliers")

### only run this code if cleaned_data is returning data
if len(cleaned_data) > 0:
    features_cleaned, targets_cleaned, errors = zip(*cleaned_data)
    features_cleaned_reshaped = np.reshape(np.array(features_cleaned), (len(features_cleaned), 1))

    targets_cleaned_reshaped = np.reshape(np.array(targets_cleaned), (len(targets_cleaned), 1))

    feature_cleaned_train, feature_cleaned_test, target_cleaned_train, target_cleaned_test = train_test_split(
        features_cleaned_reshaped, targets_cleaned_reshaped, test_size=0.1,
        random_state=42)

    print("shape of cleaned feature train : ", np.shape(feature_cleaned_train))
    print("shape of cleaned target train : ", np.shape(target_cleaned_train))

    ### refit your cleaned data!
    try:
        reg.fit(feature_cleaned_train, target_cleaned_train)
        #plt.plot(feature_cleaned_test, reg.predict(feature_cleaned_test), color="blue")

        # What are the slope and intercept?
        print("Slope after cleaned data : ", reg.coef_)
        print("Intercept after cleaned data : ", reg.intercept_)

        # working with score (r-squared score)
        # r-squared score for training data is not useful
        print("r-squared score for training data after cleaned data : ",
              reg.score(feature_cleaned_train, target_cleaned_train))

        # r-squared score for testing data is very useful
        print("r-squared score for testing data after cleaned data : ",
              reg.score(feature_cleaned_test, target_cleaned_test))
    except NameError:
        print("you don't seem to have regression imported/created,")
        print("   or else your regression object isn't named reg")
        print("   either way, only draw the scatter plot of the cleaned data")

    ### labels for the legend
    plt.scatter(feature_cleaned_train, target_cleaned_train, color=color_train, label=label_train)
    plt.scatter(feature_cleaned_test, target_cleaned_test, color=color_test, label=label_test)
    plt.plot(feature_cleaned_test, reg.predict(target_cleaned_test), color=color_prediction, label=label_prediction)

    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.legend()
    plt.show()


else:
    print("outlierCleaner() is returning an empty list, no refitting to be done")





















# If I want to train with the new cleaned data and test with previous test data

### identify and remove the most outlier-y points
from outlier_cleaner import outlierCleaner

cleaned_data = []
try:
    predictions = reg.predict(feature_train)
    cleaned_data = outlierCleaner(predictions, feature_train, target_train)
except NameError:
    print("my regression object doesn't exist, or isn't name reg")
    print("can't make predictions to use in identifying outliers")

### only run this code if cleaned_data is returning data
if len(cleaned_data) > 0:
    features_cleaned, targets_cleaned, errors = zip(*cleaned_data)
    features_train_cleaned_reshaped = np.reshape(np.array(features_cleaned), (len(features_cleaned), 1))
    targets_train_cleaned_reshaped = np.reshape(np.array(targets_cleaned), (len(targets_cleaned), 1))

    ### refit your cleaned data!
    try:
        reg.fit(features_train_cleaned_reshaped, targets_train_cleaned_reshaped)

        # What are the slope and intercept?
        print("Slope after cleaned data : ", reg.coef_)
        print("Intercept after cleaned data : ", reg.intercept_)

        # working with score (r-squared score)
        # r-squared score for training data is not useful
        print("r-squared score for training data after cleaned data : ",
              reg.score(features_train_cleaned_reshaped, targets_train_cleaned_reshaped))

        # r-squared score for testing data is very useful
        print("r-squared score for testing data after cleaned data : ",
              reg.score(features_train_cleaned_reshaped, targets_train_cleaned_reshaped))
    except NameError:
        print("you don't seem to have regression imported/created,")
        print("   or else your regression object isn't named reg")
        print("   either way, only draw the scatter plot of the cleaned data")

    ### labels for the legend
    plt.scatter(features_train_cleaned_reshaped, targets_train_cleaned_reshaped, color=color_train, label=label_train)
    plt.scatter(feature_test, target_test, color=color_test, label=label_test)
    plt.plot(feature_test, reg.predict(target_test), color=color_prediction, label=label_prediction)

    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.legend()
    plt.show()


else:
    print("outlierCleaner() is returning an empty list, no refitting to be done")

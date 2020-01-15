import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from tools.data_controller import *

df = get_data_from_my_dataset()
print(df.head())

original_acc1_x_df = df["a1_x"]
original_position_x_t_df = df["positon_x_t"]
original_time_step_df = df["simu_time"]

original_acc1_x = np.asfarray(original_acc1_x_df.to_numpy(), np.float)
original_position_x_t = np.asfarray(original_position_x_t_df.to_numpy(), np.float)
original_time_step = np.asfarray(original_time_step_df.to_numpy(), np.float)
time_step = original_time_step - original_time_step[0]
delta_t = np.diff(original_time_step, axis=0)
delta_t = delta_t / 1000 # delta_t is in milli sec. ## Converting ms to sec

# print(original_acc1_x[0:5])
# print(original_position_x_t[0:5])
# print(original_time_step[0:5])
# print(delta_t[0:5])



limit = None
plt.plot(time_step[0:limit], original_position_x_t[0:limit], "g.-")
plt.show()


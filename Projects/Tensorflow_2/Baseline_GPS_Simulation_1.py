import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from tools.data_controller import *

np.random.seed(13)

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


v_t = 0.1
v_t_plus_1 = []
v_t_plus_1.append(v_t)
loopRange = len(delta_t)


s_t_plus_1 = []
s_t_plus_1.append(original_position_x_t[0])

# Assumed GPS speed 10Hz
# So every 100ms the position data will be updated
gpsTimeToUpdate = 100
currentSumOfTime = 0

dumping_factor_for_velocity = .5

for i in range(loopRange):
    currentSumOfTime = currentSumOfTime + (delta_t[i] * 1000)
    if(currentSumOfTime >= gpsTimeToUpdate):
        currentSumOfTime = 0
        s_t_plus_1.append(np.random.normal(original_acc1_x[i], 7.8, 1))
        v_t_plus_1.append(0.1 + (original_acc1_x[i] * delta_t[i]))
    else:
        # Calculating velocity
        # v = u + at
        v_t_plus_1.append(v_t_plus_1[i] + (original_acc1_x[i] * delta_t[i]))

        # Calculating position
        # s_t_plus_1 = s_t + (u * t) + (.5 * a * t^2)
        s_t_plus_1.append(s_t_plus_1[i] + (v_t * delta_t[i]) + (0.5 * original_acc1_x[i] * (delta_t[i]**2)))


# print("a : ", original_acc1_x[0:5])
# print("t : ", delta_t[0:5])
# print("v : ", v_t_plus_1[0:5])
# print("calculated s : ", s_t_plus_1[0:5])
# print("measured s : ", original_position_x_t[0:5])
# print("diff of position : ", abs(original_position_x_t - s_t_plus_1))

limit = 500
# plt.plot(time_step[0:limit], original_position_x_t[0:limit], "g.-")
# plt.show()

plt.plot(time_step[0:limit], s_t_plus_1[0:limit], "r.-", time_step[0:limit], original_position_x_t[0:limit], "g.-")
plt.show()


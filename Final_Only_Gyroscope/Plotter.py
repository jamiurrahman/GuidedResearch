import matplotlib.pyplot as plt
import tensorflow as tf

class Plotter:
    def plot_2D(self, X, Y, xlabel, ylabel, color="b", marker="o"):
        limit = len(X)
        # plt.plot(time_step[0:limit], original_position_x_t[0:limit], "g.-")
        # plt.show()

        # plt.plot(delta_t[0:limit], delta_s[0:limit], "b.")
        # plt.show()

        # Visualizing 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # ax.scatter(delta_t[0:limit], delta_s[0:limit], original_acc1_x[0:limit], c=color, marker=marker)
        ax.scatter(X[0:limit], Y[0:limit], c=color, marker=marker)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        plt.show()

    def plot_3D(self, X, Y, Z, xlabel, ylabel, zlabel, color="b", marker="o"):
        limit = 6000
        # plt.plot(time_step[0:limit], original_position_x_t[0:limit], "g.-")
        # plt.show()

        # plt.plot(delta_t[0:limit], delta_s[0:limit], "b.")
        # plt.show()

        # Visualizing 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # ax.scatter(delta_t[0:limit], delta_s[0:limit], original_acc1_x[0:limit], c=color, marker=marker)
        ax.scatter(X[0:limit], Y[0:limit], Z[0:limit], c=color, marker=marker)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)

        plt.show()

    def plot(self,  X, Y_1, Y_2, title, xlabel, ylabel, label_Y_1, label_Y_2, color_Y_1="b", color_Y_2 = "r", linestyle="None", marker="."):
        plt.plot(X, Y_1, color=color_Y_1, marker=marker, linestyle=linestyle, label=label_Y_1)
        plt.plot(X, Y_2, color=color_Y_2, marker=marker, linestyle=linestyle, label=label_Y_2)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend()
        plt.title(title)
        plt.show();

    def plot_graphs(history, string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_' + string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_' + string])
        plt.show()
import csv
import matplotlib.pyplot as plt
import numpy as np


def draw():
    """draws a figure from sparsity_out.txt csv file"""
    data = np.genfromtxt("sparsity_out.csv", delimiter=",")
    fig, ax0 = plt.subplots()

    ax0.plot(data[:, 0], data[:, 1], color="red", label="accuracy")
    ax0.set_ylabel("final test accuracy")
    ax0.set_xlabel("sparsity")
    ax0.set_ylim([0.9, 1.0])

    ax1 = ax0.twinx()
    ax1.plot(data[:, 0], data[:, 2], color="green", label="time")
    ax1.set_ylabel("time in seconds")

    h0, l0 = ax0.get_legend_handles_labels()
    h1, l1 = ax1.get_legend_handles_labels()

    ax1.legend(h0 + h1, l0 + l1, loc="upper left")

    fig.savefig("sparsity_accuracy_time-plot.png")


if __name__ == "__main__":
    draw()

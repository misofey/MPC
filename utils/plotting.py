import matplotlib.pyplot as plt
import numpy as np


def plot_path(path):
    plt.figure(figsize=(10, 10))
    plt.plot(path[:, 0], path[:, 1])


def plot_path_and_heading(data, ref=None):
    # plt.figure(figsize=(10, 10))
    # plt.plot(data[:, 0], data[:, 1])
    if ref is None:
        plot_directions(data)
    else:
        plot_directions(data, ref)


def plot_directions(x, ref=None):
    t = np.arange(0, x.shape[0])
    plt.scatter(
        x[:, 0],
        x[:, 1],
        label="position"
    )
    if ref is not None:
        t = np.arange(0, ref.shape[0])
        plt.scatter(
            ref[:, 0],
            ref[:, 1],
            label="reference"
        )
    plt.legend()
    #ax = plt.gca()
    #ax.set_aspect("equal")
    #xmin = np.min(x[:, 0]) - 1
    #xmax = np.max(x[:, 0]) + 1
    #ymin = np.min(x[:, 1]) - 1
    #ymax = np.max(x[:, 1]) + 1
    plt.xlim([0, 30])
    plt.ylim([-15, 15])

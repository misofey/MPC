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
    plt.quiver(
        x[:, 0],
        x[:, 1],
        x[:, 2],
        x[:, 3],
        t,
        label="position",
        angles="xy",
        scale_units="xy",
        scale=30,
        cmap="turbo",
    )
    if ref is not None:
        t = np.arange(0, ref.shape[0])
        plt.quiver(
            ref[:, 0],
            ref[:, 1],
            ref[:, 2],
            ref[:, 3],
            t,
            label="reference",
            angles="xy",
            scale_units="xy",
            scale=30,
            cmap="brg",
        )
    plt.legend()
    ax = plt.gca()
    ax.set_aspect("equal")
    xmin = np.min(x[:, 0]) - 1
    xmax = np.max(x[:, 0]) + 1
    ymin = np.min(x[:, 1]) - 1
    ymax = np.max(x[:, 1]) + 1
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])


def plot_steering(x, u, t):
    plt.plot(t, x[:, 6], label="wheel angle")
    plt.plot(t[:-1], u, label="steering rate")
    plt.legend()

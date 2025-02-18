import numpy as np

# import scipy as sp
import matplotlib.pyplot as plt


def system_properties(A):
    (val, vec) = np.linalg.eig(A)
    print("linear system: ", A[0, :])
    print("               ", A[1, :])
    print("eigenvalues: ", val)
    print("eigenvectors: ", vec)


def plot_results(x, u, p, t, ref):
    plt.plot(x[:, 0], x[:, 1], label="position")
    plt.plot(t, u, label="steering angle")
    plt.plot(ref[:, 0], ref[:, 1], label="reference")
    plt.legend()
    # plt.plot(t, p)
    # plt.title("paramter trajectory")
    plt.show()


def plot_directions(x, u, p, t, ref):
    plt.quiver(x[:, 0], x[:, 1], x[:, 2], x[:, 3], label="position", angles="xy", scale_units="xy", scale=30, color="r")
    plt.plot(t, u, label="steering angle")
    plt.quiver(ref[:, 0], ref[:, 1], ref[:, 2], ref[:, 3], label="reference", angles="xy", scale_units="xy", scale=30)
    plt.legend()
    # plt.plot(t, p)
    # plt.title("paramter trajectory")
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.xlim([-1, 3])
    plt.ylim([-1, 1])
    plt.show()

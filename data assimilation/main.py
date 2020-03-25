import numpy as np
import matplotlib.pyplot as plt
from helper import *


def main():
    # parameters
    n = 3  # state size
    p = 3  # observation size
    nb = 1000  # number of times
    Ne = 100 # number of ensembles
    time = np.array(range(nb))  # time vector

    var_Q = 0.01  # true error variance of the model
    var_R = 1  # true error variance of the observations

    R_factor = 0.1 # multiplicative error factor of the estimated R matrix
    Q_factor = 10 # multiplicative error factor of the estimated Q matrix

    # variables
    Q_true = var_Q * np.eye(n, n)
    R_true = var_R * np.eye(p, p)

    Q = Q_factor * Q_true
    R = R_factor * R_true

    # observation operator
    H = np.eye((3))

    # plot trajectory and confidence interval of the assimilation
    #plot_trajectory(Q_true, R_true, Q, R, Ne, n, p, nb, H, time, obs_rate=3000)

    # plot the impact on assimilation of Q & R estimated values
    Q_R_estimation_impact(Q_true, R_true, Ne, n, p, nb, H, time, obs_rate=100, window_size=100)

    plt.show()


if __name__ == '__main__':
    main()


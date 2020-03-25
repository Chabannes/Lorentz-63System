import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import numpy as np
from numpy.random import normal
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.linear_model import Lasso


def main():
    run_Lorentz_resolution(dt=0.01)
    plt.show()


#                 GENERATION OF THE LORENZ-63 SYSTEM
#-----------------------------------------------------------------------------------------------------------------------
# defining the parameters of the system
x0 = np.array([8,0,30]) # initial condition
T = 100 # number of Lorenz-63 times
sigma = 10
rho = 28
beta = 8/3

# Creation of the lorenz-63 dynamical model
def Lorenz_63(x, dx, sigma, rho, beta):
    dx = np.zeros(3)
    dx[0] = sigma*(x[1]-x[0])
    dx[1] = x[0]*(rho-x[2])-x[1]
    dx[2] = x[0]*x[1] - beta*x[2]
    return dx


# generating the Lorenz-63 system
def generating_lorentz63(dt):
    x = odeint(Lorenz_63, x0, np.arange(0.01,T, dt), args=(sigma, rho, beta))
    time = np.arange(0.01, T, dt)
    return x,time


#                                      VISUALISING THE THEORITICAL MODEL
#-----------------------------------------------------------------------------------------------------------------------
def plot_2D(time, x):
    plt.figure()
    plt.plot(time, x)
    plt.xlabel('Lorenz-63 model', size=20)
    plt.legend(['$x_1$', '$x_2$', '$x_3$'], prop={'size': 20})


def plot_3D(x):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x[:, 0], x[:, 1], x[:, 2], 'k')
    plt.title('True model')
    ax.set_xlabel('$x_1$', size=20)
    ax.set_ylabel('$x_2$', size=20)
    ax.set_zlabel('$x_3$', size=20)

#                                          CREATING A TRAINAING AND TESTING DATASETS
#-----------------------------------------------------------------------------------------------------------------------


def datasets(x, dt):
    # we aim to solve the system 𝑌=𝑓(𝑋) with  X = [x1, x2, x3, x1x1, x1x2, x1x3, x2x2, x2x3, x3x3]

    # constructing the output 𝑌 corresponding to the ODE formulation of the Lorenz-63
    Y = (x[1:, ]-x[:-1, :])/dt

    # construct the input 𝑋 with the information of the Lorenz-63 at time 𝑡
    # to build X, we take the 3 dimensions and their products. We might have to introduce sine functions in other problems
    X = np.vstack((x[:-1, 0], x[:-1, 1], x[:-1, 2],
                x[:-1, 0] * x[:-1, 0], x[:-1, 0] * x[:-1, 1], x[:-1, 0] * x[:-1, 2],
                x[:-1, 1] * x[:-1, 1], x[:-1, 1] * x[:-1, 2], x[:-1, 2] * x[:-1, 2])).transpose()

    #  training/test dataset creation
    sep = int(T/dt*2/3)
    X_train = X[:sep, ]
    X_test = X[sep:, ]
    y_train = Y[:sep, ]
    y_test = Y[sep:, ]

    # adding noise to training data
    X_train = X_train + np.random.normal(0, 1, np.shape(X_train))
    y_train = y_train + np.random.normal(0, 1, np.shape(y_train))
    return(X_train, y_train, X_test, y_test)


#                                          CREATING THE MODELS
#-----------------------------------------------------------------------------------------------------------------------

def alpha_tunning_lasso(X_train, y_train):
    # declaring the cross-validation procedure (cvp)
    cvp = ShuffleSplit(n_splits=10, test_size=1/3, train_size=2/3)

    # defining the alphas between 10^(-5) and 1
    n_alphas = 10
    alphas = np.logspace(-5, 0, n_alphas)

    # loop on the alpha parameters and compute mean RMSE
    tab_RMSE_lasso = np.zeros(n_alphas)
    for i in range(n_alphas):
        reg_lasso = Lasso(alphas[i])
        tab_RMSE_lasso[i] = np.mean(np.sqrt(-cross_val_score(
            reg_lasso, X_train, y_train, scoring='neg_mean_squared_error', cv=cvp)))

    # plot results
    plt.figure()
    plt.plot(alphas, tab_RMSE_lasso)
    plt.xscale('log')
    plt.xlabel('Alpha coefficients for Lasso method', size=20)
    plt.ylabel('RMSE', size=20)
    plt.title('Cross-validation', size=20)


def XY_to_L63(X, Y, dt):
    L63 = Y * dt + X
    # return the L63 coordinate
    return L63


# function to transform a L3 coordinate to the input of the regression, assuming that:
# X = [x1, x2, x3, x1x1, x1x2, x1x3, x2x2, x2x3, x3x3]
def L63_to_X(trajectory):
    X = np.vstack((trajectory[0], trajectory[1], trajectory[2],
                   trajectory[0] * trajectory[0], trajectory[0] * trajectory[1], trajectory[0] * trajectory[2],
                   trajectory[1] * trajectory[1], trajectory[1] * trajectory[2],
                   trajectory[2] * trajectory[2])).transpose()
    return X


def get_trajectories(y_test, y_mlr, y_lasso, X_test, reg_mlr, reg_lasso, time, dt):
    # apply sequentially the linear regressions from the initial value of X_test
    traj_true = y_test*0
    traj_mlr = y_test*0
    traj_lasso = y_test*0
    traj_true[0, :] = XY_to_L63(X_test[0, 0:3], y_test[0, :], dt)
    traj_mlr[0, :] = XY_to_L63(X_test[0, 0:3], y_mlr[0, :], dt)
    traj_lasso[0, :] = XY_to_L63(X_test[0, 0:3], y_lasso[0,:], dt)
    for t in range(1, len(X_test)):
        traj_true[t, :] = XY_to_L63(X_test[t, 0:3], y_test[t,:], dt)
        # applying the linear regressions recursively
        traj_mlr[t, :] = XY_to_L63(traj_mlr[t-1, :], reg_mlr.predict(L63_to_X(traj_mlr[t-1, :])), dt)
        traj_lasso[t, :] = XY_to_L63(traj_lasso[t-1, :], reg_lasso.predict(L63_to_X(traj_lasso[t-1, :])), dt)
    sep = int(T/dt*2/3)
    plt.figure()
    plt.plot(time[sep + 1:], traj_true[:, 1], 'r')
    plt.plot(time[sep + 1:], traj_mlr[:, 1], 'k')
    plt.plot(time[sep + 1:], traj_lasso[:, 1], 'b')
    plt.xlabel('Lorenz-63 time', size=20)
    plt.ylabel('$x_2$', size=20)
    plt.legend(['truth', 'linear regression model', 'lasso regression model'], prop={'size': 10})

    # phase-space representation
    fig = plt.figure()
    plt.ax = fig.gca(projection='3d')
    plt.ax.plot(traj_true[:, 0], traj_true[:, 1], traj_true[:, 2], 'r')
    plt.ax.plot(traj_mlr[:, 0], traj_mlr[:, 1], traj_mlr[:, 2], 'k')
    plt.ax.plot(traj_lasso[:, 0], traj_lasso[:, 1], traj_lasso[:, 2], 'b')
    plt.ax.set_xlabel('$x_1$', size=20)
    plt.ax.set_ylabel('$x_2$', size=20)
    plt.ax.set_zlabel('$x_3$', size=20)
    plt.legend(['truth', 'linear regression model', 'lasso regression model'], prop={'size': 10})


def run_Lorentz_resolution(dt):
    # generating Lorentz model
    x, time = generating_lorentz63(dt)

    #  Plotting theoritical model Lorentz-63
    plot_2D(time, x)
    plot_3D(x)
    #  training and testing datasets creation
    X_train, y_train, X_test, y_test = datasets(x, dt)

    #  models creation and predictions
    # alpha_tunning_lasso(X_train, y_train)
    reg_mlr = LinearRegression(fit_intercept=False)
    reg_mlr.fit(X_train, y_train)
    y_mlr = reg_mlr.predict(X_test)

    reg_lasso = Lasso(alpha=0.02, fit_intercept=False)
    reg_lasso.fit(X_train, y_train)
    y_lasso = reg_lasso.predict(X_test)

    # Plot the modeled trajectories
    get_trajectories(y_test, y_mlr, y_lasso, X_test, reg_mlr, reg_lasso, time, dt)


if __name__ == "__main__":
    main()
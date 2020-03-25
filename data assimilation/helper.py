import numpy as np
import matplotlib.pyplot as plt
from helper import *
from mpl_toolkits.mplot3d import Axes3D
import math


# define the nonlinear dynamic system (Lorenz-63) using the Runge-Kutta integration method
def m(x_past):
    # physical parameters
    dT = 0.01
    sigma = 10
    rho = 28
    beta = 8 / 3

    # Runge-Kutta (4,5) integration method
    X1 = np.copy(x_past)
    k1 = np.zeros(X1.shape)
    k1[0] = sigma * (X1[1] - X1[0])
    k1[1] = X1[0] * (rho - X1[2]) - X1[1]
    k1[2] = X1[0] * X1[1] - beta * X1[2]
    X2 = np.copy(x_past + k1 / 2 * dT)
    k2 = np.zeros(x_past.shape)
    k2[0] = sigma * (X2[1] - X2[0])
    k2[1] = X2[0] * (rho - X2[2]) - X2[1]
    k2[2] = X2[0] * X2[1] - beta * X2[2]
    X3 = np.copy(x_past + k2 / 2 * dT)
    k3 = np.zeros(x_past.shape)
    k3[0] = sigma * (X3[1] - X3[0])
    k3[1] = X3[0] * (rho - X3[2]) - X3[1]
    k3[2] = X3[0] * X3[1] - beta * X3[2]
    X4 = np.copy(x_past + k3 * dT)
    k4 = np.zeros(x_past.shape)
    k4[0] = sigma * (X4[1] - X4[0])
    k4[1] = X4[0] * (rho - X4[2]) - X4[1]
    k4[2] = X4[0] * X4[1] - beta * X4[2]

    # return the state in the near future
    x_future = x_past + dT / 6. * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_future


def lorenz_assimilation(Q_true, R_true, Q, R, Ne, obs_rate, n, p, nb, H):

    # true state and noisy observations
    x_t = np.zeros((n, nb))
    y_o = np.zeros((p, nb))
    x_t[:, 0] = np.array([8, 0, 30])
    for t in range(1, nb):
        x_t[:, t] = m(x_t[:, t - 1]) # true state
        y_o[:, t] = np.dot(H, x_t[:, t]) + np.random.multivariate_normal(np.zeros(p), R_true)  # noisy observations
    # remove observations
    i_nan = np.random.choice(range(nb), obs_rate)
    y_o[:, i_nan] = y_o[:, i_nan] * np.nan

    # ensemble Kalman initialization
    x_f_enkf=np.zeros((n,nb))   # forecast state
    P_f_enkf=np.zeros((n,n,nb)) # forecast error covariance matrix
    x_a_enkf=np.zeros((n,nb))   # analysed state
    P_a_enkf=np.zeros((n,n,nb)) # analysed error covariance matrix
    coverage_probability = 0
    count = 0

    # ensemble Kalman filter
    x_a_enkf_tmp=np.zeros((n,Ne)) # initial state
    x_f_enkf_tmp=np.zeros((n,Ne))
    y_f_enkf_tmp=np.zeros((p,Ne))
    P_a_enkf_tmp=0*np.eye(n,n) # initial state covariance
    x_a_enkf[:,0]=np.mean(x_a_enkf_tmp)
    P_a_enkf[:,:,0]=P_a_enkf_tmp
    for k in range(1,nb): # forward in time
        # prediction step
        for i in range(Ne):
            x_f_enkf_tmp[:,i]=m(x_a_enkf_tmp[:,i]) + np.random.multivariate_normal(np.zeros(n),Q)
            y_f_enkf_tmp[:,i]=np.dot(H,x_f_enkf_tmp[:,i]) + np.random.multivariate_normal(np.zeros(p),R)
        P_f_enkf_tmp=np.cov(x_f_enkf_tmp)
        K=np.dot(np.dot(P_f_enkf_tmp,H.T),np.linalg.inv(np.dot(np.dot(H,P_f_enkf_tmp),H.T)+R))  # Kalman gain
        # update step
        if(sum(np.isfinite(y_o[:,k]))>0):
            for i in range(Ne):
                x_a_enkf_tmp[:,i]=x_f_enkf_tmp[:,i]+np.dot(K,(y_o[:,k]-y_f_enkf_tmp[:,i]))
            P_a_enkf_tmp=np.cov(x_a_enkf_tmp)
        else:
            x_a_enkf_tmp=x_f_enkf_tmp
            P_a_enkf_tmp=P_f_enkf_tmp

        # store results
        x_f_enkf[:,k]=np.mean(x_f_enkf_tmp,1)
        P_f_enkf[:,:,k]=P_f_enkf_tmp
        x_a_enkf[:,k]=np.mean(x_a_enkf_tmp,1)
        P_a_enkf[:,:,k]=P_a_enkf_tmp

        if not math.isnan(y_o[0][k]):
            count = count + 1
            if (y_o[0][k] > x_a_enkf[0][k] - 1.96*np.sqrt(P_a_enkf[0,0,k])) and (y_o[0][k] < x_a_enkf[0][k] + 1.96*np.sqrt(P_a_enkf[0,0,k])):
                coverage_probability = coverage_probability + 1

    coverage_probability = coverage_probability/count

    return x_t, y_o, x_a_enkf, P_a_enkf, coverage_probability


def plot_trajectory(Q_true, R_true, Q, R, Ne, n, p, nb, H, time, obs_rate):

    x_t, y_o, x_a_enkf, P_a_enkf, coverage_probability = lorenz_assimilation(Q_true, R_true, Q, R, Ne, obs_rate, n, p, nb, H)

    plt.figure()
    plt.plot('Trajectory and confidence interval of the assimilated model')
    plt.plot(time, x_t[0,:])
    plt.plot(time, y_o[0,:], 'k*')
    plt.plot(time,x_a_enkf[0,:],'r')
    plt.fill_between(time,x_a_enkf[0,:]-1.96*np.sqrt(P_a_enkf[0,0,:]),x_a_enkf[0,:]+1.96*np.sqrt(P_a_enkf[0,0,:]),facecolor='red',alpha=0.5)
    plt.xlabel('Lorenz-63 time', size=20)
    plt.legend(['truth','obs','EnKF'], prop={'size': 20})


def Q_R_estimation_impact(Q_true, R_true, Ne, n, p, nb, H, time, obs_rate, window_size):

    x_t_1, y_o_1, x_a_enkf_1, P_a_enkf_1, coverage_probability_1 = lorenz_assimilation(Q_true, R_true, 0.1*Q_true, R_true, Ne, obs_rate,n, p, nb, H)
    x_t_2, y_o_2, x_a_enkf_2, P_a_enkf_2, coverage_probability_2 = lorenz_assimilation(Q_true, R_true, Q_true, 0.1*R_true, Ne, obs_rate,n, p, nb, H)
    x_t_3, y_o_3, x_a_enkf_3, P_a_enkf_3, coverage_probability_3 = lorenz_assimilation(Q_true, R_true, 10*Q_true, R_true, Ne, obs_rate,n, p, nb, H)
    x_t_4, y_o_4, x_a_enkf_4, P_a_enkf_4, coverage_probability_4 = lorenz_assimilation(Q_true, R_true, Q_true, 10*R_true, Ne, obs_rate,n, p, nb, H)
    x_t_5, y_o_5, x_a_enkf_5, P_a_enkf_5, coverage_probability_5 = lorenz_assimilation(Q_true, R_true, 0.1*Q_true, 0.1*R_true, Ne, obs_rate,n, p, nb, H)
    x_t_6, y_o_6, x_a_enkf_6, P_a_enkf_6, coverage_probability_6 = lorenz_assimilation(Q_true, R_true, 10*Q_true, 10*R_true, Ne, obs_rate,n, p, nb, H)

    fig, ax = plt.subplots(3, 2, figsize=(40, 40))

    plt.suptitle("Q & R estimated values impact on x_1 assimilation performence.\nOne observation every %s time step" %(obs_rate), fontsize=15)

    ax[0,0].plot(time[200:200+window_size], x_t_1[0,:][200:200+window_size], c='k',linewidth=0.5)
    ax[0,0].plot(time[200:200+window_size], x_a_enkf_1[0,:][200:200+window_size], c='b',linewidth=0.5)
    ax[0,0].fill_between(time[200:200+window_size], x_a_enkf_1[0,:][200:200+window_size]-1.96*np.sqrt(P_a_enkf_1[0,0,:][200:200+window_size]), \
                    x_a_enkf_1[0,:][200:200+window_size]+1.96*np.sqrt(P_a_enkf_1[0,0,:][200:200+window_size]), alpha=0.3)
    ax[0,0].scatter(time[200:200+window_size], y_o_1[0,:][200:200+window_size], c='r',s=0.1)
    ax[0,0].legend(['True Model','Ensemble Kalman analysis','Coverage Probability = %s' %(coverage_probability_1),'Oberservations', ], prop={'size': 10})
    ax[0,0].title.set_text("R = %sR_true  Q = %sQ_true" %(0.1, 1))


    ax[0,1].plot(time[200:200+window_size], x_t_2[0,:][200:200+window_size], c='k',linewidth=0.5)
    ax[0,1].plot(time[200:200+window_size], x_a_enkf_2[0,:][200:200+window_size], c='b',linewidth=0.5)
    ax[0,1].fill_between(time[200:200+window_size], x_a_enkf_2[0,:][200:200+window_size]-1.96*np.sqrt(P_a_enkf_2[0,0,:][200:200+window_size]), \
                    x_a_enkf_2[0,:][200:200+window_size]+1.96*np.sqrt(P_a_enkf_2[0,0,:][200:200+window_size]), alpha=0.3)
    ax[0,1].scatter(time[200:200+window_size], y_o_2[0,:][200:200+window_size], c='r',s=0.1)
    ax[0,1].legend(['True Model','Ensemble Kalman analysis','Coverage Probability = %s' %(coverage_probability_2),'Oberservations', ], prop={'size': 10})
    ax[0,1].title.set_text("R = %sR_true  Q = %sQ_true" %(1, 0.1))


    ax[1,0].plot(time[200:200+window_size], x_t_3[0,:][200:200+window_size], c='k',linewidth=0.5)
    ax[1,0].plot(time[200:200+window_size], x_a_enkf_3[0,:][200:200+window_size], c='b',linewidth=0.5)
    ax[1,0].fill_between(time[200:200+window_size], x_a_enkf_3[0,:][200:200+window_size]-1.96*np.sqrt(P_a_enkf_3[0,0,:][200:200+window_size]), \
                    x_a_enkf_3[0,:][200:200+window_size]+1.96*np.sqrt(P_a_enkf_3[0,0,:][200:200+window_size]), alpha=0.3)
    ax[1,0].scatter(time[200:200+window_size], y_o_3[0,:][200:200+window_size], c='r',s=0.1)
    ax[1,0].legend(['True Model','Ensemble Kalman analysis','Coverage Probability = %s' %(coverage_probability_3),'Oberservations', ], prop={'size': 10})
    ax[1,0].title.set_text("R = %sR_true  Q = %sQ_true" %(10, 1))


    ax[1,1].plot(time[200:200+window_size], x_t_4[0,:][200:200+window_size], c='k',linewidth=0.5)
    ax[1,1].plot(time[200:200+window_size], x_a_enkf_4[0,:][200:200+window_size], c='b',linewidth=0.5)
    ax[1,1].fill_between(time[200:200+window_size], x_a_enkf_4[0,:][200:200+window_size]-1.96*np.sqrt(P_a_enkf_4[0,0,:][200:200+window_size]), \
                    x_a_enkf_4[0,:][200:200+window_size]+1.96*np.sqrt(P_a_enkf_4[0,0,:][200:200+window_size]), alpha=0.3)
    ax[1,1].scatter(time[200:200+window_size], y_o_4[0,:][200:200+window_size], c='r',s=0.1)
    ax[1,1].legend(['True Model','Ensemble Kalman analysis','Coverage Probability = %s' %(coverage_probability_4),'Oberservations', ], prop={'size': 10})
    ax[1,1].title.set_text("R = %sR_true  Q = %sQ_true" %(1, 10))


    ax[2,0].plot(time[200:200+window_size], x_t_5[0,:][200:200+window_size], c='k',linewidth=0.5)
    ax[2,0].plot(time[200:200+window_size], x_a_enkf_5[0,:][200:200+window_size], c='b',linewidth=0.5)
    ax[2,0].fill_between(time[200:200+window_size], x_a_enkf_5[0,:][200:200+window_size]-1.96*np.sqrt(P_a_enkf_5[0,0,:][200:200+window_size]), \
                    x_a_enkf_5[0,:][200:200+window_size]+1.96*np.sqrt(P_a_enkf_5[0,0,:][200:200+window_size]), alpha=0.3)
    ax[2,0].scatter(time[200:200+window_size], y_o_5[0,:][200:200+window_size], c='r',s=0.1)
    ax[2,0].legend(['True Model','Ensemble Kalman analysis','Coverage Probability = %s' %(coverage_probability_5),'Oberservations', ], prop={'size': 10})
    ax[2,0].title.set_text("R = %sR_true  Q = %sQ_true" %(0.1, 0.1))


    ax[2,1].plot(time[200:200+window_size], x_t_6[0,:][200:200+window_size], c='k',linewidth=0.5)
    ax[2,1].plot(time[200:200+window_size], x_a_enkf_6[0,:][200:200+window_size], c='b',linewidth=0.5)
    ax[2,1].fill_between(time[200:200+window_size], x_a_enkf_6[0,:][200:200+window_size]-1.96*np.sqrt(P_a_enkf_6[0,0,:][200:200+window_size]), \
                    x_a_enkf_6[0,:][200:200+window_size]+1.96*np.sqrt(P_a_enkf_6[0,0,:][200:200+window_size]), alpha=0.3)
    ax[2,1].scatter(time[200:200+window_size], y_o_6[0,:][200:200+window_size], c='r',s=0.1)
    ax[2,1].legend(['True Model','Ensemble Kalman analysis','Coverage Probability = %s' %(coverage_probability_6),'Oberservations', ], prop={'size': 10})
    ax[2,1].title.set_text("R = %sR_true  Q = %sQ_true" %(10, 10))
import numpy as np
import matplotlib.pyplot as plt

# Initial values
mu = np.array([1, -10])
var = np.array([[1, 0], [0, 1]])
a_final = []
pl = []
a_true = -1
x = np.zeros((101, 1))
y_obs = np.zeros((100, 1))
x[0] = 1

# Generating data
for i in range(100):
    x[i+1] = a_true*x[i] + np.random.normal(0, 1)
    y_obs[i] = np.sqrt(x[i]**2 + 1) + np.random.normal(0, 0.5)

# EKF
for i in range(100):    #100 iterations
    a = mu[1]
    y = y_obs[i]
    mu_pred = [a*mu[0], a]      #  mu_k+1|k
    F = np.array([[a, mu[0]], [0, 1]])   #Jacobian(a.x_k)
    var_pred = F @ var @ F.T + np.diag([1,0.1])  # sigma_k+1|k
    C = np.array([[mu_pred[0]/np.sqrt(mu_pred[0]**2+1), 0]])  #Jacobian(sqrt(x^2 + 1))
    K = var_pred @ C.T @ np.linalg.inv(C @ var_pred @ C.T + np.array([[0.5]]))  # Kalman Gain
    x_lat = mu_pred + K@(y - np.array([np.sqrt(mu_pred[0]**2 + 1)]))  # mu_k+1|k+1 - mean
    P_lat = (np.eye(2) - K @ C) @ var_pred     #sigma_k+1|k+1 - covariance
    a_final.append(x_lat[1])
    pl.append(x_lat[1] + np.sqrt(P_lat[1][1]))  #mu_k + sigma_k
    mu = x_lat
#print(a_final)
a_true = [-1]*100
plt.figure(1)
plt.plot(a_true, 'k')
plt.plot(pl)
plt.show()
import numpy as np
import scipy
from scipy import io
from quaternion import Quaternion
import math
from scipy.linalg import sqrtm

#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter
def sigma_points(xk_ini,cov_k_ini, sigma, weight, Q, R, dt):
    q1 = Quaternion()
    weight = np.array(weight)
    xk_ini = np.array(xk_ini)
    S = sqrtm(12*(cov_k_ini  +R*dt))
    #print(S)
    weight[:,:6] = S
    weight[:, 6:] = -S
    q_w = np.zeros((4,12))
    q_x = np.zeros((4,12))
    for i in range(weight.shape[1]):
        temp = Quaternion(xk_ini[0][0], xk_ini[1:4].flatten())
        q1.from_axis_angle(weight[:3,[i]].flatten())
        q_w[:,[i]] =   q1.q.reshape((4,1))
        #print(q_w[:,[i]])
        temp2 = Quaternion(q_w[0][i], q_w[1:4, [i]].flatten())
        q_x[:,[i]] = (temp).__mul__(temp2 ).q.reshape((4,1))
    #print(q_w)
    sigma[:4,:-1] = q_x
    sigma[4:,:-1] = xk_ini[4:]+ weight[3:,:]
    sigma[:, [-1]] = xk_ini
    #print(sigma)
    return(sigma)

def sigma_transform(sigma, w_prime, q_delta, dt, Z, R, g, Q, weight):
    q2 = Quaternion()
    q_delta = np.array(q_delta)
    Z = np.array(Z)
    sigma = np.array(sigma)
    qk_g = np.zeros((4, sigma.shape[1]))
    qk_inverse = np.zeros((4, sigma.shape[1]))
    g = np.array(g)
    wk_mag = np.linalg.norm(sigma[4:,:],axis=0)
    wk_mag[wk_mag==0]=0.001
    e_delta = sigma[4:, :] /wk_mag
    q_delta[0,:] = np.cos(wk_mag*(dt/2))
    q_delta[1:,:] = np.sin(wk_mag*(dt/2)) *e_delta
    
    for i in range(sigma.shape[1]):
        #temp4 = Quaternion(q_delta[0][i], q_delta[1:4,[i]].reshape((1,3)))
        temp = Quaternion(sigma[0][i], sigma[1:4,[i]].flatten())
        temp1 = Quaternion(q_delta[0][i], q_delta[1:4,[i]].flatten())
        sigma[:4, [i]] = (temp).__mul__(temp1).q.reshape((4,1))
    #q = sigma[:4,:]
    wk_mean = np.array(np.mean(sigma[4:,:],axis=1)).reshape((-1,1))
    q_mean, q_e = mean(sigma[:4,:])
    xk_mean = np.vstack((q_mean, wk_mean))    #mu k+1|k
    w_prime[:3,:] = q_e
    w_prime[3:,:] = sigma[4:,:] - xk_mean[4:]
    Pk = np.matmul(w_prime, w_prime.T)/13      #cov k+1|k
    #sigma = sigma_points(xk_mean, Pk, sigma, weight, Q, R, dt)
    for i in range(sigma.shape[1]):
        temp3 = Quaternion(g[0][0], g[1:4].flatten())
        
        temp5 = Quaternion(sigma[0][i], sigma[1:4, [i]].flatten())
        qk_g[:,[i]] = ((temp3).__mul__(temp5)).q.reshape((4,1))
        temp1 = np.array(temp3.inv().q)
        qk_inverse[:,[i]] = temp1.reshape((4,1))
        temp4 = Quaternion(qk_inverse[0][i], qk_inverse[1:4, [i]].flatten())
        temp6 = Quaternion(qk_g[0][i], qk_g[1:4, [i]].flatten())
        temp2 = temp4.__mul__(temp6).q
        Z[:3,[i]] = temp2.reshape((4,1))[1:]
    Z[3:,:] = sigma[4:,:]
    z_mean = np.array(np.mean(Z, axis=1)).reshape((-1,1))
    Pzz = np.matmul((Z-z_mean), (Z-z_mean).T)/13
    Pxz = np.matmul(w_prime, (Z-z_mean).T)/13
    Pvv = Pzz + Q
    #Pzz = np.matmul() 
    return xk_mean, Pk, Pxz, Pvv, z_mean

def mean(quat):
    quat = np.array(quat)
    q4 = Quaternion()
    q_t = np.zeros((4,1))
    q_t[0][0] = 1
    e = np.zeros((4, 13))
    e_convert = np.zeros((3, 13))
    condition = 1
    threshold = 0.1
    count = 0
    while(condition>threshold) and count<50:
        count += 1
        q_tc = Quaternion(q_t[0][0], q_t[1:4].flatten())
        for i in range(e.shape[1]):
            
            #a = q_tc.inv().q
            temp = Quaternion(quat[0][i], quat[1:4, [i]].flatten())
            #e[:,[i]] =  ((quat[:,[i]]).__mul__(q_tc.inv().q)).reshape((4,1))
            e[:,[i]] = ((temp)).__mul__(q_tc.inv()).q.reshape((4,1))
        
            e_tc = Quaternion(e[0][i], e[1:4,[i]].flatten())
            #e_tc.normalize()
            e_convert[:,[i]] = e_tc.vec().reshape((3,1))
        #q_tc = Quaternion(q_t[0][0], q_t[1:4].reshape((1,3)))
        e_convert_mean = np.array(np.mean(e_convert, axis=1)).reshape((-1,1))
        condition = np.linalg.norm(e_convert_mean, axis=0)
        q4.from_axis_angle(e_convert_mean.flatten())
        e_quat_mean = q4.q.reshape((4,1))
        temp1 = Quaternion(e_quat_mean[0][0], e_quat_mean[1:4].flatten())
        q_t = (q_tc).__mul__(temp1).q.reshape((4,1))
    return q_t, e_convert

        
def kalman(Pk, Pvv, Pxz, ax, ay, az, wx, wy, wz, z_mean, xk_mean, K_inno_q):
    q3 = Quaternion()
    xk_mean = np.array(xk_mean)
    K_inno_q = np.array(K_inno_q)
    K = np.array(np.matmul(Pxz,np.linalg.inv(Pvv)))
    # print(np.linalg.norm(np.array([ax, ay, az])))
    y = np.array([ax, ay, az, wx, wy, wz]).reshape((6,1))
    inno = np.array(y - z_mean)
    #K_inno_q = np.matmul(K, inno)
    q3.from_axis_angle(np.matmul(K, inno)[:3].flatten())
    K_inno_q[:4] = q3.q.reshape((4,1))
    K_inno_q[4:] = np.matmul(K, inno)[3:]
    temp1 = Quaternion(xk_mean[0][0], xk_mean[1:4].flatten())
    temp = Quaternion(K_inno_q[0][0], K_inno_q[1:4].flatten())
    xk_mean[:4] = ((temp).__mul__(temp1)).q.reshape((4,1))
    xk_mean[4:] = (xk_mean[4:] + K_inno_q[4:]).reshape((3,1))
    Pk = Pk - np.matmul(np.matmul(K,Pvv),K.T)
    return xk_mean, Pk

# def quat2rpy(quat):
#     r = np.zeros((quat.shape[1],))
#     p = np.zeros((quat.shape[1],))
#     y = np.zeros((quat.shape[1],))
#     for i in range(quat.shape[1]):
#         in_q = quat[:, i].flatten()
#         q = Quaternion(float(in_q[0]), in_q[1:])
#         q.normalize()
#         angles = q.euler_angles()
#         r[i] = float(angles[0])
#         p[i] = float(angles[1])
#         y[i] = float(angles[2])
#     return r, p, y

def estimate_rot(data_num=1):
    #load data
    imu = io.loadmat('imuRaw'+str(data_num)+'.mat')
    # vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
    acce = imu['vals']
    accel = np.array(acce[0:3,:])
    gyr = imu['vals']
    gyro = np.array(gyr[3:6,:])
    T = np.shape(imu['ts'])[1]

    # your code goes here
    ts = np.array(imu['ts'])
    a_beta_x = 511
    a_beta_y = 501
    a_beta_z = 522.1
    a_alpha = 27
    ax = -(accel[0,:] - a_beta_x*(np.ones(accel[0,:].shape)))*3300/(1023*a_alpha)
    ay = -(accel[1,:]- a_beta_y*(np.ones(accel[1,:].shape)))*3300/(1023*a_alpha)
    az = (accel[2,:] -a_beta_z*(np.ones(accel[2,:].shape)))*3300/(1023*a_alpha)

    g_alpha = 27
    g_beta_x = 374
    g_beta_y = 376
    g_beta_z = 370
    alpha_roll = 210
    alpha_pitch = 210
    alpha_yaw = 210
    acc= acce[0:3,:]

    g_ax = (acc[0][0] - 511)*3300/(1023*g_alpha)
    g_ay = (acc[1][0]- 500)*3300/(1023*g_alpha)
    g_az = (acc[2][0] - 500)*3300/(1023*g_alpha)

    wx = (gyro[1,:] - g_beta_x*(np.ones(gyro[1,:].shape)))*3300/(1023*alpha_roll)
    wy = (gyro[2,:] - g_beta_y*(np.ones(gyro[2,:].shape)))*3300/(1023*alpha_pitch)
    wz = (gyro[0,:] - g_beta_z*(np.ones(gyro[0,:].shape)))*3300/(1023*alpha_yaw)

    roll = np.ones(ts.shape[1])
    pitch = np.ones(ts.shape[1])
    yaw = np.ones(ts.shape[1])
    roll[0] = np.arctan2(g_ay, g_az)
    pitch[0] = np.arctan2(-g_ax,np.sqrt(g_ay**2+g_az**2))
    yaw[0] = 0
    for i in range(1,ts.shape[1]):
        roll[i] = roll[i-1] + wx[i-1]*(ts[0][i] - ts[0][i-1])
        pitch[i] = pitch[i-1] + wy[i-1]*(ts[0][i] - ts[0][i-1])
        yaw[i] = yaw[i-1] + wz[i-1]*(ts[0][i] - ts[0][i-1])
    print(roll)
    print(pitch)
    print(yaw)
    q6 = Quaternion()
    xk_ini = np.zeros((7, 1))
    xk_ini[:4] = np.array([1,0,0,0]).reshape((4,1))
    xk_ini[4:] = np.array([wx[0], wy[0], wz[0]]).reshape((3,1))
    print(xk_ini)
    #Q = 11*np.eye(6)
    Q = np.diag([5, 5, 5, 25, 25, 25])
    #R = 20*np.eye(6)
    R = np.diag([2200, 2200, 2200, 1000, 1000, 1000])
    cov_k_ini = 0*np.eye(6)
    Z = np.zeros((6, 13))
    weight = np.zeros((6, 12))
    sigma_ini = np.zeros((7, 13))
    K_inno_q = np.zeros((7, 1))
    g = np.array([0,0,0,10]).reshape((4,1))
    quaternion = np.zeros((4, 13))
    w_prime = np.zeros((6, 13))
    roll_final = np.zeros(T)
    roll_final[0] = roll[0]
    pitch_final = np.zeros(T)
    pitch_final[0] = pitch[0]
    yaw_final = np.zeros(T)
    yaw_final[0] = yaw[0]
    # return np.zeros((T,1)),np.zeros((T,1)),np.zeros((T,1))
    for i in range(1,T):
        if i%100==0:
            print(i)
        dt = ts[0][i] - ts[0][i-1]
        sigma = sigma_points(xk_ini, cov_k_ini, sigma_ini, weight, Q, R, dt)
        #dt = ts[0][i] - ts[0][i-1]
        # print(sigma)
        
        xk_mean, Pk, Pxz, Pvv, z_mean = sigma_transform(sigma, w_prime, quaternion, dt, Z, R, g, Q, weight )
        #print(xk_mean)
        xk_ini, cov_k_ini = kalman(Pk, Pvv, Pxz, ax[i], ay[i], az[i], wx[i], wy[i], wz[i], z_mean, xk_mean, K_inno_q )
        # cov_k_ini = np.where(cov_k_ini < 0, -cov_k_ini, cov_k_ini)
        #print(xk_ini)
        #roll_final[i], pitch_final[i], yaw_final[i] =quat2rpy(xk_ini[0:4].reshape((-1,1)))
        temp = Quaternion(xk_ini[0][0], xk_ini[1:4].reshape((1,3)))
        final = temp.euler_angles().reshape((3,1))
        # print(final)
        roll_final[i] = final[0][0]
        pitch_final[i] = final[1][0]
        yaw_final[i] = final[2][0]

    # roll, pitch, yaw are numpy arrays of length T
    return pitch_final,yaw_final,roll_final

    # vicon_data = io.loadmat(vicon)
    # vicon_data_rots = vicon['rots']
    # vicon_roll = np.arctan2(vicon_data_rots[2, 1, :], vicon_data_rots[2, 2, :])
    # vicon_pitch = np.arctan2(-vicon_data_rots[2, 0,:],np.sqrt(vicon_data_rots[2, 1, :] ** 2 + vicon_data_rots[2, 2, :] ** 2))
    # vicon_yaw = np.arctan2(vicon_data_rots[1,0,:],vicon_data_rots[0,0,:])

    # plt.figure(1)
    # plt.plot(vicon_roll,'k')
    # plt.plot(pitch_final)
    # plt.figure(2)
    # plt.plot(vicon_pitch,'k')
    # plt.plot(roll_final)
    # plt.figure(3)
    # plt.plot(vicon_yaw, 'k')
    # plt.plot(yaw_final)
    # plt.show()

estimate_rot(1)


import matplotlib.pyplot as plt

num = 1
vicon = io.loadmat("viconRot" + str(num) + ".mat")
roll, pitch, yaw = estimate_rot(num)

vicon2Sens = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
r = []
p = []
y = []
quat = Quaternion()
for i in range(vicon["rots"].shape[-1]):
    R = vicon["rots"][:, :, i].reshape(3, 3)
    quat.from_rotm(R)
    ang = quat.euler_angles()
    r.append(float(ang[0]))
    p.append(float(ang[1]))
    y.append(float(ang[2]))
r = np.array(r)
plt.figure()
plt.subplot(2, 2, 1)
plt.plot(r[: roll.shape[0]])
plt.plot(roll)
plt.legend(["Vicon","Filtered"])
plt.title("Roll Angle")
plt.ylabel("rad")

plt.subplot(2, 2, 2)
plt.plot(p[: roll.shape[0]])
plt.plot(pitch)
plt.legend(["Vicon","Filtered"])
plt.title("Pitch Angle")
plt.ylabel("rad")

plt.subplot(2, 2, 3)
plt.plot(y[: roll.shape[0]])
plt.plot(yaw)
plt.legend(["Vicon","Filtered"])
plt.title("Yaw Angle")
plt.ylabel("rad")
plt.show()
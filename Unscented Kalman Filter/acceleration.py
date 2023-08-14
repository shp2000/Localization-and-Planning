import numpy as np
from scipy import io
import matplotlib.pyplot as plt

imu = io.loadmat('imuRaw1.mat')['vals']
vicon = io.loadmat('viconRot1.mat')['rots']
#vicon_data_rots = vicon_data['rots']

v_roll = np.arctan2(vicon[2,1,:],vicon[2,2,:])
v_pitch = np.arctan2(-vicon[2,0],np.sqrt(vicon[2,1,:]**2+vicon[2,2,:]**2))

#print(vicon)
beta_x = 511
beta_y = 500
beta_z = 500
alpha = 20
acc= imu[0:3,:]
ax = (acc[0,:] - beta_x)*3300/(1023*alpha)
ay = (acc[1,:]- beta_y)*3300/(1023*alpha)
az = (acc[2,:] - beta_z)*3300/(1023*alpha)
print(az)
print(acc[2,:])
i_roll = np.arctan2(-ay, az)
i_pitch = np.arctan2(ax,np.sqrt(ay**2+az**2))

plt.figure(1)
plt.plot(v_roll,'k')
plt.plot(i_roll)
plt.figure(2)
plt.plot(v_pitch,'k')
plt.plot(i_pitch)
plt.show()
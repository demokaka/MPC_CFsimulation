import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

ptf = './Trajectory/traj4UAVs_Vincent_obs2.mat'
sim_data = loadmat(ptf)
for key, value in sim_data.items() :
    print (key, )

x = sim_data["x"] - 0.5
y = sim_data["y"] 
z = sim_data["z"] 

vx = sim_data["vx"]
vy = sim_data["vy"]
vz = sim_data["vz"]

ax = sim_data["ax"]
ay = sim_data["ay"]
az = sim_data["az"]

tt2 = sim_data["tt2"]

XiF = sim_data["XiF"]
Xi_2F = sim_data["Xi_2F"]

tt = np.arange(start=0.0, stop=20.1, step=0.1)

print(np.array(ax).shape)
print(np.array(vx).shape)
print(np.array(x).shape)
print(np.array(XiF).shape)
print(np.array(Xi_2F).shape)
print(np.array(tt2).shape)
print(np.array(tt).shape)
print(tt)
print(tt2)




fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot(x[0,:],y[0,:],z[0,:])

figx = plt.figure()
axx = figx.add_subplot()
# axx.plot(tt2,x[0,:])
# axx.plot(tt2,y[0,:])
# axx.plot(tt2,z[0,:])



plt.show()
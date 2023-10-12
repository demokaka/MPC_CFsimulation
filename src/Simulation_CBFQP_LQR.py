import HT_control_packagecf as ctrl
import CFmodel
import HT_generate_trajectory as trajgen_thinh
import get_solver_cmpc as cmpc
import numpy as np
import yaml
## Load parameters from config file
with open('Config_Crazyflie_V2.yaml') as f:
    system_parameters = yaml.load(f,Loader=yaml.FullLoader)

qtm_ip = system_parameters['qtm_ip']
Ts = system_parameters['Ts']
Tsim = system_parameters['Tsim']
m = system_parameters['mass']
uris = system_parameters['uris']
drone_bodies = system_parameters['drone_bodies']

## Load Trajectories for drones:
# ref = {}
# vref = {}
# full_ref1 = trajgen_thinh.get_ref_setpoints_Khanh(psi=0,Tsim=Tsim,dt=Ts,agent=1)
# full_ref2 = trajgen_thinh.get_ref_setpoints_Khanh(psi=0,Tsim=Tsim,dt=Ts,agent=2)
# full_ref3 = trajgen_thinh.get_ref_setpoints_Khanh(psi=0,Tsim=Tsim,dt=Ts,agent=3)

# full_ref1 = trajgen_thinh.get_ref_setpoints(psi=0,Tsim=Tsim,dt=Ts,version=11)
# full_ref2 = trajgen_thinh.get_ref_setpoints(psi=0,Tsim=Tsim,dt=Ts,version=12)
# full_ref3 = trajgen_thinh.get_ref_setpoints(psi=0,Tsim=Tsim,dt=Ts,version=13)
# ref = {uris[0]: full_ref1["trajectory"],
#        uris[1]: full_ref2["trajectory"],
#       uris[2]: full_ref3["trajectory"]} 

# vref = {uris[0]: full_ref1["v_ref"],
#         uris[1]: full_ref2["v_ref"],
#         uris[2]: full_ref3["v_ref"]}

# print(np.size(ref[uris[0]],1))
# vref[uris[0]] = np.zeros((np.size(ref[uris[0]],0),3))
# vref[uris[1]] = np.zeros((np.size(ref[uris[1]],0),3))
# vref[uris[2]] = np.zeros((np.size(ref[uris[2]],0),3))


ptf = './Trajectory/traj4UAVs_Vincent_obs2.mat'
# full_ref = trajgen_thinh.get_trajectory_mat(path_to_file=ptf,dt=Ts)
full_ref = trajgen_thinh.get_ref_setpoints_all(path_to_file=ptf,dt=Ts,Tsim=Tsim)
ref = {}

vref = {}
for i in range(len(drone_bodies)):
    ref[uris[i]] = full_ref[i]["trajectory"]

    vref[uris[i]] = full_ref[i]["v_ref"]


common_plant,common_controller = cmpc.load_constant_parameters(Ts=Ts)
drone_params = {}
for i in range(len(drone_bodies)):
    drone_params[i] = cmpc.droneParameters(drone_address=uris[i],body_name=drone_bodies[i],mass=m[i],
                                                         plant=common_plant,controller=common_controller) 

central_plant,central_controller,simulator = cmpc.get_stacked_drones_parameters(list_of_drones=drone_params)
simulator['Nsim'] = np.size(ref[uris[0]],0)
# CMPC_solver,CMPC_solver_variables = cmpc.get_solver_cmpc(plant=central_plant,controller=central_controller,simulator=simulator)


simulator['u_sim'] = np.zeros((common_plant['du'],simulator['Nsim'],simulator['na']))
simulator['x_sim'] = np.zeros((common_plant['dx'],simulator['Nsim']+1,simulator['na']))
simulator['h_up'] = np.zeros((common_plant['du'],simulator['Nsim'],simulator['na']))
simulator['h_down'] = np.zeros((common_plant['du'],simulator['Nsim'],simulator['na']))
for i in range(simulator['na']):
    simulator['u_sim'][:,:,i] = np.zeros((common_plant['du'],simulator['Nsim']))
    simulator['x_sim'][:,0,i] = ref[uris[i]][0,:].T 
    


simulator['un_sim'] = np.zeros((common_plant['du'],simulator['Nsim'],simulator['na']))
simulator['xn_sim'] = np.zeros((common_plant['dx'],simulator['Nsim']+1,simulator['na']))
simulator['hn_up'] = np.zeros((common_plant['du'],simulator['Nsim'],simulator['na']))
simulator['hn_down'] = np.zeros((common_plant['du'],simulator['Nsim'],simulator['na']))

for i in range(simulator['na']):
    simulator['un_sim'][:,:,i] = np.zeros((common_plant['du'],simulator['Nsim']))
    simulator['xn_sim'][:,0,i] = ref[uris[i]][0,:].T 


import get_solver_CBFQP as cbfqp
solver = {}
dc=0.1
a1=6
a2=8
for i in range(len(drone_bodies)):
    solver[drone_bodies[i]] = cbfqp.CBFQPSolver(v_ref0=vref[uris[i]][0,:],
                                             x_ref0=ref[uris[i]][0,:],
                                             x0=simulator['x_sim'][:,0,i],
                                             a1=a1, a2=a2,dc=dc)
    res = solver[drone_bodies[i]].prob.solve()
    res = solver[drone_bodies[i]].prob.solve()
    res = solver[drone_bodies[i]].prob.solve()
    print(res.x)


""" Nominal controller(LQR) and CBF-QP"""
### import LQR parameters
import get_parameter_LQR as LQR
Kf = {}
# for i in range(len(drone_bodies)):
#     Kf[uris[i]]  =  -2.0 *  np.array([[2.5, 0, 0, 1.5, 0, 0],
#                                     [0, 2.5, 0, 0, 1.5, 0],
#                                     [0, 0, 2.5, 0, 0, 1.5]]) #gain matrix obtained from LQR
for i in range(len(drone_bodies)):
    Kf[uris[i]]  =  -LQR.get_LQR_gain(Ts=Ts,qp=2000,qv=500,r=100)  #gain matrix obtained from LQR

k = 0
import time 
start = time.perf_counter()
while(k< simulator['Nsim']):
    
    tic = time.perf_counter()
    # if k==0:
    #     u_init = np.zeros((central_plant['du'],1))
    # else:
    #     u_init = simulator['U_total'][:,k-1]
    
    # simulator['X_ref'] = cmpc.get_ref_pred_horz(ref=ref,k=k,Npred=common_controller['Npred']+1,uris=uris)
    # simulator['v_ref'] = cmpc.get_ref_pred_horz(ref=vref,k=k,Npred=common_controller['Npred'],uris=uris)
    # for i in range(simulator['na']):
    #     simulator['X_total'][i*(common_plant['dx']):(i+1)*(common_plant['dx']),k] = simulator['x_sim'][:,k,i]
    
    # simulator['U_total'][:,k] = cmpc.compute_control_cmpc(solver=CMPC_solver,solver_variables=CMPC_solver_variables,X0=simulator['X_total'][:,k],
    #                           v0=u_init,Xrefk=simulator['X_ref'],vrefk = simulator['v_ref'],yawk = 0)
    controls = {}
    for i in range(simulator['na']):
        simulator["h_up"][:,k,i] = dc + ref[uris[i]][k,0:3] - simulator['x_sim'][0:3,k,i]
        simulator["h_down"][:,k,i] = dc - ref[uris[i]][k,0:3] + simulator['x_sim'][0:3,k,i]
        # simulator['u_sim'][:,k,i] = simulator['U_total'][i*(common_plant['du']):(i+1)*(common_plant['du']),k]
        solver[drone_bodies[i]].update(v_ref=vref[uris[i]][k,:],
                     x_ref=ref[uris[i]][k,:],
                    x=simulator['x_sim'][:,k,i],)
        res = solver[drone_bodies[i]].prob.solve()
        simulator['u_sim'][:,k,i] = (res.x.reshape(-1, 1) +  ctrl.compute_control(v_ref=vref[uris[i]][k,:].reshape(-1, 1),
                                                                 x0=simulator['x_sim'][:,k,i].reshape(-1, 1),
                                                                 xref=ref[uris[i]][k,:].reshape(-1, 1),
                                                                 Kf=Kf[uris[i]]) ).ravel()
        # print(simulator['u_sim'][:,k,i])
        # compute real control u
        controls[uris[i]] = ctrl.get_real_input(v_controls=simulator['u_sim'][:,k,i],yaw=0)
        
        # disturbance
        disturb = np.zeros((1,3))
        # disturb = disturb + np.random.normal(loc=0.0,scale=0.03,size=(1,3))
        simulator['x_sim'][:,k+1,i] = CFmodel.CFmodel(xk=simulator['x_sim'][:,k,i],urk= controls[uris[i]],plant=common_plant,disturb=disturb)

    k = k+1 
    
    # print(f'{time.perf_counter()-tic} sec')
end = time.perf_counter()
calcul_t = end-start


print('Calulation time : ',calcul_t)
print('Average calculation time : ',calcul_t/simulator['Nsim'])

""" Only nominal controller (LQR)"""

k = 0
Kf = {}
# for i in range(len(drone_bodies)):
#     Kf[uris[i]]  =  -2.0 *  np.array([[2.5, 0, 0, 1.5, 0, 0],
#                                     [0, 2.5, 0, 0, 1.5, 0],
#                                     [0, 0, 2.5, 0, 0, 1.5]]) #gain matrix obtained from LQR
for i in range(len(drone_bodies)):
    Kf[uris[i]]  =  -LQR.get_LQR_gain(Ts=Ts,qp=2000,qv=500,r=100)  #gain matrix obtained from LQR

import time 
start = time.perf_counter()
while(k< simulator['Nsim']):
    tic = time.perf_counter()
    # if k==0:
    #     u_init = np.zeros((central_plant['du'],1))
    # else:
    #     u_init = simulator['U_total'][:,k-1]
    
    # simulator['X_ref'] = cmpc.get_ref_pred_horz(ref=ref,k=k,Npred=common_controller['Npred']+1,uris=uris)
    # simulator['v_ref'] = cmpc.get_ref_pred_horz(ref=vref,k=k,Npred=common_controller['Npred'],uris=uris)
    # for i in range(simulator['na']):
    #     simulator['X_total'][i*(common_plant['dx']):(i+1)*(common_plant['dx']),k] = simulator['x_sim'][:,k,i]
    
    # simulator['U_total'][:,k] = cmpc.compute_control_cmpc(solver=CMPC_solver,solver_variables=CMPC_solver_variables,X0=simulator['X_total'][:,k],
    #                           v0=u_init,Xrefk=simulator['X_ref'],vrefk = simulator['v_ref'],yawk = 0)
    controls = {}
    for i in range(simulator['na']):
        simulator["hn_up"][:,k,i] = dc + ref[uris[i]][k,0:3] - simulator['xn_sim'][0:3,k,i]
        simulator["hn_down"][:,k,i] = dc - ref[uris[i]][k,0:3] + simulator['xn_sim'][0:3,k,i]
        # simulator['u_sim'][:,k,i] = simulator['U_total'][i*(common_plant['du']):(i+1)*(common_plant['du']),k]
        # solver[drone_bodies[0]].update(v_ref=vref[uris[0]][k,:],
        #              x_ref=ref[uris[0]][k,:],
        #             x=simulator['x_sim'][:,k,0],)
        # res = solver[drone_bodies[0]].prob.solve()
        simulator['un_sim'][:,k,i] = ctrl.compute_control(v_ref=vref[uris[i]][k,:],
                                                                 x0=simulator['xn_sim'][:,k,i],
                                                                 xref=ref[uris[i]][k,:],
                                                                 Kf=Kf[uris[i]])
        # compute real control u
        controls[uris[i]] = ctrl.get_real_input(v_controls=simulator['un_sim'][:,k,i],yaw=0)
        
        # disturbance
        disturb = np.zeros((1,3))
        # disturb = disturb + np.random.normal(loc=0.0,scale=0.03,size=(1,3))
        simulator['xn_sim'][:,k+1,i] = CFmodel.CFmodel(xk=simulator['xn_sim'][:,k,i],urk= controls[uris[i]],plant=common_plant,disturb=disturb)

    k = k+1 
    
    # print(f'{time.perf_counter()-tic} sec')
end = time.perf_counter()
calcul_t = end-start


print('Calulation time : ',calcul_t)
print('Average calculation time : ',calcul_t/simulator['Nsim'])


ur = np.zeros((3,simulator['Nsim'],simulator['na']))
for k in range(simulator['Nsim']):
    for i in range(simulator['na']):
        # compute real control u
        ur[:,k,i]= ctrl.get_real_input(v_controls=simulator['u_sim'][:,k,i],yaw=0)

urn = np.zeros((3,simulator['Nsim'],simulator['na']))
for k in range(simulator['Nsim']):
    for i in range(simulator['na']):
        # compute real control u
        urn[:,k,i]= ctrl.get_real_input(v_controls=simulator['un_sim'][:,k,i],yaw=0)

# plot result

import matplotlib.pyplot as plt
# import matplotlib

ts = np.linspace(start=0,stop=Tsim,num=simulator['Nsim']+1)
# plot position x
figxc = plt.figure()
for i in range(simulator['na']):
    axxc = figxc.add_subplot(simulator['na'],1,i+1)
    axxc.plot(ts,simulator['x_sim'][0,:,i],label='Trajectory CBF-QP')
    axxc.plot(ts,simulator['xn_sim'][0,:,i],label='Trajectory nominal',linestyle='dashed')
    axxc.plot(ts[1:],ref[uris[i]][:,0],label='Reference')
    axxc.grid(True)
    axxc.set_xlabel("Time (s)")
    axxc.set_ylabel(f"x{drone_bodies[i]} (m)")
    if (i==0):
        axxc.legend()

# plot position y
figyc = plt.figure()
for i in range(simulator['na']):
    axyc = figyc.add_subplot(simulator['na'],1,i+1)
    axyc.plot(ts,simulator['x_sim'][1,:,i],label='Trajectory CBF-QP')
    axyc.plot(ts,simulator['xn_sim'][1,:,i],label='Trajectory nominal',linestyle='dashed')
    axyc.plot(ts[1:],ref[uris[i]][:,1],label='Reference')
    axyc.grid(True)
    axyc.set_xlabel("Time (s)")
    axyc.set_ylabel(f"y{drone_bodies[i]} (m)")
    if (i==0):
        axyc.legend()

# plot position x
figzc = plt.figure()
for i in range(simulator['na']):
    axzc = figzc.add_subplot(simulator['na'],1,i+1)
    axzc.plot(ts,simulator['x_sim'][2,:,i],label='Trajectory CBF-QP')
    axzc.plot(ts,simulator['xn_sim'][2,:,i],label='Trajectory nominal',linestyle='dashed')
    axzc.plot(ts[1:],ref[uris[i]][:,2],label='Reference')
    axzc.grid(True)
    axzc.set_xlabel("Time (s)")
    axzc.set_ylabel(f"z{drone_bodies[i]} (m)")
    if (i==0):
        axzc.legend()





# plot position x
figvxc = plt.figure()
for i in range(simulator['na']):
    axvxc = figvxc.add_subplot(simulator['na'],1,i+1)
    axvxc.plot(ts,simulator['x_sim'][3,:,i],label='Trajectory CBF-QP')
    axvxc.plot(ts,simulator['xn_sim'][3,:,i],label='Trajectory nominal',linestyle='dashed')
    axvxc.plot(ts[1:],ref[uris[i]][:,3],label='Reference')
    axvxc.grid(True)
    axvxc.set_xlabel("Time (s)")
    axvxc.set_ylabel(f"vx{drone_bodies[i]} (m)")
    if (i==0):
        axvxc.legend()

# plot position y
figvyc = plt.figure()
for i in range(simulator['na']):
    axvyc = figvyc.add_subplot(simulator['na'],1,i+1)
    axvyc.plot(ts,simulator['x_sim'][4,:,i],label='Trajectory CBF-QP')
    axvyc.plot(ts,simulator['xn_sim'][4,:,i],label='Trajectory nominal',linestyle='dashed')
    axvyc.plot(ts[1:],ref[uris[i]][:,4],label='Reference')
    axvyc.grid(True)
    axvyc.set_xlabel("Time (s)")
    axvyc.set_ylabel(f"vy{drone_bodies[i]} (m)")
    if (i==0):
        axvyc.legend()

# plot position x
figvzc = plt.figure()
for i in range(simulator['na']):
    axvzc = figvzc.add_subplot(simulator['na'],1,i+1)
    axvzc.plot(ts,simulator['x_sim'][5,:,i],label='Trajectory CBF-QP')
    axvzc.plot(ts,simulator['xn_sim'][5,:,i],label='Trajectory nominal',linestyle='dashed')
    axvzc.plot(ts[1:],ref[uris[i]][:,5],label='Reference')
    axvzc.grid(True)
    axvzc.set_xlabel("Time (s)")
    axvzc.set_ylabel(f"vz{drone_bodies[i]} (m)")
    if (i==0):
        axvzc.legend()


# plot acceleration ax
figaxc = plt.figure()
for i in range(simulator['na']):
    axaxc = figaxc.add_subplot(simulator['na'],1,i+1)
    axaxc.plot(ts[1:],simulator['u_sim'][0,:,i],label='Trajectory CBF-QP')
    axaxc.plot(ts[1:],simulator['un_sim'][0,:,i],label='Trajectory nominal',linestyle='dashed')
    axaxc.plot(ts[1:],vref[uris[i]][:,0],label='Reference')
    axaxc.grid(True)
    axaxc.set_xlabel("Time (s)")
    axaxc.set_ylabel(f"ax{drone_bodies[i]} (m)")
    if (i==0):
        axaxc.legend()

# plot acceleration ay
figayc = plt.figure()
for i in range(simulator['na']):
    axayc = figayc.add_subplot(simulator['na'],1,i+1)
    axayc.plot(ts[1:],simulator['u_sim'][1,:,i],label='Trajectory CBF-QP')
    axayc.plot(ts[1:],simulator['un_sim'][1,:,i],label='Trajectory nominal',linestyle='dashed')
    axayc.plot(ts[1:],vref[uris[i]][:,1],label='Reference')
    axayc.grid(True)
    axayc.set_xlabel("Time (s)")
    axayc.set_ylabel(f"ay{drone_bodies[i]} (m)")
    if (i==0):
        axayc.legend()

# plot acceleration az
figazc = plt.figure()
for i in range(simulator['na']):
    axazc = figazc.add_subplot(simulator['na'],1,i+1)
    axazc.plot(ts[1:],simulator['u_sim'][2,:,i],label='Trajectory CBF-QP')
    axazc.plot(ts[1:],simulator['un_sim'][2,:,i],label='Trajectory nominal',linestyle='dashed')
    axazc.plot(ts[1:],vref[uris[i]][:,2],label='Reference')
    axazc.grid(True)
    axazc.set_xlabel("Time (s)")
    axazc.set_ylabel(f"az{drone_bodies[i]} (m)")
    if (i==0):
        axazc.legend()




# Create a figure and axis object
figcc = plt.figure()

# plot Thrust
g = 9.81
Tmax = 10.5
for i in range(simulator['na']):
    axcc = figcc.add_subplot(simulator['na'],1,i+1)
    axcc.plot(ts[1:],ur[0,:,i],label='CBF-QP + Nominal')
    axcc.plot(ts[1:],urn[0,:,i],label='Nominal')
    axcc.grid(visible=True)

    axcc.plot(ts,np.ones(np.size(ts)) * Tmax,'k--',linewidth=1.5)
    axcc.plot(ts,np.ones(np.size(ts)) * 9.7,'k--',linewidth=1.5)
    axcc.set_xlabel('Time (s)')
    axcc.set_ylabel('Thrust T(m/s²)')
    if (i==0):
        axcc.legend()
# fig.savefig("./Simulation/Thrust_MPC_tracking.eps", format='eps')

phimax = 2.5 * np.pi/180
thetamax = 2.5 * np.pi/180

figrc = plt.figure()

for i in range(simulator['na']):
    axrc = figrc.add_subplot(simulator['na'],1,i+1)
    axrc.plot(ts[1:],ur[1,:,i]*180/np.pi,label='CBF-QP + Nominal')
    axrc.plot(ts[1:],urn[1,:,i]*180/np.pi,label='Nominal')
    axrc.grid(visible=True)

    axrc.plot(ts,np.ones(np.size(ts)) * phimax*180/np.pi,'k--',linewidth=1.5)
    axrc.plot(ts,np.ones(np.size(ts)) * -phimax*180/np.pi,'k--',linewidth=1.5)
    axrc.set_xlabel('Time (s)')
    axrc.set_ylabel('Roll angle (°)')
    if (i==0):
        axrc.legend()
# figrc.savefig("./Simulation/Roll_MPC_tracking.eps", format='eps')


figpc= plt.figure()

for i in range(simulator['na']):
    axpc = figpc.add_subplot(simulator['na'],1,i+1)
    axpc.plot(ts[1:],ur[2,:,i]*180/np.pi,label='CBF-QP + Nominal')
    axpc.plot(ts[1:],urn[2,:,i]*180/np.pi,label='Nominal')
    axpc.grid(visible=True)

    axpc.plot(ts,np.ones(np.size(ts)) * thetamax *180/np.pi,'k--',linewidth=1.5)
    axpc.plot(ts,np.ones(np.size(ts)) * -thetamax *180/np.pi,'k--',linewidth=1.5)
    axpc.set_xlabel('Time (s)')
    axpc.set_ylabel('Pitch angle (°)')
    if (i==0):
        axpc.legend()

fighqx = plt.figure()
for i in range(simulator['na']):
    axhqx = fighqx.add_subplot(simulator['na'],1,i+1)
    axhqx.plot(ts[1:],simulator['h_up'][0,:,i],label='CBF-QP + Nominal')
    axhqx.plot(ts[1:],simulator['hn_up'][0,:,i],label='Nominal')
    axhqx.grid(visible=True)

    axhqx.plot(ts,np.ones(np.size(ts)) * dc ,'k--',linewidth=1.5)
    axhqx.plot(ts,np.ones(np.size(ts)) * -dc ,'k--',linewidth=1.5)
    axhqx.set_xlabel('Time (s)')
    axhqx.set_ylabel('hx (m)')
    if (i==0):
        axhqx.legend()

fighqy = plt.figure()
for i in range(simulator['na']):
    axhqy = fighqy.add_subplot(simulator['na'],1,i+1)
    axhqy.plot(ts[1:],simulator['h_up'][1,:,i],label='CBF-QP + Nominal')
    axhqy.plot(ts[1:],simulator['hn_up'][1,:,i],label='Nominal')
    axhqy.grid(visible=True)

    axhqy.plot(ts,np.ones(np.size(ts)) * dc * 2,'k--',linewidth=1.5)
    axhqy.plot(ts,np.ones(np.size(ts)) * -dc * 2,'k--',linewidth=1.5)
    axhqy.set_xlabel('Time (s)')
    axhqy.set_ylabel('hy (m)')
    if (i==0):
        axhqy.legend()

fighqz = plt.figure()
for i in range(simulator['na']):
    axhqz = fighqz.add_subplot(simulator['na'],1,i+1)
    axhqz.plot(ts[1:],simulator['h_up'][2,:,i],label='CBF-QP + Nominal')
    axhqz.plot(ts[1:],simulator['hn_up'][2,:,i],label='Nominal')
    axhqz.grid(visible=True)

    axhqz.plot(ts,np.ones(np.size(ts)) * dc * 2,'k--',linewidth=1.5)
    axhqz.plot(ts,np.ones(np.size(ts)) * -dc * 2,'k--',linewidth=1.5)
    axhqz.set_xlabel('Time (s)')
    axhqz.set_ylabel('hz (m)')
    if (i==0):
        axhqz.legend()

fighqx_ = plt.figure()
for i in range(simulator['na']):
    axhqx_ = fighqx_.add_subplot(simulator['na'],1,i+1)
    axhqx_.plot(ts[1:],simulator['h_down'][0,:,i],label='CBF-QP + Nominal')
    axhqx_.plot(ts[1:],simulator['hn_down'][0,:,i],label='Nominal')
    axhqx_.grid(visible=True)

    axhqx_.plot(ts,np.ones(np.size(ts)) * dc ,'k--',linewidth=1.5)
    axhqx_.plot(ts,np.ones(np.size(ts)) * -dc ,'k--',linewidth=1.5)
    axhqx_.set_xlabel('Time (s)')
    axhqx_.set_ylabel('hx_ (m)')
    if (i==0):
        axhqx_.legend()

fighqy_ = plt.figure()
for i in range(simulator['na']):
    axhqy_ = fighqy_.add_subplot(simulator['na'],1,i+1)
    axhqy_.plot(ts[1:],simulator['h_down'][1,:,i],label='CBF-QP + Nominal')
    axhqy_.plot(ts[1:],simulator['hn_down'][1,:,i],label='Nominal')
    axhqy_.grid(visible=True)

    axhqy_.plot(ts,np.ones(np.size(ts)) * dc * 2,'k--',linewidth=1.5)
    axhqy_.plot(ts,np.ones(np.size(ts)) * -dc * 2,'k--',linewidth=1.5)
    axhqy_.set_xlabel('Time (s)')
    axhqy_.set_ylabel('hy_ (m)')
    if (i==0):
        axhqy_.legend()

fighqz_ = plt.figure()
for i in range(simulator['na']):
    axhqz_ = fighqz_.add_subplot(simulator['na'],1,i+1)
    axhqz_.plot(ts[1:],simulator['h_down'][2,:,i],label='CBF-QP + Nominal')
    axhqz_.plot(ts[1:],simulator['hn_down'][2,:,i],label='Nominal')
    axhqz_.grid(visible=True)

    axhqz_.plot(ts,np.ones(np.size(ts)) * dc * 2,'k--',linewidth=1.5)
    axhqz_.plot(ts,np.ones(np.size(ts)) * -dc * 2,'k--',linewidth=1.5)
    axhqz_.set_xlabel('Time (s)')
    axhqz_.set_ylabel('hz_ (m)')
    if (i==0):
        axhqz_.legend()

plt.show()
from casadi import *
import numpy as np
import dill as pickle
from Bspline_package import Bspline_conversionMatrix as BsplineM
from scipy.interpolate import BSpline
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.io import loadmat


def get_ref(W, psi, Tsim, dt):
    k = 8  # d=k+1, k: polynomial degree of the spline
    n_ctrl_pts = 28  # number of control points
    knot = [0, Tsim]
    g = 9.81
    # psi = 0 * np.pi / 180
    knot = BsplineM.knot_vector(k, n_ctrl_pts, knot)
    tt = numpy.arange(min(knot), max(knot), dt)
    bs_list = BsplineM.b_spline_basis_functions(n_ctrl_pts, k, knot)
    # Conversion matrix M
    M = BsplineM.bsplineConversionMatrices(n_ctrl_pts, k, knot)
    # W = np.array([[0, 0.3, 0.6, 0.6, 0.3, 0, -0.3, -0.3, 0],
    #               [0, -0.3, 0, 0.3, 0.6, 0.6, 0.3, 0, 0],
    #               [0.35, 0.4, 0.75, 0.8, 0.8, 0.8, 0.8, 0.5, 0.35]  # 3D test
    #               ])
    waypoint_time_stamps = np.linspace(min(knot), max(knot), W.shape[1])
    ctrl_pts_timestamps = np.linspace(min(knot), max(knot), n_ctrl_pts)
    # ------ Optimization problem construction --------------------
    solver = casadi.Opti()
    # Control point as optimization variable
    P = solver.variable(W.shape[0], n_ctrl_pts)
    # Objective function
    objective = 0
    P1 = mtimes(P, M[0])
    for i in range(n_ctrl_pts + 1):
        for j in range(n_ctrl_pts + 1):
            f_lamb = lambda t, it=i, jt=j: bs_list[1][it](t) * bs_list[1][jt](t)
            buff_int = quad(f_lamb, min(knot), max(knot))[0]
            objective = objective + mtimes(transpose(mtimes(buff_int, P1[:, i])), P1[:, j])
    # Implementing waypoint constraints
    for i in range(W.shape[1]):
        tmp_bs = np.zeros((len(bs_list[0]), 1))
        for j in range(len(bs_list[0])):
            tmp_bs[j] = bs_list[0][j](waypoint_time_stamps[i])
        # Mathematically, mtimes(P, tmp_bs) = P * tmp_bs
        solver.subject_to(mtimes(P, tmp_bs) == W[:, i])
    # Final velocity is zero
    for i in range(W.shape[1]):
        tmp_bs = np.zeros((len(bs_list[1]), 1))
        for j in range(len(bs_list[1])):
            tmp_bs[j] = bs_list[1][j](waypoint_time_stamps[-1] - dt)
        solver.subject_to(mtimes(P1, tmp_bs) == 0)
    # Final acceleration is zero
    P2 = mtimes(P, M[1])
    for i in range(W.shape[1]):
        tmp_bs = np.zeros((len(bs_list[2]), 1))
        for j in range(len(bs_list[2])):
            tmp_bs[j] = bs_list[2][j](waypoint_time_stamps[-1] - dt)
        solver.subject_to(mtimes(P2, tmp_bs) == 0)

    solver.minimize(objective)

    solver_options = {'ipopt': {'print_level': 0, 'sb': 'yes'}, 'print_time': 0}
    solver.solver('ipopt', solver_options)
    # ============================================================================================
    print('Generating reference ...')
    sol = solver.solve()  # Solve for the control points
    # ============================================================================================
    # Construct the result curve
    P = sol.value(P)
    print('Optimal control-points found')
    # Compute the Bspline with the solution of P
    z = []
    for i in range(P.shape[0]):
        z.append(BSpline(knot, P[i], k))

    # First derivative of the flat output
    P1 = np.array(P * M[0])
    z_d = []
    for i in range(P1.shape[0]):
        z_d.append(BSpline(knot, P1[i], k - 1))
    # Second derivative of the flat output
    P2 = np.array(P * M[1])
    z_dd = []
    for i in range(P2.shape[0]):
        z_dd.append(BSpline(knot, P2[i], k - 2))

    x = z[0](tt)
    y = z[1](tt)
    z = z[2](tt)
    dx = z_d[0](tt)
    dy = z_d[1](tt)
    dz = z_d[2](tt)
    ddx = z_dd[0](tt)
    ddy = z_dd[1](tt)
    ddz = z_dd[2](tt)
    # ref = np.stack([x, y, z, dx, dy, dz, ddx, ddy, ddz])
    v_ref = np.stack([ddx, ddy, ddz])
    thrust = np.sqrt(ddx ** 2 + ddy ** 2 + (ddz + 9.81) ** 2)
    phi = np.arcsin((ddx * sin(psi) - ddy * cos(psi)) / thrust)
    theta = np.arctan((ddx * cos(psi) + ddy * sin(psi)) / (ddz + g))

    ref = {"trajectory": (np.round(np.stack([x, y, z, dx, dy, dz]), 3)).transpose(),
           "time_step": tt,
           "thrust": thrust,
           "phi": phi,
           "theta": theta,
           "Nsim": tt.shape[0],
           "v_ref": v_ref.transpose()}

    return ref




def get_ref_setpoints(psi, Tsim, dt, version=1):
    knot = [0, Tsim]
    g = 9.81
    tt = numpy.arange(min(knot), max(knot), dt)
    if version == 1:
        W = np.array([[0, 0.3, 0.6, 0.6, 0.3, 0, -0.3, -0.3, 0],
                      [0, -0.3, 0, 0.3, 0.6, 0.6, 0.3, 0, 0],
                      [0.35, 0.4, 0.75, 0.8, 0.8, 0.8, 0.8, 0.5, 0.35]  # 3D test
                      ])
    elif version == 2:
        W = np.array([[0, 0.6, 0.3, -0.3, 0],
                      [0, 0, 0.6, 0.3, 0],
                      [0.35, 0.75, 0.8, 0.8, 0.35]  # 3D test
                      ])
    elif version == 3:
        W = np.array([[0.3, 0.3],
                      [0.3, 0.3],
                      [0.8, 0.8]  # 3D test
                      ])
    elif version == 5:
        W = np.array([[-0.2, -0.2, -0.2,-0.2,0.3,0.8,0.8,0.8,0.3,-0.2,-0.2,-0.2],
                      [0, 0, 0, 0, 0.25,0.5,0.5,0.5,0.25,0,0,0],
                      [0.15, 0.3,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.2,0.15]  # 3D test
                      ])
    elif version == 6:
        W = np.array([[-0.2, -0.2 , -0.2,-0.2,0.8, 0.8,0.8,0.8,0.8,-0.2,-0.2,-0.2],
                      [0,    0,      0,   0,  0.5, 0.5,0.5,0.5,0.5,0,0,0],
                      [0.2,  0.2,    0.2, 0.2,0.4, 0.4,0.4,0.4,0.4,0.2,0.2,0.2]  # 3D test
                      ])
    elif version == 11:
        W = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0.2, 0.7, 0.7, 0.7]  # 3D test
                    ])
    elif version == 12:
        W = np.array([[-0.5, -0.5, -0.5, -0.5],
                    [0, 0, 0, 0],
                    [0.2, 0.7, 0.7, 0.7]  # 3D test
                    ])
    elif version == 13:
        W = np.array([[0.5, 0.5, 0.5, 0.5],
                    [0, 0, 0, 0],
                    [0.2, 0.7, 0.7, 0.7]  # 3D test
                    ])
    k_pass = 1
    ref_tmp = np.empty((0, 3))
    waypoint_time_stamps = np.linspace(min(knot), max(knot), W.shape[1] + 1)
    for i_tmp in range(waypoint_time_stamps.shape[0] - 1):
        cur = np.array(W[:, i_tmp])
        while dt * k_pass <= waypoint_time_stamps[i_tmp + 1]:
            ref_tmp = np.vstack((ref_tmp, cur))
            k_pass = k_pass + 1

    ref_full = np.block([
        [ref_tmp, ref_tmp * 0]
    ])
    v_ref = 0 * ref_tmp.transpose()
    v_ref[2, :] = v_ref[2, :] - 0.075
    v_ref[0, :] = v_ref[0, :] - 0.125
    ddx, ddy, ddz = v_ref[0, :], v_ref[1, :], v_ref[2, :]
    thrust = np.sqrt(ddx ** 2 + ddy ** 2 + (ddz + 9.81) ** 2)
    phi = np.arcsin((ddx * sin(psi) - ddy * cos(psi)) / thrust)
    theta = np.arctan((ddx * cos(psi) + ddy * sin(psi)) / (ddz + g))
    ref = {
        "trajectory": ref_full,
        "time_step": tt,
        "thrust": thrust,
        "phi": phi,
        "theta": theta,
        "Nsim": tt.shape[0],
        "v_ref": v_ref.transpose()}

    return ref



def get_ref_setpoints_Vincent(psi, Tsim, dt, agent=1):
    knot = [0, Tsim]
    g = 9.81
    tt = numpy.arange(min(knot), max(knot), dt)
    with open('sim_data123_Ts02_steps100_agents3_Rcomm07.pkl', 'rb') as inp:
        sim_data = pickle.load(inp)
    x_profile=sim_data['x_profile']
    y_profile=sim_data['y_profile']
    z_profile=sim_data['z_profile']
    vx_profile=sim_data['vx_profile']
    vy_profile=sim_data['vy_profile']
    vz_profile=sim_data['vz_profile']
    ux_profile=sim_data['ux_profile']
    uy_profile=sim_data['uy_profile']
    uz_profile=sim_data['uz_profile']
    # print(x_profile[:,agent-1])
    init_pos=sim_data['initial_positions']
    target_pos=sim_data['target_positions']
    W = np.zeros((6,x_profile.shape[0]))
    # print(W)
    W[0,:] = x_profile[:,agent-1].transpose()
    W[1,:] = y_profile[:,agent-1].transpose()
    W[2,:] = z_profile[:,agent-1].transpose()
    W[3,:] = vx_profile[:,agent-1].transpose()
    W[4,:] = vy_profile[:,agent-1].transpose()
    W[5,:] = vz_profile[:,agent-1].transpose()
    # if agent == 1:

    # elif agent == 2:
    #     W = np.array([[0, 0.6, 0.3, -0.3, 0],
    #                   [0, 0, 0.6, 0.3, 0],
    #                   [0.35, 0.75, 0.8, 0.8, 0.35]  # 3D test
    #                   ])
    # elif agent == 3:
    #     W = np.array([[0.3, 0.3],
    #                   [0.3, 0.3],
    #                   [0.8, 0.8]  # 3D test
    #                   ])

    k_pass = 1
    ref_tmp = np.empty((0, 6))
    waypoint_time_stamps = np.linspace(min(knot), max(knot), W.shape[1] + 1)
    for i_tmp in range(waypoint_time_stamps.shape[0] - 1):
        cur = np.array(W[:, i_tmp])
        while dt * k_pass <= waypoint_time_stamps[i_tmp + 1]:
            ref_tmp = np.vstack((ref_tmp, cur))
            k_pass = k_pass + 1

    ref_full = ref_tmp
    k_pass = 1
    ref_tmp = np.empty((0, 3))
    accel = np.zeros((3,x_profile.shape[0]))
    # print(W[0:3,-3:-1])
    accel[0,:-1] = ux_profile[:,agent-1].transpose()
    accel[0,:] = np.append(accel[0,:-1],0)
    accel[1,:-1] = uy_profile[:,agent-1].transpose()
    accel[1,:] = np.append(accel[1,:-1],0)
    accel[2,:-1] = uz_profile[:,agent-1].transpose()
    accel[2,:] = np.append(accel[2,:-1],0)
    # waypoint_time_stamps = np.linspace(min(knot), max(knot), W.shape[1] + 1)
    for i_tmp in range(waypoint_time_stamps.shape[0] - 1):
        cur = np.array(accel[:, i_tmp])
        while dt * k_pass <= waypoint_time_stamps[i_tmp + 1]:
            ref_tmp = np.vstack((ref_tmp, cur))
            k_pass = k_pass + 1
    v_ref = ref_tmp.transpose()
    ref = {
        "trajectory": ref_full,
        "time_step": tt,
        "Nsim": tt.shape[0],
        "v_ref": v_ref.transpose()}

    return ref



def get_ref_setpoints_Khanh(psi, Tsim, dt, agent=1):
    knot = [0, Tsim]
    g = 9.81
    tt = numpy.arange(min(knot), max(knot), dt)
    # with open('khanh.mat', 'rb') as inp:
    #     sim_data = pickle.load(inp)
    sim_data = loadmat('./Trajectory/khanh_3.mat')
    x_profile=sim_data['x_profile']
    y_profile=sim_data['y_profile']
    z_profile=sim_data['z_profile']
    vx_profile=sim_data['vx_profile']
    vy_profile=sim_data['vy_profile']
    vz_profile=sim_data['vz_profile']
    ux_profile=sim_data['ux_profile']
    uy_profile=sim_data['uy_profile']
    uz_profile=sim_data['uz_profile']
    # print(x_profile[:,agent-1])
    # init_pos=sim_data['initial_positions']
    # target_pos=sim_data['target_positions']
    W = np.zeros((6,x_profile.shape[0]))
    # print(W)
    W[0,:] = x_profile[:,agent-1].transpose()
    W[1,:] = y_profile[:,agent-1].transpose()
    W[2,:] = z_profile[:,agent-1].transpose()
    W[3,:] = vx_profile[:,agent-1].transpose()
    W[4,:] = vy_profile[:,agent-1].transpose()
    W[5,:] = vz_profile[:,agent-1].transpose()
    # if agent == 1:

    # elif agent == 2:
    #     W = np.array([[0, 0.6, 0.3, -0.3, 0],
    #                   [0, 0, 0.6, 0.3, 0],
    #                   [0.35, 0.75, 0.8, 0.8, 0.35]  # 3D test
    #                   ])
    # elif agent == 3:
    #     W = np.array([[0.3, 0.3],
    #                   [0.3, 0.3],
    #                   [0.8, 0.8]  # 3D test
    #                   ])

    k_pass = 1
    ref_tmp = np.empty((0, 6))
    waypoint_time_stamps = np.linspace(min(knot), max(knot), W.shape[1] + 1)
    for i_tmp in range(waypoint_time_stamps.shape[0] - 1):
        cur = np.array(W[:, i_tmp])
        while dt * k_pass <= waypoint_time_stamps[i_tmp + 1]:
            ref_tmp = np.vstack((ref_tmp, cur))
            k_pass = k_pass + 1

    ref_full = ref_tmp
    k_pass = 1
    ref_tmp = np.empty((0, 3))
    accel = np.zeros((3,x_profile.shape[0]))
    # print(W[0:3,-3:-1])
    accel[0,:] = ux_profile[:,agent-1].transpose()
    # accel[0,:] = np.append(accel[0,:-1],0)
    accel[1,:] = uy_profile[:,agent-1].transpose()
    # accel[1,:] = np.append(accel[1,:-1],0)
    accel[2,:] = uz_profile[:,agent-1].transpose()
    # accel[2,:] = np.append(accel[2,:-1],0)
    # waypoint_time_stamps = np.linspace(min(knot), max(knot), W.shape[1] + 1)
    for i_tmp in range(waypoint_time_stamps.shape[0] - 1):
        cur = np.array(accel[:, i_tmp])
        while dt * k_pass <= waypoint_time_stamps[i_tmp + 1]:
            ref_tmp = np.vstack((ref_tmp, cur))
            k_pass = k_pass + 1
    v_ref = ref_tmp.transpose()
    ref = {
        "trajectory": ref_full,
        "time_step": tt,
        "Nsim": tt.shape[0],
        "v_ref": v_ref.transpose()}

    return ref


def get_ref_setpoints_takeoff(psi, Tto, dt, ref):
    knot = [0, Tto]
    g = 9.81
    tt = numpy.arange(min(knot), max(knot), dt)
    k_pass = 1
    dest = ref[0,0:3].reshape(-1,1)
    # nbr_step = int(Tto/dt)
    nbr_step = 2
    W12 = np.repeat(dest[:2],nbr_step,axis=1)
    W3 = np.repeat(dest[2],nbr_step,axis=0)
    # W3 = np.linspace(0.15,dest[2],nbr_step).reshape((1,nbr_step))
    W = np.vstack((W12,W3))
    print(W)
    ref_tmp = np.empty((0, 3))
    waypoint_time_stamps = np.linspace(min(knot), max(knot), W.shape[1] + 1)
    for i_tmp in range(waypoint_time_stamps.shape[0] - 1):
        cur = np.array(W[:, i_tmp])
        while dt * k_pass <= waypoint_time_stamps[i_tmp + 1]:
            ref_tmp = np.vstack((ref_tmp, cur))
            k_pass = k_pass + 1

    ref_full = np.block([
        [ref_tmp, ref_tmp * 0]
    ])
    v_ref = 0 * ref_tmp.transpose()
    # v_ref[2, :] = v_ref[2, :] - 0.075
    # v_ref[0, :] = v_ref[0, :] - 0.125
    ddx, ddy, ddz = v_ref[0, :], v_ref[1, :], v_ref[2, :]
    thrust = np.sqrt(ddx ** 2 + ddy ** 2 + (ddz + 9.81) ** 2)
    phi = np.arcsin((ddx * sin(psi) - ddy * cos(psi)) / thrust)
    theta = np.arctan((ddx * cos(psi) + ddy * sin(psi)) / (ddz + g))
    ref = {
        "trajectory": ref_full,
        "time_step": tt,
        "thrust": thrust,
        "phi": phi,
        "theta": theta,
        "Nsim": tt.shape[0],
        "v_ref": v_ref.transpose()}

    return ref


def get_ref_setpoints_stage(psi, Tsim, dt):
    knot = [0, Tsim]
    g = 9.81
    tt = numpy.arange(min(knot), max(knot), dt)
    # with open('khanh.mat', 'rb') as inp:
    #     sim_data = pickle.load(inp)
    sim_data = loadmat('./Trajectory/data7.mat')
    W = sim_data['valeurs']
    

    k_pass = 1
    ref_tmp = np.empty((0, 3))
    waypoint_time_stamps = np.linspace(min(knot), max(knot), W.shape[1] + 1)
    for i_tmp in range(waypoint_time_stamps.shape[0] - 1):
        cur = np.array(W[:, i_tmp])
        while dt * k_pass <= waypoint_time_stamps[i_tmp + 1]:
            ref_tmp = np.vstack((ref_tmp, cur))
            k_pass = k_pass + 1

    ref_full = np.block([
        [ref_tmp, ref_tmp * 0]
    ])
    v_ref = 0 * ref_tmp.transpose()
    v_ref[2, :] = v_ref[2, :] - 0.075
    v_ref[0, :] = v_ref[0, :] - 0.125
    ddx, ddy, ddz = v_ref[0, :], v_ref[1, :], v_ref[2, :]
    thrust = np.sqrt(ddx ** 2 + ddy ** 2 + (ddz + 9.81) ** 2)
    phi = np.arcsin((ddx * sin(psi) - ddy * cos(psi)) / thrust)
    theta = np.arctan((ddx * cos(psi) + ddy * sin(psi)) / (ddz + g))
    ref = {
        "trajectory": ref_full,
        "time_step": tt,
        "thrust": thrust,
        "phi": phi,
        "theta": theta,
        "Nsim": tt.shape[0],
        "v_ref": v_ref.transpose()}

    return ref

def get_trajectory_mat(path_to_file,dt,psi=0):
    g = 9.81
    sim_data = loadmat(path_to_file)
    x = sim_data["x"] - 0.4
    y = sim_data["y"] + 0.4
    z = sim_data["z"] + 0.30

    vx = sim_data["vx"]
    vy = sim_data["vy"]
    vz = sim_data["vz"]

    ax = sim_data["ax"]
    ay = sim_data["ay"]
    az = sim_data["az"]

    nbr_agents = np.size(x,0)
    Tsim = dt*(np.size(x,1))

    ref_full = {}
    for i in range(nbr_agents):
        ref_full[i] = np.vstack((x[i,:],y[i,:],z[i,:],vx[i,:],vy[i,:],vz[i,:])).transpose()
    
    v_ref = {}
    for i in range(nbr_agents):
        v_ref[i] = np.vstack((ax[i,:],ay[i,:],az[i,:])).transpose()
        
    tt = np.arange(0, Tsim, dt)

    thrust = {}
    phi = {}
    theta = {}
    for i in range(nbr_agents):
        thrust[i] =  np.sqrt(ax[i,:] ** 2 + ay[i,:] ** 2 + (az[i,:] + 9.81) ** 2)
        phi[i] = np.arcsin((ax[i,:] * np.sin(psi) - ay[i,:] * np.cos(psi)) / thrust[i])
        theta[i] = np.arctan((ax[i,:] * np.cos(psi) + ay[i,:] * np.sin(psi)) / (az[i,:] + g))

    ref = {}
    for i in range(nbr_agents):
        ref[i] = {
        "trajectory": ref_full[i],
        "time_step": tt,
        "thrust": thrust[i],
        "phi": phi[i],
        "theta": theta[i],
        "Nsim": tt.shape[0],
        "v_ref": v_ref[i]}

    return ref

def get_ref_setpoints_all(path_to_file,dt,Tsim,psi=0):
    g = 9.81
    sim_data = loadmat(path_to_file)
    x = sim_data["x"] - 0.5
    y = sim_data["y"] 
    z = sim_data["z"] 

    vx = sim_data["vx"]
    vy = sim_data["vy"]
    vz = sim_data["vz"]

    ax = sim_data["ax"]
    ay = sim_data["ay"]
    az = sim_data["az"]

    nbr_agents = np.size(x,0)
    # Tsim = dt*(np.size(x,1))
    knot = [0, Tsim]
    tt = np.arange(0, Tsim+dt, dt)
    W = {}
    ref_full = {}
    v_ref = {}
    thrust = {}
    phi = {}
    theta = {}
    for i in range(nbr_agents):
        W[i] = np.zeros((6,x[i,:].shape[0]))
        # print(W)
        W[i][0,:] = x[i,:]
        W[i][1,:] = y[i,:]
        W[i][2,:] = z[i,:]
        W[i][3,:] = vx[i,:]
        W[i][4,:] = vy[i,:]
        W[i][5,:] = vz[i,:]


        k_pass = 1
        ref_tmp = np.empty((0, 6))
        waypoint_time_stamps = np.linspace(min(knot), max(knot), W[i].shape[1] + 1)
        for i_tmp in range(waypoint_time_stamps.shape[0] - 1):
            cur = np.array(W[i][:, i_tmp])
            while dt * k_pass <= waypoint_time_stamps[i_tmp + 1]:
                ref_tmp = np.vstack((ref_tmp, cur))
                k_pass = k_pass + 1

        ref_full[i] = ref_tmp
        k_pass = 1
        ref_tmp = np.empty((0, 3))
        accel = np.zeros((3,x[i,:].shape[0]))
        # print(W[0:3,-3:-1])
        accel[0,:] = ax[i,:].transpose()
        # accel[0,:] = np.append(accel[0,:-1],0)
        accel[1,:] = ay[i,:].transpose()
        # accel[1,:] = np.append(accel[1,:-1],0)
        accel[2,:] = az[i,:].transpose()
        # accel[2,:] = np.append(accel[2,:-1],0)
        # waypoint_time_stamps = np.linspace(min(knot), max(knot), W.shape[1] + 1)
        for i_tmp in range(waypoint_time_stamps.shape[0] - 1):
            cur = np.array(accel[:, i_tmp])
            while dt * k_pass <= waypoint_time_stamps[i_tmp + 1]:
                ref_tmp = np.vstack((ref_tmp, cur))
                k_pass = k_pass + 1
        v_ref[i] = ref_tmp.transpose()

        ddx, ddy, ddz = v_ref[i][0, :], v_ref[i][1, :], v_ref[i][2, :]
        thrust[i] = np.sqrt(ddx ** 2 + ddy ** 2 + (ddz + 9.81) ** 2)
        phi[i] = np.arcsin((ddx * np.sin(psi) - ddy * np.cos(psi)) / thrust[i])
        theta[i] = np.arctan((ddx * np.cos(psi) + ddy * np.sin(psi)) / (ddz + g))

    ref = {}
    for i in range(nbr_agents):
        ref[i] = {
        "trajectory": ref_full[i],
        "time_step": tt,
        "thrust": thrust[i],
        "phi": phi[i],
        "theta": theta[i],
        "Nsim": tt.shape[0],
        "v_ref": v_ref[i].transpose()}
    return ref

if __name__=="__main__":

    """Two lists of way-points for two drones"""
    # W1 = np.array([
    #     [0.5, 0, -0.5, 0, 0.5, 0, 0],
    #     [0, 0.5, 0, -0.5, 0, 0.5, 0.5],
    #     [0.3, 0.8, 0.8, 0.8, 0.8, 0.8, 0.3]  # 3D test
    # ])
    #
    # W2 = np.array([
    #     [0.7, 0, -0.7, 0, 0.7, 0.7, 0.7],
    #     [0, -0.7, 0, 0.7, 0, 0, 0],
    #     [0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3]  # 3D test
    # ])
    # rref=get_ref(W1, psi=0, Tsim=30, dt=0.1)
    # rref2=get_ref(W2, psi=0, Tsim=30, dt=0.1)
    # #
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(1, 1, 1, projection="3d")
    # ax1.plot(rref["trajectory"][:, 0], rref["trajectory"][:, 1], rref["trajectory"][:, 2])
    # ax1.scatter(W1[0, :], W1[1, :], W1[2, :])
    # ax1.plot(rref2["trajectory"][:, 0], rref2["trajectory"][:, 1], rref2["trajectory"][:, 2])
    # ax1.scatter(W2[0, :], W2[1, :], W2[2, :])
    # #
    # # fig1 = plt.figure()
    # # ax1 = fig1.add_subplot(1, 1, 1)
    # # ax1.plot(rref["time_step"], rref["trajectory"][:, 0])
    # # ax1.plot(rref2["time_step"], rref2["trajectory"][:, 0])
    #
    # plt.show()





    # Tsim = 30
    # Ts = 0.2
    # rref = get_ref_setpoints_Khanh(psi=0,Tsim=Tsim,dt=Ts,agent=1)
    # rref2 = get_ref_setpoints_Khanh(psi=0,Tsim=Tsim,dt=Ts,agent=2)

    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(1, 1, 1, projection="3d")
    # ax1.plot(rref["trajectory"][:, 0], rref["trajectory"][:, 1], rref["trajectory"][:, 2])
    # # ax1.scatter(W1[0, :], W1[1, :], W1[2, :])
    # ax1.plot(rref2["trajectory"][:, 0], rref2["trajectory"][:, 1], rref2["trajectory"][:, 2])
    # # ax1.scatter(W2[0, :], W2[1, :], W2[2, :])
    # # #
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(1, 1, 1)
    # ax1.plot(rref["time_step"], rref["trajectory"][:, 0])
    # ax1.plot(rref2["time_step"], rref2["trajectory"][:, 0])
    # ax1.plot(rref["time_step"], rref["trajectory"][:, 3])
    # ax1.plot(rref2["time_step"], rref2["trajectory"][:, 3])
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(1, 1, 1)
    # ax1.plot(rref["time_step"], rref["v_ref"][:, 0])
    # ax1.plot(rref2["time_step"], rref2["v_ref"][:, 0])
    # plt.show()


    # W = np.array([[0.0, 0.0],
    #             [0.0, 0.0],
    #             [0.7, 0.7]  # 3D test
    #             ])
    Tsim = 20
    Ts = 0.1
    rref = get_ref_setpoints_Khanh(psi=0,Tsim=Tsim,dt=Ts,agent=1)
    # rref = get_ref_setpoints(psi=0,Tsim=Tsim,dt=Ts,version=12)
    full_ref_takeoff = get_ref_setpoints_takeoff(psi=0,Tto=10,dt=0.2,ref=rref['trajectory'])
    ref_takeoff = {'Test': full_ref_takeoff["trajectory"]}
    vref_takeoff = {'Test': full_ref_takeoff["v_ref"]}

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(full_ref_takeoff['time_step'],ref_takeoff['Test'][:,0])
    ax.plot(full_ref_takeoff['time_step'],ref_takeoff['Test'][:,1])
    ax.plot(full_ref_takeoff['time_step'],ref_takeoff['Test'][:,2])

    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.plot(full_ref_takeoff['time_step'],vref_takeoff['Test'][:,0])
    ax1.plot(full_ref_takeoff['time_step'],vref_takeoff['Test'][:,1])
    ax1.plot(full_ref_takeoff['time_step'],vref_takeoff['Test'][:,2])
    plt.show()
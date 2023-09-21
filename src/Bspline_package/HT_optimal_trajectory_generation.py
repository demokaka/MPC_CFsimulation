"""
Created on Spetember 2022
@author: Huu-Thinh DO
"""
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from Bspline_package import Bspline_conversionMatrix as BsplineM
from scipy.interpolate import BSpline
from scipy.integrate import quad
from time import time

k = 8 # d=k+1, k: polynomial degree of the spline
n_ctrl_pts = 28  # number of control points
time_steps = 300
dt = 0.1
knot = [0, 25]
g = 9.81
psi = 30 * np.pi / 180
knot = BsplineM.knot_vector(k, n_ctrl_pts, knot)
tt = numpy.arange(min(knot), max(knot), dt)
bs_list = BsplineM.b_spline_basis_functions(n_ctrl_pts, k, knot)
M = BsplineM.bsplineConversionMatrices(n_ctrl_pts, k, knot)

# Waypoints
# W = np.array([[0, 0.6, 0, -1.2, -1.2, -1.2, -0.8, 0],
#               [0, 0.6, 1.2, 1.2, 0.5, 0, -0.5, 0],
#               [0.1, 0.4, 0.8, 1.5, 1.7, 0.8, 0.5, 0.5]  # 3D test
#               ])
# W = np.array([[0.4,0.9,0.4,0,-0.6,-0.6,-0.6,-0.6],
#               [0, 0, 0.6, 0.9, 0.9, 0.4, -0.3, -0.6],
#               [0.25, 0.7, 1.2, 1.3, 1.5, 1.2, 0.7, 0.3]  # 3D test
#               ])
W = np.array([[0, 0.3, 0.6, 0.6, 0.3, 0, -0.3, -0.3, 0],
              [0, -0.3, 0, 0.3, 0.6, 0.6, 0.3, 0, 0],
              [0.35, 0.4, 0.75, 1.0, 1.2, 1.1, 0.8, 0.5, 0.35]  # 3D test
              ])
waypoint_time_stamps = np.linspace(min(knot), max(knot), W.shape[1])
ctrl_pts_timestamps = np.linspace(min(knot), max(knot), n_ctrl_pts)
# ------ Optimization problem construction --------------------
solver = casadi.Opti()
# Control point as optimization variable
P = solver.variable(W.shape[0], n_ctrl_pts)
# Objective function
objective = 0

P1 = mtimes(P, M[0])  # Conversion matrix M
for i in range(n_ctrl_pts + 1):
    for j in range(n_ctrl_pts + 1):
        f_lamb = lambda t, it=i, jt=j: bs_list[1][it](t) * bs_list[1][jt](t)
        buff_int = quad(f_lamb, min(knot), max(knot))[0]
        objective = objective + mtimes(transpose(mtimes(buff_int, P1[:, i])), P1[:, j])

# Implementing constraints
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
tic = time()
sol = solver.solve()  # Solve for the control points
toc = time()
Elapsed_time = toc - tic
print('Elapsed time for solving: ', Elapsed_time, '[second]')
# ============================================================================================
# Construct the result curve
P = sol.value(P)
print('=======================================================================================')
print('Optimal control-points found')
# Compute the Bspline with the solution of P
spn = []
if W.shape[0] == 1:
    spn.append(BSpline(knot, P, k))
else:
    for i in range(P.shape[0]):
        spn.append(BSpline(knot, P[i], k))

fig = plt.figure()
if W.shape[0] == 2:
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(spn[0](tt), spn[1](tt), lw=2)
    ax1.plot(P[0], P[1], lw=1)
    ax1.scatter(W[0, :], W[1, :], label='waypoints', color='red', lw=5)
    ax1.scatter(P[0], P[1], label='Control Points')
    ax1.grid(True)
    ax1.legend()
else:
    if W.shape[0] == 3:
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        ax1.plot(spn[0](tt), spn[1](tt), spn[2](tt), lw=2)
        ax1.plot(P[0], P[1], P[2], lw=1)
        ax1.scatter(W[0, :], W[1, :], W[2, :], label='waypoints', color='red', lw=5)
        ax1.scatter(P[0], P[1], P[2], label='Control Points')
        ax1.grid(True)
        ax1.legend()
    else:
        if W.shape[0] == 1:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.plot(tt, spn[0](tt), lw=2)
            ax1.plot(ctrl_pts_timestamps, P, lw=1)
            ax1.scatter(ctrl_pts_timestamps, P, lw=1, label='Control Points')
            ax1.scatter(waypoint_time_stamps, W, label='waypoints', color='red', lw=5)
            ax1.grid(True)
            ax1.legend()
        else:
            print('Curves with dimension higher than 3 cannot be plotted')

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

# fig2 = plt.figure()
# ax2 = fig2.add_subplot(1, 1, 1)
# ax2.plot(tt, z[0](tt), lw=2, label='x(t)')
# ax2.plot(tt, z_d[0](tt), lw=2, label='dx/dt')
# ax2.plot(tt, spn_dd[0](tt), lw=2, label='ddx/dt')
# ax2.grid(True)
# ax2.legend()
# plt.show()
dx = z_d[0](tt)
dy = z_d[1](tt)
dz = z_d[2](tt)
ddx = z_dd[0](tt)
ddy = z_dd[1](tt)
ddz = z_dd[2](tt)

fig2 = plt.figure()
ax2 = fig2.add_subplot(3, 1, 1)
ax2.plot(tt, ddx, lw=2, label='dx')
ax2.grid(True)
ax2.legend()

Thrust = np.sqrt(ddx ** 2 + ddy ** 2 + (ddz + 9.81) ** 2)
# ax2.plot(tt, Thrust, lw=2, label='Thrust')
# ax2.grid(True)
# ax2.legend()

phi = np.arcsin((ddx * sin(psi) - ddy * cos(psi)) / Thrust)
# ax2 = plt.subplot(3, 1, 2)
# ax2.plot(tt, phi * 180 / np.pi, lw=2, label='phi')
# ax2.grid(True)
# ax2.legend()

theta = np.arctan((ddx * cos(psi) + ddy * sin(psi)) / (ddz + g))
# ax2 = plt.subplot(3, 1, 3)
# ax2.plot(tt, theta * 180 / np.pi, lw=2, label='theta')
# ax2.grid(True)
# ax2.legend()

print('Max Thrust: {txta}g (m/s^2), Min Thrust: {txtb}g (m/s^2)'.format(txta=round(max(Thrust) / g, 2),
                                                                        txtb=round(min(Thrust) / g, 2)))
print('Max Roll  : {txta}  (deg),   Min Roll  : {txtb}  (deg)'.format(txta=round(max(phi) * 180 / np.pi, 2),
                                                                      txtb=round(min(phi) * 180 / np.pi, 2)))
print('Max Pitch : {txta}  (deg),   Min Pitch : {txtb}  (deg)'.format(txta=round(max(theta) * 180 / np.pi, 2),
                                                                      txtb=round(min(theta) * 180 / np.pi, 2)))
plt.show()

import numpy as np

def CFmodel(xk,urk,plant,disturb=0):
    T = urk[0]
    phi = urk[1]
    theta = urk[2]
    psi = 0
    g = 9.81
    uk = np.zeros((3,1))

    # Thrust coefficient depending on the masses of the drone and the carried batterie
    m = 33.0
    mreal = 33.0
    T = T * m / mreal
    uk[0,0] = T * (np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi)) + disturb[0,0] + .0
    uk[1,0] = T * (np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi)) + disturb[0,1] + .0
    uk[2,0] = -g + T * (np.cos(phi)*np.cos(theta)) + disturb[0,2]
    xrk = xk.reshape((-1,1))
    xk1 = plant['Ad'] @ xrk + plant['Bd'] @ uk
    return xk1.ravel()
    
def sat(x,xmin,xmax):
    if(x<xmin):
        y = xmin
    else:
        if x > xmax:
            y = xmax
        else:
            y = x
    return y
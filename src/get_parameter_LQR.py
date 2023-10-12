import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from scipy.linalg import block_diag

def get_LQR_gain(Ts,qp,qv,r):
    ## Model
    g = 9.81
    plant = {}
    plant['A'] = np.vstack( (np.hstack( (np.zeros((3,3)),np.eye(3)) ), np.hstack( (np.zeros((3,3)), np.zeros((3,3))) )) )
    plant['B'] =  np.hstack( (np.zeros((3,3)), np.eye(3)) )
    plant['C'] =  np.hstack( (np.eye(3),np.zeros((3,3))))  
    plant['D'] = np.zeros(3)
    plant['Ad'] = np.vstack( (np.hstack( (np.eye(3),Ts*np.eye(3)) ), np.hstack( (np.zeros((3,3)), np.eye(3)) )) )
    plant['Bd'] = np.vstack( (np.eye(3)*0.5*Ts**2, Ts*np.eye(3)) )
    plant['Cd'] = np.hstack( (np.eye(3),np.zeros((3,3)))) 
    plant['Dd'] = np.zeros((3,3))
    plant['Ts'] = Ts                                              # sampling time
    # dimension
    plant['dx'] = np.size(plant['Bd'],0)                        
    plant['du'] = np.size(plant['Bd'],1)
    plant['dy'] = np.size(plant['Cd'],0)

    controller = {}
    controller['Q'] = block_diag(np.eye(3)*qp,np.eye(3)*qv)

    controller['R'] = np.eye(plant['du'])*r 



    ## LQR
    [Klqr,S,e] = ctrl.dlqr(plant['Ad'],plant['Bd'],controller['Q'],controller['R'])

    return Klqr


Ts = 0.1

Klqr = get_LQR_gain(Ts,500,250,10)
print(Klqr)
# Tsim = 20

# tt = np.arange(0,Tsim+Ts,Ts)

# ## Model
# g = 9.81
# plant = {}
# plant['A'] = np.vstack( (np.hstack( (np.zeros((3,3)),np.eye(3)) ), np.hstack( (np.zeros((3,3)), np.zeros((3,3))) )) )
# plant['B'] =  np.hstack( (np.zeros((3,3)), np.eye(3)) )
# plant['C'] =  np.hstack( (np.eye(3),np.zeros((3,3))))  
# plant['D'] = np.zeros(3)
# plant['Ad'] = np.vstack( (np.hstack( (np.eye(3),Ts*np.eye(3)) ), np.hstack( (np.zeros((3,3)), np.eye(3)) )) )
# plant['Bd'] = np.vstack( (np.eye(3)*0.5*Ts**2, Ts*np.eye(3)) )
# plant['Cd'] = np.hstack( (np.eye(3),np.zeros((3,3)))) 
# plant['Dd'] = np.zeros((3,3))
# plant['Ts'] = Ts                                              # sampling time
# # dimension
# plant['dx'] = np.size(plant['Bd'],0)                        
# plant['du'] = np.size(plant['Bd'],1)
# plant['dy'] = np.size(plant['Cd'],0)

# controller = {}
# # Q = blkdiag(eye(dx/2)*1000,eye(dx/2)*500);
# controller['Q'] = block_diag(np.eye(3)*1000,np.eye(3)*500)
# # R = eye(du)*10;
# controller['R'] = np.eye(plant['du'])*10 



# ## LQR
# [Klqr,S,e] = ctrl.dlqr(plant['Ad'],plant['Bd'],controller['Q'],controller['R'])

# print(Klqr)
# ## LQI
# # Qa = blkdiag(Q/2,eye(3)*500/2);
# # model.Ada = [model.Ad,zeros(6,3);-h*[eye(3),zeros(3)],eye(3)];
# # model.Bda = [model.Bd;zeros(3,3)];
# # [Klqi,S,e] = dlqr(model.Ada,model.Bda,Qa,R);

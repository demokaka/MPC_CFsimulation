import numpy as np
import casadi as ca
from scipy.linalg import block_diag

class droneParameters:
    def __init__(self,drone_address,body_name,mass,plant,controller):
        self.uri = drone_address
        self.name = body_name
        self.mass = mass
        self.plant = plant
        self.controller = controller

def get_solver_cmpc(plant,controller,simulator):
    solver = ca.Opti()

    # Virtual inputs and states over prediction horizon
    X = solver.variable(plant['dx'],controller['Npred']+1)  # [Xi_1',Xi_2',...Xi_na']' * (Npred+1)  States
    U = solver.variable(plant['du'],controller['Npred'])    # [v_1',v_2',...,v_na']'   * (Npred)    Virtual inputs stack

    # Initial inputs, states and reference states
    
    X_init = solver.parameter(plant['dx'],1)
    U_init = solver.parameter(plant['du'],1)
    X_ref = solver.parameter(plant['dx'],controller['Npred']+1)
    v_ref = solver.parameter(plant['du'],controller['Npred'])
    yaw = solver.parameter(1,1)

    solver_variables = {'X':X, 'U':U, 'X_init':X_init, 'U_init':U_init, 'X_ref':X_ref, 'v_ref':v_ref, 'yaw':yaw}

    objective = 0

    ## Add constraints to the solver object
    solver.subject_to(X[:,0] == X_init)

    for k in range(controller['Npred']):
        # Inputs bounds
        solver.subject_to(X[:,k+1] == plant['A'] @ X[:,k] + plant['B'] @ U[:,k] )
        solver.subject_to(plant['umin'] <= U[:,k])
        solver.subject_to(U[:,k]  <= plant['umax'] )
        # States bounds
        solver.subject_to(plant['xmin'] <= X[:,k+1])
        solver.subject_to(X[:,k+1]  <= plant['xmax'] )
        # Control input rate bounds
        # solver.subject_to(np.tile(plant['deltaUmin'], (simulator['na'],1)) <= U[:, k] - U_init)
        # solver.subject_to(U[:, k] - U_init <= np.tile(plant['deltaUmax'], (simulator['na'],1)))

        
        objective = objective + \
                        ca.transpose(X[:,k] - X_ref[:,k]) @ controller['Q'] @ (X[:,k] - X_ref[:,k]) + \
                        ca.transpose(U[:,k] - v_ref[:,k]) @ controller['R'] @ (U[:,k] - v_ref[:,k])
        
    objective = objective + \
                ca.transpose(X[:,controller['Npred']] - X_ref[:,controller['Npred']]) @ controller['P'] @ (X[:,controller['Npred']] - X_ref[:,controller['Npred']])


    solver.minimize(objective)
    solver_opts = {'ipopt':{'print_level':0,'sb':'yes'},
                   'print_time':0,
                   'error_on_fail':False}
    solver.solver('ipopt', solver_opts)
    return solver,solver_variables

    
def load_constant_parameters(Ts,Q=None,P=None,R=None,phy_lim=None,Npred=None):
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
    if not phy_lim: # by default
        plant['amin'] = np.array([[-1],[-1],[-1]])*1.5              # m/s²
        plant['amax'] = np.array([[1],[1],[1]])*1.5                 
        plant['vmin'] = np.array([[-1],[-1],[-1]])*1.5              # m/s
        plant['vmax'] = np.array([[1],[1],[1]])*1.5
        plant['pmin'] = np.array([[-1.8],[-1.8],[0]])*1.0           # m
        plant['pmax'] = np.array([[1.8],[1.8],[1.8]])*1.0
    else:
        plant['amin'] = phy_lim['amin']
        plant['amax'] = phy_lim['amax']               
        plant['vmin'] = phy_lim['vmin']           
        plant['vmax'] = phy_lim['vmax']
        plant['pmin'] = phy_lim['pmin']
        plant['pmax'] = phy_lim['pmax']
    controller = {}
    if not (Q): # by default
        controller['Q'] = block_diag(np.eye(3)*50,np.eye(3)*5)  # Q = blkdiag(Qp,Qv)
                                                                    # Qp = qp*I(dp),Qv = qv*I(dv)
    else:
        controller['Q'] = Q
    if not (P): # by default
        controller['P'] = block_diag(np.eye(3)*50,np.eye(3)*5)  # P = blkdiag(Pp,Pv)
                                                                    # Pp = pp*I(dp),Pv = pv*I(dv)
    else:
        controller['P'] = P
    if not (R): # by default
        controller['R'] = np.eye(plant['du'])*10                 # R = r*I(du)
    else:
        controller['R'] = R
    
    if not (Npred): # by default
        controller['Npred'] = 5
    else:
        controller['Npred'] = Npred
    return plant,controller

def get_stacked_drones_parameters(list_of_drones):
    plant = {}
    controller = {}
    simulator = {}
    simulator['na'] = len(list_of_drones)
    plant['Ts'] = list_of_drones[0].plant['Ts']                                              # sampling time
    # dimension
    plant['dx'] = list_of_drones[0].plant['dx'] * simulator['na']                   
    plant['du'] = list_of_drones[0].plant['du'] * simulator['na']
    plant['dy'] = list_of_drones[0].plant['dy'] * simulator['na']
    plant['A'] = np.kron(np.eye(simulator['na']),list_of_drones[0].plant['Ad'])
    plant['B'] = np.kron(np.eye(simulator['na']),list_of_drones[0].plant['Bd'])
    plant['xmin'] = np.zeros((list_of_drones[0].plant['dx']*simulator['na'],1))
    plant['xmax'] = np.zeros((list_of_drones[0].plant['dx']*simulator['na'],1))
    plant['umin'] = np.zeros((list_of_drones[0].plant['du']*simulator['na'],1))
    plant['umax'] = np.zeros((list_of_drones[0].plant['du']*simulator['na'],1))
    controller['Q'] = np.zeros((list_of_drones[0].plant['dx']*simulator['na'],list_of_drones[0].plant['dx']*simulator['na']))
    controller['P'] = np.zeros((list_of_drones[0].plant['dx']*simulator['na'],list_of_drones[0].plant['dx']*simulator['na']))
    controller['R'] = np.zeros((list_of_drones[0].plant['du']*simulator['na'],list_of_drones[0].plant['du']*simulator['na']))

    controller['Npred'] = list_of_drones[0].controller['Npred']
    simulator['name_Drone'] = []
    for i in range(simulator['na']):
        plant['xmin'][i*6:(i+1)*6] = np.vstack( (list_of_drones[i].plant['pmin']
                                                ,list_of_drones[i].plant['vmin']) )
                                    
        plant['xmax'][i*6:(i+1)*6]  = np.vstack( (list_of_drones[i].plant['pmax']
                                                ,list_of_drones[i].plant['vmax']) )
                                    
        plant['umin'][i*3:(i+1)*3]  = list_of_drones[i].plant['amin']
                                    
        plant['umax'][i*3:(i+1)*3]  = list_of_drones[i].plant['amax']
                                    
        controller['Q'][i*6:(i+1)*6,i*6:(i+1)*6]  = list_of_drones[i].controller['Q']
                                    
        controller['P'][i*6:(i+1)*6,i*6:(i+1)*6] = list_of_drones[i].controller['P']
                            
        controller['R'][i*3:(i+1)*3,i*3:(i+1)*3] = list_of_drones[i].controller['R']
                                    
        simulator['name_Drone'].append(list_of_drones[i].name
                                    )
    return plant,controller,simulator

def compute_control_cmpc(solver,solver_variables,X0,v0,Xrefk,vrefk,yawk):
    X = solver_variables['X']
    X_init = solver_variables['X_init']
    U = solver_variables['U']
    U_init = solver_variables['U_init']
    X_ref = solver_variables['X_ref']
    v_ref = solver_variables['v_ref']
    yaw = solver_variables['yaw']

    solver.set_value(X_init, X0)
    solver.set_value(X_ref,Xrefk)
    solver.set_value(U_init, v0)
    solver.set_value(v_ref,vrefk)
    solver.set_value(yaw,yawk)

    solver.set_initial(U, vrefk)
    solver.set_initial(X,Xrefk)

    sol = solver.solve()

    return sol.value(U[:,0])


def get_ref_pred_horz(ref,k,Npred,uris):
    d = np.size(ref[uris[0]],1)
    dim = d*len(uris)
    Nsim = np.size(ref[uris[0]],0)
    for i in range(len(uris)):
        pass
    refk = np.zeros((dim,Npred))

    if k <= (Nsim-Npred):
        for i in range(len(uris)):
            refk[i*d:(i+1)*d,:] = ref[uris[i]][k:(k+Npred),:].T
    else:
        for i in range(len(uris)):
            refk[i*d:(i+1)*d,0:(Nsim-k)] = ref[uris[i]][k:,:].T
            refk[i*d:(i+1)*d,(Nsim-k):] = np.tile( ref[uris[i]][(Nsim-1):,:].T ,(1, Npred-Nsim+k))    
    return refk


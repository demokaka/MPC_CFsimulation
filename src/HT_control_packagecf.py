import numpy as np

def Thrust_to_PWM_v1(Thrust):
    a1 = 2.130295e-11
    a2 = 1.032633e-6
    a3 = 5.484560e-4
    m = 33.0 # quadcopter mass : from 28 to 33 gram
    g = 9.81 # gravitational acceleration
    kc = m*g/4000
    m_ratio = 1.9
    PWM_theo = (-a2 + np.sqrt(a2**2 + 4*a1*(Thrust * kc - a3)))/(2*a1*m_ratio)
    # Mapping 0-65535 to 10000-60000 
    alpha = 0.7630
    beta = 10000
    pwm_signal = alpha * PWM_theo + beta
    return pwm_signal

def Thrust_to_PWM_modified(Thrust,m=33.0):
    a1 = 2.130295e-11
    a2 = 1.032633e-6
    a3 = 5.484560e-4
    # a2 = 1.00133e-6
    # a3 = -9.51544e-4

    g = 9.81 # gravitational acceleration
    kc = m*g/4000
    PWM_signal = (-a2 + np.sqrt(a2**2 + 4*a1*(Thrust * kc - a3)))/(2*a1)
    return PWM_signal

def Thrust_to_PWM(Thrust,alpha):
    pwm_signal = 65535 * (-140.5e-3*alpha + np.sqrt(140.5e-3 ** 2 - 4 * 0.409e-3 * (-0.099 - Thrust))) / (2 * 0.409e-3 * 256)
    return pwm_signal

def compute_control(v_ref, x0, xref, Kf):
    v = v_ref + np.matmul(Kf, x0 - xref)
    return v

def get_real_input(v_controls,yaw):
    g = 9.81
    T = np.round(np.sqrt(v_controls[0] ** 2 + v_controls[1] ** 2 + (v_controls[2] + g) ** 2), 5)
    # print("Vcontrols 0 = ",v_controls[0])
    # print("Vcontrols 1 = ",v_controls[1])
    # print("Vcontrols 2 = ",v_controls[2])
    # print(T)
    phi = np.round(np.arcsin((v_controls[0] * np.sin(yaw) - v_controls[1] * np.cos(yaw)) / T), 5)
    theta = np.round(np.arctan((v_controls[0] * np.cos(yaw) + v_controls[1] * np.sin(yaw)) / (
                v_controls[2] + g)), 5)
    controls = [T, phi, theta]
    return controls

def get_cf_input(v_controls, yaw, T_coeff=23.5, desired_yaw = 0, alpha=1.000, mass = 33.0, bias=[0,0]):
    g = 9.81
    T = np.round(np.sqrt(v_controls[0] ** 2 + v_controls[1] ** 2 + (v_controls[2] + g) ** 2), 5)
    phi = np.round(np.arcsin((v_controls[0] * np.sin(yaw) - v_controls[1] * np.cos(yaw)) / T), 5)
    theta = np.round(np.arctan((v_controls[0] * np.cos(yaw) + v_controls[1] * np.sin(yaw)) / (
                v_controls[2] + g)), 5)
    controls = [T, phi, theta]
    # Thrust_pwm = int(T_coeff*Thrust_to_PWM(controls[0] / g,alpha))
    Thrust_pwm = int(T_coeff*Thrust_to_PWM_modified(controls[0]/g,mass)*alpha)
    print('Thrust_pwm = ',Thrust_pwm)
    Roll = (controls[1] * 180) / np.pi  
    Pitch = (controls[2] * 180) / np.pi
    Yawrate = 0.000*(yaw - desired_yaw) * 180 / np.pi
    controls_cf = [Roll+bias[0], Pitch+bias[1], Yawrate, Thrust_pwm]
    return controls_cf

def saturation(x,min,max):
    if x < min:
        return min
    elif x>max:
        return max
    else:
        return x
if __name__=='__main__':

    Thrust_calcul = 2
    PWM_Thinh = Thrust_to_PWM(23.8*Thrust_calcul,1)
    PWM_Modified = Thrust_to_PWM_modified(Thrust_calcul)
    print(PWM_Thinh)
    print(PWM_Modified)

    Kf  =  -1.0 *  np.array([[2.5, 0, 0, 1.5, 0, 0],
                    [0, 2.5, 0, 0, 1.5, 0],
                    [0, 0, 2.5, 0, 0, 1.5]]) 

    x = np.array([[3],[3],[0.522],[3],[3],[3]])
    xref = np.array([[3],[3],[0.2],[3],[3],[3]])

    x1 = np.array([[3],[3],[0.88],[3],[3],[3]])
    xref1 = np.array([[3],[3],[0.5],[3],[3],[3]])

    print('x-xref = ',x-xref)
    v = np.matmul(Kf,x-xref)
    v1 = np.matmul(1.5*Kf,x1-xref1)
    print('v = ',v)
    g = 9.81
    T = np.sqrt(v[0]**2 + v[1]**2 + (v[2] + g)**2)
    T1 = np.sqrt(v1[0]**2 + v1[1]**2 + (v1[2] + g)**2)
    Tunit = T/g
    T1unit = T1/g
    print('T = ',T)
    print('Tunit = ',Tunit)
    print('T1 = ',T1)
    print('T1unit = ',T1unit)

    PWM_Thinh = Thrust_to_PWM(23.8*Tunit,1)
    PWM_Thinh1 = Thrust_to_PWM(23.8*T1unit,1)
    PWM_Modified = Thrust_to_PWM_modified(1)
    print(PWM_Thinh)
    print(PWM_Thinh1)
    print(PWM_Modified)
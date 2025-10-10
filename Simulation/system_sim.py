import numpy as np
import scipy
from scipy.integrate import solve_ivp
import numpy 
import matplotlib.pyplot as plt 
from  matplotlib import animation
from math import sin, cos, pi

from render import SimplePendulumCartRenderer, DoublePendulumCartRender
import sys


G = 9.81

#mass of cart/pendulum, length of pendulum
m1 = 5
m2 = 10
m3 = 10
l1= 0.5
l2 = 0.5

#friction coefficients
d1 = 0.1
d2 = 0.1
d3 = 0.1

sim_length = 30 #seconds 
frame_rate = 25 




def system_dynamics(t, Y):
    x, dx, t1, dt1, t2, dt2 = Y
    sin1, cos1 = sin(t1), cos(t1)
    sin2, cos2 = sin(t2), cos(t2)
    w1, w2, w3 = 0,0,0
    u=0

    M =  np.array([[m1+m2+m3, l1*(m2+m3)*cos1, m3*l2*cos2], 
                   [l1*(m2+m3)*cos1, l1**2 * (m2+m3), l1*l2*m3*cos(t1-t2)],
                   [l2*m3*cos2,l1*l2*m3*cos(t1-t2), l2**2 * m3]])
    
    system = np.array([[l1*(m2+m3)*(dt1**2)*sin1 + m3*l2*(dt2**2)*sin2],
                 [-l1*l2*m3*(dt2**2)*sin(t1-t2)+G*(m2+m3)*l1*sin1],
                 [l1*l2*m3*(dt1**2)*sin(t1-t2)+G*l2*m3*sin2]])

    friction = -1* np.array([[d1 * dx],[d2 * dt1],[d3 * dt2]])
    control = np.array([[u],[0],[0]])
    disturbances = np.array([[w1], [w2], [w3]])

    f = system + friction + control + disturbances
    y = np.linalg.solve(M, f) 
    y = y.T

    return [dx, y[0][0], dt1, y[0][1], dt2, y[0][2]]

Y0 = [0,0,-0.3,0.6,0,0]
t_span = [0, sim_length]
t_eval = np.linspace(t_span[0], t_span[1], sim_length*100)
sol = solve_ivp(system_dynamics, t_span, Y0, t_eval=t_eval)
   
if __name__ == "__main__":



    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(sol.t, sol.y[0], label='x')
    plt.plot(sol.t, sol.y[2], label='t1')
    plt.plot(sol.t, sol.y[4], label='t2')
    plt.title('cart and angle')
    plt.xlabel('time')
    plt.ylabel('cart pos')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(sol.t, sol.y[1], label='dx')
    plt.plot(sol.t, sol.y[3], label='dt1')
    plt.plot(sol.t, sol.y[5], label='dt2')
    plt.title('cart and angle derivatives')
    plt.xlabel('time')
    plt.ylabel('d')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    rend = DoublePendulumCartRender(sim_length,frame_rate,[m1,m2,m3], [l1,l2])
    rend.draw_cart_double_pendulum(sol)
        



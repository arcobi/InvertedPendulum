import numpy as np
import scipy
from scipy.integrate import solve_ivp
import numpy 
import matplotlib.pyplot as plt 
from  matplotlib import animation
from math import sin, cos, pi

G = 9.81

#mass of cart/pendulum, length of pendulum
m1 = 10
m2 = 3
m3 = 3
l1= 1
l2 = 1

#friction coefficients
d1 = 0.1
d2 = 0.1
d3 = 0.1


def system_dynamics(t, Y):
    x, dx, t1, dt1, t2, dt2 = Y
    sin1, cos1 = sin(t1), cos(t1)
    sin2, cos2 = sin(t2), cos(t2)
    u = 0
    w1, w2, w3 = 0,0,0


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
    y = np.linalg.inv(M) @ f
    y = y.T

    return [dx, y[0][0], dt1, y[0][1], dt2, y[0][2]]

Y0 = [0,0,3*pi/4,0,0,0]
t_span = [0, 80]
t_eval = np.linspace(t_span[0], t_span[1], 50000)
sol = solve_ivp(system_dynamics, t_span, Y0, t_eval=t_eval)


def draw_system_state():
    fig, ax = plt.subplots()
    pendulum1, = ax.plot([], [], 'b-', lw=3)
    pendulum2, = ax.plot([], [], 'b-', lw=3)
    ball, = ax.plot([], [], 'ro', markersize=12)

    ax.set_title("Single Pendulum")
    frame_size = l1*3
    ax.set_xlim(-1*frame_size, frame_size)
    ax.set_ylim(-1*frame_size, frame_size)
    ax.set_aspect('equal')

    def init():
        pendulum1.set_data([], [])
        pendulum2.set_data([], [])
        ball.set_data([], [])
        return pendulum1, pendulum2,  ball

    def animate(i):
        index = i*50000 // (2500 * 100)

        x, theta1, theta2 = sol.y[0][i*10], sol.y[2][i*10], sol.y[4][i*10]

        x_points1 = [x, l1 * sin(theta1)+x]
        y_points1 = [0, l1 * cos(theta1)]
        x_points2 = [x_points1[1], x_points1[1] + l2 * sin(theta2)]
        y_points2 = [y_points1[1], y_points1[1] + l2 * cos(theta2)]

        pendulum1.set_data(x_points1, y_points1)
        pendulum2.set_data(x_points2, y_points2)
        ball.set_data([x_points1[-1]], [y_points1[-1]])

        return pendulum1,pendulum2, ball
        

    ani = animation.FuncAnimation(fig,animate, frames=2500, init_func=init, interval=10, blit=True)
    plt.show()


if __name__ == "__main__":

    print(sol.t.shape)
    print(sol.status)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(sol.t, sol.y[0], label='x')
    plt.plot(sol.t, sol.y[2], label='t1')
    plt.plot(sol.t, sol.y[4], label='t2')
    plt.title('cart pos')
    plt.xlabel('Time')
    plt.ylabel('cart pos')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(sol.t, sol.y[1], label='dx')
    plt.plot(sol.t, sol.y[3], label='dt1')
    plt.plot(sol.t, sol.y[5], label='dt2')
    plt.title('angle1')
    plt.xlabel('Time')
    plt.ylabel('angle1')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    draw_system_state()
        



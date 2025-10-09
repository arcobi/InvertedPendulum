import numpy as np
import scipy
import numpy 
import matplotlib.pyplot as plt 
from  matplotlib import animation
from math import sin, cos, pi

G = 9.81

m1 = 1 
m2 = 2
m3 = 3
l1= 1
l2 = 1


def system_dynamics(self, x, dx, t1, dt1, t2, dt2):
    sin1, cos1 = sin(t1), cos(t1)
    sin2, cos2 = sin(t2), cos(t2)
    #magic
    ddt1_denominator = ((m2+m3)*l1**2) - (cos(t1-t2))**2 * m3 * l1**2 + l1 * cos1 * (m2+m3) * ((m2+m3)* l1 * cos1 - l1 *cos2 *cos(t1-t2)) / (m1+m2+m3)
    ddt1_numerator = - m3 * l1 * G * sin2 + m3 * (l1**2) * sin(t1-t2) * (dt1**2) - l1*cos1*(m2+m3) / (m1+m2+m3) *( (l2*G*sin2+l1*l2*(dt1**2)*sin(t1-t2)) / (l2**2) + m2*l1*(dt1**2)*sin1 +m3*l1*(dt1**2)*sin1 + m3*l2*(dt2**2)*sin2)
    ddt1 = ddt1_numerator / ddt1_denominator

    ddx_denominator = m1+m2+m3
    ddx_numerator = m2*l1*(dt1**2)*sin1 + m3*(l1*(dt1**2)*sin1 + l2 * (dt2**2)*sin2) - (m2*l1*cos1*ddt1 + m3*l1*ddt1*cos1 + m3*cos2*(g*sin2 - l1*(dt1**2)*sin(t1-t2)-l1*ddt1*cos(t1-t2)))
    ddx = ddx_numerator / ddx_denominator

    ddt2_denominator = m3*(l2**2)
    ddt2_numerator = m3*l2*g*sin2 - m3*l1*l2*(dt1**2)*sin(t1-t2) - m3*l1*l2*ddt1*cos(t1-t2)
    ddt2 = ddt2_numerator / ddt2_denominator

    pass

def draw_system_state(self):
    fig, ax = plt.subplots()
    pendulum1, = ax.plot([], [], 'b-', lw=3)
    pendulum2, = ax.plot([], [], 'b-', lw=3)
    ball, = ax.plot([], [], 'ro', markersize=12)

    ax.set_title("Single Pendulum")
    frame_size = self.l*3
    ax.set_xlim(-1*frame_size, frame_size)
    ax.set_ylim(-1*frame_size, frame_size)
    ax.set_aspect('equal')

    def init():
        pendulum1.set_data([], [])
        pendulum2.set_data([], [])
        ball.set_data([], [])
        return pendulum1, pendulum2,  ball

    def animate(i):
        num_frame = 1000 

        x_points1 = [self.x, self.l * sin(self.theta1)+self.x]
        y_points1 = [0, self.l * cos(self.theta1)]
        x_points2 = [x_points1[1], x_points1[1] + self.l * sin(self.theta2)]
        y_points2 = [y_points1[1], y_points1[1] + self.l * cos(self.theta2)]

        pendulum1.set_data(x_points1, y_points1)
        pendulum2.set_data(x_points2, y_points2)
        ball.set_data([x_points1[-1]], [y_points1[-1]])

        s_x_dot, s_x_ddot, s_theta_dot, s_theta_ddot = self.system_dynamics(0.01)
        self.x -= s_x_dot 
        self.x_dot += s_x_ddot 
        self.theta += s_theta_dot
        self.theta_dot += s_theta_ddot

        return pendulum1, ball

    ani = animation.FuncAnimation(fig,animate, frames=10000, init_func=init, interval=10, blit=True)
    plt.show()


if __name__ == "__main__":
    instance = (1, 1, 2, 1, 1, 0, 0, 0.3, 0, 0.1, 0)
    instance.draw_system_state()


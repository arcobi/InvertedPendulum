import numpy as np
import scipy
import numpy 
import matplotlib.pyplot as plt 
from  matplotlib import animation
from math import sin, cos, pi

G = 9.81


class SinglePendulum:
    #state s is given by x, x_dot, theta, theta_dot

    def __init__(self, L, Mcart, Mpendulum, X0, X0_dot, Theta0, Theta0_dot):
        self.l = L                 # length
        self.m_c = Mcart            #cart mass
        self.m_p = Mpendulum        #pendulum mass
        self.x = X0                 #cart initial position
        self.x_dot = X0_dot         #cart initial speed
        self.theta = Theta0        #pendulum initial angle
        self.theta_dot = Theta0_dot#pendulum initial angular velocity
        self.t = 0                  #simulation time

    def system_dynamics(self, dt):
        sin_theta = np.sin(self.theta)
        cos_theta = np.cos(self.theta)

        numerator = G * sin_theta - cos_theta * ( ( self.m_p * self.l * self.theta_dot**2 * sin_theta) / (self.m_c + self.m_p) )
        denominator = self.l * (4/3 - (self.m_p * cos_theta**2) / (self.m_c + self.m_p))
        theta_ddot = numerator / denominator
        
        x_ddot = ( self.m_p * self.l * (self.theta_dot**2 * sin_theta - theta_ddot * cos_theta)) / (self.m_c + self.m_p)

        return [self.x_dot*dt , x_ddot*dt , self.theta_dot*dt, theta_ddot*dt]
    

    def draw_system_state(self):
        fig, ax = plt.subplots()
        line, = ax.plot([], [], 'b-', lw=3)
        ball, = ax.plot([], [], 'ro', markersize=12)

        ax.set_title("Single Pendulum")
        frame_size = self.l*3
        ax.set_xlim(-1*frame_size, frame_size)
        ax.set_ylim(-1*frame_size, frame_size)
        ax.set_aspect('equal')

        def init():
            line.set_data([], [])
            ball.set_data([], [])
            return line, ball

        def animate(i):
            num_frame = 1000 

            x_points = [self.x, self.l * cos(self.theta+ pi/2)+self.x]
            y_points = [0, self.l * sin(self.theta+pi/2)]

            line.set_data(x_points, y_points)
            ball.set_data([x_points[-1]], [y_points[-1]])

            s_x_dot, s_x_ddot, s_theta_dot, s_theta_ddot = self.system_dynamics(0.01)
            self.x -= s_x_dot 
            self.x_dot += s_x_ddot 
            self.theta += s_theta_dot
            self.theta_dot += s_theta_ddot

            return line, ball
 
        ani = animation.FuncAnimation(fig,animate, frames=10000, init_func=init, interval=10, blit=True)
        plt.show()


class DoublePendulum:
    def __init__(self, L1, L2):
        pass
    def draw_system_state(x, theta1, theta2):
        pass


if __name__ == "__main__":
    instance = SinglePendulum(10, 1, 1, 0, 0, -0.3, 0)
    instance.draw_system_state()


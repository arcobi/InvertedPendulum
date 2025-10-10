import matplotlib.pyplot as plt 
from  matplotlib import animation
import matplotlib.patches as patches
import matplotlib.animation as animation

from math import sin, cos, pi, sqrt
import numpy as np

class DoublePendulumCartRender:
    def __init__(self, duration, frameRate, masses, lengths):
        self.sim_length = duration
        self.frame_rate = frameRate
        self.m1, self.m2, self.m3 = masses
        self.l1, self.l2 = lengths


    def print_progress_bar(self, iteration, total, prefix='Progress:', suffix='Complete', length=50, fill='█'):
        percent_str = f"{100 * (iteration / float(total)):.1f}"
        percent_val = float(percent_str) 
        filled_length = int(length * iteration // total) 
        bar = fill * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} |{bar}| {percent_str}% {suffix}', end='', flush=True)
        if iteration == total:
            print()
        
    def draw_cart_double_pendulum(self,sol):
        fig, ax = plt.subplots()
        max_weight = max(self.m1, self.m2, self.m3)
        area = 14/max_weight
        pendulum1, = ax.plot([], [], 'b-', lw=3)
        pendulum2, = ax.plot([], [], 'b-', lw=3)
        ball1, = ax.plot([], [], 'ro', markersize=sqrt(area*self.m2 / pi))
        ball2, = ax.plot([], [], 'ro', markersize=sqrt(area*self.m3 / pi))
        rect_side = sqrt(area*self.m1)/16
        rect = patches.Rectangle((-rect_side/2, -rect_side/2), rect_side, rect_side, facecolor='skyblue')

        ax.set_title("Double Inverted Pendulum")
        frame_size = max(self.l1, self.l2)*3
        ax.set_xlim(-1*frame_size, frame_size)
        ax.set_ylim(-1*frame_size, frame_size)
        ax.set_aspect('equal')

        def init():
            pendulum1.set_data([], [])
            pendulum2.set_data([], [])
            ball1.set_data([], [])
            ball2.set_data([], [])
            ax.add_patch(rect)
            return pendulum1, pendulum2,  ball1, ball2, rect

        def animate(i):
            frame_count = self.sim_length*self.frame_rate
            self.print_progress_bar(i,frame_count-1)
            index = i*100 // self.frame_rate

            x, theta1, theta2 = sol.y[0][i], sol.y[2][i], sol.y[4][i]
            dx, dt1, dt2 = sol.y[1][i], sol.y[3][i], sol.y[5][i]

            x_points1 = [x, self.l1 * sin(theta1)+x]
            y_points1 = [0, self.l1 * cos(theta1)]
            x_points2 = [x_points1[1], x_points1[1] + self.l2 * sin(theta2)]
            y_points2 = [y_points1[1], y_points1[1] + self.l2 * cos(theta2)]

            pendulum1.set_data(x_points1, y_points1)
            pendulum2.set_data(x_points2, y_points2)

            ball1.set_data([x_points1[-1]], [y_points1[-1]])
            ball2.set_data([x_points2[-1]], [y_points2[-1]])

            rect.set_xy((x-rect_side/2, -rect_side/2))

            return pendulum1,pendulum2, ball1, ball2, rect
        ani = animation.FuncAnimation(fig,animate, frames=self.sim_length*self.frame_rate, init_func=init, interval=1000/self.frame_rate, blit=True)
        print("saving")
        ani.save('double_pendulum.gif', writer='pillow', fps=25)
        



class SimplePendulumCartRenderer:
    def __init__(self, duration, frameRate, masses, length):
        self.sim_length = duration
        self.frame_rate = frameRate
        self.m1, self.m2= masses
        self.l1 = length

    def print_progress_bar(self, iteration, total, prefix='Progress:', suffix='Complete', length=50, fill='█'):
        percent_str = f"{100 * (iteration / float(total)):.1f}"
        percent_val = float(percent_str) 
        filled_length = int(length * iteration // total) 
        bar = fill * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} |{bar}| {percent_str}% {suffix}', end='', flush=True)
        if iteration == total:
            print()
        
    def draw_cart_double_pendulum(self,sol):
        fig, ax = plt.subplots()
        max_weight = max(self.m1, self.m2, self.m3)
        area = 14/max_weight
        pendulum1, = ax.plot([], [], 'b-', lw=3)
        ball1, = ax.plot([], [], 'ro', markersize=sqrt(area*self.m2 / pi))
        rect_side = sqrt(area*self.m1)
        rect = patches.Rectangle((-rect_side/2, -rect_side/2), rect_side, rect_side, facecolor='skyblue')

        ax.set_title("Simple Inverted Pendulum")
        frame_size = max(self.l1, self.l2)*3
        ax.set_xlim(-1*frame_size, frame_size)
        ax.set_ylim(-1*frame_size, frame_size)
        ax.set_aspect('equal')

        def init():
            pendulum1.set_data([], [])
            ball1.set_data([], [])
            ax.add_patch(rect)
            return pendulum1, ball1, rect

        def animate(i):
            frame_count = self.sim_length*self.frame_rate
            self.print_progress_bar(i,frame_count-1)
            index = i*100 // self.frame_rate

            x, theta1, theta2 = sol.y[0][i], sol.y[2][i], sol.y[4][i]
            dx, dt1, dt2 = sol.y[1][i], sol.y[3][i], sol.y[5][i]

            x_points1 = [x, self.l1 * sin(theta1)+x]
            y_points1 = [0, self.l1 * cos(theta1)]

            pendulum1.set_data(x_points1, y_points1)
            ball1.set_data([x_points1[-1]], [y_points1[-1]])
            rect.set_xy((x-rect_side/2, -rect_side/2))

            return pendulum1, ball1, rect
        ani = animation.FuncAnimation(fig,animate, frames=self.sim_length*self.frame_rate, init_func=init, interval=1000/self.frame_rate, blit=True)
        print("saving...")
        ani.save('simple_pendulum.gif', writer='pillow', fps=25)
        
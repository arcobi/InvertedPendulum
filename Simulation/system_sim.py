import numpy as np
import scipy
from scipy.integrate import solve_ivp
import numpy 
import matplotlib.pyplot as plt 
from  matplotlib import animation
from math import sin, cos, pi

G = 9.81

m1 = 10
m2 = 2
m3 = 3
l1= 1
l2 = 1


def system_dynamics(t, Y):
    x, dx, t1, dt1, t2, dt2 = Y
    sin1, cos1 = sin(t1), cos(t1)
    sin2, cos2 = sin(t2), cos(t2)

    u = 0
    #magic
    #ddt1_denominator = ((m2+m3)*(l1**2)) - (cos(t1-t2)**2)*m3*(l1**2)  + l1*cos1 * (m2+m3) * ((m2+m3)* l1 * cos1 - l1 *cos2 *cos(t1-t2)) / (m1+m2+m3)
    #ddt1_numerator = - m3 * l1 * G * sin2 + m3 * (l1**2) * sin(t1-t2) * (dt1**2) - l1*cos1*(m2+m3) / (m1+m2+m3) *( (-l2*G*sin2+l1*l2*(dt1**2)*sin(t1-t2)) / (l2**2) + m2*l1*(dt1**2)*sin1 +m3*l1*(dt1**2)*sin1 + m3*l2*(dt2**2)*sin2) + (m2+m3)*l1*G*sin1 + m3*l1*l2*(dt2**2)*sin(t1-t2)
    #ddt1 = ddt1_numerator / ddt1_denominator

    #ddx_denominator = m1+m2+m3
    #ddx_numerator = m2*l1*(dt1**2)*sin1 + m3*(l1*(dt1**2)*sin1 + l2 * (dt2**2)*sin2) - (m2*l1*cos1*ddt1 + m3*l1*ddt1*cos1 + m3*cos2*(G*sin2 - l1*(dt1**2)*sin(t1-t2)-l1*ddt1*cos(t1-t2)))
    #ddx = ddx_numerator / ddx_denominator

    #ddt2_denominator = m3*(l2**2)
    #ddt2_numerator = m3*l2*G*sin2 - m3*l1*l2*(dt1**2)*sin(t1-t2) - m3*l1*l2*ddt1*cos(t1-t2)
    #ddt2 = ddt2_numerator / ddt2_denominator

    #denominator = m2**2 + 2*m1*m2 + m1*m3 + m2*m3 - m1*m3*(cos(2*t1 - 2*t2) - m2*m3 * cos(2*t1) - m2**2 * cos(2*t1))

    #ddx = -(2*l1*m2**2*sin1+2*l1*m2*m3*sin1)*(dt1**2) - (m2*m3*l2*sin(2*t1-t2)+m2*m3*l2*sin(t2))*(dt2**2) - (-m2**2*sin(2*t1) - m2*m3*sin(2*t1))*G
    #ddx /= denominator

    #ddt1 = (l1*m2**2*sin(2*t1) + l1 * m2*m3*sin(2*t1)+l1*m1*m2*sin(2*t1-2*t2))*(dt1**2) - (m2*m3*l2 *sin(t1+t2) + m2*m3*l2*sin(t1-t2)+2*m1*m3*l2*sin(t1-t2)) * (dt2**2) - (m1*m2*(-2*sin(t1)) - m1*m3*sin(t1) - 2*m2**2*sin(t1) - 2*m2*m3*sin(t1) - m1*m3*sin(t1-2*t2))*G
    #ddt1 /= denominator * l1

    #ddt2 =-(-2*m1*m2*l1*sin(t1-t2) - 2*m1*m3*l2*sin(t1-t2))*(dt1**2) + (m1*m3*l2*sin(2*t1 - 2*t2))*(dt2**2) - (m1*m2*sin(2*t1-t2) - m1*m2*sin(t2) + m1*m3*sin(2*t1-t2) - m1*m3*sin(t2))*G
    #ddt2/=denominator*l2

    denominator = (m2**2 + 2*m1*m2 + m1*m3 + m2*m3 - m1*m3*cos(2*t1 - 2*t2)
                - m2*m3*cos(2*t1) - m2**2*cos(2*t1))

    # --- Calculation for ddx (x acceleration) ---
    # Numerator is already quite complex, breaking it down is safer but left as is for comparison
    ddx_num = (-(2*l1*m2**2*sin(2*t1) + 2*l1*m2*m3*sin(t1))*(dt1**2)
            -(m2*m3*l2*sin(2*t1-t2) + m2*m3*l2*sin(t2))*(dt2**2)
            - (-m2**2*sin(2*t1) - m2*m3*sin(2*t1))*G)
    ddx = ddx_num / denominator

    # --- Calculation for ddt1 (theta acceleration) ---
    # Numerator broken into parts for clarity
    num1_dtheta_sq = (l1*m2**2*sin(2*t1) + l1*m2*m3*sin(2*t1) + l2*m1*m3*sin(2*t1-2*t2))*(dt1**2) # FIXED: ADDED THIS TERM
    num1_dphi_sq = -(m2*m3*l2*sin(t1+t2) + m2*m3*l2*sin(t1-t2) + 2*m1*m3*l2*sin(t1-t2))*(dt2**2)
    num1_gravity = -(m1*m2*(-2*sin(t1)) - m1*m3*sin(t1) - 2*m2**2*sin(t1)
                    - 2*m2*m3*sin(t1) - m1*m3*sin(t1-2*t2))*G # FIXED: ADDED LEADING '-'

    ddt1_num = num1_dtheta_sq + num1_dphi_sq + num1_gravity
    ddt1 = ddt1_num / (denominator * l1)

    # --- Calculation for ddt2 (phi acceleration) ---
    # Numerator broken into parts for clarity
    num2_dtheta_sq = -(-2*m1*m2*l1*sin(t1-t2) - 2*m1*m3*l2*sin(t1-t2))*(dt1**2)
    num2_dphi_sq = (m1*m3*l2*sin(2*t1 - 2*t2))*(dt2**2)
    num2_gravity = -(m1*m2*sin(2*t1-t2) - m1*m2*sin(t2) + m1*m3*sin(2*t1-t2)
                    - m1*m3*sin(t2))*G # FIXED: sin(t1-t2) -> sin(t2)

    ddt2_num = num2_dtheta_sq + num2_dphi_sq + num2_gravity
    ddt2 = ddt2_num / (denominator * l2)

 

    return [dx, ddx, dt1, ddt1, dt2, ddt2]

Y0 = [0,0,3*pi/4,0,-3*pi/4,0]
t_span = [0, 80]
t_eval = np.linspace(t_span[0], t_span[1], 50000)
sol = solve_ivp(system_dynamics, t_span, Y0, t_eval=t_eval)


def draw_system_state():
    fig, ax = plt.subplots()
    pendulum1, = ax.plot([], [], 'b-', lw=3)
    pendulum2, = ax.plot([], [], 'b-', lw=3)
    ball, = ax.plot([], [], 'ro', markersize=12)

    ax.set_title("Single Pendulum")
    frame_size = l1*5
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
        



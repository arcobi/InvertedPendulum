import matplotlib.pyplot as plt
from double_pendulum_env import DoubleCartPoleEnv
import numpy as np

def plot_energy_conservation():
    """
    1エピソードを実行し、系のエネルギー保存が成り立っているかをグラフで可視化する。
    """
    env = DoubleCartPoleEnv()
    obs, info = env.reset()

    times, kinetic_energies, potential_energies = [], [], []
    system_energies, work_inputs, conserved_quantities = [], [], []

    cumulative_work = 0.0
    time = 0.0
    done = False
    step_count = 0
    
    print("--- Starting energy conservation test (Corrected Version) ---")

    while not done:
        # 方策: 最初の1秒間だけ右に押し、あとは成り行きに任せる
        is_done_by_policy = step_count > 1000
        force = 5.0 if step_count < 50 else 0.0
        action = np.array([force])
        
        state = env.state
        x, x_dot, theta1, theta1_dot, theta2, theta2_dot = state
        
        # --- 運動エネルギー (KE) の計算 (変更なし) ---
        v1_x = x_dot + env.l_1 * theta1_dot * np.cos(theta1)
        v1_y = -env.l_1 * theta1_dot * np.sin(theta1) # y軸上向きを正とする
        
        v2_x = v1_x + env.l_2 * theta2_dot * np.cos(theta2)
        v2_y = v1_y - env.l_2 * theta2_dot * np.sin(theta2) # y軸上向きを正とする

        ke_cart = 0.5 * env.m_c * x_dot**2
        ke_p1 = 0.5 * env.m_1 * (v1_x**2 + v1_y**2)
        ke_p2 = 0.5 * env.m_2 * (v2_x**2 + v2_y**2)
        total_ke = ke_cart + ke_p1 + ke_p2

        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼【ここを修正】▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # --- 位置エネルギー (PE) の計算 (基準: カートのピボット位置 y=0) ---
        # y座標は上向きを正とするため、高さ h = l * cos(theta) となる
        h1 = env.l_1 * np.cos(theta1)
        h2 = h1 + env.l_2 * np.cos(theta2)
        
        # 初期状態(theta=pi)で PE が負の最小値になるように基準を調整
        pe_offset = (env.m_1 * env.g * env.l_1) + (env.m_2 * env.g * (env.l_1 + env.l_2))

        pe_p1 = env.m_1 * env.g * h1
        pe_p2 = env.m_2 * env.g * h2
        total_pe = pe_p1 + pe_p2 + pe_offset
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲【ここまで】▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        total_sys_energy = total_ke + total_pe
        work_this_step = force * x_dot * env.dt
        cumulative_work += work_this_step
        conserved_quantity = total_sys_energy - cumulative_work

        times.append(time)
        kinetic_energies.append(total_ke)
        potential_energies.append(total_pe)
        system_energies.append(total_sys_energy)
        work_inputs.append(cumulative_work)
        conserved_quantities.append(conserved_quantity)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated or is_done_by_policy
        time += env.dt
        step_count += 1
        
    env.close()
    print("Simulation finished. Plotting results...")

    # --- グラフの描画 (変更なし) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Energy Analysis of the Double Pendulum System (Corrected)', fontsize=16)

    ax1.plot(times, kinetic_energies, label='Kinetic Energy (KE)', color='red')
    ax1.plot(times, potential_energies, label='Potential Energy (PE)', color='blue')
    ax1.plot(times, system_energies, label='Total System Energy (KE + PE)', color='black', linewidth=2)
    ax1.set_ylabel('Energy (Joules)')
    ax1.set_title('Energy Components')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(times, work_inputs, label='Work Done by Input Force ($W_{in}$)', color='green')
    ax2.plot(times, conserved_quantities, label='Conserved Quantity ($E_{sys} - W_{in}$)', color='purple', linewidth=3)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Energy (Joules)')
    ax2.set_title('Energy Conservation Check')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    plot_energy_conservation()
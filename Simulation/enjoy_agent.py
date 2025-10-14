import time
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from tqdm import tqdm
from double_pendulum_env import DoubleCartPoleEnv

def evaluate_and_plot():
    """
    学習済みエージェントを実行してアニメーションを表示し、
    終了後にそのエピソードのエネルギー分析グラフをプロットする。
    """
    # --- 1. モデルと環境の準備 ---
    model_path = "models/ppo_double_pendulum_final.zip"

    print(f"Loading trained model from {model_path}...")
    model = PPO.load(model_path)

    env = DoubleCartPoleEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    
    # --- データを記録するためのリスト ---
    history_states = [env.state.copy()] # 初期状態を記録
    history_forces = []
    
    print("\n--- Phase 1: Running Agent Animation ---")
    
    # --- 2. アニメーションの実行とデータ記録 ---
    progress_bar = tqdm(desc="Evaluating Agent", unit=" steps")
    try:
        while not done:
            env.render()
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 今回のステップの状態と行動を記録
            history_states.append(env.state.copy())
            history_forces.append(action[0])
            
            done = terminated or truncated
            progress_bar.update(1)
    finally:
        progress_bar.close()
        env.close()

    print(f"\n--- Phase 2: Calculating Energy from History ---")
    print(f"Episode finished after {progress_bar.n} steps. Analyzing...")

    # --- 3. 記録したデータからエネルギーを計算 ---
    times = np.arange(len(history_forces)) * env.dt
    kinetic_energies, potential_energies, system_energies = [], [], []
    work_inputs, conserved_quantities = [], []
    cumulative_work = 0.0

    # 最初の状態のエネルギーを計算
    initial_state = history_states[0]
    # (エネルギー計算は省略。ループ内で計算するため)

    for i in range(len(history_forces)):
        state = history_states[i]
        force = history_forces[i]
        x, x_dot, theta1, theta1_dot, theta2, theta2_dot = state
        
        # 運動エネルギー (KE)
        v1_x = x_dot + env.l_1 * theta1_dot * np.cos(theta1)
        v1_y = -env.l_1 * theta1_dot * np.sin(theta1)
        v2_x = v1_x + env.l_2 * theta2_dot * np.cos(theta2)
        v2_y = v1_y - env.l_2 * theta2_dot * np.sin(theta2)
        ke_cart = 0.5 * env.m_c * x_dot**2
        ke_p1 = 0.5 * env.m_1 * (v1_x**2 + v1_y**2)
        ke_p2 = 0.5 * env.m_2 * (v2_x**2 + v2_y**2)
        total_ke = ke_cart + ke_p1 + ke_p2

        # 位置エネルギー (PE)
        h1 = env.l_1 * np.cos(theta1)
        h2 = h1 + env.l_2 * np.cos(theta2)
        pe_offset = (env.m_1 * env.g * env.l_1) + (env.m_2 * env.g * (env.l_1 + env.l_2))
        pe_p1 = env.m_1 * env.g * h1
        pe_p2 = env.m_2 * env.g * h2
        total_pe = pe_p1 + pe_p2 + pe_offset
        
        total_sys_energy = total_ke + total_pe
        work_this_step = force * x_dot * env.dt
        cumulative_work += work_this_step
        conserved_quantity = total_sys_energy - cumulative_work

        kinetic_energies.append(total_ke)
        potential_energies.append(total_pe)
        system_energies.append(total_sys_energy)
        work_inputs.append(cumulative_work)
        conserved_quantities.append(conserved_quantity)

    # --- 4. グラフの描画 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Energy Analysis of Trained Agent Performance', fontsize=16)

    ax1.plot(times, kinetic_energies, label='Kinetic Energy (KE)', color='red')
    ax1.plot(times, potential_energies, label='Potential Energy (PE)', color='blue')
    ax1.plot(times, system_energies, label='Total System Energy (KE + PE)', color='black', linewidth=2)
    ax1.set_ylabel('Energy (Joules)')
    ax1.set_title('Energy Components During Episode')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(times, work_inputs, label='Work Done by Agent ($W_{in}$)', color='green')
    ax2.plot(times, conserved_quantities, label='Conserved Quantity ($E_{sys} - W_{in}$)', color='purple', linewidth=3)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Energy (Joules)')
    ax2.set_title('Energy Conservation Check')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    print("\nDisplaying energy analysis plot. Close the plot window to exit.")
    plt.show()

if __name__ == "__main__":
    evaluate_and_plot()
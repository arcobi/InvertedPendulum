import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class DoubleCartPoleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # --- 物理パラメータ (変更なし) ---
        self.g = 9.81
        self.m_c = 1.0
        self.m_1 = 0.5
        self.m_2 = 0.5
        self.l_1 = 0.6
        self.l_2 = 0.6
        self.dt = 0.02

        # --- 状態空間と行動空間 (変更なし) ---
        high = np.array([
            4.0, np.finfo(np.float32).max,
            np.pi, np.finfo(np.float32).max,
            np.pi, np.finfo(np.float32).max
        ], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)

        # --- レンダリング用 (高さを16の倍数に修正) ---
        self.render_mode = render_mode
        self.screen_width = 800
        self.screen_height = 608 # 600 -> 608 に変更
        self.screen = None
        self.clock = None
        self.state = None

    # double_pendulum_env.py 内の、この関数だけを置き換えてください

    def _get_derivatives(self, state, force):
        """
        与えられた状態と力に対する状態の時間微分を計算する。
        戻り値: [x_dot, x_ddot, theta1_dot, theta1_ddot, theta2_dot, theta2_ddot]
        """
        x, x_dot, theta1, theta1_dot, theta2, theta2_dot = state
        
        m_t = self.m_c + self.m_1 + self.m_2
        c1, s1 = np.cos(theta1), np.sin(theta1)
        c2, s2 = np.cos(theta2), np.sin(theta2)
        c12, s12 = np.cos(theta1 - theta2), np.sin(theta1 - theta2)

        M = np.zeros((3, 3))
        M[0, 0] = m_t
        M[0, 1] = (self.m_1 + self.m_2) * self.l_1 * c1
        M[0, 2] = self.m_2 * self.l_2 * c2
        M[1, 0] = M[0, 1]
        M[1, 1] = (self.m_1 + self.m_2) * self.l_1**2
        M[1, 2] = self.m_2 * self.l_1 * self.l_2 * c12
        M[2, 0] = M[0, 2]
        M[2, 1] = M[1, 2]
        M[2, 2] = self.m_2 * self.l_2**2

        F = np.zeros(3)
        F[0] = (self.m_1 + self.m_2) * self.l_1 * theta1_dot**2 * s1 + \
               self.m_2 * self.l_2 * theta2_dot**2 * s2 + force
        
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼【ここが修正点】▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # 重力項の符号を反転させ、theta=0 が「真上」になるように座標系を統一
        F[1] = -self.m_2 * self.l_1 * self.l_2 * theta2_dot**2 * s12 + \
               (self.m_1 + self.m_2) * self.g * self.l_1 * s1
        F[2] = self.m_2 * self.l_1 * self.l_2 * theta1_dot**2 * s12 + \
               self.m_2 * self.g * self.l_2 * s2
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲【ここまで】▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        try:
            accel = np.linalg.solve(M, F)
            x_ddot, theta1_ddot, theta2_ddot = accel
        except np.linalg.LinAlgError:
            x_ddot, theta1_ddot, theta2_ddot = 1e10, 1e10, 1e10
        
        return np.array([x_dot, x_ddot, theta1_dot, theta1_ddot, theta2_dot, theta2_ddot])

    def step(self, action):
        force = action[0]
        
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼【ここから4次のルンゲ＝クッタ法】▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        k1 = self._get_derivatives(self.state, force)
        k2 = self._get_derivatives(self.state + 0.5 * self.dt * k1, force)
        k3 = self._get_derivatives(self.state + 0.5 * self.dt * k2, force)
        k4 = self._get_derivatives(self.state + self.dt * k3, force)
        
        # 状態の更新
        self.state += (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲【ここまで】▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # 状態変数を更新後のものに展開
        x, _, theta1, theta1_dot, theta2, theta2_dot = self.state

        # --- 終了判定と報酬計算 (変更なし) ---
        terminated = bool(abs(x) > 3.0)

        reward = np.cos(theta1) + np.cos(theta2)
        reward -= 0.1 * np.abs(x)
        reward -= 0.01 * (theta1_dot**2 + theta2_dot**2)
        reward -= 0.001 * (force**2)

        return self._get_obs(), reward, terminated, False, {}

    # reset, _get_obs, render, close 関数は変更なし
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0, 0, np.pi, 0, np.pi, 0], dtype=np.float32)
        self.state += self.np_random.uniform(low=-0.01, high=0.01, size=self.state.shape)
        return self._get_obs(), {}

    def _get_obs(self):
        s = self.state.copy()
        s[2] = (s[2] + np.pi) % (2 * np.pi) - np.pi
        s[4] = (s[4] + np.pi) % (2 * np.pi) - np.pi
        return s
    
    def render(self):
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.set_caption("Double Cart-Pole RL Environment (RK4)")
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        self.screen.fill((255, 255, 255))
        world_width = 6.0
        scale = self.screen_width / world_width
        
        if self.state is None: return None
        
        x, _, theta1, _, theta2, _ = self.state
        cart_x = x * scale + self.screen_width / 2
        cart_y = self.screen_height * 2 / 3
        cart_w, cart_h = 100, 50
        pygame.draw.rect(self.screen, (0, 0, 0), (cart_x - cart_w / 2, cart_y - cart_h / 2, cart_w, cart_h))
        
        pole1_len = self.l_1 * scale
        pole1_x_end = cart_x + pole1_len * np.sin(theta1)
        pole1_y_end = cart_y - pole1_len * np.cos(theta1)
        pygame.draw.line(self.screen, (200, 100, 100), (cart_x, cart_y), (pole1_x_end, pole1_y_end), 10)

        pole2_len = self.l_2 * scale
        pole2_x_end = pole1_x_end + pole2_len * np.sin(theta2)
        pole2_y_end = pole1_y_end - pole2_len * np.cos(theta2)
        pygame.draw.line(self.screen, (100, 100, 200), (pole1_x_end, pole1_y_end), (pole2_x_end, pole2_y_end), 10)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        elif self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
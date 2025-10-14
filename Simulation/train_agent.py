import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from double_pendulum_env import DoubleCartPoleEnv

# --- メインの実行部分 ---
if __name__ == "__main__":
    # ログとモデルの保存先フォルダ
    log_dir = "logs/"
    model_dir = "models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # --- 1. 環境の準備 ---
    # 複数の環境を並列で動かして、効率的に学習させる (VecEnv)
    # n_envs=4 は、同時に4つのシミュレータを動かすという意味
    env = make_vec_env(DoubleCartPoleEnv, n_envs=4)

    # --- 2. モデルの定義 (AIの頭脳の設計) ---
    # PPOアルゴリズムと、多層パーセプトロン(MlpPolicy)のネットワークを使用
    # いくつかの重要なパラメータを設定
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1, # 学習の進捗状況をターミナルに表示する
        learning_rate=0.0003, # 学習の速さ
        n_steps=2048, # 何ステップ分の経験を貯めてから学習するか
        batch_size=64, # 1回の学習で使うデータ量
        gamma=0.99, # 未来の報酬をどれだけ重視するか
        tensorboard_log=log_dir # 学習の進捗をグラフで見るためのログの保存先
    )

    # --- 3. 学習の開始 ---
    # 合計で100万ステップ分のシミュレーションを実行して学習させる
    # この数字が大きいほど、賢くなる可能性が高い (時間もかかる)
    total_timesteps = 1_000_000
    
    print("--- Starting training ---")
    model.learn(total_timesteps=total_timesteps)
    print("--- Finished training ---")

    # --- 4. 学習済みモデルの保存 ---
    # 学習後の賢くなったAIの脳（ニューラルネットワークの重み）をファイルに保存
    model.save(os.path.join(model_dir, "ppo_double_pendulum_final"))
    print(f"Trained model saved to {model_dir}ppo_double_pendulum_final.zip")

    # 環境を閉じる
    env.close()
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

# C++でビルドしたモジュールのあるパスを解決
# このファイルの場所から見て、2つ上の階層にあるbuildディレクトリを探す
build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../build/controller/vnoid_rl_env"))
sys.path.append(build_path)

try:
    # C++モジュールをインポート
    import vnoid_rl_env
except ImportError as e:
    print(f"C++モジュールのインポートに失敗しました。パスを確認してください: {build_path}")
    print(f"エラー詳細: {e}")
    sys.exit(1)

class HumanoidVnoidEnv(gym.Env):
    """
    vnoidコントローラを内蔵したC++製の高速なHumanoid環境のラッパー
    """
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 67, # この値はシミュレーションに合わせて要調整
    }

    def __init__(self, **kwargs):
        super().__init__()
        
        # モデルファイルのパスを解決
        # このファイルの場所から見た相対パスで指定
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/sample_robot/sample_robot_mujoco.xml"))
        
        # C++側の環境インスタンスを作成
        self.cpp_env = vnoid_rl_env.VnoidEnv(model_path)
        
        # 行動空間と観測空間をC++の仕様から設定
        # C++側 (bindings.cpp) の実装と一致させる必要がある
        
        # 行動空間: 着地点オフセット(x, y)の2次元
        self.action_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)
        # 観測空間: qpos(35) + qvel(30) = 65次元 (あなたのモデルに合わせて調整)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(73,), dtype=np.float64)

    def step(self, action):
        obs, reward, terminated, info = self.cpp_env.step(action)
        # C++側でtruncated(時間切れ)は実装していないのでFalseを返す
        truncated = False
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.cpp_env.reset()
        info = {}
        return obs, info

    def render(self):
        # ★★★ C++側のrender関数を呼び出して、その結果を返す ★★★
        return self.cpp_env.render()

    def close(self):
        del self.cpp_env

# 新しい環境として登録
register(
    id="HumanoidVnoid-v0",
    entry_point="my_humanoid_env:HumanoidVnoidEnv",
    max_episode_steps=1000,
)
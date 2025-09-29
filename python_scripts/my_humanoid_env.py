# ★★★ 最終版 my_humanoid_env.py ★★★

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

# C++でビルドしたモジュールのあるパスを解決
build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../build/controller/vnoid_rl_env"))
sys.path.append(build_path)

try:
    import vnoid_rl_env
except ImportError as e:
    print(f"C++モジュールのインポートに失敗しました。パスを確認してください: {build_path}")
    print(f"エラー詳細: {e}")
    sys.exit(1)

class HumanoidVnoidEnv(gym.Env):
    """
    vnoidコントローラを内蔵したC++製の高速なHumanoid環境のラッパー
    - enable_rendering=False: 学習用（超軽量、GLFW初期化なし）
    - enable_rendering=True: 録画・デバッグ用（インタラクティブ表示付き）
    """
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 60,
    }

    def __init__(self, enable_rendering=False, render_mode=None, **kwargs):
        super().__init__()
        
        # render_modeが指定されている場合は自動的にレンダリングを有効化
        if render_mode in ["human", "rgb_array", "depth_array"]:
            enable_rendering = True
        
        # モデルファイルのパスを解決
        model_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            "../model/sample_robot/sample_robot_mujoco.xml"
        ))
        
        # C++側の環境インスタンスを作成
        self.cpp_env = vnoid_rl_env.VnoidEnv(model_path, enable_rendering)
        self.render_mode = render_mode
        self.enable_rendering = enable_rendering
        
        # 行動空間と観測空間の設定
        self.action_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(73,), dtype=np.float64)
        
        print(f"HumanoidVnoidEnv初期化完了 (レンダリング: {'有効' if enable_rendering else '無効'})")

    def step(self, action):
        obs, reward, terminated, info = self.cpp_env.step(action)
        truncated = False
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.cpp_env.reset()
        info = {}
        return obs, info

    def render(self):
        """
        レンダリング機能
        - レンダリング無効: Noneを返す
        - レンダリング有効: 録画用ピクセル配列を返す（画面表示は自動更新）
        """
        if not self.enable_rendering:
            if self.render_mode == "rgb_array":
                print("警告: レンダリングが無効化されています。enable_rendering=Trueで環境を作成してください。")
                return None
            return None
        
        try:
            return self.cpp_env.render()
        except Exception as e:
            print(f"レンダリングエラー: {e}")
            return None

    def close(self):
        # ウィンドウが開いている場合は閉じる確認
        if hasattr(self.cpp_env, 'should_close') and self.cpp_env.should_close():
            print("ウィンドウが閉じられました。")
        del self.cpp_env

    def is_rendering_enabled(self):
        """レンダリング状態を確認"""
        return self.cpp_env.is_rendering_enabled()

    def should_close(self):
        """ウィンドウが閉じられたかを確認"""
        if self.enable_rendering:
            return self.cpp_env.should_close()
        return False


# ★★★ 用途別環境登録 ★★★

# 学習用：レンダリングなしで超高速
def make_training_env(config=None):
    config = config or {}
    return HumanoidVnoidEnv(enable_rendering=False, **config)

# 録画用：レンダリングありでインタラクティブ表示 + 録画機能
def make_recording_env(config=None):
    config = config or {}
    return HumanoidVnoidEnv(enable_rendering=True, render_mode="rgb_array", **config)

# 学習専用環境（軽量）
register(
    id="HumanoidVnoid-v0",
    entry_point=make_training_env,
    max_episode_steps=1000,
)

# 録画専用環境（インタラクティブ）
register(
    id="HumanoidVnoidRecording-v0", 
    entry_point=make_recording_env,
    max_episode_steps=1000,
)

# デバッグ用環境（手動操作可能）
register(
    id="HumanoidVnoidDebug-v0",
    entry_point=lambda config: HumanoidVnoidEnv(enable_rendering=True, render_mode="human"),
    max_episode_steps=1000,
)
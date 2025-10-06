# ★★★ 学習/録画モード対応版 my_humanoid_env.py ★★★

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

# C++モジュールのインポート
build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../build/controller/vnoid_rl_env"))
sys.path.append(build_path)

try:
    import vnoid_rl_env
except ImportError as e:
    print(f"❌ C++モジュールのインポートに失敗しました: {build_path}")
    print(f"エラー詳細: {e}")
    sys.exit(1)


class HumanoidVnoidEnv(gym.Env):
    """
    Vnoidヒューマノイド環境
    
    使用モード:
    - 学習用: enable_rendering=False (OpenGL不使用、超高速)
    - 録画用: enable_rendering=True, render_mode="rgb_array" (OpenGL使用)
    
    Note:
        frame_skipはsample_controller_mujocoと同じ60fps制御に設定されます
        (1/60秒 ÷ MuJoCo timestep)
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,  # sample_controller_mujocoと同じ
    }

    def __init__(self, enable_rendering=False, render_mode=None, **kwargs):
        super().__init__()
        
        # render_modeが指定されている場合は自動的にレンダリング有効化
        if render_mode in ["human", "rgb_array"]:
            enable_rendering = True
        
        # モデルファイルのパスを解決
        model_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            "../model/sample_robot/sample_robot_mujoco.xml"
        ))
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
        
        # C++環境インスタンスを作成
        # frame_skipはC++側でvnoid control_cycleから自動設定される
        self.cpp_env = vnoid_rl_env.VnoidEnv(model_path, enable_rendering)
        self.render_mode = render_mode
        self.enable_rendering = enable_rendering
        
        # 行動空間と観測空間の設定
        self.action_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(73,), dtype=np.float64)
        
        mode_str = "録画モード(OpenGL有効)" if enable_rendering else "学習モード(OpenGL無効)"
        print(f"✅ HumanoidVnoidEnv初期化完了: {mode_str}")

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
        - 学習モード: Noneを返す（OpenGL不使用）
        - 録画モード: ピクセル配列を返す（OpenGL使用）
        """
        if not self.enable_rendering:
            return None
        
        try:
            return self.cpp_env.render()
        except Exception as e:
            print(f"⚠️ レンダリングエラー: {e}")
            return None

    def close(self):
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


# ★★★ 環境登録 ★★★

# 学習専用環境（OpenGL不使用、超高速、並列向け）
def make_training_env(config=None):
    config = config or {}
    return HumanoidVnoidEnv(enable_rendering=False, **config)

# 録画専用環境（OpenGL使用、単一環境向け）
def make_recording_env(config=None):
    config = config or {}
    return HumanoidVnoidEnv(enable_rendering=True, render_mode="rgb_array", **config)

# 学習用環境ID
register(
    id="HumanoidVnoid-v0",
    entry_point=make_training_env,
    max_episode_steps=1000,
)

# 録画用環境ID
register(
    id="HumanoidVnoidRecording-v0", 
    entry_point=make_recording_env,
    max_episode_steps=1000,
)
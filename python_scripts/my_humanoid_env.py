# ★★★ 学習/録画モード対応版 my_humanoid_env.py ★★★

import os
import sys
import numpy as np
import gymnasium as gym
from ray import tune
#パイソンが探すものの中にパス入れる
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
    """離散的な歩行制御環境 (1ステップ = 1歩)"""
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

    def __init__(self, enable_rendering=False, render_mode=None, max_episode_steps=50, **kwargs):
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
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        
        # 行動空間と観測空間の設定
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float64)
        
        mode_str = "録画モード(OpenGL有効)" if enable_rendering else "学習モード(OpenGL無効)"
        print(f"✅ HumanoidVnoidEnv初期化完了: {mode_str}")

    def step(self, action):
         """
        1歩分の実行
        
        Args:
            action: [foot_offset_x, foot_offset_y, 
                    foot_angle_roll, foot_angle_pitch, foot_angle_yaw]
        
        Returns:
            obs, reward, terminated, truncated, info,frames
        """
        # actionをスケーリング
         rl_action = np.zeros(2, dtype=np.float64)
         rl_action[0] = action[0] * 0.15   
         rl_action[1] = action[1] * 0.15   
         #rl_action[2] = action[2] * 0.15 
         #rl_action[3] = action[3] * 0.1
         #rl_action[4] = action[4] * 0.2  
            #actionはすべて-1~1で入ってくる
         '''
          Step step;
	        step.stride   = 0.1 ;
	        step.turn     = 0.0 ;
	        step.spacing  = 0.2 + rl_params.spacing_offset;
	        step.climb    = 0.0 + rl_params.climb_offset;
	        step.duration = 0.4 + rl_params.duration_offset;

            stb1.foot_pos[swg].x() += rl_params.foot_offset.x();
            stb1.foot_pos[swg].y() += rl_params.foot_offset.y();
         '''

         print(rl_action)

        
        # ★ 1歩完了まで実行（C++側で制御サイクルは1000Hz）
        # C++のstep()は内部でframe_skip回のmj_step()を実行
         obs, reward, terminated, info,frames = self.cpp_env.step(rl_action)
         print(reward)
         self._step_count += 1

         #C++側で出ているrewardの値とPython側で受け取っているrewardの値が違う。
         #並列環境無しで行うと報酬が同じ。８環境で行うと報酬異なる
         #なぜこうなるのか、並列環境一個だけで試す必要あり
        
         truncated = (self._step_count >= self.max_episode_steps) and (not terminated)

         # ★ enable_renderingで分岐
         if self.enable_rendering:
            # 録画モード：フレームも返す
            return obs, reward, terminated, truncated, info, frames
         else:
            # 学習モード：フレームなし（互換性のため）
            return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0  # リセット時にカウンタをゼロに
        obs = self.cpp_env.reset()
        self._prev_step_completed = False
        info = {}
        #_get_obs()と_get_info()作ってもいいかも
        return obs, info

    def close(self):
        if hasattr(self.cpp_env, 'should_close') and self.cpp_env.should_close():
            print("ウィンドウが閉じられました。")
        del self.cpp_env

    def get_control_log(self):
        """C++側からログデータを取得"""
        return self.cpp_env.get_control_log()
    
    def clear_control_log(self):
        """C++側のログをクリア"""
        self.cpp_env.clear_control_log()

tune.register_env("HumanoidVnoid-v0", 
                  lambda config: HumanoidVnoidEnv(
                      enable_rendering=config.get("enable_rendering", False),
                      render_mode=config.get("render_mode", None)
                  ))
"""
Choreonoid Gymnasium環境
End-to-End強化学習用
"""

import os
import numpy as np
import gymnasium as gym
from typing import Optional, Tuple
from shm_interface import ChoreonoidShmClient, NUM_JOINTS, NUM_OBSERVATIONS


class ChoreonoidEnv(gym.Env):
    """
    Choreonoid End-to-End RL環境
    
    観測空間: 73次元（センサー生値）
    行動空間: 29次元（全関節目標角度、正規化）
    """
    
    metadata = {
        "render_modes": [],
        "render_fps": 1000,  # 1ms timestep
    }
    
    def __init__(self, env_id: int = 0, max_episode_steps: int = 1000):
        super().__init__()
        
        self.env_id = env_id
        self.max_episode_steps = max_episode_steps
        self._step_count = 0
        
        # 観測空間・行動空間の定義
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(NUM_OBSERVATIONS,),
            dtype=np.float64
        )
        
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(NUM_JOINTS,),
            dtype=np.float32
        )
        
        # 共有メモリクライアント
        self.shm_client = None
        
        print(f"✅ ChoreonoidEnv初期化完了 (env_id={env_id})")
        print(f"   観測空間: {self.observation_space.shape}")
        print(f"   行動空間: {self.action_space.shape}")
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """環境をリセット"""
        super().reset(seed=seed)
        
        # 初回のみ共有メモリを作成
        if self.shm_client is None:
            self.shm_client = ChoreonoidShmClient(self.env_id)
            print(f"[Env {self.env_id}] 共有メモリ作成完了。Choreonoidを起動してください...")
            print(f"  ENV_ID={self.env_id} choreonoid <project_file>")
        
        # リセット要求
        obs = self.shm_client.reset(timeout=30.0)
        
        self._step_count = 0
        info = {}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """1ステップ実行"""
        # 行動をクリップ
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        
        # ステップ実行
        obs, reward, done, info = self.shm_client.step(action, timeout=10.0)
        
        self._step_count += 1
        
        # 最大ステップ数チェック
        truncated = (self._step_count >= self.max_episode_steps) and (not done)
        
        return obs, reward, done, truncated, info
    
    def close(self):
        """環境をクローズ"""
        if self.shm_client is not None:
            self.shm_client.close()
            self.shm_client = None


# Ray RLlib用の登録
def make_choreonoid_env(config):
    """Ray RLlib用のファクトリ関数"""
    env_id = config.get("worker_index", 0) + config.get("vector_index", 0)
    return ChoreonoidEnv(env_id=env_id)


if __name__ == "__main__":
    # 簡易テスト
    print("=" * 70)
    print("Choreonoid環境テスト")
    print("=" * 70)
    
    env = ChoreonoidEnv(env_id=0)
    
    print("\nリセット中...")
    obs, info = env.reset()
    print(f"観測: shape={obs.shape}, mean={obs.mean():.3f}, std={obs.std():.3f}")
    
    print("\nランダム行動で10ステップ実行...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"  Step {i+1}: reward={reward:.3f}, done={done}, truncated={truncated}")
        
        if done or truncated:
            print("エピソード終了")
            break
    
    env.close()
    print("\n✅ テスト完了")

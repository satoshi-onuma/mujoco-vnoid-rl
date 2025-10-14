# test_single_step.py

import os
import sys
import numpy as np

build_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    "../build/controller/vnoid_rl_env"
))
sys.path.append(build_path)

from my_humanoid_env import HumanoidVnoidEnv

print("=" * 70)
print("🔍 単体テスト：描画ありで1歩ずつ確認")
print("=" * 70)

# 描画ありで環境作成
env = HumanoidVnoidEnv(enable_rendering=True, render_mode="human")

obs, info = env.reset()
print(f"✅ 初期化完了")
print(f"   観測空間: {obs.shape}")
print(f"   初期観測の一部: {obs[:10]}")

print("\n🎮 ランダムアクションで10歩実行...")
print("-" * 70)

episode_counter=0
for i in range(10):
    # ランダムアクション
    
    action = np.array([
         np.random.uniform(-1.0, 1.0),  # foot_offset_x
         np.random.uniform(-1.0, 1.0),  # foot_offset_y
         np.random.uniform(-1.0, 1.0),  # foot_angle_roll
         np.random.uniform(-1.0, 1.0),  # foot_angle_pitch
         np.random.uniform(-1.0, 1.0),  # foot_angle_yaw これないかも
     ])
    
    
    print(f"\nステップ {i+1}:")
    
    
    # 1歩実行（描画される）
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"  報酬: {reward:.4f}")
    print(f"  終了: {terminated}")
    print(f"  ベース位置(x,y,z): ({obs[0]:.3f}, {obs[1]:.3f}, {obs[2]:.3f})")
    
    if terminated or truncated:
        print(f"  ⚠️ エピソード終了、リセット")
        episode_counter = 0
        obs, info = env.reset()

print("\n" + "=" * 70)
print("✅ テスト完了")
print("=" * 70)

env.close()
# 簡単なテストスクリプト作成
# test_env.py
import sys
import os
sys.path.append("build/controller/vnoid_rl_env")

import vnoid_rl_env

print("1. モジュールインポート成功")

model_path = os.path.abspath("model/sample_robot/sample_robot_mujoco.xml")
print(f"2. モデルパス: {model_path}")

print("3. レンダリングなし環境作成中...")
env = vnoid_rl_env.VnoidEnv(model_path, False)
print("4. レンダリングなし環境作成成功！")

print("5. reset()実行中...")
obs = env.reset()
print(f"6. reset()成功！観測次元: {obs.shape}")

print("7. step()実行中...")
import numpy as np
action = np.array([0.0, 0.0])
obs, reward, done, info = env.step(action)
print(f"8. step()成功！報酬: {reward}")

print("9. レンダリングあり環境作成中...")
env2 = vnoid_rl_env.VnoidEnv(model_path, True)  # ← ここでクラッシュするはず
print("10. レンダリングあり環境作成成功！")

print("✅ すべてのテスト成功！")
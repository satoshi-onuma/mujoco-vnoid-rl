# ★★★ 学習用スクリプト：並列環境でOpenGL不使用 ★★★

import os
import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
import matplotlib.pyplot as plt

from my_humanoid_env import HumanoidVnoidEnv

print("=" * 70)
print("🚀 Vnoid Humanoid 学習スクリプト")
print("=" * 70)

# 設定パラメータ
NUM_WORKERS = 8   # 並列環境数

print("設定:")
print(f"  - 並列環境数: {NUM_WORKERS}")
print("  - 制御周波数: 60fps (sample_controller_mujocoと同じ)")
print("  - OpenGL: 無効（全環境で超高速動作）")
print("  - 学習アルゴリズム: PPO")
print("=" * 70)

# Ray初期化
ray.init(logging_level="ERROR")



# PPO設定
config = (
    PPOConfig()
    .environment(env="HumanoidVnoid-v0",
                 env_config={"enable_rendering": False}  # ★ 描画なし
                 )
    .env_runners(
        num_env_runners=NUM_WORKERS,
        rollout_fragment_length=50,  # ★ 適度な長さ
        sample_timeout_s=700.0,      # ★ 余裕を持たせる
    )
    .framework("torch")
    .training(
        train_batch_size=400,  # ★ 8×20×10 = 1600
        lr=1e-4,
        gamma=0.99,
        lambda_=0.95,
        clip_param=0.2,
        entropy_coeff=0.0,
        num_sgd_iter=20,
    )
    .resources(num_gpus=0)
    .debugging(seed=42)
)

# sgd_minibatch_sizeは別途設定
config.sgd_minibatch_size = 256

print("\n📚 アルゴリズム構築中...")
algo = config.build()
print("✅ アルゴリズム構築完了\n")

# 学習ループ
rewards = []
checkpoint_dir = os.path.abspath("./humanoid_vnoid_checkpoint")

print("🎓 学習開始...")
print("-" * 70)


for i in range(100):
    result = algo.train()
    
    # 報酬取得
    try:
        reward = result["env_runners"]["episode_return_mean"]
    except KeyError:
        reward = 0.0
        print("can't get reward")
    
    rewards.append(reward)
    
    print(f"Iteration {i+1:3d} | Mean Reward: {reward:8.2f}")

    # 10イテレーションごとにチェックポイント保存
    if (i + 1) % 10 == 0:
        checkpoint_result = algo.save(checkpoint_dir)
        print(f"💾 チェックポイント保存: {checkpoint_result.checkpoint.path}")

print("-" * 70)

# 最終チェックポイント保存
checkpoint_result = algo.save(checkpoint_dir)
print(f"\n💾 最終チェックポイント保存: {checkpoint_result.checkpoint.path}")

# 学習曲線を保存
plt.figure(figsize=(10, 6))
plt.plot(rewards, linewidth=2)
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Mean Episode Reward", fontsize=12)
plt.title("Vnoid Humanoid PPO Training (Parallel, No OpenGL)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_curve.png", dpi=150)
print(f"📊 学習曲線を保存: training_curve.png")

plt.show()

ray.shutdown()

print("\n" + "=" * 70)
print("✅ 学習完了！")
print("=" * 70)
print(f"📁 チェックポイント: {checkpoint_dir}")
print(f"📊 学習曲線: training_curve.png")
print("\n次のステップ:")
print("  python record_humanoid.py  # 学習済みポリシーを録画")
print("=" * 70)
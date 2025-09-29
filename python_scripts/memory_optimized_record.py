# ★★★ 最終版 train_humanoid_soft.py ★★★

import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig
import matplotlib.pyplot as plt
import os

from ray import tune
from my_humanoid_env import HumanoidVnoidEnv

# ★★★ 学習用環境：7つは軽量、1つは表示付き ★★★
def make_mixed_training_env(config):
    """
    worker_index=0のみレンダリング有効で進捗確認
    残りは超軽量で高速学習
    """
    worker_index = config.get("worker_index", 0)
    enable_rendering = (worker_index == 0)
    
    if enable_rendering:
        print(f"🎯 Worker {worker_index}: インタラクティブ表示で学習進捗を確認")
        return HumanoidVnoidEnv(enable_rendering=True)
    else:
        print(f"⚡ Worker {worker_index}: 高速学習モード")
        return HumanoidVnoidEnv(enable_rendering=False)

tune.register_env("HumanoidVnoidMixed-v0", make_mixed_training_env)

# Ray初期化
ray.init(logging_level="ERROR")

# PPO設定
config = (
    PPOConfig()
    .environment(env="HumanoidVnoidMixed-v0")
    .env_runners(
        # 8並列：1つだけ表示付き、7つは超高速
        num_env_runners=8,
        rollout_fragment_length=1000,
    )
    .framework("torch")
    .training(
        train_batch_size=8000,
        lr=1e-4,
        gamma=0.99,
        lambda_=0.95,
        clip_param=0.2,
        entropy_coeff=0.0,
    )
    .resources(
        num_gpus=0
    )
    .debugging(
        seed=42
    )
)

config.sgd_minibatch_size = 256
config.num_sgd_iter = 20

print("=" * 50)
print("🚀 混合モード学習を開始します")
print("   📺 Worker 0: インタラクティブ表示付き（進捗確認用）")
print("   ⚡ Worker 1-7: 超高速学習")
print("   🎮 Worker 0のウィンドウではマウスでカメラ操作可能")
print("   ⌨️  Backspaceキーでリセット")
print("=" * 50)

algo = config.build()

# 学習ループ
rewards = []
checkpoint_dir = os.path.abspath("./humanoid_vnoid_policy_checkpoint_dir")

for i in range(100):
    result = algo.train()
    try:
        reward = result["env_runners"]["episode_return_mean"]
    except KeyError:
        reward = 0.0
    rewards.append(reward)
    
    print(f"📈 Iteration {i+1:3d}: Mean Reward = {reward:.2f}")

    # 10イテレーションごとにチェックポイント保存
    if (i + 1) % 10 == 0:
        checkpoint_result = algo.save(checkpoint_dir)
        print(f"💾 チェックポイントを保存: {checkpoint_result.checkpoint.path}")
        print(f"🎯 Worker 0のウィンドウで学習の進捗を確認してください")

# 最終チェックポイント保存
checkpoint_result = algo.save(checkpoint_dir)
print(f"✅ 最終チェックポイント: {checkpoint_result.checkpoint.path}")

# 学習曲線を保存
plt.plot(rewards)
plt.xlabel("Iteration")
plt.ylabel("Mean Episode Reward")
plt.title("HumanoidVnoid-v0 PPO Training (Mixed Mode)")
plt.grid(True)
plt.savefig("humanoid_vnoid_training_curve_mixed.png")
plt.show()

ray.shutdown()

print("=" * 50)
print("🎉 混合モード学習が完了しました！")
print("📊 学習曲線: humanoid_vnoid_training_curve_mixed.png")
print("📁 チェックポイント: ./humanoid_vnoid_policy_checkpoint_dir")
print("🎥 次は memory_optimized_record.py で学習済みポリシーを録画できます")
print("=" * 50)
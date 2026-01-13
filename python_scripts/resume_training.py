import os
import gymnasium as gym
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray import tune
import matplotlib.pyplot as plt

from my_humanoid_env import HumanoidVnoidEnv

print("=" * 70)
print("🔄 チェックポイントから学習再開")
print("=" * 70)

# チェックポイント確認
checkpoint_dir = os.path.abspath("./humanoid_vnoid_checkpoint")
if not os.path.exists(checkpoint_dir):
    print(f"❌ エラー: チェックポイントが見つかりません: {checkpoint_dir}")
    exit(1)

print(f"📂 チェックポイント: {checkpoint_dir}")

# Ray初期化
ray.init(logging_level="ERROR")

# チェックポイントからアルゴリズムをロード
print("\n📥 チェックポイントをロード中...")
NUM_WORKERS = 16  # i9-7900X (20スレッド) の80%を活用
NUM_GPUS = 1      # TITAN V を活用

algo = Algorithm.from_checkpoint(
    checkpoint_dir,
    config_overrides={
        "num_env_runners": NUM_WORKERS,
        "num_gpus": NUM_GPUS,
        "num_cpus_per_env_runner": 1,
    }
)
print(f"並列数:")
print("✅ ロード完了")

# 学習再開
print("\n🎓 学習再開...")
print("-" * 70)

NUM_ADDITIONAL_ITERATIONS = 10  # 追加で何イテレーション実行するか

rewards = []
for i in range(NUM_ADDITIONAL_ITERATIONS):
    result = algo.train()
    
    try:
        reward_mean = result["env_runners"]["episode_return_mean"]
        reward_min = result["env_runners"]["episode_return_min"]
        reward_max = result["env_runners"]["episode_return_max"]
        num_episodes = result["env_runners"]["num_episodes"]
        episode_len_mean = result["env_runners"]["episode_len_mean"]
        
    except KeyError:
        reward_mean = 0.0
        reward_min=0.0
        reward_max = 0.0
        num_episodes = 0.0
    rewards.append(reward_mean)
    
    print(f"Iteration {i+1:3d} | Mean Reward: {reward_mean:8.2f}")
    print(f"episode_return_min {reward_min:8.2f}")
    print(f"episode_return_max {reward_max:8.2f}")
    print(f"num_episodes {num_episodes:8.2f}")
    print(f"  episode_len_mean: {episode_len_mean:.2f}")
    

print("-" * 70)

# 最終チェックポイント保存
checkpoint_result = algo.save(checkpoint_dir)
print(f"\n💾 チェックポイント更新: {checkpoint_result.checkpoint.path}")

ray.shutdown()

print("\n✅ 学習再開完了！")
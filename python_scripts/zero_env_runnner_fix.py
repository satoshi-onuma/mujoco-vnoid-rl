import os
import gymnasium as gym
import ray
from ray.rllib.algorithms.algorithm import Algorithm
import torch
import numpy as np
import imageio
from ray.rllib.models.torch.torch_distributions import TorchDiagGaussian

from ray import tune
from my_humanoid_env import HumanoidVnoidEnv

# 完全にクリーンな状態からスタート
try:
    ray.shutdown()
except:
    pass

# 最小限のRay初期化
print("Ray初期化中...")
ray.init(
    logging_level="ERROR",
    num_cpus=2,  # さらに削減
    num_gpus=0,
    object_store_memory=1000000000,  # 1GB
    ignore_reinit_error=True,
    include_dashboard=False,
    configure_logging=False,
)

print("Ray初期化完了")

tune.register_env("HumanoidVnoid-v0", lambda config: HumanoidVnoidEnv(**config))

checkpoint_dir = os.path.abspath("humanoid_vnoid_policy_checkpoint_dir") 
print(f"チェックポイント: {checkpoint_dir}")

# env_runnerを完全に0に設定
print("アルゴリズム読み込み中...")
try:
    algo = Algorithm.from_checkpoint(
        checkpoint_dir,
        config_overrides={
            "num_env_runners": 0,  # 完全に無効化
            "num_gpus": 0,
            "num_cpus_per_env_runner": 0,
            "evaluation_config": {
                "env_runners": {
                    "num_env_runners": 0,
                }
            },
            # 学習関連設定も最小に
            "train_batch_size": 1,
            "sgd_minibatch_size": 1,
            "rollout_fragment_length": 1,
        },
    )
    print("アルゴリズム読み込み成功")
except Exception as e:
    print(f"アルゴリズム読み込みエラー: {e}")
    ray.shutdown()
    exit(1)

# Rayアクターを使わずに直接環境作成
print("推論環境作成中...")
env = HumanoidVnoidEnv()  # gym.makeではなく直接作成
obs, info = env.reset(seed=42)
frames = []

module = algo.get_module("default_policy")

print("推論ループ開始...")
try:
    for i in range(1500):  # 短めに設定
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            fwd_outs = module.forward_inference({"obs": obs_tensor})
        
        action_dist = TorchDiagGaussian.from_logits(fwd_outs["action_dist_inputs"])
        action_tensor = action_dist.sample()
        action = action_tensor[0].cpu().numpy()
        
        obs, reward, terminated, truncated, info = env.step(action)

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if terminated or truncated:
            obs, info = env.reset()

        if i % 150 == 0:
            print(f"ステップ {i}: {len(frames)} フレーム")

except Exception as e:
    print(f"推論エラー: {e}")
finally:
    env.close()
    ray.shutdown()

# 動画保存
output_path = "humanoid_zero_runner.mp4"
if frames:
    print(f"保存中: {len(frames)} フレーム...")
    imageio.mimsave(output_path, frames, fps=30)
    print(f"保存完了: {output_path}")
else:
    print("フレームが空です")

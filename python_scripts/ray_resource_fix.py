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

# より詳細なRay初期化
print("Ray初期化中...")
ray.init(
    logging_level="ERROR",
    num_cpus=4,  # 12個中4個を明示的に使用
    num_gpus=0,
    object_store_memory=2000000000,  # 2GB
    ignore_reinit_error=True,
    include_dashboard=False,  # ダッシュボードを無効化してリソース節約
    configure_logging=False,
    _enable_object_reconstruction=False,
)

print("Ray初期化完了")
print(f"利用可能リソース: {ray.cluster_resources()}")

tune.register_env("HumanoidVnoid-v0", lambda config: HumanoidVnoidEnv(**config))

checkpoint_dir = os.path.abspath("humanoid_vnoid_policy_checkpoint_dir") 
print(f"チェックポイント: {checkpoint_dir}")

# より保守的な設定でアルゴリズム読み込み
print("アルゴリズム読み込み中...")
try:
    algo = Algorithm.from_checkpoint(
        checkpoint_dir,
        config_overrides={
            "num_env_runners": 1,  # 0ではなく1に設定
            "num_gpus": 0,
            "num_cpus_per_env_runner": 1,  # 明示的に1CPU割り当て
            "evaluation_config": {
                "env_runners": {
                    "num_env_runners": 0,  # 評価は無効
                }
            },
            "train_batch_size": 512,  # より小さなバッチサイズ
            "sgd_minibatch_size": 64,
        },
    )
    print("アルゴリズム読み込み成功")
except Exception as e:
    print(f"アルゴリズム読み込みエラー: {e}")
    ray.shutdown()
    exit(1)

# 推論実行
print("推論環境作成中...")
env = gym.make("HumanoidVnoid-v0", render_mode="rgb_array")
obs, info = env.reset(seed=42)
frames = []

module = algo.get_module("default_policy")

print("推論ループ開始...")
try:
    for i in range(2000):
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

        if i % 200 == 0:
            print(f"ステップ {i}: {len(frames)} フレーム")

except Exception as e:
    print(f"推論エラー: {e}")
finally:
    env.close()
    ray.shutdown()

# 動画保存
output_path = "humanoid_fixed.mp4"
if frames:
    print(f"保存中: {len(frames)} フレーム...")
    imageio.mimsave(output_path, frames, fps=30)
    print(f"保存完了: {output_path}")
else:
    print("フレームが空です")

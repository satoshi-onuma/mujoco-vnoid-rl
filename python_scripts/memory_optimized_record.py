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

tune.register_env("HumanoidVnoid-v0", lambda config: HumanoidVnoidEnv(**config))

checkpoint_dir = os.path.abspath("humanoid_vnoid_policy_checkpoint_dir") 

# あなたの成功した設定をそのまま使用
ray.init(logging_level="ERROR", num_cpus=os.cpu_count() - 1)

print(f"チェックポイントからポリシーをロード中: {checkpoint_dir}")

algo = Algorithm.from_checkpoint(
    checkpoint_dir,
    config_overrides={
        "num_env_runners": 0,
        "num_gpus": 0,
        "evaluation_config": {
            "env_runners": {
                "num_env_runners": 0,
            }
        }
    },
)

print("ロード完了。")

env = gym.make("HumanoidVnoid-v0", render_mode="rgb_array")
obs, info = env.reset(seed=42)

# メモリ対策：フレーム数だけ制限
frames = []
max_frames = 750  # 25秒の動画（メモリ節約）

module = algo.get_module("default_policy")

print(f"推論ループを開始（最大{max_frames}フレーム）...")

for i in range(2000):  # ステップ数は適度に制限
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        fwd_outs = module.forward_inference({"obs": obs_tensor})
    
    action_dist = TorchDiagGaussian.from_logits(fwd_outs["action_dist_inputs"])
    action_tensor = action_dist.sample()
    action = action_tensor[0].cpu().numpy()
    
    obs, reward, terminated, truncated, info = env.step(action)

    # フレーム数制限でメモリ使用量をコントロール
    if len(frames) < max_frames:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    if terminated or truncated:
        print(f"エピソード終了 (ステップ {i+1})。リセットします。")
        obs, info = env.reset()
    
    # 進捗表示
    if i % 200 == 0:
        print(f"ステップ {i}: {len(frames)} フレーム")
    
    # フレーム数上限に達したら終了
    if len(frames) >= max_frames:
        print(f"フレーム上限（{max_frames}）に達しました。")
        break

env.close()
ray.shutdown()

# 動画保存
output_path = "humanoid_vnoid_demo.mp4"
if not frames:
    raise RuntimeError("❌ 録画フレームが空です。保存できません。")

print(f"✅ {len(frames)} フレームを録画。保存中...")
imageio.mimsave(output_path, frames, fps=30)
print(f"🎥 録画完了: {output_path}")
print(f"録画時間: {len(frames)/30:.1f}秒")
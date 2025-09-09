import os
import gymnasium as gym
import ray
from ray.rllib.algorithms.algorithm import Algorithm
import torch
import numpy as np
import imageio
from ray.rllib.models.torch.torch_distributions import TorchDiagGaussian

# Ray RLlibにカスタム環境を正しく認識させるための設定
from ray import tune
# ★ Humanoid環境クラスをインポート
from my_humanoid_env import HumanoidVnoidEnv

# ★ Humanoid環境を登録
tune.register_env("HumanoidVnoid-v0", lambda config: HumanoidVnoidEnv(**config))

# ✅ チェックポイント保存先 (★ Humanoid用に変更)
checkpoint_dir = os.path.abspath("humanoid_vnoid_policy_checkpoint_dir") 

# ✅ Ray 初期化
ray.init(logging_level="ERROR") 

# ✅ 学習済みポリシーをロード
print(f"チェックポイントからポリシーをロード中: {checkpoint_dir}")
# ★★★ ここからが変更箇所 ★★★
# ★★★ こちらが正しい復元方法です ★★★
algo = Algorithm.from_checkpoint(
    checkpoint_dir,
    # 設定を上書きして、並列化をオフにする
    config_overrides={"num_env_runners": 0},
)

# ★★★ ここまで変更 ★★★

print("ロード完了。")

# ✅ MuJoCo 環境（録画モード）でカスタムXMLを読み込む (★ Humanoid環境を指定)
env = gym.make("HumanoidVnoid-v0", render_mode="rgb_array")
obs, info = env.reset(seed=42)
frames = []

# ✅ RLModule取得
module = algo.get_module("default_policy")

print("推論ループを開始...")
for i in range(1000):
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        fwd_outs = module.forward_inference({"obs": obs_tensor})
    
    # アクションを決定
    action_dist = TorchDiagGaussian.from_logits(fwd_outs["action_dist_inputs"])
    action_tensor = action_dist.sample()

    action = action_tensor[0].cpu().numpy()
    
    obs, reward, terminated, truncated, info = env.step(action)

    frame = env.render()
    if frame is not None:
        frames.append(frame)

    if terminated or truncated:
        print(f"エピソード終了 (ステップ {i+1})。リセットします。")
        obs, info = env.reset()

env.close()
ray.shutdown()

# ✅ フレーム確認と保存 (★ 出力ファイル名を変更)
output_path = "humanoid_vnoid_demo.mp4"
if not frames:
    raise RuntimeError("❌ 録画フレームが空です。保存できません。")
print(f"✅ {len(frames)} フレームを録画。保存中...")

imageio.mimsave(output_path, frames, fps=30)
print(f"🎥 録画完了: {output_path}")

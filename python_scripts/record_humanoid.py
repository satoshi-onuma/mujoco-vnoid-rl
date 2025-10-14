# ★★★ 録画用スクリプト：単一環境でOpenGL使用 ★★★

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

print("=" * 70)
print("🎥 Vnoid Humanoid 録画スクリプト")
print("=" * 70)

# 設定パラメータ
OUTPUT_FPS = 30       # 出力動画のFPS
TOTAL_STEPS = 500    # 録画するステップ数

print("設定:")
print("  - 制御周波数: 60fps (sample_controller_mujocoと同じ)")
print(f"  - 出力FPS: {OUTPUT_FPS}")
print("  - 並列環境数: なし（単一環境）")
print("  - OpenGL: 有効（録画のため）")
print("  - 推論モード: 学習済みポリシー使用")
print("=" * 70)

# チェックポイント確認
checkpoint_dir = os.path.abspath("./humanoid_vnoid_checkpoint")
if not os.path.exists(checkpoint_dir):
    print(f"\n❌ エラー: チェックポイントが見つかりません")
    print(f"パス: {checkpoint_dir}")
    print("\n先に学習を実行してください:")
    print("  python train_humanoid.py")
    exit(1)

print(f"\n📂 チェックポイント: {checkpoint_dir}")

# Ray初期化
print("\n🔧 Ray初期化中...")
ray.init(
    logging_level="ERROR",
    ignore_reinit_error=True,
)
print("✅ Ray初期化完了")

# 録画用環境を登録（OpenGL使用）
tune.register_env("HumanoidVnoidRecording-v0", 
                  lambda config: HumanoidVnoidEnv(enable_rendering=True, render_mode="rgb_array"))

# ★★★ 重要：学習時と同じ環境名も登録 ★★★
tune.register_env("HumanoidVnoid-v0",
                  lambda config: HumanoidVnoidEnv(enable_rendering=False))

# 学習済みアルゴリズムをロード
print("\n📥 学習済みポリシーをロード中...")
try:
    algo = Algorithm.from_checkpoint(
        checkpoint_dir,
        config_overrides={
            "num_env_runners": 0,  # 並列環境を無効化
            "num_gpus": 0,
            "evaluation_config": {
                "env_runners": {
                    "num_env_runners": 0,
                }
            }
        },
    )
    print("✅ ポリシーのロード完了")
except Exception as e:
    print(f"❌ ポリシーのロード失敗: {e}")
    ray.shutdown()
    exit(1)

# 録画用環境を作成（OpenGL有効）
print("\n🎬 録画環境を作成中...")
env = gym.make("HumanoidVnoidRecording-v0")
obs, info = env.reset(seed=42)
print("✅ 録画環境作成完了")

# RLModuleを取得
module = algo.get_module("default_policy")

# フレームバッファ
frames = []

print("\n🎥 録画開始...")
print("-" * 70)

try:
    for i in range(TOTAL_STEPS):

        if i < 2:
            action = np.zeros(5)  # 何もしない
        else:
            # 観測をテンソルに変換
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        
            # 推論
            with torch.no_grad():
                fwd_outs = module.forward_inference({"obs": obs_tensor})
        
            # アクション決定
            action_dist = TorchDiagGaussian.from_logits(fwd_outs["action_dist_inputs"])
            action_tensor = action_dist.sample()
            action = action_tensor[0].cpu().numpy()
        
        # ステップ実行
        obs, reward, terminated, truncated, info = env.step(action)

        # フレーム取得
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        # エピソード終了時にリセット
        if terminated or truncated:
            print(f"  エピソード終了 (ステップ {i+1})、リセット")
            obs, info = env.reset()

        # 進捗表示
        if (i + 1) % 200 == 0:
            print(f"  ステップ {i+1:4d} / {TOTAL_STEPS} | フレーム数: {len(frames)}")

except KeyboardInterrupt:
    print("\n⚠️ 録画が中断されました")
except Exception as e:
    print(f"\n❌ 録画エラー: {e}")
finally:
    print("-" * 70)
    env.close()
    ray.shutdown()

# 動画保存
output_path = "humanoid_demo.mp4"
print(f"\n💾 動画を保存中...")

if not frames:
    print("❌ エラー: フレームが空です。録画できませんでした。")
    exit(1)

print(f"  フレーム数: {len(frames)}")
print(f"  解像度: {frames[0].shape[1]}x{frames[0].shape[0]}")
print(f"  出力FPS: {OUTPUT_FPS}")

imageio.mimsave(output_path, frames, fps=OUTPUT_FPS)
print(f"✅ 動画保存完了: {output_path}")

print("\n" + "=" * 70)
print("🎉 録画完了！")
print("=" * 70)
print(f"📹 動画ファイル: {output_path}")
print(f"📊 総フレーム数: {len(frames)}")
print(f"⏱️  動画の長さ: {len(frames)/OUTPUT_FPS:.1f}秒")
print(f"🎮 制御周波数: 60fps (sample_controller_mujocoと同じ)")
print("=" * 70)

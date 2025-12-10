# ★★★ 録画用スクリプト：単一環境でOpenGL使用 ★★★

import os
import gymnasium as gym
import ray
from ray.rllib.algorithms.algorithm import Algorithm
import numpy as np
import imageio
from ray import tune
import torch
from ray.rllib.core.rl_module import RLModule
from pathlib import Path
from pprint import pprint

from my_humanoid_env import HumanoidVnoidEnv

print("=" * 70)
print("🎥 Vnoid Humanoid 録画スクリプト")
print("=" * 70)

# 設定パラメータ
OUTPUT_FPS = 30       # 出力動画のFPS
TOTAL_STEPS = 500    # 録画するステップ数

print("設定:")
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

# ★ os.path.join()でパスを結合
rl_module_path = os.path.join(
    checkpoint_dir,
    "learner_group",
    "learner",
    "rl_module",
    "default_policy"
)

print(f"\n📂 チェックポイント: {checkpoint_dir}")
print(f"📦 RLModule パス: {rl_module_path}")


# 学習済みアルゴリズムをロード
print("\n📥 学習済みポリシーをロード中...")
try:
    rl_module = RLModule.from_checkpoint(rl_module_path)
    print("\n📥 RLModuleをロード中...")
    
except Exception as e:
    print(f"❌ ポリシーのロード失敗: {e}")
    exit(1)

# 録画用環境を作成（OpenGL有効）
print("\n🎬 録画環境を作成中...")
env = HumanoidVnoidEnv(enable_rendering=True, render_mode="rgb_array")
obs, info = env.reset(seed=42)
print("✅ 録画環境作成完了")

# フレームバッファ
frames = []

print("\n🎥 録画開始...")
print("-" * 70)

try:
    for i in range(TOTAL_STEPS):

        '''if i < 2:
            action = np.zeros(6)  # 何もしない
        else:'''
             # ★ Rayで推論（explore=Falseで決定的行動）
        obs_batch = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
             # 推論（勾配計算不要）
        with torch.no_grad():
            model_outputs = rl_module.forward_inference({"obs": obs_batch})
            
        action_dist_params = model_outputs["action_dist_inputs"][0].numpy()

            
        """
            RL介入あり

            action = np.clip(
                action_dist_params[:6],  # 0=mean, 1=log(stddev), [0:1]=use mean, but keep shape=(1,)
                a_min=env.action_space.low,
                a_max=env.action_space.high,
            )

            RL介入なし
            action = np.zeros(6)
            """

        action = np.clip(
            action_dist_params[:6],  # 0=mean, 1=log(stddev), [0:1]=use mean, but keep shape=(1,)
            a_min=env.action_space.low,
            a_max=env.action_space.high,
        )
            
        # ステップ実行
        obs, reward, terminated, truncated, info , step_frames= env.step(action)

        # ★ 取得したフレームを全て追加
        frames.extend(step_frames)

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

    # ★★★ close()の前にログを取得 ★★★
    print("\n📊 制御データを取得中...")
    try:
        log = env.get_control_log()
    except Exception as e:
        print(f"⚠️ ログ取得エラー: {e}")
        log = None

    env.close()

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
# ★ ログをプロット（既に取得済み）
if log is not None and len(log.get('time', [])) > 0:
    print("\n📊 制御データをプロット中...")
    try:
        from plot_control_data import plot_control_analysis
        plot_control_analysis(log)
    except Exception as e:
        print(f"⚠️ グラフ生成エラー: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n⚠️ ログデータが空です（レンダリングが無効の可能性）")

print("\n✅ 全処理完了")
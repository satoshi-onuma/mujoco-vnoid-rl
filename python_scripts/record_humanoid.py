# ★★★ 録画用スクリプト：単一環境でOpenGL使用 ★★★

import os
import argparse
import json
import math
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
import csv

from my_humanoid_env import HumanoidVnoidEnv

print("=" * 70)
print("🎥 Vnoid Humanoid 録画スクリプト")
print("=" * 70)

# ★ 実験管理アプリ対応: コマンドライン引数（未指定なら従来と同じ動き）
parser = argparse.ArgumentParser(description="Vnoid Humanoid recording / evaluation")
parser.add_argument("--checkpoint-dir", type=str, default="./humanoid_vnoid_checkpoint")
parser.add_argument("--run-dir", type=str, default=None, help="出力先。未指定時はCWDに従来のファイル名で出力")
parser.add_argument("--terrain", type=str, default=None, choices=["hard", "soft", "debug", "random"],
                    help="歩行途中で切り替わる先の地盤（開始時は常に硬地盤）")
parser.add_argument("--total-steps", type=int, default=500)
parser.add_argument("--output-fps", type=int, default=30)
parser.add_argument("--no-rl-policy", action="store_true", help="学習済み方策を使わずゼロアクションで実行")
args = parser.parse_args()

# 設定パラメータ
# ↓ argparse化前の従来のデフォルト値（引数省略時はこの値のまま動く）
# OUTPUT_FPS = 30       # 出力動画のFPS
# TOTAL_STEPS = 500    # 録画するステップ数
# USE_RL_POLICY = True  # [一時] obs19 vs checkpoint16 不一致。reward検証時はFalse
OUTPUT_FPS = args.output_fps       # 出力動画のFPS
TOTAL_STEPS = args.total_steps    # 録画するステップ数
USE_RL_POLICY = not args.no_rl_policy  # [一時] obs19 vs checkpoint16 不一致。reward検証時はFalse

print("設定:")
print(f"  - 出力FPS: {OUTPUT_FPS}")
print("  - 並列環境数: なし（単一環境）")
print("  - OpenGL: 有効（録画のため）")
print("  - 推論モード: 学習済みポリシー使用")
print(f"  - 切り替え先地盤: {args.terrain or '(C++側デフォルト: soft)'}")
print("=" * 70)

# チェックポイント確認
checkpoint_dir = os.path.abspath(os.path.expanduser(args.checkpoint_dir))
if not os.path.exists(checkpoint_dir):
    print(f"\n❌ エラー: チェックポイントが見つかりません")
    print(f"パス: {checkpoint_dir}")
    print("\n先に学習を実行してください:")
    print("  python train_humanoid.py")
    exit(1)

# ★ 出力先: --run-dir 指定時は <terrain>_demo.mp4 等、未指定時は従来通り
if args.run_dir:
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.terrain}_" if args.terrain else ""
    output_path = str(run_dir / f"{prefix}demo.mp4")
    csv_output_path = str(run_dir / f"{prefix}recording_log.csv")
else:
    run_dir = Path.cwd()
    output_path = "humanoid_demo.mp4"
    csv_output_path = "recording_log.csv"

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
terrain_config = {"mode": args.terrain} if args.terrain else None
env = HumanoidVnoidEnv(enable_rendering=True, render_mode="rgb_array", terrain_config=terrain_config)
obs, info = env.reset(seed=42)
print("✅ 録画環境作成完了")

# フレームバッファ
frames = []

# CSV記録用のリスト
csv_data = []

print("\n🎥 録画開始...")
print("-" * 70)

try:
    for i in range(TOTAL_STEPS):

        #if i < 2:
            #action = np.zeros(2)  # 何もしない
        #else:
             # ★ Rayで推論（explore=Falseで決定的行動）
        if USE_RL_POLICY:
            obs_batch = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
             # 推論（勾配計算不要）
            with torch.no_grad():
                model_outputs = rl_module.forward_inference({"obs": obs_batch})
            
            action_dist_params = model_outputs["action_dist_inputs"][0].numpy()

            
        """
        RL介入あり

        action = np.clip(
            action_dist_params[:2],  # 0=mean, 1=log(stddev), [0:1]=use mean, but keep shape=(1,)
            a_min=env.action_space.low,
            a_max=env.action_space.high,
        )

            RL介入なし
            action = np.zeros(2)
        """

            

        if USE_RL_POLICY:
            action = np.clip(
                action_dist_params[:2],  # 0=mean, 1=log(stddev), [0:1]=use mean, but keep shape=(1,)
                a_min=env.action_space.low,
                a_max=env.action_space.high,
            )
        else:
            action = np.zeros(2)
        # ステップ実行
        obs, reward, terminated, truncated, info , step_frames= env.step(action)

        print(f"reward: {reward}")

        # ★ 取得したフレームを全て追加
        frames.extend(step_frames)

        # CSV用データを記録
        csv_row = {
            'step': i,
            'reward': reward,
            'terminated': terminated,
            'truncated': truncated,
        }
        # obsの各要素を列として追加
        for j, obs_val in enumerate(obs):
            csv_row[f'obs_{j}'] = obs_val
        csv_data.append(csv_row)

        # エピソード終了時にリセット
        if terminated or truncated:
            print(f"  エピソード終了 (ステップ {i+1})、リセット")
            obs, info = env.reset()
            break

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

# 動画保存
print(f"\n💾 動画を保存中...")

if not frames:
    print("❌ エラー: フレームが空です。録画できませんでした。")
    exit(1)

print(f"  フレーム数: {len(frames)}")
print(f"  解像度: {frames[0].shape[1]}x{frames[0].shape[0]}")
print(f"  出力FPS: {OUTPUT_FPS}")

imageio.mimsave(output_path, frames, fps=OUTPUT_FPS)
print(f"✅ 動画保存完了: {output_path}")

# CSVファイルを保存
print(f"\n💾 ログデータを保存中...")
if csv_data:
    with open(csv_output_path, 'w', newline='') as csvfile:
        fieldnames = list(csv_data[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    print(f"✅ ログ保存完了: {csv_output_path}")
else:
    print("⚠️ ログデータが空です")

# ★ 実験管理アプリ対応: control_log.csv から歩行距離を算出してJSONで出力
walk_distance = 0.0
control_log_src = Path.cwd() / "control_log.csv"
if control_log_src.exists():
    try:
        xs, ys = [], []
        with open(control_log_src, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "base_pos_x" in row:
                    xs.append(float(row["base_pos_x"]))
                    ys.append(float(row["base_pos_y"]))
        if xs:
            walk_distance = math.sqrt((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2)
        if args.run_dir:
            prefix = f"{args.terrain}_" if args.terrain else ""
            import shutil
            shutil.move(str(control_log_src), str(run_dir / f"{prefix}control_log.csv"))
    except Exception as e:
        print(f"⚠️ control_log からの歩行距離算出に失敗: {e}")

# eval_launcher が標準出力からJSONを読む
print("EVAL_RESULT_JSON:" + json.dumps({
    "terrain_mode": args.terrain or "default",
    "walk_distance": walk_distance,
    "video_path": str(output_path),
    "log_csv_path": str(csv_output_path),
    "checkpoint_dir": checkpoint_dir,
    "total_steps": TOTAL_STEPS,
    "num_frames": len(frames),
}))

print("\n" + "=" * 70)
print("🎉 録画完了！")
print("=" * 70)
print(f"📹 動画ファイル: {output_path}")
print(f"📊 総フレーム数: {len(frames)}")
print(f"⏱️  動画の長さ: {len(frames)/OUTPUT_FPS:.1f}秒")
print(f"🎮 制御周波数: 60fps (sample_controller_mujocoと同じ)")
print(f"📝 ログファイル: {csv_output_path} ({len(csv_data)}ステップ)")
print(f"📏 歩行距離: {walk_distance:.4f} m")
print("=" * 70)
print("\n💡 制御データのプロット:")
print("   CSVファイルが生成されました: control_log.csv")
print("   プロットする場合は以下を実行してください:")
print("   python python_scripts/plot_control_data.py")
print("=" * 70)

print("\n✅ 全処理完了")

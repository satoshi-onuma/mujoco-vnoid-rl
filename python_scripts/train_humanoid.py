# ★★★ 学習用スクリプト：並列環境でOpenGL不使用 ★★★

import os
import argparse
import json
import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from my_humanoid_env import HumanoidVnoidEnv
import csv
import time
from datetime import datetime
from pathlib import Path

print("=" * 70)
print("🚀 Vnoid Humanoid 学習スクリプト")
print("=" * 70)

# ★ 実験管理アプリ対応: コマンドライン引数（未指定なら従来と同じ値で動く）
parser = argparse.ArgumentParser(description="Vnoid Humanoid PPO training")
parser.add_argument("--run-id", type=str, default=None, help="実験ID。未指定時は adhoc_<timestamp>")
parser.add_argument("--run-dir", type=str, default=None, help="出力先。未指定時は ~/vnoid-experiments/runs/<run-id>/")
parser.add_argument("--w-track", type=float, default=1.0)
parser.add_argument("--w-act", type=float, default=0.1)
parser.add_argument("--w-healthy", type=float, default=1.0)
parser.add_argument("--tracking-sigma", type=float, default=0.02)
parser.add_argument("--terrain", type=str, default="soft", choices=["hard", "soft", "debug", "random"],
                    help="歩行途中で切り替わる先の地盤（開始時は常に硬地盤）")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--num-gpus", type=int, default=1)
parser.add_argument("--num-iterations", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# 設定パラメータ
# ↓ argparse化前の従来のデフォルト値（引数省略時はこの値のまま動く）
# NUM_WORKERS = 8   # 並列環境数
# NUM_GPUS = 1
NUM_WORKERS = args.num_workers   # 並列環境数
NUM_GPUS = args.num_gpus
ROLLOUT_FRAGMENT_LENGTH = 50

# ★ 出力先: ~/vnoid-experiments/runs/<run_id>/ に統一
run_id = args.run_id or f"adhoc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
if args.run_dir:
    run_dir = Path(args.run_dir).expanduser().resolve()
else:
    run_dir = Path.home() / "vnoid-experiments" / "runs" / run_id
run_dir.mkdir(parents=True, exist_ok=True)

reward_weights = {
    "w_track": args.w_track,
    "w_act": args.w_act,
    "w_healthy": args.w_healthy,
    "tracking_sigma": args.tracking_sigma,
}

print("設定:")
print(f"  - run_id: {run_id}")
print(f"  - run_dir: {run_dir}")
print(f"  - 並列環境数: {NUM_WORKERS}")
print(f"  - 切り替え先地盤: {args.terrain}")
print(f"  - reward_weights: {reward_weights}")
print("  - OpenGL: 無効（全環境で超高速動作）")
print("  - 学習アルゴリズム: PPO")
print("=" * 70)


# # Ray初期化
# if not ray.is_initialized():
#     ray.init(
#         address='auto',  # 既存クラスタを検出
#         logging_level="ERROR",
#         #ignore_reinit_error=True
#     )


# PPO設定
config = (
    PPOConfig()
    .environment(env="HumanoidVnoid-v0",
                 env_config={"enable_rendering": False,  # ★ 描画なし
                             "reward_weights": reward_weights,
                             "terrain_config": {"mode": args.terrain}}
                 )
    .env_runners(
        num_env_runners=NUM_WORKERS,
        rollout_fragment_length=ROLLOUT_FRAGMENT_LENGTH,  # ★ 適度な長さ
        sample_timeout_s=1000.0,      # ★ 余裕を持たせる
        #num_cpus_per_env_runner = 1.2, # ★ 環境ごとに1.5CPUを割り当て
    )
    .framework("torch")
    .training(
        train_batch_size=NUM_WORKERS*ROLLOUT_FRAGMENT_LENGTH, #16*50
        lr=args.lr,
        gamma=args.gamma,
        lambda_=0.95,
        clip_param=0.2,
        entropy_coeff=0.0,
        num_sgd_iter=20,  # ★ num_sgd_iterから変更
    )
    .resources(
        num_gpus=NUM_GPUS,  # ★ TITAN V を活用
    )
    .debugging(seed=args.seed)
)

# sgd_minibatch_sizeは別途設定
config.sgd_minibatch_size = 256 #16*50/4 =NUM_WORKERS*ROLLOUT_FRAGMENT_LENGTH/4

print("\n📚 アルゴリズム構築中...")
algo = config.build()
print("✅ アルゴリズム構築完了\n")

# 学習ループ
checkpoint_dir = str(run_dir / "checkpoint")

# 学習統計用のCSV（軽量）
training_csv_filename = str(run_dir / "training_stats.csv")
training_csv = open(training_csv_filename, 'w', newline='')
training_writer = csv.writer(training_csv)
training_writer.writerow(['iteration', 'reward_mean', 'episode_len_mean', 'sample_time_s', 'learn_time_s', 'elapsed_time_s', 'iter_time_s'])

start_time = time.time()

print("🎓 学習開始...")
print("-" * 70)


for i in range(args.num_iterations):
    result = algo.train()
    
    # 報酬取得
    try:
        reward = result["env_runners"]["episode_return_mean"]
    except KeyError:
        reward = 0.0
        print("can't get reward")
    
    # エピソード長取得
    try:
        episode_len = result["env_runners"]["episode_len_mean"]
    except KeyError:
        episode_len = 0.0
        print("can't get episode_len")
    
    # タイマー情報取得
    try:
        sample_time = result.get("timers", {}).get("sample_time_ms", 0.0) / 1000.0
    except (KeyError, AttributeError):
        sample_time = 0.0
    
    try:
        learn_time = result.get("timers", {}).get("learn_time_ms", 0.0) / 1000.0
    except (KeyError, AttributeError):
        learn_time = 0.0

    try:
        iter_time = result.get("time_this_iter_s", 0.0)
    except (KeyError, AttributeError):
        iter_time = 0.0
    
    elapsed = time.time() - start_time
    
    # ★ Iteration統計をCSVに書き込む（軽量）
    training_writer.writerow([i, reward, episode_len, sample_time, learn_time, elapsed, iter_time])
    training_csv.flush()  # 即座に書き込み

    # 10イテレーションごとにチェックポイント保存
    if (i + 1) % 10 == 0:
        checkpoint_result = algo.save(checkpoint_dir)
        print(f"💾 チェックポイント保存: {checkpoint_result.checkpoint.path}")

training_csv.close()

print("-" * 70)

# 最終チェックポイント保存
checkpoint_result = algo.save(checkpoint_dir)
print(f"\n💾 最終チェックポイント保存: {checkpoint_result.checkpoint.path}")

# ★ 実験管理アプリ対応: 結果サマリをJSONで書き出す
result_payload = {
    "run_id": run_id,
    "run_dir": str(run_dir),
    "checkpoint_dir": checkpoint_dir,
    "csv_path": training_csv_filename,
    "final_reward_mean": reward,
    "final_episode_len_mean": episode_len,
    "elapsed_time_s": elapsed,
    "num_iterations": args.num_iterations,
    "reward_weights": reward_weights,
    "terrain_mode": args.terrain,
    "hyperparams": {"lr": args.lr, "gamma": args.gamma, "num_workers": NUM_WORKERS,
                    "num_gpus": NUM_GPUS, "seed": args.seed},
}
with open(run_dir / "result.json", "w") as f:
    json.dump(result_payload, f, indent=2)

ray.shutdown()

print("\n" + "=" * 70)
print("✅ 学習完了！")
print("=" * 70)
print(f"📁 チェックポイント: {checkpoint_dir}")
print(f"📈 学習統計CSV: {training_csv_filename}")
print(f"📄 結果JSON: {run_dir / 'result.json'}")
print("\n次のステップ:")
print(f"  python record_humanoid.py --checkpoint-dir {checkpoint_dir}  # 学習済みポリシーを録画")
print(f"  python plot_training_stats.py {training_csv_filename}  # 学習統計を可視化")
print("=" * 70)

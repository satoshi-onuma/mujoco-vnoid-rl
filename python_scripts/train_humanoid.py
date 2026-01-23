# ★★★ 学習用スクリプト：並列環境でOpenGL不使用 ★★★

import os
import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from my_humanoid_env import HumanoidVnoidEnv
import csv
import time
from datetime import datetime

print("=" * 70)
print("🚀 Vnoid Humanoid 学習スクリプト")
print("=" * 70)

# 設定パラメータ
NUM_WORKERS = 16   # 並列環境数
NUM_GPUS = 1

print("設定:")
print(f"  - 並列環境数: {NUM_WORKERS}")
print("  - OpenGL: 無効（全環境で超高速動作）")
print("  - 学習アルゴリズム: PPO")
print("=" * 70)


# Ray初期化
if not ray.is_initialized():
    ray.init(
        address='auto',  # 既存クラスタを検出
        logging_level="ERROR",
        ignore_reinit_error=True
    )


# PPO設定
config = (
    PPOConfig()
    .environment(env="HumanoidVnoid-v0",
                 env_config={"enable_rendering": False}  # ★ 描画なし
                 )
    .env_runners(
        num_env_runners=NUM_WORKERS,
        rollout_fragment_length=50,  # ★ 適度な長さ
        sample_timeout_s=2000.0,      # ★ 余裕を持たせる
    )
    .framework("torch")
    .training(
        train_batch_size=800, #16*50
        lr=1e-4,
        gamma=0.99,
        lambda_=0.95,
        clip_param=0.2,
        entropy_coeff=0.0,
        num_epochs=12,  # ★ num_sgd_iterから変更
    )
    .resources(
        num_gpus=NUM_GPUS,  # ★ TITAN V を活用
    )
    .debugging(seed=42)
)

# sgd_minibatch_sizeは別途設定
config.sgd_minibatch_size = 256 #16*50/4

print("\n📚 アルゴリズム構築中...")
algo = config.build()
print("✅ アルゴリズム構築完了\n")

# 学習ループ
checkpoint_dir = os.path.abspath("./humanoid_vnoid_checkpoint")

# 学習統計用のCSV（軽量）
training_csv_filename = f"training_stats_.csv"
training_csv = open(training_csv_filename, 'w', newline='')
training_writer = csv.writer(training_csv)
training_writer.writerow(['iteration', 'reward_mean', 'sample_time_s', 'learn_time_s', 'elapsed_time_s', 'iter_time_s'])

start_time = time.time()

print("🎓 学習開始...")
print("-" * 70)


for i in range(50):
    result = algo.train()
    
    # 報酬取得
    try:
        reward = result["env_runners"]["episode_return_mean"]
    except KeyError:
        reward = 0.0
        print("can't get reward")
    
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
    training_writer.writerow([i, reward, sample_time, learn_time, elapsed,iter_time])
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


ray.shutdown()

print("\n" + "=" * 70)
print("✅ 学習完了！")
print("=" * 70)
print(f"📁 チェックポイント: {checkpoint_dir}")
print(f"📊 学習曲線: training_curve.png")
print(f"📈 学習統計CSV: {training_csv_filename}")
print("\n次のステップ:")
print("  python record_humanoid.py  # 学習済みポリシーを録画")
print(f"  python plot_training_stats.py {training_csv_filename}  # 学習統計を可視化")
print("=" * 70)
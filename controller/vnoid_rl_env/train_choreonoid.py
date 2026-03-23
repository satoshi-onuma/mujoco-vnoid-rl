"""
Choreonoid End-to-End RL学習スクリプト
RLlib PPOを使用
"""

import os
import sys
import subprocess
import time
import csv
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from choreonoid_env import ChoreonoidEnv, make_choreonoid_env

print("=" * 70)
print("🚀 Choreonoid End-to-End RL 学習スクリプト")
print("=" * 70)

# 設定パラメータ
NUM_WORKERS = 1  # まずは1環境でテスト
NUM_GPUS = 1
ROLLOUT_FRAGMENT_LENGTH = 100
MAX_EPISODE_STEPS = 1000

# Choreonoid実行ファイルのパス
CHOREONOID_BIN = os.path.expanduser("~/choreonoid/build/bin/choreonoid")
CHOREONOID_PROJECT = os.path.expanduser("~/choreonoid/ext/vnoid/project/vnoid_rl_project.cnoid")

print("設定:")
print(f"  - 並列環境数: {NUM_WORKERS}")
print(f"  - 学習アルゴリズム: PPO")
print(f"  - 最大エピソード長: {MAX_EPISODE_STEPS}")
print("=" * 70)

# Ray初期化
if not ray.is_initialized():
    ray.init(
        logging_level="ERROR",
    )

print("\n📚 環境登録...")
tune.register_env("ChoreonoidEnv-v0", make_choreonoid_env)

# PPO設定（MuJoCo版と同じハイパーパラメータ）
config = (
    PPOConfig()
    .environment(
        env="ChoreonoidEnv-v0",
        env_config={
            "max_episode_steps": MAX_EPISODE_STEPS
        }
    )
    .env_runners(
        num_env_runners=NUM_WORKERS,
        rollout_fragment_length=ROLLOUT_FRAGMENT_LENGTH,
        sample_timeout_s=1000.0,
    )
    .framework("torch")
    .training(
        train_batch_size=NUM_WORKERS * ROLLOUT_FRAGMENT_LENGTH,
        lr=1e-4,
        gamma=0.99,
        lambda_=0.95,
        clip_param=0.2,
        entropy_coeff=0.0,
        num_sgd_iter=20,
    )
    .resources(
        num_gpus=NUM_GPUS,
    )
    .debugging(seed=42)
)

config.sgd_minibatch_size = 256

print("\n🔧 Choreonoidプロセスを起動中...")
choreonoid_processes = []

for worker_id in range(NUM_WORKERS):
    env_id = worker_id
    env_vars = os.environ.copy()
    env_vars["ENV_ID"] = str(env_id)
    
    # Choreonoidをバックグラウンドで起動
    proc = subprocess.Popen(
        [CHOREONOID_BIN, CHOREONOID_PROJECT, "--start-simulation"],
        env=env_vars,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    choreonoid_processes.append(proc)
    print(f"  Choreonoid起動 (ENV_ID={env_id}, PID={proc.pid})")
    time.sleep(2)  # プロセス起動の間隔を空ける

print(f"✅ {NUM_WORKERS}個のChoreonoidプロセスを起動完了")
print("⏳ 5秒待機してからアルゴリズムを構築...")
time.sleep(5)

try:
    print("\n📚 アルゴリズム構築中...")
    algo = config.build()
    print("✅ アルゴリズム構築完了\n")
    
    # 学習ループ
    checkpoint_dir = os.path.abspath("./choreonoid_checkpoint")
    
    # 学習統計用のCSV
    training_csv_filename = "training_stats_choreonoid.csv"
    training_csv = open(training_csv_filename, 'w', newline='')
    training_writer = csv.writer(training_csv)
    training_writer.writerow([
        'iteration', 'reward_mean', 'episode_len_mean',
        'sample_time_s', 'learn_time_s', 'elapsed_time_s', 'iter_time_s'
    ])
    
    start_time = time.time()
    
    print("🎓 学習開始...")
    print("-" * 70)
    
    for i in range(100):
        result = algo.train()
        
        # 報酬取得
        try:
            reward = result["env_runners"]["episode_return_mean"]
        except KeyError:
            reward = 0.0
        
        # エピソード長取得
        try:
            episode_len = result["env_runners"]["episode_len_mean"]
        except KeyError:
            episode_len = 0.0
        
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
        
        # コンソール出力
        print(f"Iter {i:3d} | Reward: {reward:8.2f} | Len: {episode_len:6.1f} | Time: {iter_time:6.1f}s")
        
        # CSV書き込み
        training_writer.writerow([i, reward, episode_len, sample_time, learn_time, elapsed, iter_time])
        training_csv.flush()
        
        # 10イテレーションごとにチェックポイント保存
        if (i + 1) % 10 == 0:
            checkpoint_result = algo.save(checkpoint_dir)
            print(f"💾 チェックポイント保存: {checkpoint_result.checkpoint.path}")
    
    training_csv.close()
    
    print("-" * 70)
    
    # 最終チェックポイント保存
    checkpoint_result = algo.save(checkpoint_dir)
    print(f"\n💾 最終チェックポイント保存: {checkpoint_result.checkpoint.path}")

except KeyboardInterrupt:
    print("\n⚠️  学習が中断されました")

except Exception as e:
    print(f"\n❌ エラーが発生しました: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Choreonoidプロセスを終了
    print("\n🛑 Choreonoidプロセスを終了中...")
    for proc in choreonoid_processes:
        proc.terminate()
        proc.wait(timeout=5)
        print(f"  PID {proc.pid} 終了")
    
    ray.shutdown()
    
    print("\n" + "=" * 70)
    print("✅ 学習完了！")
    print("=" * 70)
    print(f"📁 チェックポイント: {checkpoint_dir}")
    print(f"📈 学習統計CSV: {training_csv_filename}")
    print("=" * 70)

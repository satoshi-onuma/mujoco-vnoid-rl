"""
Choreonoid End-to-End RL 学習スクリプト (Ray RLlib PPO)
=======================================================

## env_id の対応について

Choreonoid プロセスと Ray ワーカーは共有メモリ名 /vnoid_rl_shm_<env_id> で
1対1に対応する。この対応を保証するために:

  - num_envs_per_env_runner = 1 に固定する (ベクトル化環境を使わない)
  - Choreonoid を NUM_WORKERS 個、ENV_ID=0..NUM_WORKERS-1 で起動する
  - make_choreonoid_env は worker_index * 1 + 0 = worker_index を env_id とする

num_envs_per_env_runner > 1 にすると、Python 側の env_id 計算が
worker_index * num_envs + vector_index になり、事前に起動する
Choreonoid の数と整合が取れなくなるため、このスクリプトでは禁止する。
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
print("Choreonoid End-to-End RL 学習スクリプト")
print("=" * 70)

# ---------------------------------------------------------------------------
# 設定パラメータ
# ---------------------------------------------------------------------------

NUM_WORKERS            = 1    # 並列環境数 (Choreonoid プロセス数と一致させる)
NUM_GPUS               = 1
ROLLOUT_FRAGMENT_LENGTH = 100
MAX_EPISODE_STEPS      = 1000

# num_envs_per_env_runner は必ず 1 に固定する。
# 変更すると Choreonoid の ENV_ID と Python の env_id がずれる。
NUM_ENVS_PER_WORKER = 1

# Choreonoid 実行ファイルのパス
CHOREONOID_BIN     = os.path.expanduser("~/choreonoid/build/bin/choreonoid")
CHOREONOID_PROJECT = os.path.expanduser("~/choreonoid/ext/vnoid/project/vnoid_rl_project.cnoid")

print("設定:")
print(f"  - 並列環境数 (NUM_WORKERS): {NUM_WORKERS}")
print(f"  - ワーカーあたり環境数:       {NUM_ENVS_PER_WORKER} (固定)")
print(f"  - 学習アルゴリズム:           PPO")
print(f"  - 最大エピソード長:           {MAX_EPISODE_STEPS}")
print("=" * 70)

# ---------------------------------------------------------------------------
# Ray 初期化・環境登録
# ---------------------------------------------------------------------------

if not ray.is_initialized():
    ray.init(logging_level="ERROR")

tune.register_env("ChoreonoidEnv-v0", make_choreonoid_env)

# ---------------------------------------------------------------------------
# PPO 設定
# ---------------------------------------------------------------------------

config = (
    PPOConfig()
    .environment(
        env="ChoreonoidEnv-v0",
        env_config={
            "max_episode_steps":       MAX_EPISODE_STEPS,
            "num_envs_per_env_runner": NUM_ENVS_PER_WORKER,  # make_choreonoid_env に渡す
        }
    )
    .env_runners(
        num_env_runners=NUM_WORKERS,
        num_envs_per_env_runner=NUM_ENVS_PER_WORKER,  # 1 に固定
        rollout_fragment_length=ROLLOUT_FRAGMENT_LENGTH,
        sample_timeout_s=1000.0,
    )
    .framework("torch")
    .training(
        train_batch_size=NUM_WORKERS * ROLLOUT_FRAGMENT_LENGTH,
        minibatch_size=64,
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

# ---------------------------------------------------------------------------
# Choreonoid プロセス起動
# ---------------------------------------------------------------------------
# ENV_ID=worker_id で起動することで、Ray の worker_index と 1対1 に対応させる。

print("\nChoreonoid プロセスを起動中...")
choreonoid_processes = []

for worker_id in range(NUM_WORKERS):
    env_id    = worker_id  # worker_index と一致 (num_envs_per_worker=1 前提)
    env_vars  = os.environ.copy()
    env_vars["ENV_ID"] = str(env_id)

    proc = subprocess.Popen(
        [CHOREONOID_BIN, CHOREONOID_PROJECT, "--start-simulation"],
        env=env_vars,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    choreonoid_processes.append(proc)
    print(f"  Choreonoid 起動 (ENV_ID={env_id}, PID={proc.pid})")
    time.sleep(2)  # プロセス起動の間隔を空ける

print(f"{NUM_WORKERS} 個の Choreonoid プロセスを起動完了")
print("5 秒待機してからアルゴリズムを構築...")
time.sleep(5)

# ---------------------------------------------------------------------------
# 学習ループ
# ---------------------------------------------------------------------------

try:
    print("\nアルゴリズム構築中...")
    algo = config.build()
    print("アルゴリズム構築完了\n")

    checkpoint_dir        = os.path.abspath("./choreonoid_checkpoint")
    training_csv_filename = "training_stats_choreonoid.csv"

    with open(training_csv_filename, 'w', newline='') as training_csv:
        training_writer = csv.writer(training_csv)
        training_writer.writerow([
            'iteration', 'reward_mean', 'episode_len_mean',
            'sample_time_s', 'learn_time_s', 'elapsed_time_s', 'iter_time_s',
        ])

        start_time = time.time()
        print("学習開始...")
        print("-" * 70)

        for i in range(100):
            result = algo.train()

            reward      = result.get("env_runners", {}).get("episode_return_mean", 0.0)
            episode_len = result.get("env_runners", {}).get("episode_len_mean",    0.0)
            sample_time = result.get("timers",      {}).get("sample_time_ms",      0.0) / 1000.0
            learn_time  = result.get("timers",      {}).get("learn_time_ms",       0.0) / 1000.0
            iter_time   = result.get("time_this_iter_s", 0.0)
            elapsed     = time.time() - start_time

            print(f"Iter {i:3d} | Reward: {reward:8.2f} | Len: {episode_len:6.1f} | Time: {iter_time:6.1f}s")

            training_writer.writerow([i, reward, episode_len, sample_time, learn_time, elapsed, iter_time])
            training_csv.flush()

            if (i + 1) % 10 == 0:
                save_result = algo.save(checkpoint_dir)
                print(f"  チェックポイント保存: {save_result.checkpoint.path}")

    print("-" * 70)

    save_result = algo.save(checkpoint_dir)
    print(f"\n最終チェックポイント保存: {save_result.checkpoint.path}")

except KeyboardInterrupt:
    print("\n学習が中断されました")

except Exception as e:
    print(f"\nエラーが発生しました: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("\nChoreonoid プロセスを終了中...")
    for proc in choreonoid_processes:
        proc.terminate()
        proc.wait(timeout=5)
        print(f"  PID {proc.pid} 終了")

    ray.shutdown()

    print("\n" + "=" * 70)
    print("学習完了")
    print("=" * 70)
    if 'checkpoint_dir' in locals():
        print(f"  チェックポイント : {checkpoint_dir}")
    if 'training_csv_filename' in locals():
        print(f"  学習統計 CSV    : {training_csv_filename}")
    print("=" * 70)

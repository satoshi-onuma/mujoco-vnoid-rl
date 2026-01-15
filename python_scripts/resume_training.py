import os
import gymnasium as gym
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray import tune
import csv
import time
from datetime import datetime
import glob

from my_humanoid_env import HumanoidVnoidEnv

print("=" * 70)
print("🔄 チェックポイントから学習再開")
print("=" * 70)

# チェックポイント確認
checkpoint_dir = os.path.abspath("./humanoid_vnoid_checkpoint")
if not os.path.exists(checkpoint_dir):
    print(f"❌ エラー: チェックポイントが見つかりません: {checkpoint_dir}")
    exit(1)

print(f"📂 チェックポイント: {checkpoint_dir}")

# 既存のCSVファイルを検索
existing_csvs = sorted(glob.glob("training_stats_*.csv"))
start_iteration = 0
csv_mode = 'w'
training_csv_filename = f"training_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

if existing_csvs:
    latest_csv = existing_csvs[-1]
    print(f"\n📊 既存の統計ファイル発見: {latest_csv}")
    
    # 最後のイテレーション番号を取得
    try:
        with open(latest_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # ヘッダーをスキップ
            rows = list(reader)
            if rows:
                start_iteration = int(rows[-1][0]) + 1
                print(f"   最終イテレーション: {start_iteration - 1}")
                print(f"   → Iteration {start_iteration} から再開します")
                training_csv_filename = latest_csv
                csv_mode = 'a'
    except Exception as e:
        print(f"   ⚠️  CSVの読み込みに失敗: {e}")
        print(f"   → 新しいファイルを作成します")
else:
    print(f"\n📊 新しい統計ファイルを作成: {training_csv_filename}")

print(f"📈 統計出力先: {training_csv_filename}")

# Ray初期化
ray.init(logging_level="ERROR")

# チェックポイントからアルゴリズムをロード
print("\n📥 チェックポイントをロード中...")
NUM_WORKERS = 16  # i9-7900X (20スレッド) の80%を活用
NUM_GPUS = 1      # TITAN V を活用

algo = Algorithm.from_checkpoint(
    checkpoint_dir,
    config_overrides={
        "num_env_runners": NUM_WORKERS,
        "num_gpus": NUM_GPUS,
        "num_cpus_per_env_runner": 1,
    }
)
print(f"並列数:")
print("✅ ロード完了")

# CSVファイルを開く
training_csv = open(training_csv_filename, csv_mode, newline='')
training_writer = csv.writer(training_csv)

# 新規ファイルの場合はヘッダーを書き込む
if csv_mode == 'w':
    training_writer.writerow(['iteration', 'reward_mean', 'sample_time_s', 'learn_time_s', 'elapsed_time_s'])

# 学習再開
print("\n🎓 学習再開...")
print("-" * 70)

NUM_ADDITIONAL_ITERATIONS = 10  # 追加で何イテレーション実行するか

start_time = time.time()

for i in range(NUM_ADDITIONAL_ITERATIONS):
    result = algo.train()
    
    # 報酬取得
    try:
        reward_mean = result["env_runners"]["episode_return_mean"]
        reward_min = result["env_runners"]["episode_return_min"]
        reward_max = result["env_runners"]["episode_return_max"]
        num_episodes = result["env_runners"]["num_episodes"]
        episode_len_mean = result["env_runners"]["episode_len_mean"]
    except KeyError:
        reward_mean = 0.0
        reward_min = 0.0
        reward_max = 0.0
        num_episodes = 0
        episode_len_mean = 0.0
    
    # タイマー情報取得
    try:
        sample_time = result.get("timers", {}).get("sample_time_ms", 0.0) / 1000.0
    except (KeyError, AttributeError):
        sample_time = 0.0
    
    try:
        learn_time = result.get("timers", {}).get("learn_time_ms", 0.0) / 1000.0
    except (KeyError, AttributeError):
        learn_time = 0.0
    
    elapsed = time.time() - start_time
    
    # CSVに統計を書き込む
    current_iteration = start_iteration + i
    training_writer.writerow([current_iteration, reward_mean, sample_time, learn_time, elapsed])
    training_csv.flush()  # 即座に書き込み
    
    print(f"Iteration {current_iteration+1:3d} | Mean Reward: {reward_mean:8.2f} | Sample: {sample_time:6.2f}s | Learn: {learn_time:6.2f}s | Elapsed: {elapsed:8.2f}s")
    print(f"  Min/Max: {reward_min:8.2f} / {reward_max:8.2f} | Episodes: {num_episodes} | Avg Length: {episode_len_mean:.2f}")

training_csv.close()
print("-" * 70)

# 最終チェックポイント保存
checkpoint_result = algo.save(checkpoint_dir)
print(f"\n💾 チェックポイント更新: {checkpoint_result.checkpoint.path}")

ray.shutdown()

print("\n" + "=" * 70)
print("✅ 学習再開完了！")
print("=" * 70)
print(f"📁 チェックポイント: {checkpoint_dir}")
print(f"📈 学習統計CSV: {training_csv_filename}")
print(f"📊 統計の可視化: python plot_training_stats.py {training_csv_filename}")
print("=" * 70)
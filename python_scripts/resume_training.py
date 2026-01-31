import os
import gymnasium as gym
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray import tune
import csv
import time

from my_humanoid_env import HumanoidVnoidEnv

print("=" * 70)
print("🔄 チェックポイントから学習再開")
print("=" * 70)

# 設定パラメータ（train_humanoid.py と同じ）
NUM_WORKERS_CONFIG = 18
NUM_GPUS_CONFIG = 1
NUM_ADDITIONAL_ITERATIONS_CONFIG = 20

print("設定:")
print(f"  - 並列環境数: {NUM_WORKERS_CONFIG}")
print(f"  - GPU数: {NUM_GPUS_CONFIG}")
print(f"  - 追加イテレーション数: {NUM_ADDITIONAL_ITERATIONS_CONFIG}")
print("=" * 70)

# チェックポイント確認
checkpoint_dir = os.path.abspath("./humanoid_vnoid_checkpoint_id10")
if not os.path.exists(checkpoint_dir):
    print(f"❌ エラー: チェックポイントが見つかりません: {checkpoint_dir}")
    exit(1)

print(f"📂 チェックポイント: {checkpoint_dir}")

# CSVファイル名（train_humanoid.py と同じ）
training_csv_filename = "training_stats_.csv"

# Ray初期化
ray.init(logging_level="ERROR")

# チェックポイントからアルゴリズムをロード
print("\n📥 チェックポイントをロード中...")

algo = Algorithm.from_checkpoint(
    checkpoint_dir,
    config_overrides={
        "num_env_runners": NUM_WORKERS_CONFIG,
        "num_gpus": NUM_GPUS_CONFIG,
        "num_cpus_per_env_runner": 1.2,  # train_humanoid.py と同じ設定
    }
)
print(f"並列数: {NUM_WORKERS_CONFIG}")
print("✅ ロード完了")

# チェックポイントから正確なイテレーション番号を取得
checkpoint_iteration = algo.iteration
print(f"\n📍 チェックポイントのイテレーション: {checkpoint_iteration}")

# CSVとチェックポイントの整合性を確認
if os.path.exists(training_csv_filename):
    try:
        with open(training_csv_filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # ヘッダーをスキップ
            rows = list(reader)
            if rows:
                csv_last_iteration = int(rows[-1][0])
                print(f"📊 CSV最終イテレーション: {csv_last_iteration}")
                
                if csv_last_iteration >= checkpoint_iteration:
                    # CSVの方が進んでいる場合、チェックポイントに合わせる
                    start_iteration = checkpoint_iteration
                    csv_mode = 'a'
                    print(f"⚠️  CSVの方が進んでいます（CSV:{csv_last_iteration} > CP:{checkpoint_iteration}）")
                    print(f"   → チェックポイントに合わせて Iteration {start_iteration} から再開します")
                else:
                    # チェックポイントの方が進んでいる（通常はこちら）
                    start_iteration = checkpoint_iteration
                    csv_mode = 'a'
                    print(f"✅ チェックポイントから Iteration {start_iteration} で再開します")
    except Exception as e:
        print(f"⚠️  CSV読み込みエラー: {e}")
        start_iteration = checkpoint_iteration
        csv_mode = 'a'
else:
    start_iteration = checkpoint_iteration
    csv_mode = 'w'
    print(f"📊 新しいCSVファイルを作成: {training_csv_filename}")

# CSVファイルを開く
training_csv = open(training_csv_filename, csv_mode, newline='')
training_writer = csv.writer(training_csv)

# 新規ファイルの場合はヘッダーを書き込む
if csv_mode == 'w':
    training_writer.writerow(['iteration', 'reward_mean', 'episode_len_mean', 'sample_time_s', 'learn_time_s', 'elapsed_time_s', 'iter_time_s'])

# 学習再開
print("\n🎓 学習再開...")
print("-" * 70)

start_time = time.time()

for i in range(NUM_ADDITIONAL_ITERATIONS_CONFIG):
    result = algo.train()
    
    # 報酬取得（train_humanoid.py と同じ変数名）
    try:
        reward = result["env_runners"]["episode_return_mean"]
        reward_min = result["env_runners"]["episode_return_min"]
        reward_max = result["env_runners"]["episode_return_max"]
        num_episodes = result["env_runners"]["num_episodes"]
    except KeyError:
        reward = 0.0
        reward_min = 0.0
        reward_max = 0.0
        num_episodes = 0
    
    # エピソード長取得（train_humanoid.py と同じ変数名）
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
    
    # CSVに統計を書き込む（train_humanoid.py と完全に同じフォーマット）
    current_iteration = start_iteration + i
    training_writer.writerow([current_iteration, reward, episode_len, sample_time, learn_time, elapsed, iter_time])
    training_csv.flush()  # 即座に書き込み
    
    # 進捗表示（resume時は進捗を確認しやすいように表示）
    print(f"Iteration {current_iteration:3d} | Mean Reward: {reward:8.2f} | Ep Length: {episode_len:6.1f} | Sample: {sample_time:6.2f}s | Learn: {learn_time:6.2f}s | Iter: {iter_time:6.2f}s | Elapsed: {elapsed:8.2f}s")
    print(f"  Min/Max: {reward_min:8.2f} / {reward_max:8.2f} | Episodes: {num_episodes}")
    
    # 10イテレーションごとにチェックポイント保存（train_humanoid.py と同じ）
    if (current_iteration + 1) % 10 == 0:
        checkpoint_result = algo.save(checkpoint_dir)
        print(f"💾 チェックポイント保存: {checkpoint_result.checkpoint.path}")

training_csv.close()
print("-" * 70)

# 最終チェックポイント保存
checkpoint_result = algo.save(checkpoint_dir)
print(f"\n💾 最終チェックポイント保存: {checkpoint_result.checkpoint.path}")

ray.shutdown()

print("\n" + "=" * 70)
print("✅ 学習再開完了！")
print("=" * 70)
print(f"📁 チェックポイント: {checkpoint_dir}")
print(f"📈 学習統計CSV: {training_csv_filename}")
print("\n次のステップ:")
print("  python record_humanoid.py  # 学習済みポリシーを録画")
print(f"  python plot_training_stats.py {training_csv_filename}  # 学習統計を可視化")
print("=" * 70)
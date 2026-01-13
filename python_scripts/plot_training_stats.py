"""
学習統計データを可視化するスクリプト
CSVファイルから読み込んで報酬変化のグラフと各計算時間をターミナルに出力
"""

import csv
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from datetime import datetime

matplotlib.use('Agg')  # GUIなしバックエンドを明示的に設定

def load_training_stats(csv_filename):
    """CSVファイルから学習統計データを読み込む"""
    if not os.path.exists(csv_filename):
        print(f"❌ CSVファイルが見つかりません: {csv_filename}")
        return None
    
    iterations = []
    rewards = []
    sample_times = []
    learn_times = []
    elapsed_times = []
    
    with open(csv_filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            iterations.append(int(row['iteration']))
            rewards.append(float(row['reward_mean']))
            sample_times.append(float(row['sample_time_s']))
            learn_times.append(float(row['learn_time_s']))
            elapsed_times.append(float(row['elapsed_time_s']))
    
    if len(iterations) == 0:
        print("⚠️ CSVファイルが空です")
        return None
    
    print(f"✅ CSVファイル読み込み完了: {csv_filename} ({len(iterations)} イテレーション)")
    
    return {
        'iterations': iterations,
        'rewards': rewards,
        'sample_times': sample_times,
        'learn_times': learn_times,
        'elapsed_times': elapsed_times
    }

def print_time_statistics(data):
    """各計算時間をターミナルに出力"""
    print("\n" + "=" * 70)
    print("⏱️  計算時間統計")
    print("=" * 70)
    
    sample_times = np.array(data['sample_times'])
    learn_times = np.array(data['learn_times'])
    elapsed_times = np.array(data['elapsed_times'])
    
    # Sample Time統計
    print("\n📊 Sample Time (サンプリング時間):")
    print(f"  平均: {np.mean(sample_times):8.3f} s")
    print(f"  最小: {np.min(sample_times):8.3f} s")
    print(f"  最大: {np.max(sample_times):8.3f} s")
    print(f"  合計: {np.sum(sample_times):8.3f} s")
    
    # Learn Time統計
    print("\n📚 Learn Time (学習時間):")
    print(f"  平均: {np.mean(learn_times):8.3f} s")
    print(f"  最小: {np.min(learn_times):8.3f} s")
    print(f"  最大: {np.max(learn_times):8.3f} s")
    print(f"  合計: {np.sum(learn_times):8.3f} s")
    
    # Elapsed Time統計
    print("\n⏰ Elapsed Time (経過時間):")
    print(f"  開始: {elapsed_times[0]:8.3f} s")
    print(f"  終了: {elapsed_times[-1]:8.3f} s")
    print(f"  総時間: {elapsed_times[-1]:8.3f} s ({elapsed_times[-1]/60:.2f} 分)")
    
    # 1イテレーションあたりの平均時間
    if len(elapsed_times) > 1:
        iteration_durations = np.diff(elapsed_times)
        print(f"\n🔄 1イテレーションあたりの平均時間:")
        print(f"  平均: {np.mean(iteration_durations):8.3f} s")
        print(f"  最小: {np.min(iteration_durations):8.3f} s")
        print(f"  最大: {np.max(iteration_durations):8.3f} s")
    
    # 時間の内訳
    total_sample = np.sum(sample_times)
    total_learn = np.sum(learn_times)
    total_elapsed = elapsed_times[-1]
    other_time = total_elapsed - total_sample - total_learn
    
    print(f"\n📈 時間の内訳:")
    print(f"  Sample Time: {total_sample:8.3f} s ({total_sample/total_elapsed*100:5.1f}%)")
    print(f"  Learn Time:  {total_learn:8.3f} s ({total_learn/total_elapsed*100:5.1f}%)")
    print(f"  その他:      {other_time:8.3f} s ({other_time/total_elapsed*100:5.1f}%)")
    print(f"  合計:        {total_elapsed:8.3f} s (100.0%)")
    
    print("=" * 70)

def plot_training_stats(data, output_filename="training_stats_plot.png"):
    """報酬変化のグラフと各計算時間のグラフを生成"""
    
    iterations = data['iterations']
    rewards = data['rewards']
    sample_times = data['sample_times']
    learn_times = data['learn_times']
    elapsed_times = data['elapsed_times']
    
    print(f"\n📊 グラフ生成中... ({len(iterations)} イテレーション)")
    
    # 図1: 報酬変化と計算時間（3パネル）
    fig1 = plt.figure(figsize=(15, 10))
    
    # --- パネル1: 報酬変化 ---
    plt.subplot(3, 1, 1)
    plt.plot(iterations, rewards, 'b-', linewidth=2, label='Mean Reward')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Mean Episode Reward', fontsize=12)
    plt.title('Training Reward Progress', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 統計情報をテキストで追加
    max_reward = max(rewards)
    max_idx = rewards.index(max_reward)
    final_reward = rewards[-1]
    plt.axhline(y=max_reward, color='r', linestyle='--', alpha=0.5, label=f'Max: {max_reward:.2f}')
    plt.axhline(y=final_reward, color='g', linestyle='--', alpha=0.5, label=f'Final: {final_reward:.2f}')
    plt.legend()
    
    # --- パネル2: Sample Time と Learn Time ---
    plt.subplot(3, 1, 2)
    plt.plot(iterations, sample_times, 'g-', linewidth=2, label='Sample Time', alpha=0.7)
    plt.plot(iterations, learn_times, 'r-', linewidth=2, label='Learn Time', alpha=0.7)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Time [s]', fontsize=12)
    plt.title('Computation Time per Iteration', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # --- パネル3: 累積経過時間 ---
    plt.subplot(3, 1, 3)
    plt.plot(iterations, elapsed_times, 'purple', linewidth=2, label='Elapsed Time')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Elapsed Time [s]', fontsize=12)
    plt.title('Cumulative Elapsed Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 経過時間を分単位でも表示
    ax3 = plt.gca()
    ax3_sec = ax3.twinx()
    ax3_sec.set_ylabel('Elapsed Time [min]', fontsize=12)
    ax3_sec.plot(iterations, np.array(elapsed_times) / 60, 'purple', linewidth=0, alpha=0)
    ax3_sec.set_ylim(np.array(ax3.get_ylim()) / 60)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    print(f"✅ グラフ保存完了: {output_filename}")
    
    # 図2: 時間の内訳（積み上げ棒グラフ）
    fig2 = plt.figure(figsize=(15, 6))
    
    # 各イテレーションの時間内訳を計算
    iteration_durations = []
    for i in range(len(iterations)):
        if i == 0:
            duration = elapsed_times[0]
        else:
            duration = elapsed_times[i] - elapsed_times[i-1]
        iteration_durations.append(duration)
    
    # 積み上げ棒グラフ用のデータ
    sample_array = np.array(sample_times)
    learn_array = np.array(learn_times)
    other_array = np.array(iteration_durations) - sample_array - learn_array
    
    plt.subplot(1, 2, 1)
    x = np.arange(len(iterations))
    width = 0.8
    plt.bar(x, sample_array, width, label='Sample Time', color='green', alpha=0.7)
    plt.bar(x, learn_array, width, bottom=sample_array, label='Learn Time', color='red', alpha=0.7)
    plt.bar(x, other_array, width, bottom=sample_array + learn_array, label='Other Time', color='gray', alpha=0.7)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Time [s]', fontsize=12)
    plt.title('Time Breakdown per Iteration (Stacked)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # 移動平均で滑らかに
    window_size = min(10, len(iterations) // 5) if len(iterations) > 5 else 1
    if window_size > 1:
        rewards_smooth = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        iterations_smooth = iterations[window_size-1:]
        
        plt.subplot(1, 2, 2)
        plt.plot(iterations, rewards, 'b-', linewidth=1, alpha=0.3, label='Raw')
        plt.plot(iterations_smooth, rewards_smooth, 'b-', linewidth=2, label=f'Moving Average (window={window_size})')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Mean Episode Reward', fontsize=12)
        plt.title('Reward Progress (Smoothed)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    output_filename2 = output_filename.replace('.png', '_breakdown.png')
    plt.savefig(output_filename2, dpi=150)
    print(f"✅ グラフ保存完了: {output_filename2}")
    
    plt.close('all')

if __name__ == "__main__":
    print("=" * 70)
    print("📊 学習統計データの可視化")
    print("=" * 70)
    
    # CSVファイル名を取得（コマンドライン引数から、またはデフォルト）
    if len(sys.argv) > 1:
        csv_filename = sys.argv[1]
    else:
        # 最新のtraining_stats_*.csvファイルを探す
        csv_files = [f for f in os.listdir('.') if f.startswith('training_stats_') and f.endswith('.csv')]
        if csv_files:
            csv_files.sort(reverse=True)
            csv_filename = csv_files[0]
            print(f"📂 最新のCSVファイルを使用: {csv_filename}")
        else:
            print("❌ CSVファイルが見つかりません")
            print("使用方法: python plot_training_stats.py <training_stats_*.csv>")
            sys.exit(1)
    
    # CSVファイルから読み込む
    data = load_training_stats(csv_filename)
    
    if data is not None:
        # ターミナルに計算時間統計を出力
        print_time_statistics(data)
        
        # グラフを生成
        output_filename = csv_filename.replace('.csv', '_plot.png')
        plot_training_stats(data, output_filename)
        
        print("\n" + "=" * 70)
        print("✅ 可視化完了！")
        print("=" * 70)
    else:
        print("⚠️ データの読み込みに失敗しました")
        sys.exit(1)

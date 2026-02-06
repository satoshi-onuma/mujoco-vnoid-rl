"""
卒論用：学習曲線プロット（Iteration vs. Mean Reward）
工学部論文の慣習に従った形式で出力
"""

import csv
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 論文用の設定
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 150
plt.rcParams['lines.linewidth'] = 1.5

def load_training_stats(csv_filename):
    """CSVファイルから学習統計データを読み込む"""
    if not os.path.exists(csv_filename):
        print(f"Error: CSV file not found: {csv_filename}")
        return None
    
    iterations = []
    rewards = []
    
    with open(csv_filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            iterations.append(int(row['iteration']))
            rewards.append(float(row['reward_mean']))
    
    if len(iterations) == 0:
        print("Warning: CSV file is empty")
        return None
    
    print(f"Loaded data: {csv_filename} ({len(iterations)} iterations)")
    
    return {
        'iterations': np.array(iterations),
        'rewards': np.array(rewards)
    }

def plot_thesis_style(data, output_filename="training_curve.pdf", 
                      title=None, xlabel="Iteration", ylabel="Mean Episode Reward",
                      figsize=(10, 4)):
    """
    論文用のシンプルな学習曲線をプロット
    
    Parameters:
    -----------
    data : dict
        'iterations' と 'rewards' のキーを持つ辞書
    output_filename : str
        出力ファイル名（.pdf, .png, .epsなど）
    title : str or None
        グラフのタイトル（Noneの場合は表示しない）
    xlabel : str
        x軸ラベル
    ylabel : str
        y軸ラベル
    figsize : tuple
        図のサイズ (width, height) in inches
    """
    
    iterations = data['iterations']
    rewards = data['rewards']
    
    print(f"\nGenerating thesis-style plot...")
    print(f"  Iterations: {len(iterations)}")
    print(f"  Max reward: {np.max(rewards):.2f}")
    print(f"  Final reward: {rewards[-1]:.2f}")
    print(f"  Mean reward: {np.mean(rewards):.2f}")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 生データをプロット
    ax.plot(iterations, rewards, 'b-', linewidth=1.5, label='Training reward')
    
    # 軸ラベルとタイトル
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # 軸目盛りを内向きに
    ax.tick_params(direction='in')
    
    # レイアウト調整
    plt.tight_layout()
    
    # 保存
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_filename}")
    
    plt.close()

if __name__ == "__main__":
    print("=" * 70)
    print("Thesis-style Training Curve Plot Generator")
    print("=" * 70)
    
    # CSVファイル名を取得
    if len(sys.argv) > 1:
        csv_filename = sys.argv[1]
    else:
        # 最新のtraining_stats_*.csvファイルを探す
        csv_files = [f for f in os.listdir('.') if f.startswith('training_stats_') and f.endswith('.csv')]
        if csv_files:
            csv_files.sort(reverse=True)
            csv_filename = csv_files[0]
            print(f"Using latest CSV file: {csv_filename}")
        else:
            print("Error: No CSV file found")
            print("Usage: python plot_training_stats_thesis.py <training_stats_*.csv>")
            sys.exit(1)
    
    # データ読み込み
    data = load_training_stats(csv_filename)
    
    if data is None:
        print("Error: Failed to load data")
        sys.exit(1)
    
    # 出力ファイル名
    output_filename = csv_filename.replace('.csv', '_thesis.pdf')
    
    # シンプルな学習曲線（生データのみ）
    print("\nGenerating thesis plot...")
    plot_thesis_style(
        data, 
        output_filename=output_filename,
        title=None  # タイトルなし（キャプションで説明するため）
    )
    
    print("\n" + "=" * 70)
    print("✓ Plot generated successfully!")
    print("=" * 70)
    print(f"\nGenerated file: {output_filename}")
    print("=" * 70)

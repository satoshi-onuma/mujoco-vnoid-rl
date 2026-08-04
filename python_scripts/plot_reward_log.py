"""
[VNOID_REWARD_LOG_DEBUG] reward_tracking 検証用の一時プロットスクリプト
不要になったらこのファイルごと削除すること
"""

import matplotlib.pyplot as plt
import sys
import os
import csv
import numpy as np


def load_csv_log(csv_filename="reward_log.csv"):
    """CSVファイルからログデータを読み込む"""
    if not os.path.exists(csv_filename):
        print(f"❌ CSVファイルが見つかりません: {csv_filename}")
        return None

    log = {}

    with open(csv_filename, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

        if len(rows) == 0:
            print("⚠️ CSVファイルが空です")
            return None

        for key in rows[0].keys():
            log[key] = [float(row[key]) for row in rows]

    n = len(log.get('step', []))
    print(f"✅ CSVファイル読み込み完了: {csv_filename} ({n} steps)")
    return log


def plot_reward_analysis(log, output_prefix="reward_log"):
    """報酬内訳をプロット"""
    step = log['step']
    n = len(step)
    if n == 0:
        print("⚠️ ログデータが空です")
        return

    print(f"\n📊 グラフ生成中... ({n} steps)")

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Reward Tracking Analysis', fontsize=14)

    # --- パネル1: 指令 vs 実際変位（body frame） ---
    ax = axes[0, 0]
    ax.plot(step, log['cmd_stride'], 'r--', label='cmd stride', linewidth=2)
    ax.plot(step, log['ex_disp'], 'r-', label='ex_disp (actual)', alpha=0.8)
    ax.plot(step, log['cmd_sway'], 'g--', label='cmd sway', linewidth=2)
    ax.plot(step, log['ey_disp'], 'g-', label='ey_disp (actual)', alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('[m]')
    ax.set_title('Command vs Actual Displacement (body frame)')
    ax.legend(fontsize=8)
    ax.grid(True)

    # --- パネル2: 追従誤差 ex, ey ---
    ax = axes[0, 1]
    ax.plot(step, log['ex'], 'r-', label='ex (stride error)')
    ax.plot(step, log['ey'], 'g-', label='ey (sway error)')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('[m]')
    ax.set_title('Tracking Error')
    ax.legend(fontsize=8)
    ax.grid(True)

    # --- パネル3: tracking 報酬成分 ---
    ax = axes[1, 0]
    ax.plot(step, log['tracking'], 'b-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('tracking')
    ax.set_title('tracking = exp(-(ex²+ey²)/sigma)')
    ax.set_ylim(0, 1.05)
    ax.grid(True)

    # --- パネル4: 報酬合算内訳 ---
    ax = axes[1, 1]
    ax.plot(step, log['total'], 'k-', linewidth=2, label='total')
    ax.plot(step, log['tracking'], 'b-', alpha=0.7, label='tracking')
    ax.plot(step, log['healthy'], 'g-', alpha=0.7, label='healthy')
    ax.plot(step, log['action_penalty'], 'r-', alpha=0.7, label='action_penalty')
    ax.set_xlabel('Step')
    ax.set_ylabel('reward')
    ax.set_title('Reward Components')
    ax.legend(fontsize=8)
    ax.grid(True)

    # --- パネル5: ワールド座標変位 dx, dy ---
    ax = axes[2, 0]
    ax.plot(step, log['dx'], 'r-', label='dx (world)')
    ax.plot(step, log['dy'], 'g-', label='dy (world)')
    ax.set_xlabel('Step')
    ax.set_ylabel('[m]')
    ax.set_title('World-frame Displacement per Step')
    ax.legend(fontsize=8)
    ax.grid(True)

    # --- パネル6: yaw 基準 ---
    ax = axes[2, 1]
    ax.plot(step, log['step_start_yaw'], 'b-', label='step_start_yaw')
    ax.plot(step, log['base_yaw'], 'r--', alpha=0.7, label='base_yaw (after step)')
    ax.set_xlabel('Step')
    ax.set_ylabel('[rad]')
    ax.set_title('Yaw Reference')
    ax.legend(fontsize=8)
    ax.grid(True)

    plt.tight_layout()

    png_path = f"{output_prefix}_analysis.png"
    pdf_path = f"{output_prefix}_analysis.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    print(f"✅ 保存: {png_path}, {pdf_path}")

    # 統計サマリ
    print("\n--- 統計サマリ ---")
    for key in ['ex', 'ey', 'ex_disp', 'tracking', 'total']:
        arr = np.array(log[key])
        print(f"  {key:16s}: mean={arr.mean():.6f}  std={arr.std():.6f}  "
              f"min={arr.min():.6f}  max={arr.max():.6f}")


if __name__ == "__main__":
    print("=" * 70)
    print("📊 reward_log プロット")
    print("=" * 70)

    csv_filename = sys.argv[1] if len(sys.argv) > 1 else "reward_log.csv"
    print(f"📂 CSVファイル: {csv_filename}")

    log = load_csv_log(csv_filename)

    if log is not None:
        prefix = os.path.splitext(os.path.basename(csv_filename))[0]
        plot_reward_analysis(log, output_prefix=prefix)

    print("=" * 70)

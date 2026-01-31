"""
卒論用：CoM位置のプロットスクリプト
工学論文の慣習に則り、実測値と所望値を比較
"""

import matplotlib.pyplot as plt
import sys
import os
import csv
import numpy as np

# 論文用のスタイル設定
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 150
plt.rcParams['lines.linewidth'] = 1.5

def load_csv_log(csv_filename="control_log.csv"):
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
        
        # 各カラムをリストに変換
        for key in rows[0].keys():
            log[key] = [float(row[key]) for row in rows]
    
    print(f"✅ CSVファイル読み込み完了: {csv_filename} ({len(log.get('time', []))} サンプル)")
    return log


def plot_com_position_thesis(log, output_dir=".", time_range=None):
    """
    卒論用：CoM位置のプロット（X, Y, Top Viewの3枚）
    実測値と所望値を比較
    
    Parameters:
    -----------
    log : dict
        CSVから読み込んだログデータ
    output_dir : str
        出力ディレクトリ
    time_range : tuple or None
        プロット範囲 (t_min, t_max)。Noneの場合は全範囲
    """
    
    if len(log['time']) == 0:
        print("⚠️ ログデータが空です")
        return
    
    print(f"\n📊 卒論用グラフ生成中... ({len(log['time'])} サンプル)")
    
    # 時間範囲の設定
    time = np.array(log['time'])
    if time_range is not None:
        t_min, t_max = time_range
        mask = (time >= t_min) & (time <= t_max)
        time = time[mask]
        com_pos_x = np.array(log['com_pos_x'])[mask]
        com_pos_y = np.array(log['com_pos_y'])[mask]
        com_pos_z = np.array(log['com_pos_z'])[mask]
        com_pos_ref_x = np.array(log['com_pos_ref_x'])[mask]
        com_pos_ref_y = np.array(log['com_pos_ref_y'])[mask]
        com_pos_ref_z = np.array(log['com_pos_ref_z'])[mask]
    else:
        t_min, t_max = time[0], time[-1]
        com_pos_x = np.array(log['com_pos_x'])
        com_pos_y = np.array(log['com_pos_y'])
        com_pos_z = np.array(log['com_pos_z'])
        com_pos_ref_x = np.array(log['com_pos_ref_x'])
        com_pos_ref_y = np.array(log['com_pos_ref_y'])
        com_pos_ref_z = np.array(log['com_pos_ref_z'])
    
    print(f"   時間範囲: {t_min:.2f} - {t_max:.2f} 秒")
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # --- グラフ1: CoM Position X（実測値 vs 所望値） ---
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(time, com_pos_x, 'b-', linewidth=1.5, label='Actual')
    ax1.plot(time, com_pos_ref_x, 'k--', linewidth=1.5, label='Desired')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('CoM Position X [m]')
    ax1.set_xlim(t_min, t_max)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    output_file1 = os.path.join(output_dir, "thesis_com_position_x.pdf")
    plt.savefig(output_file1)
    print(f"   ✅ 保存: {output_file1}")
    plt.close(fig1)
    
    # --- グラフ2: CoM Position Y（実測値 vs 所望値） ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(time, com_pos_y, 'b-', linewidth=1.5, label='Actual')
    ax2.plot(time, com_pos_ref_y, 'k--', linewidth=1.5, label='Desired')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('CoM Position Y [m]')
    ax2.set_xlim(t_min, t_max)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    output_file2 = os.path.join(output_dir, "thesis_com_position_y.pdf")
    plt.savefig(output_file2)
    print(f"   ✅ 保存: {output_file2}")
    plt.close(fig2)
    
    # --- グラフ3: CoM Trajectory (Top View)（実測値 vs 所望値） ---
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    # 実測値の軌跡
    ax3.plot(com_pos_x, com_pos_y, 'b-', linewidth=1.5, label='Actual')
    # 所望値の軌跡
    ax3.plot(com_pos_ref_x, com_pos_ref_y, 'k--', linewidth=1.5, label='Desired')
    
    ax3.set_xlabel('CoM Position X [m]')
    ax3.set_ylabel('CoM Position Y [m]')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='best')
    ax3.axis('equal')
    plt.tight_layout()
    
    output_file3 = os.path.join(output_dir, "thesis_com_trajectory_topview.pdf")
    plt.savefig(output_file3)
    print(f"   ✅ 保存: {output_file3}")
    plt.close(fig3)
    
    print(f"\n✅ CoM位置グラフ生成完了")


def plot_rl_action_thesis(log, output_dir=".", time_range=None):
    """
    卒論用：RL Actionのプロット（X, Yの2枚）
    
    Parameters:
    -----------
    log : dict
        CSVから読み込んだログデータ
    output_dir : str
        出力ディレクトリ
    time_range : tuple or None
        プロット範囲 (t_min, t_max)
    """
    
    if len(log['time']) == 0:
        print("⚠️ ログデータが空です")
        return
    
    print(f"\n📊 RL Actionグラフ生成中...")
    
    # 時間範囲の設定
    time = np.array(log['time'])
    if time_range is not None:
        t_min, t_max = time_range
        mask = (time >= t_min) & (time <= t_max)
        time = time[mask]
        rl_action_x = np.array(log['rl_action_foot_offset_x'])[mask]
        rl_action_y = np.array(log['rl_action_foot_offset_y'])[mask]
    else:
        t_min, t_max = time[0], time[-1]
        rl_action_x = np.array(log['rl_action_foot_offset_x'])
        rl_action_y = np.array(log['rl_action_foot_offset_y'])
    
    os.makedirs(output_dir, exist_ok=True)
    
    # --- グラフ1: RL Action X ---
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(time, rl_action_x, 'b-', linewidth=1.5)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('RL Action Foot Offset X [m]')
    ax1.set_xlim(t_min, t_max)
    ax1.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    output_file1 = os.path.join(output_dir, "thesis_rl_action_x.pdf")
    plt.savefig(output_file1)
    print(f"   ✅ 保存: {output_file1}")
    plt.close(fig1)
    
    # --- グラフ2: RL Action Y ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(time, rl_action_y, 'b-', linewidth=1.5)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('RL Action Foot Offset Y [m]')
    ax2.set_xlim(t_min, t_max)
    ax2.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    output_file2 = os.path.join(output_dir, "thesis_rl_action_y.pdf")
    plt.savefig(output_file2)
    print(f"   ✅ 保存: {output_file2}")
    plt.close(fig2)
    
    print(f"✅ RL Actionグラフ生成完了")


def plot_dcm_offset_thesis(log, output_dir=".", time_range=None):
    """
    卒論用：DCM Offsetのプロット（X, Yの2枚）
    実測値と所望値を比較
    
    Parameters:
    -----------
    log : dict
        CSVから読み込んだログデータ
    output_dir : str
        出力ディレクトリ
    time_range : tuple or None
        プロット範囲 (t_min, t_max)
    """
    
    if len(log['time']) == 0:
        print("⚠️ ログデータが空です")
        return
    
    print(f"\n📊 DCM Offsetグラフ生成中...")
    
    # 時間範囲の設定
    time = np.array(log['time'])
    if time_range is not None:
        t_min, t_max = time_range
        mask = (time >= t_min) & (time <= t_max)
        time = time[mask]
        dcm_offset_actual_x = np.array(log['dcm_offset_actual_x'])[mask]
        dcm_offset_actual_y = np.array(log['dcm_offset_actual_y'])[mask]
        dcm_offset_desired_x = np.array(log['dcm_offset_desired_x'])[mask]
        dcm_offset_desired_y = np.array(log['dcm_offset_desired_y'])[mask]
    else:
        t_min, t_max = time[0], time[-1]
        dcm_offset_actual_x = np.array(log['dcm_offset_actual_x'])
        dcm_offset_actual_y = np.array(log['dcm_offset_actual_y'])
        dcm_offset_desired_x = np.array(log['dcm_offset_desired_x'])
        dcm_offset_desired_y = np.array(log['dcm_offset_desired_y'])
    
    os.makedirs(output_dir, exist_ok=True)
    
    # --- グラフ1: DCM Offset X（実測値 vs 所望値） ---
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(time, dcm_offset_actual_x, 'b-', linewidth=1.5, label='Actual')
    ax1.plot(time, dcm_offset_desired_x, 'k--', linewidth=1.5, label='Desired')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('DCM Offset X [m]')
    ax1.set_xlim(t_min, t_max)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    output_file1 = os.path.join(output_dir, "thesis_dcm_offset_x.pdf")
    plt.savefig(output_file1)
    print(f"   ✅ 保存: {output_file1}")
    plt.close(fig1)
    
    # --- グラフ2: DCM Offset Y（実測値 vs 所望値） ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(time, dcm_offset_actual_y, 'b-', linewidth=1.5, label='Actual')
    ax2.plot(time, dcm_offset_desired_y, 'k--', linewidth=1.5, label='Desired')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('DCM Offset Y [m]')
    ax2.set_xlim(t_min, t_max)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    output_file2 = os.path.join(output_dir, "thesis_dcm_offset_y.pdf")
    plt.savefig(output_file2)
    print(f"   ✅ 保存: {output_file2}")
    plt.close(fig2)
    
    print(f"✅ DCM Offsetグラフ生成完了")


def plot_moment_thesis(log, output_dir=".", time_range=None):
    """
    卒論用：回復モーメント関連のプロット（X, Y方向でそれぞれ2枚、合計4枚）
    - 実測値（Angular Moment）と所望値（Recovery Moment Desired）を同じグラフに
    - 差分（Moment Difference）
    
    Parameters:
    -----------
    log : dict
        CSVから読み込んだログデータ
    output_dir : str
        出力ディレクトリ
    time_range : tuple or None
        プロット範囲 (t_min, t_max)
    """
    
    if len(log['time']) == 0:
        print("⚠️ ログデータが空です")
        return
    
    print(f"\n📊 回復モーメントグラフ生成中...")
    
    # 時間範囲の設定
    time = np.array(log['time'])
    if time_range is not None:
        t_min, t_max = time_range
        mask = (time >= t_min) & (time <= t_max)
        time = time[mask]
        angular_moment_x = np.array(log['angular_moment_x'])[mask]
        angular_moment_y = np.array(log['angular_moment_y'])[mask]
        recovery_moment_x = np.array(log['recovery_moment_desired_x'])[mask]
        recovery_moment_y = np.array(log['recovery_moment_desired_y'])[mask]
        moment_diff_x = np.array(log['moment_diff_x'])[mask]
        moment_diff_y = np.array(log['moment_diff_y'])[mask]
    else:
        t_min, t_max = time[0], time[-1]
        angular_moment_x = np.array(log['angular_moment_x'])
        angular_moment_y = np.array(log['angular_moment_y'])
        recovery_moment_x = np.array(log['recovery_moment_desired_x'])
        recovery_moment_y = np.array(log['recovery_moment_desired_y'])
        moment_diff_x = np.array(log['moment_diff_x'])
        moment_diff_y = np.array(log['moment_diff_y'])
    
    os.makedirs(output_dir, exist_ok=True)
    
    # --- グラフ1: Recovery Moment X（実測値 vs 所望値） ---
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(time, angular_moment_x, 'b-', linewidth=1.5, label='Actual')
    ax1.plot(time, recovery_moment_x, 'k--', linewidth=1.5, label='Desired')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Recovery Moment X [Nm]')
    ax1.set_xlim(t_min, t_max)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    output_file1 = os.path.join(output_dir, "thesis_recovery_moment_x.pdf")
    plt.savefig(output_file1)
    print(f"   ✅ 保存: {output_file1}")
    plt.close(fig1)
    
    # --- グラフ2: Recovery Moment Y（実測値 vs 所望値） ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(time, angular_moment_y, 'b-', linewidth=1.5, label='Actual')
    ax2.plot(time, recovery_moment_y, 'k--', linewidth=1.5, label='Desired')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Recovery Moment Y [Nm]')
    ax2.set_xlim(t_min, t_max)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    output_file2 = os.path.join(output_dir, "thesis_recovery_moment_y.pdf")
    plt.savefig(output_file2)
    print(f"   ✅ 保存: {output_file2}")
    plt.close(fig2)
    
    # --- グラフ3: Moment Difference X（差分） ---
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(time, moment_diff_x, 'b-', linewidth=1.5)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Moment Difference X [Nm]')
    ax3.set_xlim(t_min, t_max)
    ax3.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    output_file3 = os.path.join(output_dir, "thesis_moment_diff_x.pdf")
    plt.savefig(output_file3)
    print(f"   ✅ 保存: {output_file3}")
    plt.close(fig3)
    
    # --- グラフ4: Moment Difference Y（差分） ---
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.plot(time, moment_diff_y, 'b-', linewidth=1.5)
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Moment Difference Y [Nm]')
    ax4.set_xlim(t_min, t_max)
    ax4.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    output_file4 = os.path.join(output_dir, "thesis_moment_diff_y.pdf")
    plt.savefig(output_file4)
    print(f"   ✅ 保存: {output_file4}")
    plt.close(fig4)
    
    print(f"✅ 回復モーメントグラフ生成完了")


if __name__ == "__main__":
    print("=" * 70)
    print("📊 卒論用：制御データプロット（実測値 vs 所望値）")
    print("=" * 70)
    
    # コマンドライン引数の処理
    csv_filename = "control_log.csv"
    output_dir = "."
    time_range = None
    
    if len(sys.argv) > 1:
        csv_filename = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    if len(sys.argv) > 4:
        # 時間範囲の指定（例: python script.py control_log.csv . 0.0 10.0）
        t_min = float(sys.argv[3])
        t_max = float(sys.argv[4])
        time_range = (t_min, t_max)
        print(f"📌 時間範囲指定: {t_min} - {t_max} 秒")
    
    print(f"📂 入力CSVファイル: {csv_filename}")
    print(f"📁 出力ディレクトリ: {output_dir}")
    
    # CSVファイルから読み込む
    log = load_csv_log(csv_filename)
    
    if log is not None and len(log.get('time', [])) > 0:
        # 全てのグラフを生成
        plot_com_position_thesis(log, output_dir=output_dir, time_range=time_range)
        plot_rl_action_thesis(log, output_dir=output_dir, time_range=time_range)
        plot_dcm_offset_thesis(log, output_dir=output_dir, time_range=time_range)
        plot_moment_thesis(log, output_dir=output_dir, time_range=time_range)
        
        print("\n" + "=" * 70)
        print("✅ 全てのグラフ生成完了")
        print("   - CoM位置: 3枚")
        print("   - RL Action: 2枚")
        print("   - DCM Offset: 2枚")
        print("   - 回復モーメント: 4枚")
        print("   合計: 11枚のPDFグラフ")
    else:
        print("⚠️ ログデータが空です")
    
    print("=" * 70)

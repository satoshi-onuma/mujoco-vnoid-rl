"""
制御データをプロットするスクリプト
record_humanoid.pyの後に実行する
"""

import matplotlib.pyplot as plt
import sys
import os

# vnoid_rl_envモジュールをインポート
build_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    "../build/controller/vnoid_rl_env"
))
sys.path.append(build_path)

from vnoid_rl_env import VnoidEnv

def plot_control_analysis(log):
    """制御データを6パネルでプロット"""
    
    if len(log['time']) == 0:
        print("⚠️ ログデータが空です")
        return
    
    print(f"\n📊 グラフ生成中... ({len(log['time'])} サンプル)")
    
    fig = plt.figure(figsize=(15, 10))
    
    # --- パネル1: CoM Position ---
    plt.subplot(3, 2, 1)
    plt.plot(log['time'], log['com_pos_x'], 'r-', label='Actual X')
    plt.plot(log['time'], log['com_pos_ref_x'], 'r--', label='Ref X')
    plt.plot(log['time'], log['com_pos_y'], 'g-', label='Actual Y')
    plt.plot(log['time'], log['com_pos_ref_y'], 'g--', label='Ref Y')
    plt.plot(log['time'], log['com_pos_z'], 'b-', label='Actual Z')
    plt.plot(log['time'], log['com_pos_ref_z'], 'b--', label='Ref Z')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('CoM Position')
    plt.legend()
    plt.grid(True)
    
    # --- パネル2: CoM Velocity ---
    plt.subplot(3, 2, 2)
    plt.plot(log['time'], log['com_vel_x'], 'r-', label='X')
    plt.plot(log['time'], log['com_vel_y'], 'g-', label='Y')
    plt.plot(log['time'], log['com_vel_z'], 'b-', label='Z')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.title('CoM Velocity')
    plt.legend()
    plt.grid(True)
    
    # --- パネル3: ZMP ---
    plt.subplot(3, 2, 3)
    plt.plot(log['time'], log['zmp_x'], 'r-', label='Actual X')
    plt.plot(log['time'], log['zmp_ref_x'], 'r--', label='Ref X')
    plt.plot(log['time'], log['zmp_y'], 'g-', label='Actual Y')
    plt.plot(log['time'], log['zmp_ref_y'], 'g--', label='Ref Y')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('ZMP (Zero Moment Point)')
    plt.legend()
    plt.grid(True)
    
    # --- パネル4: DCM ---
    plt.subplot(3, 2, 4)
    plt.plot(log['time'], log['dcm_x'], 'r-', label='Actual X')
    plt.plot(log['time'], log['dcm_ref_x'], 'r--', label='Ref X')
    plt.plot(log['time'], log['dcm_y'], 'g-', label='Actual Y')
    plt.plot(log['time'], log['dcm_ref_y'], 'g--', label='Ref Y')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('DCM (Divergent Component of Motion)')
    plt.legend()
    plt.grid(True)
    
    # --- パネル5: CoM Trajectory (XY平面) ---
    plt.subplot(3, 2, 5)
    plt.plot(log['com_pos_x'], log['com_pos_y'], 'b-', label='Actual')
    plt.plot(log['com_pos_ref_x'], log['com_pos_ref_y'], 'b--', label='Ref')
    plt.plot(log['com_pos_x'][0], log['com_pos_y'][0], 'go', markersize=10, label='Start')
    plt.plot(log['com_pos_x'][-1], log['com_pos_y'][-1], 'rx', markersize=10, label='End')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('CoM Trajectory (Top View)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # --- パネル6: ZMP & DCM Trajectory (XY平面) ---
    plt.subplot(3, 2, 6)
    plt.plot(log['zmp_x'], log['zmp_y'], 'r-', label='ZMP Actual')
    plt.plot(log['zmp_ref_x'], log['zmp_ref_y'], 'r--', label='ZMP Ref')
    plt.plot(log['dcm_x'], log['dcm_y'], 'b-', label='DCM Actual')
    plt.plot(log['dcm_ref_x'], log['dcm_ref_y'], 'b--', label='DCM Ref')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('ZMP & DCM Trajectory (Top View)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('control_analysis.png', dpi=150)
    print(f"✅ グラフ保存完了: control_analysis.png")
    plt.show()

if __name__ == "__main__":
    print("=" * 70)
    print("📊 制御データのプロット")
    print("=" * 70)
    print("このスクリプトは単独では実行できません")
    print("record_humanoid.py を使用してください")
    print("=" * 70)
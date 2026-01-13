"""
制御データをプロットするスクリプト
CSVファイルから読み込んでプロットする
"""

import matplotlib.pyplot as plt
import sys
import os
import csv
import numpy as np

def plot_control_analysis(log):
    """制御データを複数パネルでプロット"""
    
    if len(log['time']) == 0:
        print("⚠️ ログデータが空です")
        return
    
    print(f"\n📊 グラフ生成中... ({len(log['time'])} サンプル)")
    
    # データの存在チェック用ヘルパー関数
    def has_key(key):
        return key in log and len(log[key]) > 0
    
    # 図1: 基本制御データ（6パネル）
    fig1 = plt.figure(figsize=(15, 10))
    
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
    plt.plot(log['time'], log['zmp_ref_x'], 'r--', label='Ref X')
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
    
    # 図2: DCM Offset, RL Action, 接触状態（6パネル）
    fig2 = plt.figure(figsize=(15, 10))
    
    # --- パネル1: DCM Offset ---
    plt.subplot(3, 2, 1)
    if has_key('dcm_offset_actual_x'):
        plt.plot(log['time'], log['dcm_offset_actual_x'], 'r-', label='Actual X')
    if has_key('dcm_offset_actual_y'):
        plt.plot(log['time'], log['dcm_offset_actual_y'], 'g-', label='Actual Y')
    if has_key('dcm_offset_desired_x'):
        plt.plot(log['time'], log['dcm_offset_desired_x'], 'r--', label='Desired X')
    if has_key('dcm_offset_desired_y'):
        plt.plot(log['time'], log['dcm_offset_desired_y'], 'g--', label='Desired Y')
    plt.xlabel('Time [s]')
    plt.ylabel('Offset [m]')
    plt.title('DCM Offset')
    plt.legend()
    plt.grid(True)
    
    # --- パネル2: RL Action ---
    plt.subplot(3, 2, 2)
    if has_key('rl_action_foot_offset_x'):
        plt.plot(log['time'], log['rl_action_foot_offset_x'], 'r-', label='Foot Offset X')
    if has_key('rl_action_foot_offset_y'):
        plt.plot(log['time'], log['rl_action_foot_offset_y'], 'g-', label='Foot Offset Y')
    plt.xlabel('Time [s]')
    plt.ylabel('Offset [m]')
    plt.title('RL Action (Foot Offset)')
    plt.legend()
    plt.grid(True)
    
    # --- パネル3: 接触状態 ---
    plt.subplot(3, 2, 3)
    if has_key('obs_contact_left'):
        plt.plot(log['time'], log['obs_contact_left'], 'b-', label='Left Foot', linewidth=2)
    if has_key('obs_contact_right'):
        plt.plot(log['time'], log['obs_contact_right'], 'r-', label='Right Foot', linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Contact [0/1]')
    plt.title('Foot Contact State')
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.1, 1.1)
    
    # --- パネル4: 足の沈み込み ---
    plt.subplot(3, 2, 4)
    if has_key('obs_foot_sink_left'):
        plt.plot(log['time'], log['obs_foot_sink_left'], 'b-', label='Left Foot')
    if has_key('obs_foot_sink_right'):
        plt.plot(log['time'], log['obs_foot_sink_right'], 'r-', label='Right Foot')
    plt.xlabel('Time [s]')
    plt.ylabel('Sink [m]')
    plt.title('Foot Sink (Actual - Ref)')
    plt.legend()
    plt.grid(True)
    
    # --- パネル5: 足の高さ ---
    plt.subplot(3, 2, 5)
    if has_key('obs_foot_height_left'):
        plt.plot(log['time'], log['obs_foot_height_left'], 'b-', label='Left Foot')
    if has_key('obs_foot_height_right'):
        plt.plot(log['time'], log['obs_foot_height_right'], 'r-', label='Right Foot')
    plt.xlabel('Time [s]')
    plt.ylabel('Height [m]')
    plt.title('Foot Height (Ref)')
    plt.legend()
    plt.grid(True)
    
    # --- パネル6: 足の沈み込みと接触状態の関係 ---
    plt.subplot(3, 2, 6)
    if has_key('obs_foot_sink_left') and has_key('obs_contact_left'):
        # 接触時のみプロット
        contact_left = np.array(log['obs_contact_left']) > 0.5
        sink_left_contact = np.array(log['obs_foot_sink_left'])
        sink_left_contact[~contact_left] = np.nan
        plt.plot(log['time'], sink_left_contact, 'b-', label='Left (Contact)', linewidth=2)
    if has_key('obs_foot_sink_right') and has_key('obs_contact_right'):
        contact_right = np.array(log['obs_contact_right']) > 0.5
        sink_right_contact = np.array(log['obs_foot_sink_right'])
        sink_right_contact[~contact_right] = np.nan
        plt.plot(log['time'], sink_right_contact, 'r-', label='Right (Contact)', linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Sink [m]')
    plt.title('Foot Sink (Contact Only)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('control_analysis_foot.png', dpi=150)
    print(f"✅ グラフ保存完了: control_analysis_foot.png")
    
    # 図3: ベース角度と角速度（6パネル）
    fig3 = plt.figure(figsize=(15, 10))
    
    # --- パネル1: ベース角度 (Roll, Pitch, Yaw) ---
    plt.subplot(3, 2, 1)
    if has_key('base_angle_roll'):
        plt.plot(log['time'], log['base_angle_roll'], 'r-', label='Actual Roll')
    if has_key('base_angle_pitch'):
        plt.plot(log['time'], log['base_angle_pitch'], 'g-', label='Actual Pitch')
    if has_key('base_angle_yaw'):
        plt.plot(log['time'], log['base_angle_yaw'], 'b-', label='Actual Yaw')
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [rad]')
    plt.title('Base Angle (Actual)')
    plt.legend()
    plt.grid(True)
    
    # --- パネル2: ベース角度参照値 ---
    plt.subplot(3, 2, 2)
    if has_key('base_angle_ref_roll'):
        plt.plot(log['time'], log['base_angle_ref_roll'], 'r--', label='Ref Roll')
    if has_key('base_angle_ref_pitch'):
        plt.plot(log['time'], log['base_angle_ref_pitch'], 'g--', label='Ref Pitch')
    if has_key('base_angle_ref_yaw'):
        plt.plot(log['time'], log['base_angle_ref_yaw'], 'b--', label='Ref Yaw')
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [rad]')
    plt.title('Base Angle (Reference)')
    plt.legend()
    plt.grid(True)
    
    # --- パネル3: ベース角度エラー ---
    plt.subplot(3, 2, 3)
    if has_key('base_angle_roll') and has_key('base_angle_ref_roll'):
        angle_error_roll = np.array(log['base_angle_roll']) - np.array(log['base_angle_ref_roll'])
        plt.plot(log['time'], angle_error_roll, 'r-', label='Roll Error')
    if has_key('base_angle_pitch') and has_key('base_angle_ref_pitch'):
        angle_error_pitch = np.array(log['base_angle_pitch']) - np.array(log['base_angle_ref_pitch'])
        plt.plot(log['time'], angle_error_pitch, 'g-', label='Pitch Error')
    if has_key('base_angle_yaw') and has_key('base_angle_ref_yaw'):
        angle_error_yaw = np.array(log['base_angle_yaw']) - np.array(log['base_angle_ref_yaw'])
        plt.plot(log['time'], angle_error_yaw, 'b-', label='Yaw Error')
    plt.xlabel('Time [s]')
    plt.ylabel('Angle Error [rad]')
    plt.title('Base Angle Error')
    plt.legend()
    plt.grid(True)
    
    # --- パネル4: 角速度 ---
    plt.subplot(3, 2, 4)
    if has_key('obs_angvel_x'):
        plt.plot(log['time'], log['obs_angvel_x'], 'r-', label='X')
    if has_key('obs_angvel_y'):
        plt.plot(log['time'], log['obs_angvel_y'], 'g-', label='Y')
    if has_key('obs_angvel_z'):
        plt.plot(log['time'], log['obs_angvel_z'], 'b-', label='Z')
    plt.xlabel('Time [s]')
    plt.ylabel('Angular Velocity [rad/s]')
    plt.title('Base Angular Velocity')
    plt.legend()
    plt.grid(True)
    
    # --- パネル5: 加速度 ---
    plt.subplot(3, 2, 5)
    if has_key('obs_acc_x'):
        plt.plot(log['time'], log['obs_acc_x'], 'r-', label='X')
    if has_key('obs_acc_y'):
        plt.plot(log['time'], log['obs_acc_y'], 'g-', label='Y')
    if has_key('obs_acc_z'):
        plt.plot(log['time'], log['obs_acc_z'], 'b-', label='Z')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s²]')
    plt.title('Base Acceleration')
    plt.legend()
    plt.grid(True)
    
    # --- パネル6: 姿勢（クォータニオン） ---
    plt.subplot(3, 2, 6)
    if has_key('obs_ori_w'):
        plt.plot(log['time'], log['obs_ori_w'], 'k-', label='w', linewidth=2)
    if has_key('obs_ori_x'):
        plt.plot(log['time'], log['obs_ori_x'], 'r-', label='x')
    if has_key('obs_ori_y'):
        plt.plot(log['time'], log['obs_ori_y'], 'g-', label='y')
    if has_key('obs_ori_z'):
        plt.plot(log['time'], log['obs_ori_z'], 'b-', label='z')
    plt.xlabel('Time [s]')
    plt.ylabel('Quaternion')
    plt.title('Base Orientation (Quaternion)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('control_analysis_base.png', dpi=150)
    print(f"✅ グラフ保存完了: control_analysis_base.png")
    
    # 図4: 回復モーメントと角運動量の時間微分（モーメント）（6パネル）
    fig4 = plt.figure(figsize=(15, 10))
    
    # --- パネル1: 回復モーメント ---
    plt.subplot(3, 2, 1)
    if has_key('recovery_moment_desired_x'):
        plt.plot(log['time'], log['recovery_moment_desired_x'], 'r-', label='X')
    if has_key('recovery_moment_desired_y'):
        plt.plot(log['time'], log['recovery_moment_desired_y'], 'g-', label='Y')
    if has_key('recovery_moment_desired_z'):
        plt.plot(log['time'], log['recovery_moment_desired_z'], 'b-', label='Z')
    plt.xlabel('Time [s]')
    plt.ylabel('Moment [Nm]')
    plt.title('Recovery Moment (Desired)')
    plt.legend()
    plt.grid(True)
    
    # --- パネル2: 角運動量の時間微分（モーメント） ---
    plt.subplot(3, 2, 2)
    if has_key('angular_moment_x'):
        plt.plot(log['time'], log['angular_moment_x'], 'r-', label='X')
    if has_key('angular_moment_y'):
        plt.plot(log['time'], log['angular_moment_y'], 'g-', label='Y')
    if has_key('angular_moment_z'):
        plt.plot(log['time'], log['angular_moment_z'], 'b-', label='Z')
    plt.xlabel('Time [s]')
    plt.ylabel('Moment [Nm]')
    plt.title('Angular Moment (dL/dt)')
    plt.legend()
    plt.grid(True)
    
    # --- パネル3: 回復モーメントと角運動量の時間微分の関係 (X) ---
    plt.subplot(3, 2, 3)
    if has_key('recovery_moment_desired_x') and has_key('angular_moment_x'):
        ax1 = plt.gca()
        ax1.plot(log['time'], log['recovery_moment_desired_x'], 'r-', label='Recovery Moment X', linewidth=2)
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Recovery Moment [Nm]', color='r')
        ax1.tick_params(axis='y', labelcolor='r')
        ax1.grid(True)
        
        ax2 = ax1.twinx()
        ax2.plot(log['time'], log['angular_moment_x'], 'b--', label='Angular Moment X', linewidth=2)
        ax2.set_ylabel('Angular Moment [Nm]', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        plt.title('Recovery Moment vs Angular Moment (X)')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    # --- パネル4: 回復モーメントと角運動量の時間微分の関係 (Y) ---
    plt.subplot(3, 2, 4)
    if has_key('recovery_moment_desired_y') and has_key('angular_moment_y'):
        ax1 = plt.gca()
        ax1.plot(log['time'], log['recovery_moment_desired_y'], 'g-', label='Recovery Moment Y', linewidth=2)
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Recovery Moment [Nm]', color='g')
        ax1.tick_params(axis='y', labelcolor='g')
        ax1.grid(True)
        
        ax2 = ax1.twinx()
        ax2.plot(log['time'], log['angular_moment_y'], 'b--', label='Angular Moment Y', linewidth=2)
        ax2.set_ylabel('Angular Moment [Nm]', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        plt.title('Recovery Moment vs Angular Moment (Y)')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    # --- パネル5: モーメントと回復モーメントの差 ---
    plt.subplot(3, 2, 5)
    if has_key('moment_diff_x'):
        plt.plot(log['time'], log['moment_diff_x'], 'r-', label='X')
    if has_key('moment_diff_y'):
        plt.plot(log['time'], log['moment_diff_y'], 'g-', label='Y')
    if has_key('moment_diff_z'):
        plt.plot(log['time'], log['moment_diff_z'], 'b-', label='Z')
    plt.xlabel('Time [s]')
    plt.ylabel('Moment Difference [Nm]')
    plt.title('Moment Difference (Angular Moment - Recovery Moment)')
    plt.legend()
    plt.grid(True)
    
    # --- パネル6: 角運動量の時間微分の大きさ ---
    plt.subplot(3, 2, 6)
    if has_key('angular_moment_x') and has_key('angular_moment_y') and has_key('angular_moment_z'):
        angular_moment_mag = np.sqrt(
            np.array(log['angular_moment_x'])**2 +
            np.array(log['angular_moment_y'])**2 +
            np.array(log['angular_moment_z'])**2
        )
        plt.plot(log['time'], angular_moment_mag, 'b-', linewidth=2)
        plt.xlabel('Time [s]')
        plt.ylabel('Magnitude [Nm]')
        plt.title('Angular Moment Magnitude')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('control_analysis_moment.png', dpi=150)
    print(f"✅ グラフ保存完了: control_analysis_moment.png")
    
    # 図5: ZMP Local（上限・下限も表示）（3パネル）
    fig5 = plt.figure(figsize=(15, 8))
    
    # zmp_minとzmp_maxの値（myrobot.cppから）
    zmp_min_x, zmp_min_y, zmp_min_z = -0.1, -0.05, -0.1
    zmp_max_x, zmp_max_y, zmp_max_z = 0.1, 0.05, 0.1
    
    # --- パネル1: ZMP Local X ---
    plt.subplot(3, 1, 1)
    if has_key('zmp_local_x'):
        plt.plot(log['time'], log['zmp_local_x'], 'r-', label='ZMP Local X', linewidth=2)
    plt.axhline(y=zmp_min_x, color='r', linestyle='--', linewidth=1, label='ZMP Min X')
    plt.axhline(y=zmp_max_x, color='r', linestyle='--', linewidth=1, label='ZMP Max X')
    plt.fill_between(log['time'], zmp_min_x, zmp_max_x, alpha=0.2, color='gray', label='Admissible Range')
    plt.xlabel('Time [s]')
    plt.ylabel('ZMP Local X [m]')
    plt.title('ZMP Local X (with Admissible Range)')
    plt.legend()
    plt.grid(True)
    
    # --- パネル2: ZMP Local Y ---
    plt.subplot(3, 1, 2)
    if has_key('zmp_local_y'):
        plt.plot(log['time'], log['zmp_local_y'], 'g-', label='ZMP Local Y', linewidth=2)
    plt.axhline(y=zmp_min_y, color='g', linestyle='--', linewidth=1, label='ZMP Min Y')
    plt.axhline(y=zmp_max_y, color='g', linestyle='--', linewidth=1, label='ZMP Max Y')
    plt.fill_between(log['time'], zmp_min_y, zmp_max_y, alpha=0.2, color='gray', label='Admissible Range')
    plt.xlabel('Time [s]')
    plt.ylabel('ZMP Local Y [m]')
    plt.title('ZMP Local Y (with Admissible Range)')
    plt.legend()
    plt.grid(True)
    
    # --- パネル3: ZMP Local Z ---
    plt.subplot(3, 1, 3)
    if has_key('zmp_local_z'):
        plt.plot(log['time'], log['zmp_local_z'], 'b-', label='ZMP Local Z', linewidth=2)
    plt.axhline(y=zmp_min_z, color='b', linestyle='--', linewidth=1, label='ZMP Min Z')
    plt.axhline(y=zmp_max_z, color='b', linestyle='--', linewidth=1, label='ZMP Max Z')
    plt.fill_between(log['time'], zmp_min_z, zmp_max_z, alpha=0.2, color='gray', label='Admissible Range')
    plt.xlabel('Time [s]')
    plt.ylabel('ZMP Local Z [m]')
    plt.title('ZMP Local Z (with Admissible Range)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('control_analysis_zmp_local.png', dpi=150)
    print(f"✅ グラフ保存完了: control_analysis_zmp_local.png")
    
    plt.show()

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

if __name__ == "__main__":
    print("=" * 70)
    print("📊 制御データのプロット（CSV版）")
    print("=" * 70)
    
    # CSVファイル名を取得（コマンドライン引数から、またはデフォルト）
    csv_filename = sys.argv[1] if len(sys.argv) > 1 else "control_log.csv"
    
    print(f"📂 CSVファイル: {csv_filename}")
    
    # CSVファイルから読み込む
    log = load_csv_log(csv_filename)
    
    if log is not None and len(log.get('time', [])) > 0:
        plot_control_analysis(log)
    else:
        print("⚠️ ログデータが空です")
    
    print("=" * 70)
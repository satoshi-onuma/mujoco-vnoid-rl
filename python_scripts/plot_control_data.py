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
    
    # --- パネル1: CoM Position & Base Position ---
    plt.subplot(3, 2, 1)
    if has_key('base_pos_x'):
        plt.plot(log['time'], log['base_pos_x'], 'k-', label='Base X', linewidth=2.5, alpha=0.8)
    plt.plot(log['time'], log['com_pos_x'], 'r-', label='CoM Actual X')
    plt.plot(log['time'], log['com_pos_ref_x'], 'r--', label='CoM Ref X')
    plt.plot(log['time'], log['com_pos_y'], 'g-', label='CoM Actual Y')
    plt.plot(log['time'], log['com_pos_ref_y'], 'g--', label='CoM Ref Y')
    plt.plot(log['time'], log['com_pos_z'], 'b-', label='CoM Actual Z')
    plt.plot(log['time'], log['com_pos_ref_z'], 'b--', label='CoM Ref Z')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('CoM & Base Position')
    plt.legend(fontsize=8)
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
    if has_key('land_dcm_x'):
        plt.plot(log['time'], log['land_dcm_x'], 'r:', linewidth=2, label='Landing DCM X')
    if has_key('land_dcm_y'):
        plt.plot(log['time'], log['land_dcm_y'], 'g:', linewidth=2, label='Landing DCM Y')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('DCM (Divergent Component of Motion)')
    plt.legend()
    plt.grid(True)
    
    # --- パネル5: CoM & Feet & Base Trajectory (XY平面) ---
    plt.subplot(3, 2, 5)
    if has_key('base_pos_x'):
        # ベース位置は y=0 としてプロット（X方向の移動のみ）
        plt.plot(log['base_pos_x'], [0]*len(log['base_pos_x']), 'k-', linewidth=2.5, label='Base X', alpha=0.8)
        plt.plot(log['base_pos_x'][0], 0, 'ko', markersize=10, label='Base Start')
        plt.plot(log['base_pos_x'][-1], 0, 'kx', markersize=10, label='Base End')
    
    # CoM軌道
    plt.plot(log['com_pos_x'], log['com_pos_y'], 'b-', label='CoM Actual', linewidth=2)
    plt.plot(log['com_pos_ref_x'], log['com_pos_ref_y'], 'b--', label='CoM Ref', alpha=0.7)
    plt.plot(log['com_pos_x'][0], log['com_pos_y'][0], 'go', markersize=10, label='CoM Start')
    plt.plot(log['com_pos_x'][-1], log['com_pos_y'][-1], 'rx', markersize=10, label='CoM End')
    
    # 右足軌道（足0）
    if has_key('foot_pos_right_x') and has_key('foot_pos_right_y'):
        plt.plot(log['foot_pos_right_x'], log['foot_pos_right_y'], 'r-', 
                 label='Right Foot', linewidth=1.5, alpha=0.8)
        plt.plot(log['foot_pos_right_x'][0], log['foot_pos_right_y'][0], 'r^', markersize=8)
        plt.plot(log['foot_pos_right_x'][-1], log['foot_pos_right_y'][-1], 'rv', markersize=8)
    
    # 左足軌道（足1）
    if has_key('foot_pos_left_x') and has_key('foot_pos_left_y'):
        plt.plot(log['foot_pos_left_x'], log['foot_pos_left_y'], 'm-', 
                 label='Left Foot', linewidth=1.5, alpha=0.8)
        plt.plot(log['foot_pos_left_x'][0], log['foot_pos_left_y'][0], 'm^', markersize=8)
        plt.plot(log['foot_pos_left_x'][-1], log['foot_pos_left_y'][-1], 'mv', markersize=8)
    
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('CoM & Feet Trajectory (Top View)')
    plt.legend(fontsize=7, loc='best')
    plt.grid(True)
    plt.axis('equal')
    
    # --- パネル6: ZMP & DCM Trajectory (XY平面) ---
    plt.subplot(3, 2, 6)
    plt.plot(log['zmp_ref_x'], log['zmp_ref_y'], 'r--', label='ZMP Ref')
    plt.plot(log['dcm_x'], log['dcm_y'], 'b-', label='DCM Actual')
    plt.plot(log['dcm_ref_x'], log['dcm_ref_y'], 'b--', label='DCM Ref')
    if has_key('land_dcm_x') and has_key('land_dcm_y'):
        plt.plot(log['land_dcm_x'], log['land_dcm_y'], 'm:', linewidth=2, label='Landing DCM')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('ZMP & DCM Trajectory (Top View)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('control_analysis.pdf')
    print(f"✅ グラフ保存完了: control_analysis.pdf")
    
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
    plt.savefig('control_analysis_foot.pdf')
    print(f"✅ グラフ保存完了: control_analysis_foot.pdf")
    
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
    plt.savefig('control_analysis_base.pdf')
    print(f"✅ グラフ保存完了: control_analysis_base.pdf")
    
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
    plt.savefig('control_analysis_moment.pdf')
    print(f"✅ グラフ保存完了: control_analysis_moment.pdf")
    
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
    plt.savefig('control_analysis_zmp_local.pdf')
    print(f"✅ グラフ保存完了: control_analysis_zmp_local.pdf")
    
    # 図6: 足位置の時系列（3パネル、CoMと重ねて表示）
    fig6 = plt.figure(figsize=(15, 10))
    
    # --- パネル1: X座標の時系列 ---
    plt.subplot(3, 1, 1)
    # CoM
    if has_key('com_pos_x'):
        plt.plot(log['time'], log['com_pos_x'], 'b-', label='CoM X', linewidth=2.5)
    if has_key('com_pos_ref_x'):
        plt.plot(log['time'], log['com_pos_ref_x'], 'b--', label='CoM Ref X', linewidth=2, alpha=0.5)
    # 足位置
    if has_key('foot_pos_right_x'):
        plt.plot(log['time'], log['foot_pos_right_x'], 'r-', label='Right Foot X', linewidth=2)
    if has_key('foot_pos_left_x'):
        plt.plot(log['time'], log['foot_pos_left_x'], 'm-', label='Left Foot X', linewidth=2)
    # ベース位置
    if has_key('base_pos_x'):
        plt.plot(log['time'], log['base_pos_x'], 'k-', label='Base X', linewidth=2.5, alpha=0.7)
    plt.xlabel('Time [s]')
    plt.ylabel('Position X [m]')
    plt.title('X Position: CoM, Feet, and Base')
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # --- パネル2: Y座標の時系列 ---
    plt.subplot(3, 1, 2)
    # CoM
    if has_key('com_pos_y'):
        plt.plot(log['time'], log['com_pos_y'], 'b-', label='CoM Y', linewidth=2.5)
    if has_key('com_pos_ref_y'):
        plt.plot(log['time'], log['com_pos_ref_y'], 'b--', label='CoM Ref Y', linewidth=2, alpha=0.5)
    # 足位置
    if has_key('foot_pos_right_y'):
        plt.plot(log['time'], log['foot_pos_right_y'], 'r-', label='Right Foot Y', linewidth=2)
    if has_key('foot_pos_left_y'):
        plt.plot(log['time'], log['foot_pos_left_y'], 'm-', label='Left Foot Y', linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Position Y [m]')
    plt.title('Y Position: CoM and Feet')
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # --- パネル3: Z座標の時系列（接触状態も表示） ---
    plt.subplot(3, 1, 3)
    # CoM
    if has_key('com_pos_z'):
        plt.plot(log['time'], log['com_pos_z'], 'b-', label='CoM Z', linewidth=2.5)
    if has_key('com_pos_ref_z'):
        plt.plot(log['time'], log['com_pos_ref_z'], 'b--', label='CoM Ref Z', linewidth=2, alpha=0.5)
    # 足位置
    if has_key('foot_pos_right_z'):
        plt.plot(log['time'], log['foot_pos_right_z'], 'r-', label='Right Foot Z', linewidth=2)
    if has_key('foot_pos_left_z'):
        plt.plot(log['time'], log['foot_pos_left_z'], 'm-', label='Left Foot Z', linewidth=2)
    
    # # 接触状態を背景にハイライト表示
    # if has_key('obs_contact_right') and has_key('obs_contact_left'):
    #     contact_right = np.array(log['obs_contact_right']) > 0.5
    #     contact_left = np.array(log['obs_contact_left']) > 0.5
    #     for i in range(len(log['time'])-1):
    #         if contact_right[i]:
    #             plt.axvspan(log['time'][i], log['time'][i+1], alpha=0.1, color='red')
    #         if contact_left[i]:
    #             plt.axvspan(log['time'][i], log['time'][i+1], alpha=0.1, color='magenta')
    
    plt.xlabel('Time [s]')
    plt.ylabel('Position Z [m]')
    plt.title('Z Position: CoM and Feet (Red/Magenta background = Contact)')
    plt.legend(fontsize=10)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('control_analysis_foot_position.pdf')
    print(f"✅ グラフ保存完了: control_analysis_foot_position.pdf")
    
    # 図7: DCM Offset の計算要素（6パネル）
    fig7 = plt.figure(figsize=(15, 10))
    
    # --- パネル1: DCM Actual と Support Foot (X) ---
    plt.subplot(3, 2, 1)
    if has_key('dcm_x'):
        plt.plot(log['time'], log['dcm_x'], 'b-', label='DCM Actual X', linewidth=2)
    if has_key('support_foot_actual_x'):
        plt.plot(log['time'], log['support_foot_actual_x'], 'r--', label='Support Foot Actual X', linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('DCM Actual vs Support Foot (X)')
    plt.legend()
    plt.grid(True)
    
    # --- パネル2: DCM Actual と Support Foot (Y) ---
    plt.subplot(3, 2, 2)
    if has_key('dcm_y'):
        plt.plot(log['time'], log['dcm_y'], 'b-', label='DCM Actual Y', linewidth=2)
    if has_key('support_foot_actual_y'):
        plt.plot(log['time'], log['support_foot_actual_y'], 'r--', label='Support Foot Actual Y', linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('DCM Actual vs Support Foot (Y)')
    plt.legend()
    plt.grid(True)
    
    # --- パネル3: DCM Offset Actual の計算 (X) ---
    plt.subplot(3, 2, 3)
    if has_key('dcm_x') and has_key('support_foot_actual_x'):
        dcm_minus_support_x = np.array(log['dcm_x']) - np.array(log['support_foot_actual_x'])
        plt.plot(log['time'], dcm_minus_support_x, 'g-', label='DCM - Support Foot', linewidth=2)
    if has_key('dcm_offset_actual_x'):
        plt.plot(log['time'], log['dcm_offset_actual_x'], 'r--', label='DCM Offset Actual (直接計算)', linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Offset [m]')
    plt.title('DCM Offset Actual Calculation (X)')
    plt.legend()
    plt.grid(True)
    
    # --- パネル4: DCM Offset Actual の計算 (Y) ---
    plt.subplot(3, 2, 4)
    if has_key('dcm_y') and has_key('support_foot_actual_y'):
        dcm_minus_support_y = np.array(log['dcm_y']) - np.array(log['support_foot_actual_y'])
        plt.plot(log['time'], dcm_minus_support_y, 'g-', label='DCM - Support Foot', linewidth=2)
    if has_key('dcm_offset_actual_y'):
        plt.plot(log['time'], log['dcm_offset_actual_y'], 'r--', label='DCM Offset Actual (直接計算)', linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Offset [m]')
    plt.title('DCM Offset Actual Calculation (Y)')
    plt.legend()
    plt.grid(True)
    
    # --- パネル5: DCM Offset Desired の計算要素 (X) ---
    plt.subplot(3, 2, 5)
    if has_key('next_step_dcm_x'):
        plt.plot(log['time'], log['next_step_dcm_x'], 'b-', label='Next Step DCM X', linewidth=2)
    if has_key('next_step_support_foot_x'):
        plt.plot(log['time'], log['next_step_support_foot_x'], 'r--', label='Next Support Foot X', linewidth=2)
    if has_key('next_step_dcm_x') and has_key('next_step_support_foot_x'):
        next_offset_x = np.array(log['next_step_dcm_x']) - np.array(log['next_step_support_foot_x'])
        plt.plot(log['time'], next_offset_x, 'g:', label='Next DCM - Next Support', linewidth=2)
    if has_key('dcm_offset_desired_x'):
        plt.plot(log['time'], log['dcm_offset_desired_x'], 'm-.', label='DCM Offset Desired (直接計算)', linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Position/Offset [m]')
    plt.title('DCM Offset Desired Calculation (X)')
    plt.legend()
    plt.grid(True)
    
    # --- パネル6: DCM Offset Desired の計算要素 (Y) ---
    plt.subplot(3, 2, 6)
    if has_key('next_step_dcm_y'):
        plt.plot(log['time'], log['next_step_dcm_y'], 'b-', label='Next Step DCM Y', linewidth=2)
    if has_key('next_step_support_foot_y'):
        plt.plot(log['time'], log['next_step_support_foot_y'], 'r--', label='Next Support Foot Y', linewidth=2)
    if has_key('next_step_dcm_y') and has_key('next_step_support_foot_y'):
        next_offset_y = np.array(log['next_step_dcm_y']) - np.array(log['next_step_support_foot_y'])
        plt.plot(log['time'], next_offset_y, 'g:', label='Next DCM - Next Support', linewidth=2)
    if has_key('dcm_offset_desired_y'):
        plt.plot(log['time'], log['dcm_offset_desired_y'], 'm-.', label='DCM Offset Desired (直接計算)', linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Position/Offset [m]')
    plt.title('DCM Offset Desired Calculation (Y)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('control_analysis_dcm_offset_components.pdf')
    print(f"✅ グラフ保存完了: control_analysis_dcm_offset_components.pdf")
    
    #plt.show()

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


def plot_dcm_offset_analysis(log):
    """DCMオフセットの計算要素を詳細にプロット"""
    
    if len(log['time']) == 0:
        print("⚠️ ログデータが空です")
        return
    
    print(f"\n📊 DCMオフセット分析グラフ生成中... ({len(log['time'])} サンプル)")
    
    # データの存在チェック用ヘルパー関数
    def has_key(key):
        return key in log and len(log[key]) > 0
    
    # 図: DCMオフセット分析（6パネル）
    fig = plt.figure(figsize=(18, 12))
    
    # --- パネル1: DCMオフセット（Actual） ---
    plt.subplot(3, 2, 1)
    if has_key('dcm_offset_actual_x') and has_key('dcm_offset_actual_y'):
        plt.plot(log['time'], log['dcm_offset_actual_x'], 'r-', label='Actual X', linewidth=2)
        plt.plot(log['time'], log['dcm_offset_actual_y'], 'g-', label='Actual Y', linewidth=2)
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Offset [m]', fontsize=12)
    plt.title('DCM Offset (Actual)\nDCM - Support Foot', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # --- パネル2: DCMオフセット（Desired） ---
    plt.subplot(3, 2, 2)
    if has_key('dcm_offset_desired_x') and has_key('dcm_offset_desired_y'):
        plt.plot(log['time'], log['dcm_offset_desired_x'], 'r--', label='Desired X', linewidth=2)
        plt.plot(log['time'], log['dcm_offset_desired_y'], 'g--', label='Desired Y', linewidth=2)
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Offset [m]', fontsize=12)
    plt.title('DCM Offset (Desired)\nNext Step DCM - Next Support Foot', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # --- パネル3: Actual計算要素（X方向） ---
    plt.subplot(3, 2, 3)
    if has_key('dcm_x') and has_key('support_foot_actual_x'):
        plt.plot(log['time'], log['dcm_x'], 'r-', label='DCM Actual X', linewidth=2)
        plt.plot(log['time'], log['support_foot_actual_x'], 'b--', label='Support Foot X', linewidth=2)
        if has_key('dcm_offset_actual_x'):
            # オフセットを表示（差分）
            offset = np.array(log['dcm_x']) - np.array(log['support_foot_actual_x'])
            plt.fill_between(log['time'], log['support_foot_actual_x'], log['dcm_x'], 
                           alpha=0.3, label='Offset (DCM - Support)', color='orange')
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Position [m]', fontsize=12)
    plt.title('Actual DCM Offset Components (X)\nDCM_x - Support_Foot_x', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # --- パネル4: Actual計算要素（Y方向） ---
    plt.subplot(3, 2, 4)
    if has_key('dcm_y') and has_key('support_foot_actual_y'):
        plt.plot(log['time'], log['dcm_y'], 'g-', label='DCM Actual Y', linewidth=2)
        plt.plot(log['time'], log['support_foot_actual_y'], 'b--', label='Support Foot Y', linewidth=2)
        if has_key('dcm_offset_actual_y'):
            # オフセットを表示（差分）
            offset = np.array(log['dcm_y']) - np.array(log['support_foot_actual_y'])
            plt.fill_between(log['time'], log['support_foot_actual_y'], log['dcm_y'], 
                           alpha=0.3, label='Offset (DCM - Support)', color='orange')
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Position [m]', fontsize=12)
    plt.title('Actual DCM Offset Components (Y)\nDCM_y - Support_Foot_y', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # --- パネル5: Desired計算要素（X方向） ---
    plt.subplot(3, 2, 5)
    if has_key('next_step_dcm_x') and has_key('next_step_support_foot_x'):
        plt.plot(log['time'], log['next_step_dcm_x'], 'r-', label='Next Step DCM X', linewidth=2)
        plt.plot(log['time'], log['next_step_support_foot_x'], 'b--', label='Next Support Foot X', linewidth=2)
        # オフセットを表示（差分）
        offset = np.array(log['next_step_dcm_x']) - np.array(log['next_step_support_foot_x'])
        plt.fill_between(log['time'], log['next_step_support_foot_x'], log['next_step_dcm_x'], 
                       alpha=0.3, label='Offset (Next DCM - Next Support)', color='purple')
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Position [m]', fontsize=12)
    plt.title('Desired DCM Offset Components (X)\nNext_DCM_x - Next_Support_x', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # --- パネル6: Desired計算要素（Y方向） ---
    plt.subplot(3, 2, 6)
    if has_key('next_step_dcm_y') and has_key('next_step_support_foot_y'):
        plt.plot(log['time'], log['next_step_dcm_y'], 'g-', label='Next Step DCM Y', linewidth=2)
        plt.plot(log['time'], log['next_step_support_foot_y'], 'b--', label='Next Support Foot Y', linewidth=2)
        # オフセットを表示（差分）
        offset = np.array(log['next_step_dcm_y']) - np.array(log['next_step_support_foot_y'])
        plt.fill_between(log['time'], log['next_step_support_foot_y'], log['next_step_dcm_y'], 
                       alpha=0.3, label='Offset (Next DCM - Next Support)', color='purple')
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Position [m]', fontsize=12)
    plt.title('Desired DCM Offset Components (Y)\nNext_DCM_y - Next_Support_y', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_filename = "control_analysis_dcm_offset.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"💾 グラフ保存: {output_filename}")
    
    output_filename_pdf = "control_analysis_dcm_offset.pdf"
    plt.savefig(output_filename_pdf, bbox_inches='tight')
    print(f"💾 グラフ保存: {output_filename_pdf}")
    
    #plt.show()


def plot_mj_foot_analysis(log):
    """MuJoCo直読みの足位置・速度（震え確認用）"""

    def has_key(key):
        return key in log and len(log[key]) > 0

    if not has_key('mj_foot_r_pos_x'):
        print("⚠️ mj_foot_* 列がありません（新しい control_log.csv が必要です）")
        return

    print(f"\n📊 MuJoCo足データグラフ生成中... ({len(log['time'])} サンプル)")

    fig = plt.figure(figsize=(15, 12))

    # --- パネル1: 足位置 ---
    plt.subplot(3, 1, 1)
    plt.plot(log['time'], log['mj_foot_r_pos_x'], 'r-', label='R pos X', linewidth=1.5)
    plt.plot(log['time'], log['mj_foot_r_pos_y'], 'r--', label='R pos Y', linewidth=1.5)
    plt.plot(log['time'], log['mj_foot_r_pos_z'], 'r:', label='R pos Z', linewidth=1.5)
    plt.plot(log['time'], log['mj_foot_l_pos_x'], 'b-', label='L pos X', linewidth=1.5)
    plt.plot(log['time'], log['mj_foot_l_pos_y'], 'b--', label='L pos Y', linewidth=1.5)
    plt.plot(log['time'], log['mj_foot_l_pos_z'], 'b:', label='L pos Z', linewidth=1.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('MuJoCo Foot Position (R_FOOT_R / L_FOOT_R body)')
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)

    # --- パネル2: 並進速度 ---
    plt.subplot(3, 1, 2)
    plt.plot(log['time'], log['mj_foot_r_linvel_x'], 'r-', label='R linvel X', linewidth=1.5)
    plt.plot(log['time'], log['mj_foot_r_linvel_y'], 'r--', label='R linvel Y', linewidth=1.5)
    plt.plot(log['time'], log['mj_foot_r_linvel_z'], 'r:', label='R linvel Z', linewidth=1.5)
    plt.plot(log['time'], log['mj_foot_l_linvel_x'], 'b-', label='L linvel X', linewidth=1.5)
    plt.plot(log['time'], log['mj_foot_l_linvel_y'], 'b--', label='L linvel Y', linewidth=1.5)
    plt.plot(log['time'], log['mj_foot_l_linvel_z'], 'b:', label='L linvel Z', linewidth=1.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.title('MuJoCo Foot Linear Velocity')
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)

    # --- パネル3: 角速度（震えの主指標） ---
    plt.subplot(3, 1, 3)
    plt.plot(log['time'], log['mj_foot_r_angvel_x'], 'r-', label='R angvel X', linewidth=1.5)
    plt.plot(log['time'], log['mj_foot_r_angvel_y'], 'r--', label='R angvel Y', linewidth=1.5)
    plt.plot(log['time'], log['mj_foot_r_angvel_z'], 'r:', label='R angvel Z', linewidth=1.5)
    plt.plot(log['time'], log['mj_foot_l_angvel_x'], 'b-', label='L angvel X', linewidth=1.5)
    plt.plot(log['time'], log['mj_foot_l_angvel_y'], 'b--', label='L angvel Y', linewidth=1.5)
    plt.plot(log['time'], log['mj_foot_l_angvel_z'], 'b:', label='L angvel Z', linewidth=1.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Angular velocity [rad/s]')
    plt.title('MuJoCo Foot Angular Velocity (vibration check)')
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('control_analysis_mj_foot.pdf', bbox_inches='tight')
    plt.savefig('control_analysis_mj_foot.png', dpi=150, bbox_inches='tight')
    print("💾 グラフ保存: control_analysis_mj_foot.pdf / .png")


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
        plot_dcm_offset_analysis(log)
        plot_mj_foot_analysis(log)
    else:
        print("⚠️ ログデータが空です")
    
    print("=" * 70)
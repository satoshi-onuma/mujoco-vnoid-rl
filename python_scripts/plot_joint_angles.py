"""
MuJoCo直読みの脚部関節角度・角速度を描画するスクリプト。
右脚・左脚それぞれ1ファイル（各関節: 上段=角度、下段=角速度）。
"""

import sys

import matplotlib.pyplot as plt

from plot_control_data import load_csv_log

# 脚部12関節（bindings.cpp の i=18..29 と同じ順序）
R_LEG_JOINT_NAMES = [
    "R_UPPERLEG_Y",
    "R_UPPERLEG_R",
    "R_UPPERLEG_P",
    "R_LOWERLEG_P",
    "R_FOOT_P",
    "R_FOOT_R",
]

L_LEG_JOINT_NAMES = [
    "L_UPPERLEG_Y",
    "L_UPPERLEG_R",
    "L_UPPERLEG_P",
    "L_LOWERLEG_P",
    "L_FOOT_P",
    "L_FOOT_R",
]


def plot_leg_joints(time, log, joint_names, side_label, output_basename):
    """片脚分の関節角度・角速度を1ファイルに描画する。"""
    n_joints = len(joint_names)
    fig, axes = plt.subplots(n_joints * 2, 1, figsize=(12, 3 * n_joints), sharex=True)

    for i, joint_name in enumerate(joint_names):
        q_key = f"joint_{joint_name}_q"
        dq_key = f"joint_{joint_name}_dq"

        if q_key not in log or dq_key not in log:
            print(f"  skip: {joint_name} (列 {q_key} / {dq_key} がありません)")
            continue

        ax_q = axes[i * 2]
        ax_dq = axes[i * 2 + 1]

        ax_q.plot(time, log[q_key], "k-", linewidth=1.5)
        ax_q.set_ylabel("Angle [rad]")
        ax_q.set_title(joint_name)
        ax_q.grid(True, alpha=0.3)

        ax_dq.plot(time, log[dq_key], "b-", linewidth=1.5)
        ax_dq.set_ylabel("Angular velocity [rad/s]")
        ax_dq.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"Leg Joint Angles ({side_label})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_basename}.pdf", bbox_inches="tight")
    plt.savefig(f"{output_basename}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {output_basename}.pdf / .png")


def plot_joint_angles(log):
    if log is None or len(log.get("time", [])) == 0:
        print("ログデータが空です")
        return

    if "joint_R_UPPERLEG_Y_q" not in log:
        print("joint_* 列がありません（新しい control_log.csv が必要です）")
        return

    time = log["time"]
    n = len(time)
    print(f"\n脚部関節グラフ生成中... ({n} サンプル, R/L 各 {len(R_LEG_JOINT_NAMES)} 関節)")

    plot_leg_joints(time, log, R_LEG_JOINT_NAMES, "Right", "joint_angles_R")
    plot_leg_joints(time, log, L_LEG_JOINT_NAMES, "Left", "joint_angles_L")


if __name__ == "__main__":
    csv_filename = sys.argv[1] if len(sys.argv) > 1 else "control_log.csv"
    print(f"CSV: {csv_filename}")

    log = load_csv_log(csv_filename)
    plot_joint_angles(log)

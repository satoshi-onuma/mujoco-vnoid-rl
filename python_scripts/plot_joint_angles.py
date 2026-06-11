"""
MuJoCo直読みの脚部関節角度・角速度を関節ごとに描画するスクリプト。
各関節1ファイル（上段: 角度、下段: 角速度）。
"""

import sys

import matplotlib.pyplot as plt

from plot_control_data import load_csv_log

# 脚部12関節（bindings.cpp の i=18..29 と同じ順序）
LEG_JOINT_NAMES = [
    "R_UPPERLEG_Y",
    "R_UPPERLEG_R",
    "R_UPPERLEG_P",
    "R_LOWERLEG_P",
    "R_FOOT_P",
    "R_FOOT_R",
    "L_UPPERLEG_Y",
    "L_UPPERLEG_R",
    "L_UPPERLEG_P",
    "L_LOWERLEG_P",
    "L_FOOT_P",
    "L_FOOT_R",
]


def plot_single_joint(time, log, joint_name):
    """関節1つ分の角度・角速度を2段サブプロットで描画する。"""
    q_key = f"joint_{joint_name}_q"
    dq_key = f"joint_{joint_name}_dq"

    if q_key not in log or dq_key not in log:
        print(f"  skip: {joint_name} (列 {q_key} / {dq_key} がありません)")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(time, log[q_key], "k-", linewidth=1.5)
    axes[0].set_ylabel("Angle [rad]")
    axes[0].set_title(f"Joint: {joint_name}")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, log[dq_key], "b-", linewidth=1.5)
    axes[1].set_ylabel("Angular velocity [rad/s]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_basename = f"joint_{joint_name}"
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
    print(f"\n脚部関節グラフ生成中... ({n} サンプル, {len(LEG_JOINT_NAMES)} 関節)")

    for joint_name in LEG_JOINT_NAMES:
        plot_single_joint(time, log, joint_name)


if __name__ == "__main__":
    csv_filename = sys.argv[1] if len(sys.argv) > 1 else "control_log.csv"
    print(f"CSV: {csv_filename}")

    log = load_csv_log(csv_filename)
    plot_joint_angles(log)

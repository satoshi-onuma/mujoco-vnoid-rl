"""
MuJoCo直読みの足位置・並進速度・角速度を別ファイルに描画するスクリプト。
各ファイル内で X/Y/Z をサブプロット分離し、R/L のみ重ねる。
"""

import sys

import matplotlib.pyplot as plt

from plot_control_data import load_csv_log


def plot_axis_triplet(time, log, prefix_r, prefix_l, ylabel, title, output_basename):
    """X/Y/Z を縦3段のサブプロットに分けて R/L を描画する。"""
    axes_labels = ("X", "Y", "Z")
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    for ax, axis in zip(axes, axes_labels):
        key_r = f"{prefix_r}_{axis.lower()}"
        key_l = f"{prefix_l}_{axis.lower()}"
        ax.plot(time, log[key_r], "r-", label="Right", linewidth=1.5)
        ax.plot(time, log[key_l], "b-", label="Left", linewidth=1.5)
        ax.set_ylabel(f"{ylabel} {axis}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_basename}.pdf", bbox_inches="tight")
    plt.savefig(f"{output_basename}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {output_basename}.pdf / .png")


def plot_mj_foot_separate(log):
    if log is None or len(log.get("time", [])) == 0:
        print("ログデータが空です")
        return

    if "mj_foot_r_pos_x" not in log:
        print("mj_foot_* 列がありません（新しい control_log.csv が必要です）")
        return

    time = log["time"]
    n = len(time)
    print(f"\nMuJoCo足データ別グラフ生成中... ({n} サンプル)")

    plot_axis_triplet(
        time,
        log,
        "mj_foot_r_pos",
        "mj_foot_l_pos",
        "[m]",
        "MuJoCo Foot Position (R_FOOT_R / L_FOOT_R body)",
        "mj_foot_position",
    )
    plot_axis_triplet(
        time,
        log,
        "mj_foot_r_linvel",
        "mj_foot_l_linvel",
        "[m/s]",
        "MuJoCo Foot Linear Velocity",
        "mj_foot_linvel",
    )
    plot_axis_triplet(
        time,
        log,
        "mj_foot_r_angvel",
        "mj_foot_l_angvel",
        "[rad/s]",
        "MuJoCo Foot Angular Velocity",
        "mj_foot_angvel",
    )


if __name__ == "__main__":
    csv_filename = sys.argv[1] if len(sys.argv) > 1 else "control_log.csv"
    print(f"CSV: {csv_filename}")

    log = load_csv_log(csv_filename)
    plot_mj_foot_separate(log)

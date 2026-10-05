#!/usr/bin/env python3
"""論文用: control_log.csv を最大3本重ねて、信号ごと別フィギュアで出す。

subplot にはしない（論文側で並べる想定）。出力は PNG + PDF。

図（10枚）:
  t_x.png              time–base link position x（開始相対）
  t_y.png              time–base link position y（開始相対）
  t_z.png              time–base link position z
  t_deltax.png         time–RL action Δx
  x_y.png              軌道 x–y（開始相対）
  t_dcm_error_x.png    time–DCM error local x（ベース前方）
  t_dcm_error_y.png    time–DCM error local y（ベース左右）
  t_dcm_error_norm.png time–DCM error norm
  t_foot_sink_right.png time–foot sink right
  t_foot_sink_left.png  time–foot sink left

Usage:
  python plot_paper_case_compare.py \\
    --trial baseline:~/vnoid-experiments/paper_cases/baseline/seed1001_softness_1.00_control_log.csv \\
    --trial success:~/.../success/..._control_log.csv \\
    --trial fail:~/.../fail/..._control_log.csv \\
    --switch-time 2.5 \\
    -o ~/vnoid-experiments/plots/paper_case/

撮影（record_humanoid.py）:
  cd python_scripts
  # ベース（介入なし）— 動画 + control_log
  python record_humanoid.py \\
    --checkpoint-dir ~/vnoid-experiments/runs/<run>/checkpoint \\
    --intervention-mode none \\
    --terrain-softness 1.0 --seed 1001 \\
    --run-dir ~/vnoid-experiments/paper_cases/baseline
  # 成功 / 失敗は --intervention-mode full と checkpoint を変えて同様。
  # 図に使うのは *control_log.csv のみ（*_recording_log.csv は不要）。
  # --run-dir 付きだと DIR/seed*_softness_*_control_log.csv と *_demo.mp4 が出る。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 16
plt.rcParams["axes.labelsize"] = 24
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["legend.fontsize"] = 18
plt.rcParams["xtick.labelsize"] = 22
plt.rcParams["ytick.labelsize"] = 22
plt.rcParams["figure.dpi"] = 150
plt.rcParams["lines.linewidth"] = 1.5

REQUIRED_COLS = (
    "time",
    "base_pos_x",
    "base_pos_y",
    "base_pos_z",
    "rl_action_foot_offset_x",
    "dcm_error_local_x",
    "dcm_error_local_y",
    "dcm_error_norm",
    "obs_foot_sink_right",
    "obs_foot_sink_left",
)

FIGURE_SPECS: tuple[tuple[str, str, str, str, bool], ...] = (
    # stem, xlabel, ylabel, y_key, mark_switch
    ("t_x", "Time [s]", "Base link position\n$x$ [m]", "x_rel", True),
    ("t_y", "Time [s]", "Base link position\n$y$ [m]", "y_rel", True),
    ("t_z", "Time [s]", "Base link position\n$z$ [m]", "base_pos_z", True),
    ("t_deltax", "Time [s]", "RL action\n$\\Delta x$ [m]", "rl_action_foot_offset_x", True),
    ("x_y", "Base link position $x$ [m]", "Base link position\n$y$ [m]", "xy", False),
    ("t_dcm_error_x", "Time [s]", "DCM error local\n$x$ [m]", "dcm_error_local_x", True),
    ("t_dcm_error_y", "Time [s]", "DCM error local\n$y$ [m]", "dcm_error_local_y", True),
    ("t_dcm_error_norm", "Time [s]", "DCM error norm\n[m]", "dcm_error_norm", True),
    ("t_foot_sink_right", "Time [s]", "Foot sink right\n[m]", "obs_foot_sink_right", True),
    ("t_foot_sink_left", "Time [s]", "Foot sink left\n[m]", "obs_foot_sink_left", True),
)

# 色が消えても判別できるよう線種を固定（最大3本）
LINE_STYLES = ("-", "--", "-.")

# 論文で並べたとき中枠（axes）が揃うよう固定。figsize 共通 + 位置固定。
FIGSIZE = (11.0, 4.0)
# left, bottom, width, height（figure 座標）。長い ylabel 用に左余白を確保。
AXES_RECT = (0.18, 0.22, 0.78, 0.70)


def apply_fixed_axes(ax) -> None:
    """ラベル長さに依存せず中枠サイズを固定する。"""
    ax.set_position(AXES_RECT)


def resolve_control_log(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        matches = sorted(path.glob("*control_log.csv"))
        if len(matches) == 1:
            return matches[0]
        if matches:
            names = ", ".join(m.name for m in matches)
            raise FileNotFoundError(
                f"control_log が複数あります。ファイルを直接指定してください: {names}"
            )
        raise FileNotFoundError(f"control_log.csv がありません: {path}")
    raise FileNotFoundError(f"パスが存在しません: {path}")


def parse_trial(raw: str) -> tuple[str, Path]:
    if ":" not in raw:
        raise argparse.ArgumentTypeError(
            f"--trial は label:path 形式です: {raw!r}"
        )
    label, path_str = raw.split(":", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"ラベルが空です: {raw!r}")
    return label, Path(path_str.strip())


def load_control_log(path: Path) -> tuple[Path, dict[str, np.ndarray]]:
    csv_path = resolve_control_log(path)
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"ヘッダがありません: {csv_path}")
        missing = [c for c in REQUIRED_COLS if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"必要な列がありません ({csv_path}): {missing}\n"
                "dcm_error_local_x/y と dcm_error_norm 付き control_log を渡してください。"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"データが空です: {csv_path}")

    data: dict[str, np.ndarray] = {
        col: np.array([float(r[col]) for r in rows], dtype=np.float64)
        for col in REQUIRED_COLS
    }
    data["x_rel"] = data["base_pos_x"] - data["base_pos_x"][0]
    data["y_rel"] = data["base_pos_y"] - data["base_pos_y"][0]
    return csv_path, data


def plot_cases(
    members: list[tuple[str, dict[str, np.ndarray]]],
    output_dir: Path,
    switch_time: float | None = None,
    *,
    write_pdf: bool = True,
) -> list[Path]:
    if not members:
        raise ValueError("trial がありません")
    if len(members) > 3:
        raise ValueError("trial は最大3本です")

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    written: list[Path] = []

    for stem, xlabel, ylabel, y_key, mark_sw in FIGURE_SPECS:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for mi, (label, data) in enumerate(members):
            color = colors[mi % len(colors)]
            ls = LINE_STYLES[mi % len(LINE_STYLES)]
            if y_key == "xy":
                ax.plot(
                    data["x_rel"],
                    data["y_rel"],
                    color=color,
                    linestyle=ls,
                    label=label,
                )
                # datalim: 中枠サイズは変えずデータ範囲側で等方にする
                ax.set_aspect("equal", adjustable="datalim")
            else:
                ax.plot(
                    data["time"],
                    data[y_key],
                    color=color,
                    linestyle=ls,
                    label=label,
                )
        if mark_sw and switch_time is not None:
            # データ線（- / -- / -.）と被らない点線
            ax.axvline(
                switch_time,
                color="k",
                linestyle=":",
                linewidth=1.2,
                alpha=0.7,
            )
        if y_key != "xy":
            ax.axhline(0.0, color="k", linestyle=(0, (1, 3)), alpha=0.25)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(
            direction="in",
            which="both",
            top=True,
            right=True,
            width=1.2,
            length=6,
            labelsize=22,
        )
        ax.grid(True, alpha=0.3)
        ax.legend(
            loc="lower right",
            fontsize=18,
            frameon=True,
            fancybox=False,
            edgecolor="0.5",
        )
        apply_fixed_axes(ax)

        png = output_dir / f"{stem}.png"
        # bbox_inches=tight は中枠サイズを崩すので使わない
        fig.savefig(png, dpi=150)
        written.append(png)
        if write_pdf:
            pdf = output_dir / f"{stem}.pdf"
            fig.savefig(pdf)
            written.append(pdf)
        plt.close(fig)

    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="control_log 最大3本を重ね、信号ごと別フィギュアで保存"
    )
    parser.add_argument(
        "--trial",
        action="append",
        default=[],
        metavar="LABEL:PATH",
        help="label:control_log.csv（または親ディレクトリ）。最大3回",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="出力ディレクトリ（t_x.png 等を書く）",
    )
    parser.add_argument(
        "--switch-time",
        type=float,
        default=None,
        help="地盤切替の縦線 [s]（省略可）",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="PDF を出さない",
    )
    args = parser.parse_args()

    if not args.trial:
        print("❌ --trial を1つ以上指定してください", file=sys.stderr)
        return 1
    if len(args.trial) > 3:
        print("❌ --trial は最大3本です", file=sys.stderr)
        return 1

    try:
        members: list[tuple[str, dict[str, np.ndarray]]] = []
        for raw in args.trial:
            label, path = parse_trial(raw)
            src, data = load_control_log(path)
            members.append((label, data))
            print(f"  loaded {label}: {src} ({len(data['time'])} samples)")

        out_dir = Path(args.output_dir).expanduser().resolve()
        written = plot_cases(
            members,
            out_dir,
            switch_time=args.switch_time,
            write_pdf=not args.no_pdf,
        )
    except (FileNotFoundError, ValueError, argparse.ArgumentTypeError) as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    print(f"✅ 出力先: {out_dir}")
    for p in written:
        if p.suffix == ".png":
            print(f"   {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

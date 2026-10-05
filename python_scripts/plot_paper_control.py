#!/usr/bin/env python3
"""論文用 slim control.csv（1 trial）を描画する。複数 trial の7パネル比較も可。

Usage:
  python python_scripts/plot_paper_control.py \\
    ~/vnoid-experiments/paper_logs/.../seed1001_softness_1.00

  python python_scripts/plot_paper_control.py path/to/control.csv -o out.png

  # 複数 trial 比較（7パネル・色分け）。-o 必須
  python python_scripts/plot_paper_control.py \\
    path/to/trial_a path/to/trial_b \\
    --labels full,after_switch \\
    -o ~/vnoid-experiments/plots/paper_compare.png

UI からは plot_paper_trial(...) / plot_paper_trials_compare(...) を呼べる。
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.dpi"] = 150
plt.rcParams["lines.linewidth"] = 1.3

REQUIRED_COLS = (
    "time",
    "terrain_switched",
    "dcm_error_norm",
    "dcm_error_local_x",
    "dcm_error_local_y",
    "obs_foot_sink_right",
    "obs_foot_sink_left",
    "delta_x",
    "delta_y",
)

TRIAL_DIR_RE = re.compile(
    r"seed(?P<seed>\d+)_softness_(?P<softness>[0-9]+(?:\.[0-9]+)?)$"
)

# Paper比較タブ: 各サブプロット1信号
COMPARE_SERIES: tuple[tuple[str, str], ...] = (
    ("dcm_error_norm", "DCM error norm [m]"),
    ("dcm_error_local_x", "DCM error local x [m]"),
    ("dcm_error_local_y", "DCM error local y [m]"),
    ("obs_foot_sink_right", "Foot sink R [m]"),
    ("obs_foot_sink_left", "Foot sink L [m]"),
    ("delta_x", "Δx [m]"),
    ("delta_y", "Δy [m]"),
)


def resolve_control_csv(path: Path) -> Path:
    """control.csv または trial ディレクトリ → control.csv パス。"""
    path = path.expanduser().resolve()
    if path.is_file():
        if path.name != "control.csv":
            # ファイル名が違っても CSV として読む（明示指定）
            return path
        return path
    if path.is_dir():
        candidate = path / "control.csv"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"control.csv がありません: {candidate}")
    raise FileNotFoundError(f"パスが存在しません: {path}")


def load_paper_control(path: Path) -> dict[str, np.ndarray]:
    """slim control.csv を列名 → ndarray の dict で返す。"""
    csv_path = resolve_control_csv(path)
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"ヘッダがありません: {csv_path}")
        missing = [c for c in REQUIRED_COLS if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"必要な列がありません ({csv_path}): {missing}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"データが空です: {csv_path}")

    data: dict[str, np.ndarray] = {}
    for col in reader.fieldnames:
        data[col] = np.array([float(r[col]) for r in rows], dtype=np.float64)
    return data


def switch_time(data: dict[str, np.ndarray]) -> float | None:
    """terrain_switched が初めて 1 になる time。無ければ None。"""
    switched = data["terrain_switched"]
    idx = np.flatnonzero(switched >= 0.5)
    if idx.size == 0:
        return None
    return float(data["time"][idx[0]])


def infer_title(csv_path: Path, data: dict[str, np.ndarray]) -> str:
    """meta.txt または親ディレクトリ名からタイトルを作る。"""
    trial_dir = csv_path.parent
    meta_path = trial_dir / "meta.txt"
    seed: str | None = None
    soft: str | None = None
    success: str | None = None
    steps: str | None = None

    if meta_path.is_file():
        meta: dict[str, str] = {}
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()
        seed = meta.get("seed")
        soft = meta.get("softness")
        success = meta.get("success")
        steps = meta.get("steps")

    if seed is None or soft is None:
        m = TRIAL_DIR_RE.search(trial_dir.name)
        if m:
            seed = seed or m.group("seed")
            soft = soft or m.group("softness")

    parts = ["Paper control"]
    if seed is not None and soft is not None:
        parts.append(f"seed={seed} softness={soft}")
    elif seed is not None:
        parts.append(f"seed={seed}")
    if success is not None and steps is not None:
        parts.append(f"success={success} steps={steps}")

    t_sw = switch_time(data)
    if t_sw is not None:
        parts.append(f"switch@t={t_sw:.2f}s")

    return " | ".join(parts)


def _mark_switch(ax: Any, t_sw: float | None, *, label: str | None = None) -> None:
    if t_sw is None:
        return
    ax.axvline(
        t_sw,
        color="k",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label=label,
    )


def build_figure(
    data: dict[str, np.ndarray],
    title: str,
) -> plt.Figure:
    t = data["time"]
    t_sw = switch_time(data)

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1) error norm
    axes[0].plot(t, data["dcm_error_norm"], color="purple", label="‖error‖")
    _mark_switch(axes[0], t_sw, label="terrain switch")
    axes[0].set_ylabel("DCM error norm [m]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    # 2) error xy
    axes[1].plot(t, data["dcm_error_local_x"], "r-", label="error x")
    axes[1].plot(t, data["dcm_error_local_y"], "g-", label="error y")
    axes[1].axhline(0.0, color="k", linestyle=":", alpha=0.3)
    _mark_switch(axes[1], t_sw)
    axes[1].set_ylabel("DCM error local [m]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    # 3) sink
    axes[2].plot(t, data["obs_foot_sink_right"], "r-", label="sink R")
    axes[2].plot(t, data["obs_foot_sink_left"], "b-", label="sink L")
    axes[2].axhline(0.0, color="k", linestyle=":", alpha=0.3)
    _mark_switch(axes[2], t_sw)
    axes[2].set_ylabel("Foot sink [m]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    # 4) delta
    axes[3].plot(t, data["delta_x"], "r-", label="Δx")
    axes[3].plot(t, data["delta_y"], "g-", label="Δy")
    axes[3].axhline(0.0, color="k", linestyle=":", alpha=0.3)
    _mark_switch(axes[3], t_sw)
    axes[3].set_ylabel("RL delta [m]")
    axes[3].set_xlabel("Time [s]")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="upper right")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def draw_compare_on_figure(
    fig: plt.Figure,
    members: list[tuple[str, dict[str, np.ndarray]]],
    title: str | None = None,
) -> None:
    """既存 Figure を clear して7パネル比較を描く。"""
    if not members:
        raise ValueError("比較する trial がありません")

    fig.clear()
    n = len(COMPARE_SERIES)
    axes = fig.subplots(n, 1, sharex=True)
    if n == 1:
        axes = [axes]
    fig_title = title or f"Paper control compare ({len(members)} trials)"
    fig.suptitle(fig_title, fontsize=14, fontweight="bold")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    zero_keys = {
        "dcm_error_local_x",
        "dcm_error_local_y",
        "obs_foot_sink_right",
        "obs_foot_sink_left",
        "delta_x",
        "delta_y",
    }

    for mi, (label, data) in enumerate(members):
        color = colors[mi % len(colors)]
        t = data["time"]
        t_sw = switch_time(data)
        for ai, (key, _ylabel) in enumerate(COMPARE_SERIES):
            axes[ai].plot(t, data[key], color=color, label=label if ai == 0 else None)
            if t_sw is not None:
                switch_label = "switch" if mi == 0 and ai == 0 else None
                axes[ai].axvline(
                    t_sw,
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.55,
                    label=switch_label,
                )

    for ai, (key, ylabel) in enumerate(COMPARE_SERIES):
        if key in zero_keys:
            axes[ai].axhline(0.0, color="k", linestyle=":", alpha=0.3)
        axes[ai].set_ylabel(ylabel)
        axes[ai].grid(True, alpha=0.3)
        if ai == 0:
            axes[ai].legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout(rect=(0, 0, 1, 0.97))


def build_compare_figure(
    members: list[tuple[str, dict[str, np.ndarray]]],
    title: str | None = None,
) -> plt.Figure:
    """複数 trial を7パネル（各1信号）で重ね描き。members: (label, data)。"""
    fig = plt.figure(figsize=(11, 14))
    draw_compare_on_figure(fig, members, title=title)
    return fig


def _safe_label_prefix(label: str) -> str:
    """ファイル名用にラベルをサニタイズ（凡例名を先頭に付ける用）。"""
    s = re.sub(r"[^\w.\-]+", "_", label.strip(), flags=re.UNICODE)
    s = s.strip("._")
    return s or "trial"


def copy_labeled_control_csvs(
    members: list[tuple[str, str | Path]],
    dest_dir: Path,
) -> list[Path]:
    """各 trial の control.csv を `{label}_control.csv` として dest_dir へコピー。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for label, path in members:
        src = resolve_control_csv(Path(path))
        out = dest_dir / f"{_safe_label_prefix(label)}_control.csv"
        shutil.copy2(src, out)
        written.append(out)
    return written


def plot_paper_trials_compare(
    members: list[tuple[str, str | Path]],
    output_path: str | Path,
    title: str | None = None,
    *,
    write_pdf: bool = True,
    copy_control_csv: bool = True,
) -> Path:
    """複数 trial を7パネル比較して PNG 保存。members: (label, control path)。

    copy_control_csv=True のとき、PNG と同じディレクトリに
    `{凡例ラベル}_control.csv` も保存する。
    """
    if not members:
        raise ValueError("比較する trial がありません")

    loaded: list[tuple[str, dict[str, np.ndarray]]] = []
    for label, path in members:
        csv_path = resolve_control_csv(Path(path))
        loaded.append((label, load_paper_control(csv_path)))

    png_path = Path(output_path).expanduser().resolve()
    if png_path.suffix.lower() == ".pdf":
        png_path = png_path.with_suffix(".png")
    elif png_path.suffix.lower() != ".png":
        png_path = png_path.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)

    fig = build_compare_figure(loaded, title=title)
    fig.savefig(png_path, dpi=150)
    if write_pdf:
        fig.savefig(png_path.with_suffix(".pdf"))
    plt.close(fig)

    if copy_control_csv:
        copy_labeled_control_csvs(members, png_path.parent)
    return png_path


def _default_trial_label(path: Path) -> str:
    p = path.expanduser().resolve()
    if p.is_file() and p.name == "control.csv":
        return p.parent.name
    return p.name


def plot_paper_trial(
    control_path: str | Path,
    output_path: str | Path | None = None,
    title: str | None = None,
    *,
    write_pdf: bool = True,
) -> Path:
    """1 trial の paper control を描画して PNG を保存。戻り値は PNG パス。

    control_path: control.csv または trial ディレクトリ
    output_path: 未指定なら control.csv と同じディレクトリの paper_control.png
    """
    csv_path = resolve_control_csv(Path(control_path))
    data = load_paper_control(csv_path)

    if output_path is None:
        png_path = csv_path.parent / "paper_control.png"
    else:
        png_path = Path(output_path).expanduser().resolve()
        if png_path.suffix.lower() == ".pdf":
            png_path = png_path.with_suffix(".png")
        elif png_path.suffix.lower() != ".png":
            png_path = png_path.with_suffix(".png")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig_title = title if title is not None else infer_title(csv_path, data)
    fig = build_figure(data, fig_title)
    fig.savefig(png_path, dpi=150)
    if write_pdf:
        pdf_path = png_path.with_suffix(".pdf")
        fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(
        description="論文用 slim control.csv を描画（1 trial=4パネル / 複数=7パネル比較）"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=str,
        help="control.csv または trial ディレクトリ（複数で比較モード）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="出力 PNG。比較時は必須。単体時省略で <trial>/paper_control.png",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="",
        help="比較時のラベル（カンマ区切、inputs と同数）。省略時は trial ディレクトリ名",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="PDF を出力しない",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="図タイトル",
    )
    args = parser.parse_args()

    try:
        if len(args.inputs) == 1:
            png = plot_paper_trial(
                args.inputs[0],
                output_path=args.output,
                title=args.title,
                write_pdf=not args.no_pdf,
            )
        else:
            if not args.output:
                print("❌ 比較モードでは -o/--output が必須です")
                return 1
            label_parts = [p.strip() for p in args.labels.split(",") if p.strip()]
            if label_parts and len(label_parts) != len(args.inputs):
                print("❌ --labels の個数が inputs と一致しません")
                return 1
            members: list[tuple[str, Path]] = []
            for i, raw in enumerate(args.inputs):
                path = Path(raw)
                label = (
                    label_parts[i]
                    if label_parts
                    else _default_trial_label(path)
                )
                members.append((label, path))
            png = plot_paper_trials_compare(
                members,
                output_path=args.output,
                title=args.title,
                write_pdf=not args.no_pdf,
            )
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}")
        return 1

    print(f"✅ PNG: {png}")
    if not args.no_pdf:
        print(f"   PDF: {png.with_suffix('.pdf')}")
    if len(args.inputs) > 1:
        label_parts = [p.strip() for p in args.labels.split(",") if p.strip()]
        for i, raw in enumerate(args.inputs):
            lab = (
                label_parts[i]
                if label_parts
                else _default_trial_label(Path(raw))
            )
            print(f"   CSV: {png.parent / f'{_safe_label_prefix(lab)}_control.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

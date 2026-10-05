#!/usr/bin/env python3
"""複数 seed の training_stats.csv を iteration ごとに平均±std で学習曲線（Episode Length）描画。

Usage:
  python python_scripts/plot_training_curves_seeds.py \\
    --group 'hard:~/vnoid-experiments/compare/hard' \\
    --group 'soft:~/vnoid-experiments/compare/soft' \\
    --output ~/vnoid-experiments/plots/episode_length_curves.png

  # run ディレクトリを列挙する場合
  python python_scripts/plot_training_curves_seeds.py \\
    --group 'cond:~/vnoid-experiments/runs/run_a,~/vnoid-experiments/runs/run_b,~/vnoid-experiments/runs/run_c' \\
    --output ~/vnoid-experiments/plots/episode_length_curves.png

  --group は何回でも追加可。各グループ内の CSV 本数 = seed 数（1本でも可）。
  各 dir は run 直下（training_stats.csv がある場所）か、
  その配下を再帰探索して training_stats.csv を集める親ディレクトリ。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["figure.dpi"] = 150
plt.rcParams["lines.linewidth"] = 1.5

DEFAULT_OUTPUT = (
    Path.home() / "vnoid-experiments" / "plots" / "training_curves_episode_len.png"
)

# 色が分からなくても区別できるよう線種を循環
LINESTYLES = ("-", "--", "-.", ":")


def parse_group(raw: str) -> tuple[str, list[Path]]:
    """'label:dir1,dir2,...' を (label, [Path, ...]) に分解する。"""
    if ":" not in raw:
        raise argparse.ArgumentTypeError(
            f"--group は 'ラベル:dir1,dir2,...' 形式です: {raw!r}"
        )
    label, dirs_raw = raw.split(":", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"グループラベルが空です: {raw!r}")
    dirs = []
    for part in dirs_raw.split(","):
        part = part.strip()
        if not part:
            continue
        dirs.append(Path(part).expanduser().resolve())
    if not dirs:
        raise argparse.ArgumentTypeError(f"ディレクトリがありません: {raw!r}")
    return label, dirs


def find_training_stats_csvs(base: Path) -> list[Path]:
    """run 直下 or 配下の training_stats.csv を列挙（重複なし・ソート）。"""
    direct = base / "training_stats.csv"
    if direct.is_file():
        return [direct]
    found = sorted({p.resolve() for p in base.rglob("training_stats.csv")})
    return found


def load_episode_lengths(csv_path: Path) -> tuple[list[int], list[float]]:
    """training_stats.csv から iteration と episode_len_mean を読む。"""
    iterations: list[int] = []
    episode_lens: list[float] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "episode_len_mean" not in reader.fieldnames:
            raise ValueError(f"episode_len_mean 列がありません: {csv_path}")
        for row in reader:
            iterations.append(int(row["iteration"]))
            episode_lens.append(float(row["episode_len_mean"]))
    if not iterations:
        raise ValueError(f"データが空です: {csv_path}")
    return iterations, episode_lens


def aggregate_group(
    label: str, dirs: list[Path]
) -> tuple[list[int], list[float], list[float], list[int]]:
    """グループ内で iteration ごとの episode_len mean/std/n を返す。"""
    csv_paths: list[Path] = []
    for d in dirs:
        csv_paths.extend(find_training_stats_csvs(d))

    # 同一ファイルの重複を除く
    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for p in csv_paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    if not unique_paths:
        raise FileNotFoundError(f"[{label}] training_stats.csv が見つかりません")

    series: list[tuple[list[int], list[float]]] = []
    for p in unique_paths:
        iters, ep_lens = load_episode_lengths(p)
        series.append((iters, ep_lens))
        print(f"  [{label}] loaded {p} ({len(iters)} iterations)")

    min_len = min(len(ep) for _, ep in series)
    if min_len == 0:
        raise ValueError(f"[{label}] 有効な iteration がありません")

    ref_iters = series[0][0][:min_len]
    for iters, ep_lens in series[1:]:
        if iters[:min_len] != ref_iters:
            print(
                f"  ⚠ [{label}] iteration 列が一致しない CSV があります。"
                " 先頭 {min_len} 行で truncate します。"
            )
            break

    stacked = np.asarray([ep[:min_len] for _, ep in series], dtype=float)
    means = np.mean(stacked, axis=0)
    stds = np.array(
        [
            float(np.std(stacked[:, i], ddof=1)) if stacked.shape[0] >= 2 else 0.0
            for i in range(min_len)
        ],
        dtype=float,
    )
    n = int(stacked.shape[0])
    ns = [n] * min_len

    return ref_iters, means.tolist(), stds.tolist(), ns


def write_aggregate_csv(
    out_csv: Path,
    rows: list[tuple[str, int, float, float, int]],
) -> None:
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "iteration",
                "episode_len_mean",
                "episode_len_std",
                "n",
            ],
        )
        writer.writeheader()
        for group, iteration, mean, std, n in rows:
            writer.writerow(
                {
                    "group": group,
                    "iteration": iteration,
                    "episode_len_mean": f"{mean:.4f}",
                    "episode_len_std": f"{std:.4f}",
                    "n": n,
                }
            )


def draw_groups(
    ax,
    series: list[tuple[str, list[int], list[float], list[float], list[int]]],
) -> None:
    """ax 上に Episode Length 学習曲線（mean±std）を描く。"""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (label, iters, means, stds, ns) in enumerate(series):
        color = colors[i % len(colors)]
        ls = LINESTYLES[i % len(LINESTYLES)]
        x = np.asarray(iters, dtype=float)
        y = np.asarray(means, dtype=float)
        s = np.asarray(stds, dtype=float)
        ax.plot(x, y, color=color, linestyle=ls, linewidth=1.0, label=label)
        if max(ns) >= 2:
            ax.fill_between(x, y - s, y + s, color=color, alpha=0.2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Episode Length")
    ax.tick_params(direction="in", which="both", labelsize=18)
    ax.grid(True, linestyle="-", linewidth=0.4, alpha=0.35)
    ax.legend(fontsize=16)


def plot_groups(
    series: list[tuple[str, list[int], list[float], list[float], list[int]]],
    output_png: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    draw_groups(ax, series)
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    output_pdf = output_png.with_suffix(".pdf")
    fig.savefig(output_pdf)
    plt.close(fig)
    return output_pdf


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(
        description="複数 seed の training_stats を episode length 学習曲線（mean±std）で比較"
    )
    parser.add_argument(
        "--group",
        action="append",
        type=parse_group,
        required=True,
        metavar="LABEL:DIR,DIR,...",
        help="比較グループ。複数回指定可。例: 'hard:parent_or_run_a,run_b'",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"出力 PNG（既定: {DEFAULT_OUTPUT}）。同名 PDF/CSV も保存",
    )
    args = parser.parse_args()

    output_png = Path(args.output).expanduser().resolve()
    if output_png.suffix.lower() == ".pdf":
        output_png = output_png.with_suffix(".png")
    elif output_png.suffix.lower() != ".png":
        output_png = output_png.with_suffix(".png")

    series: list[tuple[str, list[int], list[float], list[float], list[int]]] = []
    csv_rows: list[tuple[str, int, float, float, int]] = []

    print("Loading groups...")
    try:
        for label, dirs in args.group:
            missing = [str(d) for d in dirs if not d.exists()]
            if missing:
                print("❌ ディレクトリがありません:")
                for m in missing:
                    print(f"  - {m}")
                return 1
            iters, means, stds, ns = aggregate_group(label, dirs)
            series.append((label, iters, means, stds, ns))
            for iteration, mean, std, n in zip(iters, means, stds, ns):
                csv_rows.append((label, iteration, mean, std, n))
            print(f"  [{label}] seeds={ns[0] if ns else 0} iterations={len(iters)}")
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}")
        return 1

    output_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv = output_png.with_name(output_png.stem + ".csv")
    write_aggregate_csv(out_csv, csv_rows)
    out_pdf = plot_groups(series, output_png)

    print(f"\nPNG: {output_png}")
    print(f"PDF: {out_pdf}")
    print(f"CSV: {out_csv}")
    print("✅ 完了")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""複数 eval の softness×成功率をグループ比較で重ね描きする。

Usage:
  python python_scripts/plot_softness_success.py \\
    --group 'w_act=1.0:~/vnoid-experiments/evals/dir_a,~/vnoid-experiments/evals/dir_b,~/vnoid-experiments/evals/dir_c' \\
    --group 'w_act=0.1:~/vnoid-experiments/evals/dir_d,~/vnoid-experiments/evals/dir_e,~/vnoid-experiments/evals/dir_f' \\
    --group 'baseline:~/vnoid-experiments/evals/humanoid_vnoid_checkpoint_YYYYMMDD_HHMMSS' \\
    --output ~/vnoid-experiments/evals/compare_wact_success.png

  --style mean_std   … 学習シード横断の mean±std（既定）
  --style per_seed   … 各学習シード（eval dir）を細線 + 平均線

  # 保存済みシード CSV から再描画（フォント変更など）
  python python_scripts/plot_softness_success.py \\
    --from-seeds-csv ~/vnoid-experiments/plots/.../compare_softness_success_seeds.csv \\
    --style per_seed \\
    --output ~/vnoid-experiments/plots/.../compare_softness_success.png

  --group は何回でも追加可。各グループのディレクトリ数も可変（1本でも10本でも可）。
  各 dir は eval_softness_grid.py の出力ディレクトリ（summary.csv がある場所）。
  1 dir = 1 学習シード。summary の成功率は dir 内の評価シード平均。
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 16
plt.rcParams["axes.labelsize"] = 24
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["legend.fontsize"] = 18
plt.rcParams["xtick.labelsize"] = 22
plt.rcParams["ytick.labelsize"] = 22
plt.rcParams["figure.dpi"] = 150
plt.rcParams["lines.linewidth"] = 1.5

DEFAULT_OUTPUT = Path.home() / "vnoid-experiments" / "evals" / "compare_softness_success.png"

PlotStyle = Literal["mean_std", "per_seed"]
STYLES: tuple[PlotStyle, ...] = ("mean_std", "per_seed")

# 追加順: 1番目=点線, 2番目=実線, …
GROUP_LINESTYLES: tuple[str, ...] = ("--", "-", "-.", ":")

FIGSIZE = (8.0, 5.0)
# left, bottom, width, height（figure 座標）。ラベルサイズに依存せず中枠を揃える。
AXES_RECT = (0.16, 0.15, 0.78, 0.78)


def apply_fixed_axes(ax: Axes) -> None:
    ax.set_position(AXES_RECT)

# seed_dir, softs, rates
MemberCurve = tuple[str, list[float], list[float]]

# label, softs, means, stds, ns, member_curves
GroupSeries = tuple[
    str,
    list[float],
    list[float],
    list[float],
    list[int],
    list[MemberCurve],
]


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


def load_summary(eval_dir: Path) -> dict[float, float]:
    """summary.csv から softness -> success_rate（overall 除外）。"""
    path = eval_dir / "summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"summary.csv がありません: {path}")
    rates: dict[float, float] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            soft = row["softness"].strip()
            if soft == "overall":
                continue
            rates[float(soft)] = float(row["success_rate"])
    if not rates:
        raise ValueError(f"softness データが空です: {path}")
    return rates


def load_group_members(
    dirs: list[Path],
) -> list[MemberCurve]:
    """各学習シード（eval dir）の (seed_dir, softs, rates) を返す。"""
    members: list[MemberCurve] = []
    for d in dirs:
        rates = load_summary(d)
        softs = sorted(rates.keys())
        members.append((d.name, softs, [rates[s] for s in softs]))
    return members


def _aggregate_from_members(
    members: list[MemberCurve],
) -> tuple[list[float], list[float], list[float], list[int]]:
    """学習シード別曲線から soft ごとの mean/std/n を返す。"""
    per_soft: dict[float, list[float]] = defaultdict(list)
    for _seed, softs, rates in members:
        for soft, rate in zip(softs, rates):
            per_soft[soft].append(rate)

    softs = sorted(per_soft.keys())
    means, stds, ns = [], [], []
    for soft in softs:
        vals = np.asarray(per_soft[soft], dtype=float)
        means.append(float(np.mean(vals)))
        # 標本標準偏差（N=1 は 0）
        stds.append(float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0)
        ns.append(int(len(vals)))
    return softs, means, stds, ns


def aggregate_group(
    label: str, dirs: list[Path], *, verbose: bool = True
) -> tuple[list[float], list[float], list[float], list[int]]:
    """グループ内で softeness ごとの mean/std/n を返す。"""
    members = load_group_members(dirs)
    if verbose:
        for seed, softs, _ in members:
            print(f"  [{label}] loaded {seed} ({len(softs)} points)")
    return _aggregate_from_members(members)


def build_group_series(
    label: str, dirs: list[Path], *, verbose: bool = True
) -> GroupSeries:
    """集計 + 学習シード別曲線をまとめて返す。"""
    members = load_group_members(dirs)
    if verbose:
        for seed, softs, _ in members:
            print(f"  [{label}] loaded {seed} ({len(softs)} points)")
    softs, means, stds, ns = _aggregate_from_members(members)
    return (label, softs, means, stds, ns, members)


def write_aggregate_csv(
    out_csv: Path,
    rows: list[tuple[str, float, float, float, int]],
) -> None:
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["group", "softness", "mean", "std", "n"]
        )
        writer.writeheader()
        for group, soft, mean, std, n in rows:
            writer.writerow(
                {
                    "group": group,
                    "softness": f"{soft:.2f}",
                    "mean": f"{mean:.4f}",
                    "std": f"{std:.4f}",
                    "n": n,
                }
            )


def write_seeds_csv(out_csv: Path, series: list[GroupSeries]) -> None:
    """学習シード別曲線を CSV 保存。1行 = group × seed_dir × softness。"""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["group", "seed_dir", "softness", "success_rate"],
        )
        writer.writeheader()
        for label, _softs, _means, _stds, _ns, members in series:
            for seed_dir, softs, rates in members:
                for soft, rate in zip(softs, rates):
                    writer.writerow(
                        {
                            "group": label,
                            "seed_dir": seed_dir,
                            "softness": f"{soft:.2f}",
                            "success_rate": f"{rate:.4f}",
                        }
                    )


def load_series_from_seed_csv(path: Path) -> list[GroupSeries]:
    """`*_seeds.csv` から GroupSeries を復元する（描画用）。"""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"seeds CSV がありません: {path}")

    # group -> seed_dir -> soft -> rate
    nested: dict[str, dict[str, dict[float, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    group_order: list[str] = []
    seed_order: dict[str, list[str]] = defaultdict(list)

    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            group = row["group"].strip()
            seed = row["seed_dir"].strip()
            soft = float(row["softness"])
            rate = float(row["success_rate"])
            if group not in nested:
                group_order.append(group)
            if seed not in nested[group]:
                seed_order[group].append(seed)
            nested[group][seed][soft] = rate

    series: list[GroupSeries] = []
    for group in group_order:
        members: list[MemberCurve] = []
        for seed in seed_order[group]:
            rates_map = nested[group][seed]
            softs = sorted(rates_map.keys())
            members.append((seed, softs, [rates_map[s] for s in softs]))
        softs, means, stds, ns = _aggregate_from_members(members)
        series.append((group, softs, means, stds, ns, members))
    if not series:
        raise ValueError(f"seeds CSV が空です: {path}")
    return series


def legend_loc_for_output(output: Path) -> str:
    """レンジごとの凡例位置。該当しなければ左上。"""
    text = str(output)
    if "0.0-1.20" in text:
        return "lower left"
    if "0.91-1.00" in text:
        return "upper right"
    return "upper left"


def _style_axes(ax: Axes, legend_loc: str = "upper left") -> None:
    ax.set_xlabel("Terrain softness")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.35)
    ax.tick_params(
        direction="in",
        which="both",
        top=True,
        right=True,
        width=1.2,
        length=6,
        labelsize=22,
    )
    ax.legend(loc=legend_loc, fontsize=18)


def draw_groups(
    ax: Axes,
    series: list[GroupSeries],
    style: PlotStyle = "mean_std",
    legend_loc: str = "upper left",
) -> None:
    """ax 上にグループ比較を描く。"""
    if style not in STYLES:
        raise ValueError(f"未知の style: {style!r}（{STYLES}）")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (label, softs, means, stds, ns, members) in enumerate(series):
        color = colors[i % len(colors)]
        ls = GROUP_LINESTYLES[i % len(GROUP_LINESTYLES)]
        x = np.asarray(softs, dtype=float)
        y = np.asarray(means, dtype=float)
        s = np.asarray(stds, dtype=float)

        if style == "mean_std":
            ax.plot(
                x,
                y,
                color=color,
                linestyle=ls,
                linewidth=1.5,
                label=label,
            )
            if max(ns) >= 2:
                ax.fill_between(x, y - s, y + s, color=color, alpha=0.2)
        else:
            for _seed, m_softs, m_rates in members:
                ax.plot(
                    np.asarray(m_softs, dtype=float),
                    np.asarray(m_rates, dtype=float),
                    color=color,
                    linestyle=ls,
                    linewidth=0.9,
                    alpha=0.35,
                    zorder=1,
                )
            ax.plot(
                x,
                y,
                color=color,
                linestyle=ls,
                linewidth=2.0,
                label=label,
                zorder=2,
            )

    _style_axes(ax, legend_loc=legend_loc)


def plot_groups(
    series: list[GroupSeries],
    output_png: Path,
    style: PlotStyle = "mean_std",
    legend_loc: str = "upper left",
) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    draw_groups(ax, series, style=style, legend_loc=legend_loc)
    apply_fixed_axes(ax)

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
        description="複数 eval の softness 成功率をグループ比較プロット"
    )
    parser.add_argument(
        "--group",
        action="append",
        type=parse_group,
        default=None,
        metavar="LABEL:DIR,DIR,...",
        help="比較グループ。複数回指定可。例: 'w_act=1.0:dir_a,dir_b,dir_c'",
    )
    parser.add_argument(
        "--from-seeds-csv",
        type=str,
        default=None,
        help="*_seeds.csv から再描画（--group の代わり）",
    )
    parser.add_argument(
        "--style",
        choices=STYLES,
        default="mean_std",
        help="mean_std=平均±std帯 / per_seed=学習シード細線+平均（既定: mean_std）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"出力 PNG（既定: {DEFAULT_OUTPUT}）。同名 PDF/CSV/_seeds.csv も保存",
    )
    args = parser.parse_args()

    if bool(args.from_seeds_csv) == bool(args.group):
        print("❌ --group か --from-seeds-csv のどちらか一方を指定してください")
        return 1

    output_png = Path(args.output).expanduser().resolve()
    if output_png.suffix.lower() == ".pdf":
        output_png = output_png.with_suffix(".png")
    elif output_png.suffix.lower() != ".png":
        output_png = output_png.with_suffix(".png")

    series: list[GroupSeries] = []
    csv_rows: list[tuple[str, float, float, float, int]] = []

    if args.from_seeds_csv:
        print(f"Loading seeds CSV: {args.from_seeds_csv}")
        series = load_series_from_seed_csv(Path(args.from_seeds_csv))
        for label, softs, means, stds, ns, members in series:
            for soft, mean, std, n in zip(softs, means, stds, ns):
                csv_rows.append((label, soft, mean, std, n))
            print(f"  [{label}] n={ns[0] if ns else 0} softs={len(softs)} seeds={len(members)}")
    else:
        print("Loading groups...")
        for label, dirs in args.group:
            missing = [str(d) for d in dirs if not (d / "summary.csv").is_file()]
            if missing:
                print("❌ summary.csv が見つかりません:")
                for m in missing:
                    print(f"  - {m}")
                return 1
            g = build_group_series(label, dirs)
            series.append(g)
            _, softs, means, stds, ns, _ = g
            for soft, mean, std, n in zip(softs, means, stds, ns):
                csv_rows.append((label, soft, mean, std, n))
            print(f"  [{label}] n={ns[0] if ns else 0} softs={len(softs)}")

    out_csv = output_png.with_name(output_png.stem + ".csv")
    out_seeds = output_png.with_name(output_png.stem + "_seeds.csv")
    write_aggregate_csv(out_csv, csv_rows)
    write_seeds_csv(out_seeds, series)
    legend_loc = legend_loc_for_output(output_png)
    out_pdf = plot_groups(
        series, output_png, style=args.style, legend_loc=legend_loc
    )

    print(f"\nPNG: {output_png}")
    print(f"PDF: {out_pdf}")
    print(f"CSV: {out_csv}")
    print(f"SEEDS: {out_seeds}")
    print(f"style: {args.style}")
    print("✅ 完了")
    return 0


if __name__ == "__main__":
    sys.exit(main())

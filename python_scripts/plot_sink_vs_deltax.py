#!/usr/bin/env python3
"""軟弱路面適用後の support-foot sink（行動入力） vs delta_x（行動）散布図。

因果:
  歩境界で delta が変わる index i について、観測は i-1 行（1制御周期前）。
  sink は i-1 で片足接地している足の obs_foot_sink_*。
  terrain_switched[i-1]==1 のみ採用。

Usage:
  python python_scripts/plot_sink_vs_deltax.py \\
    --group 'seedA:~/vnoid-experiments/paper_logs/run_a_...' \\
    --group 'seedB:~/vnoid-experiments/paper_logs/run_b_...' \\
    --output ~/vnoid-experiments/plots/sink_vs_deltax.png

  各 dir = 1 学習シード（paper_log 出力ルート）。配下の seed*_softness_*/control.csv を読む。
  UI からは extract / draw / plot_groups を import して使える。
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.dpi"] = 150

DEFAULT_OUTPUT = (
    Path.home() / "vnoid-experiments" / "plots" / "sink_vs_deltax.png"
)

FIGSIZE = (8.0, 5.5)
AXES_RECT = (0.16, 0.14, 0.78, 0.78)


def apply_fixed_axes(ax: Axes) -> None:
    ax.set_position(AXES_RECT)


REQUIRED_COLS = (
    "terrain_switched",
    "obs_foot_sink_right",
    "obs_foot_sink_left",
    "obs_contact_right",
    "obs_contact_left",
    "delta_x",
    "delta_y",
)

FootSide = Literal["right", "left"]
DELTA_EPS = 1e-12


@dataclass(frozen=True)
class SinkDeltaPoint:
    sink: float
    delta_x: float
    foot: FootSide
    source: str  # control.csv path (for debug)


# label -> list of points (pooled across eval seeds / softness within train dirs)
GroupPoints = tuple[str, list[SinkDeltaPoint]]


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
    dirs: list[Path] = []
    for part in dirs_raw.split(","):
        part = part.strip()
        if not part:
            continue
        dirs.append(Path(part).expanduser().resolve())
    if not dirs:
        raise argparse.ArgumentTypeError(f"ディレクトリがありません: {raw!r}")
    return label, dirs


def find_control_csvs(paper_dir: Path) -> list[Path]:
    """paper_log ルート配下の control.csv を列挙。"""
    paper_dir = paper_dir.expanduser().resolve()
    if not paper_dir.is_dir():
        raise FileNotFoundError(f"ディレクトリがありません: {paper_dir}")
    found = sorted(paper_dir.glob("seed*_softness_*/control.csv"))
    if not found:
        # 単一 trial ディレクトリを直接渡した場合
        direct = paper_dir / "control.csv"
        if direct.is_file():
            return [direct]
    return found


def load_control_csv(path: Path) -> dict[str, np.ndarray] | None:
    """contact 列付き control.csv を読む。旧形式は None。"""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            warnings.warn(f"ヘッダなしのためスキップ: {path}")
            return None
        missing = [c for c in REQUIRED_COLS if c not in reader.fieldnames]
        if missing:
            warnings.warn(
                f"contact 列などが無い旧 paper log のためスキップ "
                f"({path}): missing={missing}"
            )
            return None
        rows = list(reader)
    if not rows:
        warnings.warn(f"空のためスキップ: {path}")
        return None
    data: dict[str, np.ndarray] = {}
    for col in REQUIRED_COLS:
        data[col] = np.array([float(r[col]) for r in rows], dtype=np.float64)
    return data


def _delta_change_indices(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """(delta_x, delta_y) が変化する先頭 index（i>0）。"""
    if len(dx) < 2:
        return np.array([], dtype=int)
    changed = (np.abs(np.diff(dx)) > DELTA_EPS) | (np.abs(np.diff(dy)) > DELTA_EPS)
    # diff の k 番目は行 k+1 が変化したことを意味する → index = k+1
    return np.flatnonzero(changed) + 1


def _exclusive_support_at(
    contact_r: np.ndarray,
    contact_l: np.ndarray,
    sink_r: np.ndarray,
    sink_l: np.ndarray,
    idx: int,
    *,
    dx: np.ndarray,
    search_back_same_delta: bool = True,
) -> tuple[float, FootSide] | None:
    """idx 時点の片足接地 sink。両足接地なら同一 delta 区間を遡る。"""

    def exclusive(i: int) -> tuple[float, FootSide] | None:
        cr = contact_r[i] >= 0.5
        cl = contact_l[i] >= 0.5
        if cr and not cl:
            return float(sink_r[i]), "right"
        if cl and not cr:
            return float(sink_l[i]), "left"
        return None

    hit = exclusive(idx)
    if hit is not None:
        return hit
    if not search_back_same_delta:
        return None

    # 両足/非接地: 直前の RL 歩（同じ delta）内で直近の片足接地を使う
    ref_dx = dx[idx]
    j = idx - 1
    while j >= 0 and abs(dx[j] - ref_dx) <= DELTA_EPS:
        hit = exclusive(j)
        if hit is not None:
            return hit
        j -= 1
    return None


def extract_points_from_data(
    data: dict[str, np.ndarray],
    *,
    source: str = "",
) -> list[SinkDeltaPoint]:
    """1本の control.csv から因果付き (sink, delta_x) 点を抽出。"""
    dx = data["delta_x"]
    dy = data["delta_y"]
    switched = data["terrain_switched"]
    sink_r = data["obs_foot_sink_right"]
    sink_l = data["obs_foot_sink_left"]
    contact_r = data["obs_contact_right"]
    contact_l = data["obs_contact_left"]

    points: list[SinkDeltaPoint] = []
    for i in _delta_change_indices(dx, dy):
        prev = i - 1
        if switched[prev] < 0.5:
            continue
        support = _exclusive_support_at(
            contact_r,
            contact_l,
            sink_r,
            sink_l,
            prev,
            dx=dx,
            search_back_same_delta=True,
        )
        if support is None:
            continue
        sink, foot = support
        points.append(
            SinkDeltaPoint(
                sink=sink,
                delta_x=float(dx[i]),
                foot=foot,
                source=source,
            )
        )
    return points


def extract_points_from_paper_dir(
    paper_dir: Path,
    *,
    verbose: bool = True,
) -> list[SinkDeltaPoint]:
    """1 学習シードの paper_log ルートから点を集める。"""
    csvs = find_control_csvs(paper_dir)
    if not csvs:
        raise FileNotFoundError(
            f"control.csv が見つかりません: {paper_dir}"
        )
    points: list[SinkDeltaPoint] = []
    used = 0
    skipped = 0
    for csv_path in csvs:
        data = load_control_csv(csv_path)
        if data is None:
            skipped += 1
            continue
        pts = extract_points_from_data(data, source=str(csv_path))
        points.extend(pts)
        used += 1
    if verbose:
        print(
            f"  {paper_dir.name}: trials_used={used} skipped={skipped} "
            f"points={len(points)}"
        )
    if used == 0:
        raise ValueError(
            f"有効な control.csv（contact 列付き）がありません: {paper_dir}"
        )
    return points


def build_group_points(
    label: str,
    dirs: list[Path],
    *,
    verbose: bool = True,
) -> GroupPoints:
    points: list[SinkDeltaPoint] = []
    if verbose:
        print(f"[{label}]")
    for d in dirs:
        points.extend(extract_points_from_paper_dir(d, verbose=verbose))
    return label, points


def write_points_csv(out_csv: Path, groups: Iterable[GroupPoints]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["group", "foot", "sink", "delta_x", "source"],
        )
        writer.writeheader()
        for label, points in groups:
            for p in points:
                writer.writerow(
                    {
                        "group": label,
                        "foot": p.foot,
                        "sink": f"{p.sink:.8g}",
                        "delta_x": f"{p.delta_x:.8g}",
                        "source": p.source,
                    }
                )


def draw_groups(ax: Axes, series: list[GroupPoints]) -> None:
    """学習グループ色分け + 左右マーカー違いで散布。"""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = {"right": "o", "left": "^"}

    for i, (label, points) in enumerate(series):
        color = colors[i % len(colors)]
        for foot in ("right", "left"):
            xs = [p.sink for p in points if p.foot == foot]
            ys = [p.delta_x for p in points if p.foot == foot]
            if not xs:
                continue
            ax.scatter(
                xs,
                ys,
                s=18,
                alpha=0.55,
                color=color,
                marker=markers[foot],
                label=f"{label} ({foot})",
                edgecolors="none",
            )

    ax.set_xlabel("Support-foot sink [m] (before action)")
    ax.set_ylabel("delta_x")
    ax.grid(True, linestyle="--", alpha=0.4)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(markerscale=1.4, loc="best")


def plot_groups(series: list[GroupPoints], output_png: Path) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    draw_groups(ax, series)
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
        description="軟弱路面後・1歩因果付き sink vs delta_x 散布図"
    )
    parser.add_argument(
        "--group",
        action="append",
        type=parse_group,
        required=True,
        metavar="LABEL:DIR,DIR,...",
        help="比較グループ。各 DIR は paper_log 出力ルート（1学習シード）",
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

    series: list[GroupPoints] = []
    print("Loading groups...")
    for label, dirs in args.group:
        try:
            series.append(build_group_points(label, dirs))
        except (FileNotFoundError, ValueError) as e:
            print(f"❌ {e}")
            return 1

    out_csv = output_png.with_name(output_png.stem + ".csv")
    write_points_csv(out_csv, series)
    out_pdf = plot_groups(series, output_png)

    total = sum(len(pts) for _, pts in series)
    print(f"\npoints: {total}")
    print(f"PNG: {output_png}")
    print(f"PDF: {out_pdf}")
    print(f"CSV: {out_csv}")
    print("✅ 完了")
    return 0


if __name__ == "__main__":
    sys.exit(main())

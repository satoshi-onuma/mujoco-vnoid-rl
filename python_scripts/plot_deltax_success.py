#!/usr/bin/env python3
"""チェックポイント単位: Δx 平均（評価シード平均） vs paper_log overall 成功率。

指標は **1 paper_log 出力 dir のみ** から X/Y を取る。
学習↔評価の対応は ExperimentDB + paper_eval_link（UI と共有）。

X 窓 (--window):
  post_all … 切替後の全歩 Δx 平均（既定）
  pre_all  … 切替前の全歩 Δx 平均
  post_n   … 切替後の先頭 N 歩 Δx 平均（--post-n）

Usage:
  python python_scripts/plot_deltax_success.py \\
    --w-act 0.1,1,10,100 \\
    --window post_n --post-n 5 \\
    --intervention-mode after_switch \\
    --min-eval-seeds 10 \\
    --min-overall-trials 100 \\
    --output ~/vnoid-experiments/plots/deltax_vs_success.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["figure.dpi"] = 150

DEFAULT_PAPER_ROOT = Path.home() / "vnoid-experiments" / "paper_logs"
DEFAULT_OUTPUT = (
    Path.home() / "vnoid-experiments" / "plots" / "deltax_vs_success.png"
)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_experiment_app.database import DEFAULT_DB_PATH, ExperimentDB  # noqa: E402
from rl_experiment_app.paper_eval_link import (  # noqa: E402
    list_alpha_one_trials,
    list_paper_links_for_experiments,
    load_overall,
    load_softness_success,
    parse_expected_seed_count,
    w_act_from_experiment,
)

DeltaXWindow = Literal["post_all", "pre_all", "post_n"]
WINDOWS: tuple[DeltaXWindow, ...] = ("post_all", "pre_all", "post_n")
DEFAULT_POST_N = 5
INTERVENTION_MODES: tuple[str, ...] = ("none", "full", "after_switch")
DEFAULT_INTERVENTION_MODE = "full"

YMetric = Literal["overall", "alpha_1"]
Y_METRICS: tuple[YMetric, ...] = ("overall", "alpha_1")
DEFAULT_Y_METRIC: YMetric = "overall"

GROUP_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P")

FIGSIZE = (7.0, 5.0)
# left, bottom, width, height（figure 座標）
AXES_RECT = (0.18, 0.15, 0.76, 0.78)


def apply_fixed_axes(ax: Axes) -> None:
    ax.set_position(AXES_RECT)


def window_xlabel(window: DeltaXWindow, post_n: int = DEFAULT_POST_N) -> str:
    return r"RL action $\Delta x$ [m]"


def y_axis_label(y_metric: YMetric) -> str:
    if y_metric == "alpha_1":
        return "alpha=1 success"
    return "overall success"


def validate_post_n(window: DeltaXWindow, post_n: int) -> None:
    if window == "post_n" and post_n < 1:
        raise ValueError(f"post_n は 1 以上が必要です: {post_n}")


class PointStatus(str, Enum):
    READY = "ready"
    PARTIAL = "partial"
    NO_PAPER = "no_paper"


@dataclass(frozen=True)
class CheckpointPoint:
    run_id: str
    w_act: float
    delta_x_mean: float
    success_rate: float
    n_eval_seeds: int
    overall_trials: int
    paper_dir: Path
    status: PointStatus


def _footstep_dx_sequence(
    rows: list[dict[str, str]], *, switched: bool
) -> list[float]:
    """(delta_x, delta_y) が一定の区間を1歩とし、区間先頭の切替フラグで選ぶ。

    切替直後も直前の入力が保持される。その歩は開始行が切替前なら切替後に入れない。
    """
    dx_vals: list[float] = []
    prev: tuple[float, float] | None = None
    for row in rows:
        dx = float(row["delta_x"])
        dy = float(row["delta_y"])
        key = (round(dx, 12), round(dy, 12))
        if key == prev:
            continue
        prev = key
        is_sw = float(row["terrain_switched"]) >= 0.5
        if is_sw != switched:
            continue
        dx_vals.append(dx)
    return dx_vals


def mean_delta_x(
    control_csv: Path,
    window: DeltaXWindow = "post_all",
    post_n: int = DEFAULT_POST_N,
) -> float:
    """control.csv から窓付き Δx 平均（歩単位 dedupe 後）。"""
    if window not in WINDOWS:
        raise ValueError(f"未知の window: {window!r}（{WINDOWS}）")
    validate_post_n(window, post_n)

    with control_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"ヘッダがありません: {control_csv}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"データが空です: {control_csv}")

    if window == "pre_all":
        dx_vals = _footstep_dx_sequence(rows, switched=False)
        label = "切替前"
    else:
        dx_vals = _footstep_dx_sequence(rows, switched=True)
        label = "切替後"
        if window == "post_n":
            dx_vals = dx_vals[:post_n]

    if not dx_vals:
        raise ValueError(f"{label}データがありません: {control_csv}")
    return float(np.mean(dx_vals))


def mean_post_switch_delta_x(control_csv: Path) -> float:
    """互換: 切替後全歩平均。"""
    return mean_delta_x(control_csv, "post_all")


def mean_delta_x_over_eval_seeds(
    paper_run_dir: Path,
    window: DeltaXWindow = "post_all",
    post_n: int = DEFAULT_POST_N,
) -> tuple[float, int]:
    trials = list_alpha_one_trials(paper_run_dir)
    if not trials:
        raise FileNotFoundError(
            f"α=1.00 の trial がありません: {paper_run_dir}"
        )
    per_seed: list[float] = []
    for _seed, trial in trials:
        per_seed.append(
            mean_delta_x(trial / "control.csv", window, post_n=post_n)
        )
    return float(np.mean(per_seed)), len(per_seed)


def build_point_from_paper(
    db: ExperimentDB,
    experiment_id: str,
    paper_dir: Path,
    *,
    seeds_spec: str | None = None,
    min_eval_seeds: int = 10,
    min_overall_trials: int = 100,
    window: DeltaXWindow = "post_all",
    post_n: int = DEFAULT_POST_N,
    y_metric: YMetric = DEFAULT_Y_METRIC,
) -> CheckpointPoint | None:
    w_act = w_act_from_experiment(db, experiment_id)
    if w_act is None:
        return None

    meta = paper_dir / "meta.txt"
    expected = parse_expected_seed_count(meta, seeds_spec)
    if expected <= 0:
        expected = min_eval_seeds

    alpha = list_alpha_one_trials(paper_dir)
    n_alpha = len(alpha)

    if y_metric == "alpha_1":
        y_row = load_softness_success(paper_dir)
    else:
        y_row = load_overall(paper_dir)

    if y_row is None:
        return CheckpointPoint(
            run_id=experiment_id,
            w_act=w_act,
            delta_x_mean=0.0,
            success_rate=0.0,
            n_eval_seeds=n_alpha,
            overall_trials=0,
            paper_dir=paper_dir,
            status=PointStatus.PARTIAL,
        )

    success_rate, n_trials = y_row
    try:
        dx_mean, n_used = mean_delta_x_over_eval_seeds(
            paper_dir, window=window, post_n=post_n
        )
    except (FileNotFoundError, ValueError):
        return CheckpointPoint(
            run_id=experiment_id,
            w_act=w_act,
            delta_x_mean=0.0,
            success_rate=success_rate,
            n_eval_seeds=n_alpha,
            overall_trials=n_trials,
            paper_dir=paper_dir,
            status=PointStatus.PARTIAL,
        )

    # overall は格子横断 trials、α=1 は通常 n_seeds 本なので閾値を分ける
    min_y_trials = (
        min_eval_seeds if y_metric == "alpha_1" else min_overall_trials
    )
    ready = (
        n_used == expected
        and n_used >= min_eval_seeds
        and n_trials >= min_y_trials
    )
    status = PointStatus.READY if ready else PointStatus.PARTIAL
    return CheckpointPoint(
        run_id=experiment_id,
        w_act=w_act,
        delta_x_mean=dx_mean,
        success_rate=success_rate,
        n_eval_seeds=n_used,
        overall_trials=n_trials,
        paper_dir=paper_dir,
        status=status,
    )


def discover_points(
    db: ExperimentDB | None = None,
    *,
    paper_root: Path = DEFAULT_PAPER_ROOT,
    min_eval_seeds: int = 10,
    min_overall_trials: int = 100,
    experiments_only: bool = True,
    window: DeltaXWindow = "post_all",
    post_n: int = DEFAULT_POST_N,
    intervention_mode: str = DEFAULT_INTERVENTION_MODE,
    y_metric: YMetric = DEFAULT_Y_METRIC,
) -> list[CheckpointPoint]:
    """DB 連動 + scan で checkpoint ごとに paper 指標を構築。"""
    if db is None:
        db = ExperimentDB(DEFAULT_DB_PATH)
    validate_post_n(window, post_n)

    links = list_paper_links_for_experiments(
        db,
        None if experiments_only else [],
        paper_root=paper_root,
        min_eval_seeds=min_eval_seeds,
        min_overall_trials=min_overall_trials,
        intervention_mode=intervention_mode,
    )

    points: list[CheckpointPoint] = []
    for link in links:
        if link.paper_dir is None:
            w_act = w_act_from_experiment(db, link.experiment_id)
            if w_act is None:
                continue
            points.append(
                CheckpointPoint(
                    run_id=link.experiment_id,
                    w_act=w_act,
                    delta_x_mean=0.0,
                    success_rate=0.0,
                    n_eval_seeds=0,
                    overall_trials=0,
                    paper_dir=paper_root,
                    status=PointStatus.NO_PAPER,
                )
            )
            continue

        pt = build_point_from_paper(
            db,
            link.experiment_id,
            link.paper_dir,
            seeds_spec=link.seeds_spec,
            min_eval_seeds=min_eval_seeds,
            min_overall_trials=min_overall_trials,
            window=window,
            post_n=post_n,
            y_metric=y_metric,
        )
        if pt is not None:
            points.append(pt)
    return points


def w_act_label(w: float) -> str:
    if abs(w - round(w)) < 1e-9:
        return f"w_act={int(round(w))}"
    return f"w_act={w:g}"


def normalize_w_act_filter(raw: str) -> list[float]:
    vals: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    return vals


def w_act_matches(w: float, targets: list[float]) -> bool:
    if not targets:
        return True
    return any(abs(w - t) < 1e-6 for t in targets)


def write_points_csv(out_csv: Path, points: list[CheckpointPoint]) -> None:
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "w_act",
                "delta_x_mean",
                "success_rate",
                "n_eval_seeds",
                "overall_trials",
                "status",
                "paper_dir",
            ],
        )
        writer.writeheader()
        for p in points:
            writer.writerow(
                {
                    "run_id": p.run_id,
                    "w_act": f"{p.w_act:g}",
                    "delta_x_mean": f"{p.delta_x_mean:.6f}",
                    "success_rate": f"{p.success_rate:.4f}",
                    "n_eval_seeds": p.n_eval_seeds,
                    "overall_trials": p.overall_trials,
                    "status": p.status.value,
                    "paper_dir": str(p.paper_dir),
                }
            )


@dataclass(frozen=True)
class GroupCorrelation:
    """w_act グループの Δx vs success 相関係数（Pearson = 高校の相関係数）。"""

    group: str
    n: int
    pearson_r: float | None


def _canonical_w_act(w: float) -> float:
    for t in (0.1, 1.0, 10.0, 100.0):
        if abs(w - t) < 1e-6:
            return t
    return w


def _pearson_r(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if float(np.std(x)) < 1e-15 or float(np.std(y)) < 1e-15:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def compute_group_correlations(
    points: list[CheckpointPoint],
    w_act_filter: list[float] | None = None,
    *,
    ready_only: bool = True,
) -> list[GroupCorrelation]:
    """w_act ごとの Pearson 相関係数（全体まとめは出さない）。"""
    if w_act_filter is None:
        w_act_filter = []
    filtered = [p for p in points if w_act_matches(p.w_act, w_act_filter)]
    if ready_only:
        filtered = [p for p in filtered if p.status == PointStatus.READY]

    by_w: dict[float, list[CheckpointPoint]] = {}
    for p in filtered:
        by_w.setdefault(_canonical_w_act(p.w_act), []).append(p)

    out: list[GroupCorrelation] = []
    for w in sorted(by_w.keys()):
        group = by_w[w]
        xs = [p.delta_x_mean for p in group]
        ys = [p.success_rate for p in group]
        out.append(
            GroupCorrelation(
                group=w_act_label(w),
                n=len(group),
                pearson_r=_pearson_r(xs, ys),
            )
        )
    return out


def format_corr_lines(corrs: list[GroupCorrelation]) -> list[str]:
    lines: list[str] = []
    for c in corrs:
        pr = f"{c.pearson_r:+.3f}" if c.pearson_r is not None else "n/a"
        lines.append(f"{c.group}: n={c.n}  r={pr}")
    return lines


def write_corr_csv(out_csv: Path, corrs: list[GroupCorrelation]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["group", "n", "pearson_r"],
        )
        writer.writeheader()
        for c in corrs:
            writer.writerow(
                {
                    "group": c.group,
                    "n": c.n,
                    "pearson_r": (
                        f"{c.pearson_r:.6f}" if c.pearson_r is not None else ""
                    ),
                }
            )


def _style_axes(ax: Axes) -> None:
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.35)
    ax.tick_params(
        direction="in",
        which="both",
        top=True,
        right=True,
        labelsize=18,
    )


def draw_scatter(
    ax: Axes,
    points: list[CheckpointPoint],
    w_act_filter: list[float] | None = None,
    *,
    ready_only: bool = True,
    window: DeltaXWindow = "post_all",
    post_n: int = DEFAULT_POST_N,
    y_metric: YMetric = DEFAULT_Y_METRIC,
    show_corr: bool = False,
) -> list[GroupCorrelation]:
    if w_act_filter is None:
        w_act_filter = []

    filtered = [p for p in points if w_act_matches(p.w_act, w_act_filter)]
    if ready_only:
        filtered = [p for p in filtered if p.status == PointStatus.READY]

    corrs = compute_group_correlations(
        points, w_act_filter, ready_only=ready_only
    )

    xlabel = window_xlabel(window, post_n)
    ylabel = y_axis_label(y_metric)
    if not filtered:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.05, 1.05)
        _style_axes(ax)
        return corrs

    by_w: dict[float, list[CheckpointPoint]] = {}
    for p in filtered:
        by_w.setdefault(_canonical_w_act(p.w_act), []).append(p)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, w in enumerate(sorted(by_w.keys())):
        group = by_w[w]
        color = colors[i % len(colors)]
        mk = GROUP_MARKERS[i % len(GROUP_MARKERS)]
        xs = [p.delta_x_mean for p in group]
        ys = [p.success_rate for p in group]
        ax.scatter(
            xs,
            ys,
            color=color,
            marker=mk,
            s=48,
            label=w_act_label(w),
            alpha=0.85,
            zorder=2,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.05, 1.05)
    _style_axes(ax)
    ax.legend(loc="upper left", fontsize=11)
    ax.axhline(0.0, color="k", linestyle=":", alpha=0.25)
    ax.axvline(0.0, color="k", linestyle=":", alpha=0.25)

    if show_corr and corrs:
        text = "\n".join(format_corr_lines(corrs))
        ax.text(
            0.02,
            0.02,
            text,
            transform=ax.transAxes,
            fontsize=12,
            va="bottom",
            ha="left",
            family="monospace",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": "0.7",
            },
            zorder=3,
        )
    return corrs


def plot_scatter(
    points: list[CheckpointPoint],
    output_png: Path,
    w_act_filter: list[float] | None = None,
    *,
    ready_only: bool = True,
    window: DeltaXWindow = "post_all",
    post_n: int = DEFAULT_POST_N,
    y_metric: YMetric = DEFAULT_Y_METRIC,
    show_corr: bool = False,
    write_corr: bool = True,
) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    corrs = draw_scatter(
        ax,
        points,
        w_act_filter=w_act_filter,
        ready_only=ready_only,
        window=window,
        post_n=post_n,
        y_metric=y_metric,
        show_corr=show_corr,
    )
    apply_fixed_axes(ax)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    output_pdf = output_png.with_suffix(".pdf")
    fig.savefig(output_pdf)
    plt.close(fig)
    if write_corr:
        write_corr_csv(
            output_png.with_name(output_png.stem + "_corr.csv"),
            corrs,
        )
    return output_pdf


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(
        description="チェックポイント単位 Δx平均 vs paper overall 成功率"
    )
    parser.add_argument(
        "--w-act",
        type=str,
        default="",
        help="描画する w_act（カンマ区切）。未指定なら全点",
    )
    parser.add_argument(
        "--window",
        choices=WINDOWS,
        default="post_all",
        help="Δx 窓: post_all / pre_all / post_n（既定: post_all）",
    )
    parser.add_argument(
        "--post-n",
        type=int,
        default=DEFAULT_POST_N,
        help=f"window=post_n のときの先頭歩数（既定: {DEFAULT_POST_N}）",
    )
    parser.add_argument(
        "--intervention-mode",
        choices=INTERVENTION_MODES,
        default=DEFAULT_INTERVENTION_MODE,
        help="採用する paper_log の介入モード（既定: full）",
    )
    parser.add_argument(
        "--y-metric",
        choices=Y_METRICS,
        default=DEFAULT_Y_METRIC,
        help="縦軸: overall / alpha_1（既定: overall）",
    )
    parser.add_argument(
        "--min-eval-seeds",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--min-overall-trials",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=DEFAULT_PAPER_ROOT,
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="experiments.db（既定: ~/vnoid-experiments/runs/experiments.db）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
    )
    args = parser.parse_args()

    w_filter = normalize_w_act_filter(args.w_act) if args.w_act.strip() else []
    db = ExperimentDB(args.db.expanduser().resolve())
    window: DeltaXWindow = args.window
    post_n = int(args.post_n)
    try:
        validate_post_n(window, post_n)
    except ValueError as e:
        print(f"❌ {e}")
        return 1

    print(
        f"Discovering checkpoints (paper_log only, interv={args.intervention_mode}, "
        f"y={args.y_metric}, window={window}"
        + (f", post_n={post_n}" if window == "post_n" else "")
        + ")..."
    )
    points = discover_points(
        db,
        paper_root=args.paper_root.expanduser().resolve(),
        min_eval_seeds=args.min_eval_seeds,
        min_overall_trials=args.min_overall_trials,
        window=window,
        post_n=post_n,
        intervention_mode=args.intervention_mode,
        y_metric=args.y_metric,
    )
    ready = [p for p in points if p.status == PointStatus.READY]
    if not ready:
        print("❌ ready なチェックポイントがありません")
        return 1

    for p in points:
        print(
            f"  [{p.status.value}] {p.run_id} w_act={p.w_act:g} "
            f"Δx={p.delta_x_mean:.6f} (n={p.n_eval_seeds}) "
            f"Y={p.success_rate:.4f} trials={p.overall_trials}"
        )

    output_png = Path(args.output).expanduser().resolve()
    if output_png.suffix.lower() != ".png":
        output_png = output_png.with_suffix(".png")

    out_csv = output_png.with_name(output_png.stem + ".csv")
    write_points_csv(out_csv, points)
    out_pdf = plot_scatter(
        points,
        output_png,
        w_act_filter=w_filter,
        ready_only=True,
        window=window,
        post_n=post_n,
        y_metric=args.y_metric,
    )
    corr_csv = output_png.with_name(output_png.stem + "_corr.csv")

    print(f"\nPNG: {output_png}")
    print(f"PDF: {out_pdf}")
    print(f"CSV: {out_csv}")
    print(f"CORR: {corr_csv}")
    for line in format_corr_lines(
        compute_group_correlations(points, w_filter, ready_only=True)
    ):
        print(f"  {line}")
    print(f"y_metric: {args.y_metric}")
    print(f"window: {window}" + (f" post_n={post_n}" if window == "post_n" else ""))
    print("✅ 完了")
    return 0


if __name__ == "__main__":
    sys.exit(main())

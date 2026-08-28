"""
sweep CSVから論文向け2Dヒートマップ/折れ線図を生成する。

使用例:
  # ヒートマップ
  python plot_sweep_offset_x_2d.py --in-csv sweep_offset_x.csv --out-prefix figs/pitch_map

  # 折れ線 (固定 posture でスライス)
  python plot_sweep_offset_x_2d.py --in-csv sweep_offset_x.csv --out-prefix figs/pitch_lines \
    --plot-mode line --line-axis sink --line-slices -0.1 0.0 0.1

  # 両方
  python plot_sweep_offset_x_2d.py --in-csv sweep_offset_x.csv --out-prefix figs/pitch \
    --plot-mode both
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_heatmap(df: pd.DataFrame, support_foot: str, out_prefix: str,
                 dpi: int, fontsize: int):
    sub = df[df['support_foot'] == support_foot]
    if sub.empty:
        return

    posture_metric = sub['posture_metric'].iloc[0]
    sink_vals = np.sort(sub['sink'].unique())
    posture_vals = np.sort(sub['posture_value'].unique())

    pivot = sub.pivot_table(
        index='posture_value', columns='sink', values='offset_x', aggfunc='mean'
    )
    pivot = pivot.reindex(index=posture_vals, columns=sink_vals)
    grid = pivot.to_numpy()

    fig, ax = plt.subplots(figsize=(7, 5))
    extent = [sink_vals[0], sink_vals[-1], posture_vals[0], posture_vals[-1]]
    im = ax.imshow(grid, aspect='auto', origin='lower', extent=extent,
                   cmap='RdBu_r', interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Foot offset x [m]', fontsize=fontsize)

    ax.set_xlabel('Foot sink [m]', fontsize=fontsize)
    ax.set_ylabel(f'Body {posture_metric} [rad]', fontsize=fontsize)
    ax.set_title(f'Policy offset_x response ({support_foot} support)', fontsize=fontsize + 1)
    ax.tick_params(labelsize=fontsize - 1)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        path = f"{out_prefix}_heatmap_{support_foot}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_lines(df: pd.DataFrame, support_foot: str, out_prefix: str,
               line_axis: str, line_slices: list,
               dpi: int, fontsize: int):
    sub = df[df['support_foot'] == support_foot]
    if sub.empty:
        return

    posture_metric = sub['posture_metric'].iloc[0]

    if line_axis == 'sink':
        x_col, slice_col = 'sink', 'posture_value'
        xlabel = 'Foot sink [m]'
        slice_label = f'Body {posture_metric}'
    else:
        x_col, slice_col = 'posture_value', 'sink'
        xlabel = f'Body {posture_metric} [rad]'
        slice_label = 'Foot sink'

    available = np.sort(sub[slice_col].unique())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for sv in line_slices:
        closest = available[np.argmin(np.abs(available - sv))]
        slice_df = sub[sub[slice_col] == closest].sort_values(x_col)
        ax.plot(slice_df[x_col], slice_df['offset_x'],
                label=f'{slice_label}={closest:.4f}', linewidth=1.5)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel('Foot offset x [m]', fontsize=fontsize)
    ax.set_title(f'Policy offset_x slices ({support_foot} support)', fontsize=fontsize + 1)
    ax.legend(fontsize=fontsize - 2)
    ax.tick_params(labelsize=fontsize - 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        path = f"{out_prefix}_lines_{support_foot}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot sweep offset_x results (heatmap / line)")
    parser.add_argument("--in-csv", type=str, required=True)
    parser.add_argument("--out-prefix", type=str, default="sweep_offset_x")
    parser.add_argument("--plot-mode", type=str, default="both",
                        choices=["heatmap", "line", "both"])
    parser.add_argument("--line-axis", type=str, default="sink",
                        choices=["sink", "posture"])
    parser.add_argument("--line-slices", type=float, nargs='+', default=None,
                        help="Slice values for line plot. Auto-selected if omitted.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--fontsize", type=int, default=11)
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)
    print(f"Loaded {len(df)} rows from {args.in_csv}")

    out_dir = Path(args.out_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-select line slices if not provided
    if args.line_slices is None:
        if args.line_axis == 'sink':
            vals = np.sort(df['posture_value'].unique())
        else:
            vals = np.sort(df['sink'].unique())
        indices = np.linspace(0, len(vals) - 1, 5, dtype=int)
        args.line_slices = vals[indices].tolist()

    feet = df['support_foot'].unique()
    for foot in feet:
        print(f"\n--- {foot} support ---")
        if args.plot_mode in ('heatmap', 'both'):
            plot_heatmap(df, foot, args.out_prefix, args.dpi, args.fontsize)
        if args.plot_mode in ('line', 'both'):
            plot_lines(df, foot, args.out_prefix, args.line_axis,
                       args.line_slices, args.dpi, args.fontsize)

    print("\nDone.")


if __name__ == "__main__":
    main()

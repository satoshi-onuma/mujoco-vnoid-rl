#!/usr/bin/env python3
"""recording / control ログから、切替後の平均 Δx・Δy と r_act を出す。

Δx, Δy は rl_action_foot_offset_x / y。
同じ値が続く区間を1歩とみなす（1歩のあいだ入力は保持される）。

切替は terrain_switched が無いログ向けに、先頭の一定区間（通常は 0）が
終わる時点とする。after_switch では切替前の行動は 0 のままなので、
最初に行動が変わった歩以降が切替後になる。

r_act は reward_action_penalty と同じ正規化:

    r_act = (Δx / 0.15)^2 + (Δy / 0.1)^2

「平均 Δ から計算した r_act」は平均を上式に入れた値。
報酬に入るのは歩ごとの r_act の平均なので、それも並べて出す。

Usage:
  python python_scripts/calc_post_switch_r_act.py path/to/recording_log.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

DX_SCALE = 0.15  # foot_offset_x: [-0.15, 0.15] → [-1, 1]
DY_SCALE = 0.1   # foot_offset_y: [-0.1, 0.1] → [-1, 1]

DX_COLS = ("rl_action_foot_offset_x", "delta_x")
DY_COLS = ("rl_action_foot_offset_y", "delta_y")


def r_act(dx: float, dy: float) -> float:
    return (dx / DX_SCALE) ** 2 + (dy / DY_SCALE) ** 2


def _pick(fieldnames: list[str], candidates: tuple[str, ...], kind: str) -> str:
    for name in candidates:
        if name in fieldnames:
            return name
    raise ValueError(f"{kind} の列がありません（{candidates}）: {fieldnames}")


def load_offsets(path: Path) -> list[tuple[float, float]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"ヘッダがありません: {path}")
        dx_col = _pick(list(reader.fieldnames), DX_COLS, "Δx")
        dy_col = _pick(list(reader.fieldnames), DY_COLS, "Δy")
        rows: list[tuple[float, float]] = []
        for i, row in enumerate(reader, start=2):
            raw_x = (row.get(dx_col) or "").strip()
            raw_y = (row.get(dy_col) or "").strip()
            if raw_x == "" and raw_y == "":
                continue
            try:
                rows.append((float(raw_x), float(raw_y)))
            except ValueError as exc:
                raise ValueError(f"{path}:{i} の Δ を数値にできません") from exc
    if not rows:
        raise ValueError(f"Δx/Δy が空です: {path}")
    return rows


def footstep_actions(samples: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """連続して同じ (Δx, Δy) の区間を1歩にする。"""
    steps: list[tuple[float, float]] = []
    prev: tuple[float, float] | None = None
    for dx, dy in samples:
        key = (round(dx, 12), round(dy, 12))
        if key == prev:
            continue
        prev = key
        steps.append((dx, dy))
    return steps


def post_switch_steps(
    steps: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """先頭の一定行動（切替前ホールド）を除いた歩。"""
    if len(steps) < 2:
        raise ValueError("切替前後を分けるだけの歩がありません")
    return steps[1:]


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


def summarize(path: Path) -> dict[str, float | int]:
    samples = load_offsets(path)
    steps = footstep_actions(samples)
    post = post_switch_steps(steps)
    dx = [a for a, _ in post]
    dy = [b for _, b in post]
    mean_dx = mean(dx)
    mean_dy = mean(dy)
    per_step = [r_act(a, b) for a, b in post]
    return {
        "n_samples": len(samples),
        "n_steps_post": len(post),
        "mean_dx": mean_dx,
        "mean_dy": mean_dy,
        "r_act_from_means": r_act(mean_dx, mean_dy),
        "mean_r_act": mean(per_step),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="recording_log または control_log")
    args = parser.parse_args()
    path = args.csv.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"ファイルがありません: {path}")

    s = summarize(path)
    print(f"file: {path}")
    print(f"samples: {s['n_samples']}")
    print(f"post-switch steps: {s['n_steps_post']}")
    print(f"mean deltax: {s['mean_dx']:.8f}")
    print(f"mean deltay: {s['mean_dy']:.8f}")
    print(f"r_act from means: {s['r_act_from_means']:.8f}")
    print(f"mean r_act (per step): {s['mean_r_act']:.8f}")


if __name__ == "__main__":
    main()

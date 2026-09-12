#!/usr/bin/env python3
"""Softness × seed 格子評価: 50歩転倒なしの成功率だけを集計する。"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from ray.rllib.core.rl_module import RLModule

from my_humanoid_env import HumanoidVnoidEnv

DEFAULT_EXPERIMENTS_ROOT = Path.home() / "vnoid-experiments"
RL_MODULE_SUFFIX = Path("learner_group") / "learner" / "rl_module" / "default_policy"


def parse_seeds(raw: str) -> list[int]:
    """'1001-1010' または '1001,1002,1003' を受け付ける。"""
    raw = raw.strip()
    if not raw:
        raise argparse.ArgumentTypeError("seeds が空です")
    if "-" in raw and "," not in raw:
        left, right = raw.split("-", 1)
        start, end = int(left), int(right)
        if end < start:
            raise argparse.ArgumentTypeError(f"seed 範囲が不正です: {raw}")
        return list(range(start, end + 1))
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def softness_values(min_v: float, max_v: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("softness-step は正の値である必要があります")
    # 浮動小数の累積誤差を避けるため整数インデックスで生成
    n = int(round((max_v - min_v) / step)) + 1
    values = [round(min_v + i * step, 10) for i in range(n)]
    values = [v for v in values if v <= max_v + 1e-12]
    if not values or values[-1] < max_v - 1e-12:
        values.append(round(max_v, 10))
    return values


def resolve_checkpoint_dir(path: Path) -> Path:
    """RLModule が載っている checkpoint ルートを解決する。"""
    path = path.expanduser().resolve()
    candidates = [path]
    nested = path / "checkpoint"
    if nested.is_dir():
        candidates.append(nested)

    for cand in candidates:
        if (cand / RL_MODULE_SUFFIX).is_dir():
            return cand

    tried = "\n".join(f"  - {c / RL_MODULE_SUFFIX}" for c in candidates)
    raise FileNotFoundError(
        f"RLModule が見つかりません。次を探しました:\n{tried}"
    )


def checkpoint_label(checkpoint_dir: Path) -> str:
    """出力ディレクトリ名用のチェックポイント識別子。"""
    name = checkpoint_dir.name
    if name == "checkpoint" and checkpoint_dir.parent.name:
        return checkpoint_dir.parent.name
    return name


def default_output_dir(checkpoint_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = checkpoint_label(checkpoint_dir)
    return DEFAULT_EXPERIMENTS_ROOT / "evals" / f"{label}_{stamp}"


def infer_action(rl_module: RLModule | None, obs, env: HumanoidVnoidEnv) -> np.ndarray:
    if rl_module is None:
        return np.zeros(2, dtype=np.float32)
    obs_batch = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        model_outputs = rl_module.forward_inference({"obs": obs_batch})
    action_dist_params = model_outputs["action_dist_inputs"][0].numpy()
    return np.clip(
        action_dist_params[:2],
        a_min=env.action_space.low,
        a_max=env.action_space.high,
    )


def run_episode(env: HumanoidVnoidEnv, rl_module: RLModule | None, seed: int, softness: float) -> tuple[int, int]:
    """1エピソード実行。戻り値: (success 0/1, steps)。"""
    env.cpp_env.set_terrain_config(
        {"mode": "terrain_softness", "terrain_softness": float(softness)}
    )
    obs, _info = env.reset(seed=seed)
    steps = 0
    max_steps = env.max_episode_steps
    for _ in range(max_steps):
        action = infer_action(rl_module, obs, env)
        step_out = env.step(action)
        # enable_rendering=False なら 5-tuple
        obs, _reward, terminated, truncated, _info = step_out[:5]
        steps += 1
        if terminated or truncated:
            success = int(bool(truncated) and not bool(terminated))
            return success, steps
    return 0, steps


def print_matrix(seeds: list[int], softs: list[float], results: dict[tuple[int, float], tuple[int, int]]) -> None:
    soft_headers = [f"{s:.2f}" for s in softs]
    col_w = max(6, max(len(h) for h in soft_headers))
    seed_w = max(4, max(len(str(s)) for s in seeds))

    header = f"{'seed':<{seed_w}}  " + "  ".join(f"{h:>{col_w}}" for h in soft_headers) + f"  {'rate':>6}"
    print(header)
    print("-" * len(header))

    for seed in seeds:
        cells = []
        ok = 0
        for soft in softs:
            success, _ = results[(seed, soft)]
            cells.append(f"{success:>{col_w}}")
            ok += success
        rate = ok / len(softs)
        print(f"{seed:<{seed_w}}  " + "  ".join(cells) + f"  {rate:6.2f}")

    print("-" * len(header))
    soft_rates = []
    for soft in softs:
        ok = sum(results[(seed, soft)][0] for seed in seeds)
        soft_rates.append(ok / len(seeds))
    print(
        f"{'mean':<{seed_w}}  "
        + "  ".join(f"{r:{col_w}.2f}" for r in soft_rates)
    )
    total_ok = sum(v[0] for v in results.values())
    total_n = len(results)
    print(f"overall: {total_ok}/{total_n} = {total_ok / total_n:.4f}")


def write_outputs(
    output_dir: Path,
    checkpoint_dir: Path,
    seeds: list[int],
    softs: list[float],
    results: dict[tuple[int, float], tuple[int, int]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.csv"
    with results_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["seed", "softness", "success", "steps", "checkpoint_dir"]
        )
        writer.writeheader()
        for seed in seeds:
            for soft in softs:
                success, steps = results[(seed, soft)]
                writer.writerow(
                    {
                        "seed": seed,
                        "softness": f"{soft:.2f}",
                        "success": success,
                        "steps": steps,
                        "checkpoint_dir": str(checkpoint_dir),
                    }
                )

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["softness", "successes", "trials", "success_rate"]
        )
        writer.writeheader()
        for soft in softs:
            ok = sum(results[(seed, soft)][0] for seed in seeds)
            n = len(seeds)
            writer.writerow(
                {
                    "softness": f"{soft:.2f}",
                    "successes": ok,
                    "trials": n,
                    "success_rate": f"{ok / n:.4f}",
                }
            )
        total_ok = sum(v[0] for v in results.values())
        total_n = len(results)
        writer.writerow(
            {
                "softness": "overall",
                "successes": total_ok,
                "trials": total_n,
                "success_rate": f"{total_ok / total_n:.4f}",
            }
        )

    meta_path = output_dir / "meta.txt"
    meta_path.write_text(
        "\n".join(
            [
                f"checkpoint_dir={checkpoint_dir}",
                f"checkpoint_label={checkpoint_label(checkpoint_dir)}",
                f"seeds={seeds[0]}-{seeds[-1]} ({len(seeds)})"
                if seeds == list(range(seeds[0], seeds[-1] + 1))
                else f"seeds={seeds}",
                f"softness={softs[0]:.2f}-{softs[-1]:.2f} (n={len(softs)})",
                f"output_dir={output_dir}",
                f"created_at={datetime.now().isoformat(timespec='seconds')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    # C++ 側の即座 flush と順序がずれないようにする
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Softness × seed 格子で50歩生存成功率を評価する"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="RLlib checkpoint ルート（.../checkpoint でも可）",
    )
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=parse_seeds("1001-1010"),
        help="例: 1001-1010 または 1001,1002,...（既定: 1001-1010）",
    )
    parser.add_argument("--softness-min", type=float, default=0.91)
    parser.add_argument("--softness-max", type=float, default=1.0)
    parser.add_argument("--softness-step", type=float, default=0.01)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="未指定時は ~/vnoid-experiments/evals/<checkpoint名>_<日時>/",
    )
    parser.add_argument(
        "--no-rl-policy",
        action="store_true",
        help="学習済み方策を使わずゼロアクションで実行",
    )
    args = parser.parse_args()

    try:
        checkpoint_dir = resolve_checkpoint_dir(Path(args.checkpoint_dir))
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1

    softs = softness_values(args.softness_min, args.softness_max, args.softness_step)
    seeds = args.seeds
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output_dir(checkpoint_dir)
    )

    rl_module_path = checkpoint_dir / RL_MODULE_SUFFIX
    print("=" * 70)
    print("Softness 格子成功率評価")
    print("=" * 70)
    print(f"checkpoint: {checkpoint_dir}")
    print(f"RLModule:   {rl_module_path}")
    print(f"seeds:      {seeds[0]}..{seeds[-1]} ({len(seeds)})")
    print(f"softness:   {softs[0]:.2f}..{softs[-1]:.2f} ({len(softs)})")
    print(f"trials:     {len(seeds) * len(softs)}")
    print(f"output:     {output_dir}")
    print(f"policy:     {'off (zeros)' if args.no_rl_policy else 'RLModule'}")
    print("=" * 70)

    rl_module = None
    if not args.no_rl_policy:
        print("📥 RLModule をロード中...")
        try:
            rl_module = RLModule.from_checkpoint(str(rl_module_path))
        except Exception as e:
            print(f"❌ ポリシーのロード失敗: {e}")
            return 1
        print("✅ ロード完了")

    print("🎬 評価環境を作成中（描画なし）...")
    env = HumanoidVnoidEnv(
        enable_rendering=False,
        terrain_config={"mode": "terrain_softness", "terrain_softness": softs[0]},
    )

    results: dict[tuple[int, float], tuple[int, int]] = {}
    total = len(seeds) * len(softs)
    idx = 0
    try:
        for seed in seeds:
            for soft in softs:
                idx += 1
                success, steps = run_episode(env, rl_module, seed, soft)
                results[(seed, soft)] = (success, steps)
                print(
                    f"[{idx}/{total}] seed={seed} softness={soft:.2f} "
                    f"success={success} steps={steps}"
                )
    finally:
        env.close()

    print()
    print_matrix(seeds, softs, results)
    write_outputs(output_dir, checkpoint_dir, seeds, softs, results)
    print(f"\nCSV: {output_dir / 'results.csv'}")
    print(f"summary: {output_dir / 'summary.csv'}")
    print(f"meta: {output_dir / 'meta.txt'}")
    print("✅ 完了")
    return 0


if __name__ == "__main__":
    sys.exit(main())

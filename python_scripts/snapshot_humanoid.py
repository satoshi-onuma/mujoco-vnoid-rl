#!/usr/bin/env python3
"""論文用: 録画と同じ走行を斜め前カメラで進め、切替基準の5時刻をPNGで残す。

動画は出さない。時刻は reset で決まる切替ステップ（MuJoCo timestep 0.01s）基準。

  python snapshot_humanoid.py \\
    --checkpoint-dir ~/vnoid-experiments/runs/<run>/checkpoint \\
    --intervention-mode after_switch \\
    --terrain-softness 1.0 --seed 1001 \\
    --run-dir ~/vnoid-experiments/paper_cases/fail
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from eval_intervention import (
    add_intervention_args,
    needs_rl_module,
    resolve_intervention_mode,
    select_eval_action,
)
from my_humanoid_env import HumanoidVnoidEnv

# sample_robot_mujoco.xml の timestep。1フレーム = frame_skip(1) 回の mj_step。
SIM_DT = 0.01
# 絶対時刻 [s]。歩(0.4s)の位相が隣どうしでずれるよう 0.1s ずつ足す。最後は沈む 10.7秒。
SNAP_TIMES = (2.0, 4.1, 6.2, 8.3, 10.7)
# 50° は右後方になる。左前方はその反対なので +180°。
DEFAULT_AZIMUTH = 230.0
DEFAULT_ELEVATION = -18.0
DEFAULT_DISTANCE = 2.2
# 1280x720 の正立画像から切り出す枠 (left, top, right, bottom)。結果は 480x640（3:4）。
CROP_BOX = (410, 70, 890, 710)


def _prefix(seed: int, terrain_softness: float | None, terrain: str | None) -> str:
    seed_prefix = f"seed{seed}_"
    if terrain_softness is not None:
        return f"{seed_prefix}softness_{terrain_softness:.2f}_"
    if terrain:
        return f"{seed_prefix}{terrain}_"
    return seed_prefix


def main() -> int:
    parser = argparse.ArgumentParser(description="斜め前スナップショット（切替基準・5枚）")
    parser.add_argument("--checkpoint-dir", type=str, default="./humanoid_vnoid_checkpoint")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--terrain", type=str, default=None, choices=["hard", "soft", "debug", "random"])
    parser.add_argument("--terrain-softness", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--azimuth", type=float, default=DEFAULT_AZIMUTH)
    parser.add_argument("--elevation", type=float, default=DEFAULT_ELEVATION)
    parser.add_argument("--distance", type=float, default=DEFAULT_DISTANCE)
    parser.add_argument("--max-rl-steps", type=int, default=500)
    add_intervention_args(parser)
    args = parser.parse_args()

    mode = resolve_intervention_mode(args.intervention_mode, args.no_rl_policy)
    checkpoint_dir = os.path.abspath(os.path.expanduser(args.checkpoint_dir))
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = _prefix(args.seed, args.terrain_softness, args.terrain)

    rl_module = None
    if needs_rl_module(mode):
        if not os.path.isdir(checkpoint_dir):
            print(f"❌ チェックポイントがありません: {checkpoint_dir}")
            return 1
        from ray.rllib.core.rl_module import RLModule

        rl_path = os.path.join(
            checkpoint_dir, "learner_group", "learner", "rl_module", "default_policy"
        )
        print(f"📥 RLModule: {rl_path}")
        rl_module = RLModule.from_checkpoint(rl_path)

    terrain_config: dict | None = {}
    if args.terrain_softness is not None:
        terrain_config["mode"] = "terrain_softness"
        terrain_config["terrain_softness"] = args.terrain_softness
    elif args.terrain:
        terrain_config["mode"] = args.terrain
    else:
        terrain_config = None

    # C++ は CWD に control_log.csv を開く。既存の録画ログを上書きしない。
    scratch = tempfile.mkdtemp(prefix="vnoid_snap_")
    os.chdir(scratch)

    env = HumanoidVnoidEnv(
        enable_rendering=True,
        render_mode="rgb_array",
        terrain_config=terrain_config,
        max_episode_steps=args.max_rl_steps,
    )
    env.cpp_env.set_camera(args.azimuth, args.elevation, args.distance)
    obs, _info = env.reset(seed=args.seed)

    switch_at = int(env.cpp_env.get_terrain_switch_at())
    t0 = switch_at * SIM_DT
    targets = list(SNAP_TIMES)
    print(f"切替 t0={t0:.2f}s (step {switch_at})")
    print("スナップ時刻: " + ", ".join(f"{t:.2f}s" for t in targets))

    saved: dict[int, tuple[np.ndarray, float]] = {}
    last_frame: np.ndarray | None = None
    last_t = 0.0
    n_frames = 0
    terminated = False

    try:
        for i in range(args.max_rl_steps):
            action = select_eval_action(rl_module, obs, env, mode)
            obs, _reward, terminated, truncated, _info, step_frames = env.step(action)
            for frame in step_frames:
                n_frames += 1
                t = n_frames * SIM_DT
                # mjr_readPixels は下が先頭。論文用PNGは正立にする。
                last_frame = np.flipud(np.asarray(frame))
                last_t = t
                for k, target in enumerate(targets):
                    if k not in saved and t + 1e-9 >= target:
                        saved[k] = (last_frame, t)
            if all(k in saved for k in range(len(targets))):
                print(f"  5枚そろった (RL step {i + 1}, t={n_frames * SIM_DT:.2f}s)")
                break
            if terminated or truncated:
                print(f"  終了 (RL step {i + 1}, terminated={terminated}, t={n_frames * SIM_DT:.2f}s)")
                break
    finally:
        env.close()

    if last_frame is None:
        print("❌ フレームがありません")
        return 1

    for old in run_dir.glob(f"{prefix}snap_*.png"):
        old.unlink()

    written: list[Path] = []
    for k in range(len(targets)):
        img, t_save = saved.get(k, (last_frame, last_t))
        left, top, right, bottom = CROP_BOX
        img = img[top:bottom, left:right]
        out = run_dir / f"{prefix}snap_{k}_t{t_save:.2f}s.png"
        imageio.imwrite(out, img)
        written.append(out)
        tag = "captured" if k in saved else "last-frame"
        print(f"  {out.name} ({tag}, target {targets[k]:.2f}s)")

    print(f"✅ {len(written)} 枚 → {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

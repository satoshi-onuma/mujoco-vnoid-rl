"""
観測注入sweep: 沈み込み量 × 姿勢 → offset_x の方策応答を収集する。

使用例:
  python sweep_policy_offset_x.py \
    --checkpoint-dir ./humanoid_vnoid_checkpoint \
    --template-control-log ./control_log.csv \
    --posture-metric pitch \
    --sink-min -0.02 --sink-max 0.005 --sink-steps 50 \
    --posture-min -0.15 --posture-max 0.15 --posture-steps 50 \
    --out-csv sweep_result.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# obs index definitions (19-dim)
IDX_ANGVEL = slice(0, 3)       # angvel x,y,z
IDX_ORI = slice(3, 7)          # quaternion w,x,y,z
IDX_ACC = slice(7, 10)         # acc x,y,z
IDX_CONTACT_RIGHT = 10
IDX_CONTACT_LEFT = 11
IDX_FOOT_HEIGHT_RIGHT = 12
IDX_FOOT_HEIGHT_LEFT = 13
IDX_SINK_RIGHT = 14
IDX_SINK_LEFT = 15
IDX_CMD_STRIDE = 16
IDX_CMD_SWAY = 17
IDX_CMD_TURN = 18


def load_policy(checkpoint_dir: str):
    from ray.rllib.core.rl_module import RLModule
    rl_module_path = os.path.join(
        checkpoint_dir, "learner_group", "learner", "rl_module", "default_policy"
    )
    return RLModule.from_checkpoint(rl_module_path)


def build_templates_from_control_log(csv_path: str, step_threshold: int = 300):
    """control_log.csv の後半平均から右支持/左支持テンプレート(19-dim)を作成"""
    import pandas as pd
    df = pd.read_csv(csv_path)

    obs_cols_map = {
        'obs_angvel_x': 0, 'obs_angvel_y': 1, 'obs_angvel_z': 2,
        'obs_ori_w': 3, 'obs_ori_x': 4, 'obs_ori_y': 5, 'obs_ori_z': 6,
        'obs_acc_x': 7, 'obs_acc_y': 8, 'obs_acc_z': 9,
        'obs_contact_right': 10, 'obs_contact_left': 11,
        'obs_foot_height_right': 12, 'obs_foot_height_left': 13,
        'obs_foot_sink_right': 14, 'obs_foot_sink_left': 15,
    }
    cmd_cols_map = {
        'cmd_stride': 16, 'cmd_sway': 17, 'cmd_turn': 18,
    }

    missing = [c for c in ('obs_contact_right', 'obs_contact_left',
                            'obs_ori_w', 'obs_ori_x', 'obs_ori_y', 'obs_ori_z')
               if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} に必要な列がありません: {missing}\n"
            "評価録画で出る新しい control_log.csv を渡してください。"
        )

    if 'step' in df.columns:
        row_index = df['step']
    else:
        row_index = df.index
    df_late = df[row_index >= step_threshold]
    if len(df_late) == 0:
        print(f"  Warning: log has only {len(df)} rows, so threshold={step_threshold} "
              "selected nothing. Falling back to the whole log.")
        df_late = df
    print(f"  late rows: {len(df_late)} / {len(df)}")

    templates = {}
    for foot_label, contact_r, contact_l in [('right', 1, 0), ('left', 0, 1)]:
        mask = (df_late['obs_contact_right'].round() == contact_r) & \
               (df_late['obs_contact_left'].round() == contact_l)
        subset = df_late[mask]
        if len(subset) == 0:
            print(f"  Warning: no exclusive {foot_label}-support rows, "
                  "using all selected rows")
            subset = df_late
        if len(subset) == 0:
            raise ValueError(
                f"{csv_path} が空です。別の control_log.csv を指定してください。"
            )
        print(f"  {foot_label} support template rows: {len(subset)}")

        template = np.zeros(19, dtype=np.float64)
        # walk_cmd は control_log に無いので C++ 初期値をデフォルトにする
        template[IDX_CMD_STRIDE] = 0.1
        template[IDX_CMD_SWAY] = 0.0
        template[IDX_CMD_TURN] = 0.0
        for col, idx in obs_cols_map.items():
            if col in subset.columns:
                template[idx] = subset[col].mean()
        for col, idx in cmd_cols_map.items():
            if col in subset.columns:
                template[idx] = subset[col].mean()
        # 支持脚フラグは平均せず、sweep対象として固定する
        template[IDX_CONTACT_RIGHT] = float(contact_r)
        template[IDX_CONTACT_LEFT] = float(contact_l)

        q = template[IDX_ORI]
        if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
            print(f"  Warning: invalid mean quaternion for {foot_label}; using identity")
            template[IDX_ORI] = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            template[IDX_ORI] = q / np.linalg.norm(q)

        templates[foot_label] = template

    return templates


def rpy_to_quaternion_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """RPY -> quaternion [w, x, y, z]"""
    r = Rotation.from_euler('xyz', [roll, pitch, yaw])
    q_xyzw = r.as_quat()  # scipy returns [x,y,z,w]
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def quaternion_wxyz_to_rpy(q_wxyz: np.ndarray) -> np.ndarray:
    """quaternion [w,x,y,z] -> [roll, pitch, yaw]"""
    q = np.asarray(q_wxyz, dtype=np.float64)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-8:
        return np.zeros(3)
    q = q / n
    q_xyzw = np.array([q[1], q[2], q[3], q[0]])
    r = Rotation.from_quat(q_xyzw)
    return r.as_euler('xyz')


def inject_posture(obs: np.ndarray, metric: str, value: float) -> np.ndarray:
    """テンプレートの姿勢を指定メトリクスだけ上書きしてquaternionを再生成"""
    rpy = quaternion_wxyz_to_rpy(obs[IDX_ORI])
    if metric == 'roll':
        rpy[0] = value
    elif metric == 'pitch':
        rpy[1] = value
    elif metric == 'yaw':
        rpy[2] = value
    obs[IDX_ORI] = rpy_to_quaternion_wxyz(*rpy)
    return obs


def run_sweep(policy, template: np.ndarray, support_foot: str,
              sink_values: np.ndarray, posture_values: np.ndarray,
              posture_metric: str) -> list:
    rows = []
    sample_id = 0
    sink_idx = IDX_SINK_RIGHT if support_foot == 'right' else IDX_SINK_LEFT

    for sink in sink_values:
        for posture_val in posture_values:
            obs = template.copy()
            obs[sink_idx] = sink
            obs = inject_posture(obs, posture_metric, posture_val)

            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out = policy.forward_inference({"obs": obs_tensor})
            action_params = out["action_dist_inputs"][0].numpy()
            # record_humanoid.py と同じ: 正規化行動を clip してから m へ
            offset_x = float(np.clip(action_params[0], -1.0, 1.0)) * 0.15

            rows.append({
                'sample_id': sample_id,
                'sink': sink,
                'posture_metric': posture_metric,
                'posture_value': posture_val,
                'support_foot': support_foot,
                'offset_x': offset_x,
            })
            sample_id += 1

    return rows


def main():
    parser = argparse.ArgumentParser(description="Policy observation-injection sweep for offset_x")
    parser.add_argument("--checkpoint-dir", type=str, default="./humanoid_vnoid_checkpoint")
    parser.add_argument("--out-csv", type=str, default="sweep_offset_x.csv")
    parser.add_argument("--template-control-log", type=str, default="./control_log.csv")
    parser.add_argument("--template-step-threshold", type=int, default=300)
    parser.add_argument("--sink-min", type=float, default=-0.02)
    parser.add_argument("--sink-max", type=float, default=0.005)
    parser.add_argument("--sink-steps", type=int, default=50)
    parser.add_argument("--posture-min", type=float, default=-0.15)
    parser.add_argument("--posture-max", type=float, default=0.15)
    parser.add_argument("--posture-steps", type=int, default=50)
    parser.add_argument("--posture-metric", type=str, default="pitch",
                        choices=["pitch", "roll", "yaw"])
    parser.add_argument("--support-foot-mode", type=str, default="both",
                        choices=["right", "left", "both"])
    parser.add_argument("--obs-template-json", type=str, default=None,
                        help="Override template with a JSON file (19-element array)")
    args = parser.parse_args()

    print("=" * 60)
    print("Policy Observation-Injection Sweep")
    print("=" * 60)

    # Load policy
    checkpoint_dir = os.path.abspath(os.path.expanduser(args.checkpoint_dir))
    print(f"Loading policy from: {checkpoint_dir}")
    policy = load_policy(checkpoint_dir)
    print("Policy loaded.")

    # Build templates
    if args.obs_template_json:
        import json
        with open(args.obs_template_json) as f:
            raw = json.load(f)
        base_template = np.array(raw, dtype=np.float64)
        templates = {'right': base_template.copy(), 'left': base_template.copy()}
    else:
        log_path = os.path.abspath(args.template_control_log)
        print(f"Building templates from: {log_path} (step >= {args.template_step_threshold})")
        templates = build_templates_from_control_log(log_path, args.template_step_threshold)

    # Sweep axes
    sink_values = np.linspace(args.sink_min, args.sink_max, args.sink_steps)
    posture_values = np.linspace(args.posture_min, args.posture_max, args.posture_steps)

    feet = []
    if args.support_foot_mode in ('right', 'both'):
        feet.append('right')
    if args.support_foot_mode in ('left', 'both'):
        feet.append('left')

    all_rows = []
    for foot in feet:
        print(f"Sweeping support_foot={foot} ...")
        rows = run_sweep(policy, templates[foot], foot,
                         sink_values, posture_values, args.posture_metric)
        all_rows.extend(rows)
        print(f"  {len(rows)} samples collected.")

    # Write CSV
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['sample_id', 'sink', 'posture_metric', 'posture_value', 'support_foot', 'offset_x']
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone. {len(all_rows)} rows written to: {out_path}")
    print(f"  sink range: [{args.sink_min}, {args.sink_max}] ({args.sink_steps} steps)")
    print(f"  posture ({args.posture_metric}): [{args.posture_min}, {args.posture_max}] ({args.posture_steps} steps)")
    print(f"  support feet: {feet}")


if __name__ == "__main__":
    main()

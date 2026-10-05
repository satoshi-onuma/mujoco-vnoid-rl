"""評価時の RL 介入モード（zeros vs 方策）を一箇所で分岐する。"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch

INTERVENTION_MODES = ("none", "full", "after_switch")


def add_intervention_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--intervention-mode",
        type=str,
        choices=INTERVENTION_MODES,
        default="full",
        help="介入モード: none=常にzeros / full=常にRL / after_switch=地面切替後のみRL",
    )
    parser.add_argument(
        "--no-rl-policy",
        action="store_true",
        help="互換: --intervention-mode none と同じ",
    )


def resolve_intervention_mode(mode: str, no_rl_policy: bool = False) -> str:
    if no_rl_policy:
        return "none"
    if mode not in INTERVENTION_MODES:
        raise ValueError(
            f"未知の intervention-mode: {mode} (候補: {', '.join(INTERVENTION_MODES)})"
        )
    return mode


def intervention_dir_tag(mode: str) -> str:
    return f"interv-{mode}"


def needs_rl_module(mode: str) -> bool:
    return mode != "none"


def infer_rl(rl_module: Any, obs, env) -> np.ndarray:
    obs_batch = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        model_outputs = rl_module.forward_inference({"obs": obs_batch})
    action_dist_params = model_outputs["action_dist_inputs"][0].numpy()
    return np.clip(
        action_dist_params[:2],
        a_min=env.action_space.low,
        a_max=env.action_space.high,
    )


def select_eval_action(
    rl_module: Any | None,
    obs,
    env,
    intervention_mode: str,
) -> np.ndarray:
    """介入モードに応じて zeros か RL アクションを返す（評価の唯一の分岐点）。"""
    if intervention_mode == "none":
        return np.zeros(2, dtype=np.float32)
    if intervention_mode == "after_switch" and not env.cpp_env.is_terrain_switched():
        return np.zeros(2, dtype=np.float32)
    if rl_module is None:
        return np.zeros(2, dtype=np.float32)
    return infer_rl(rl_module, obs, env)

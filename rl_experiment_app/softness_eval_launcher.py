"""softness 格子評価（grid / paper_log）の subprocess 起動・監視。"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = {
    "grid": REPO_ROOT / "python_scripts" / "eval_softness_grid.py",
    "paper_log": REPO_ROOT / "python_scripts" / "eval_softness_paper_log.py",
}
DEFAULT_EXPERIMENTS_ROOT = Path.home() / "vnoid-experiments"
OUTPUT_ROOTS = {
    "grid": DEFAULT_EXPERIMENTS_ROOT / "evals",
    "paper_log": DEFAULT_EXPERIMENTS_ROOT / "paper_logs",
}
LOG_ROOT = DEFAULT_EXPERIMENTS_ROOT / "evals" / "_logs"


def predict_output_dir(
    run_id: str,
    softness_min: float,
    softness_max: float,
    softness_step: float,
    script: str = "grid",
    intervention_mode: str = "full",
) -> Path:
    """eval スクリプトの default_output_dir と同じ命名。"""
    soft_tag = f"s{softness_min:.2f}-{softness_max:.2f}_step{softness_step:.2f}"
    root = OUTPUT_ROOTS.get(script, OUTPUT_ROOTS["grid"])
    return root / f"{run_id}_{soft_tag}_interv-{intervention_mode}"


class SoftnessEvalLauncher:
    def __init__(self) -> None:
        self.process: Optional[subprocess.Popen] = None
        self.label: Optional[str] = None
        self.output_dir: Optional[Path] = None
        self.log_path: Optional[Path] = None
        self._log_file = None
        self._finalized = False
        self.last_params: dict | None = None

    def start(self, params: dict) -> Path:
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("既に評価プロセスが実行中です")

        script_key = str(params.get("script", "grid"))
        if script_key not in SCRIPTS:
            raise ValueError(f"未知の評価スクリプト: {script_key}")
        script_path = SCRIPTS[script_key]
        if not script_path.is_file():
            raise FileNotFoundError(f"評価スクリプトがありません: {script_path}")

        checkpoint_dir = Path(params["checkpoint_dir"]).expanduser()
        softness_min = float(params["softness_min"])
        softness_max = float(params["softness_max"])
        softness_step = float(params["softness_step"])
        seeds = str(params.get("seeds", "1001-1010"))
        run_id = str(params.get("run_id", checkpoint_dir.parent.name))
        intervention_mode = str(params.get("intervention_mode", "full"))

        self.output_dir = predict_output_dir(
            run_id,
            softness_min,
            softness_max,
            softness_step,
            script_key,
            intervention_mode,
        )
        self.label = str(
            params.get("label")
            or f"eval {run_id} [{script_key}] interv={intervention_mode}"
        )
        self.last_params = {
            "run_id": run_id,
            "script": script_key,
            "seeds": seeds,
            "softness_min": softness_min,
            "softness_max": softness_max,
            "softness_step": softness_step,
            "intervention_mode": intervention_mode,
        }
        self._finalized = False

        argv = [
            sys.executable,
            str(script_path),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--seeds",
            seeds,
            "--softness-min",
            str(softness_min),
            "--softness-max",
            str(softness_max),
            "--softness-step",
            str(softness_step),
            "--intervention-mode",
            intervention_mode,
        ]

        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = LOG_ROOT / f"eval_{run_id}_{stamp}.log"
        self._log_file = open(self.log_path, "w", encoding="utf-8")

        self.process = subprocess.Popen(
            argv,
            cwd=str(REPO_ROOT / "python_scripts"),
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        return self.output_dir

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def poll(self) -> Optional[int]:
        """実行中なら None。終了直後の一回だけ returncode、その後は None。"""
        if self.process is None:
            return None
        code = self.process.poll()
        if code is None:
            return None
        if self._finalized:
            return None
        self._finalize()
        return code

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if not self._finalized:
            self._finalize()

    def _finalize(self) -> None:
        self._finalized = True
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

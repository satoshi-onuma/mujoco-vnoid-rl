"""単発学習の起動・監視。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .database import DEFAULT_RUNS_ROOT, ExperimentDB, ensure_runs_root

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = REPO_ROOT / "python_scripts" / "train_humanoid.py"


def make_run_id(prefix: str = "") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    if prefix:
        return f"{prefix}_{stamp}_{short}"
    return f"{stamp}_{short}"


class TrainingLauncher:
    def __init__(self, db: Optional[ExperimentDB] = None):
        ensure_runs_root()
        self.db = db or ExperimentDB()
        self.process: Optional[subprocess.Popen] = None
        self.run_id: Optional[str] = None
        self.run_dir: Optional[Path] = None
        self.csv_path: Optional[Path] = None
        self.on_complete: Optional[Callable[[str, Path], None]] = None

    def build_argv(self, params: dict, run_id: str, run_dir: Path) -> list[str]:
        argv = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--run-id", run_id,
            "--run-dir", str(run_dir),
            "--w-track", str(params.get("w_track", 1.0)),
            "--w-act", str(params.get("w_act", 0.1)),
            "--w-healthy", str(params.get("w_healthy", 1.0)),
            "--tracking-sigma", str(params.get("tracking_sigma", 0.02)),
            "--terrain", str(params.get("terrain", "soft")),
            "--lr", str(params.get("lr", 1e-4)),
            "--gamma", str(params.get("gamma", 0.99)),
            "--num-workers", str(params.get("num_workers", 8)),
            "--num-gpus", str(params.get("num_gpus", 1)),
            "--num-iterations", str(params.get("num_iterations", 100)),
            "--seed", str(params.get("seed", 42)),
        ]
        return argv

    def start(
        self,
        params: dict,
        run_id: Optional[str] = None,
        run_dir: Optional[Path] = None,
    ) -> tuple[str, Path, Path]:
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("既に学習プロセスが実行中です")

        ensure_runs_root()
        self.run_id = run_id or make_run_id()
        if run_dir is None:
            self.run_dir = DEFAULT_RUNS_ROOT / self.run_id
        else:
            self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "training_stats.csv"
        checkpoint_dir = self.run_dir / "checkpoint"

        reward_weights = {
            "w_track": float(params.get("w_track", 1.0)),
            "w_act": float(params.get("w_act", 0.1)),
            "w_healthy": float(params.get("w_healthy", 1.0)),
            "tracking_sigma": float(params.get("tracking_sigma", 0.02)),
        }
        hyperparams = {
            "lr": float(params.get("lr", 1e-4)),
            "gamma": float(params.get("gamma", 0.99)),
            "num_workers": int(params.get("num_workers", 8)),
            "num_gpus": int(params.get("num_gpus", 1)),
            "seed": int(params.get("seed", 42)),
        }

        self.db.insert_experiment(
            run_id=self.run_id,
            run_dir=str(self.run_dir),
            reward_weights=reward_weights,
            terrain_mode=str(params.get("terrain", "soft")),
            hyperparams=hyperparams,
            num_iterations=int(params.get("num_iterations", 100)),
            status="running",
            checkpoint_dir=str(checkpoint_dir),
            csv_path=str(self.csv_path),
            note=str(params.get("note", "") or ""),
        )

        argv = self.build_argv(params, self.run_id, self.run_dir)
        log_path = self.run_dir / "train.log"
        log_file = open(log_path, "w")
        self.process = subprocess.Popen(
            argv,
            cwd=str(REPO_ROOT / "python_scripts"),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        # ログファイルハンドルはプロセス寿命まで開いたままにする
        self._log_file = log_file
        return self.run_id, self.run_dir, self.csv_path

    def poll(self) -> Optional[int]:
        """プロセスが終了していれば returncode、実行中なら None。"""
        if self.process is None:
            return None
        code = self.process.poll()
        if code is not None:
            self._finalize(code)
        return code

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            if self.run_id:
                self.db.update_experiment_status(self.run_id, "early_stopped")

    def _finalize(self, returncode: int) -> None:
        if self.run_id is None or self.run_dir is None:
            return
        result_path = self.run_dir / "result.json"
        if returncode == 0 and result_path.exists():
            with open(result_path) as f:
                result = json.load(f)
            self.db.update_experiment_status(
                self.run_id,
                status="completed",
                final_reward_mean=result.get("final_reward_mean"),
                final_episode_len_mean=result.get("final_episode_len_mean"),
                elapsed_time_s=result.get("elapsed_time_s"),
                checkpoint_dir=result.get("checkpoint_dir"),
                csv_path=result.get("csv_path"),
            )
            if self.on_complete:
                self.on_complete(self.run_id, self.run_dir)
        else:
            self.db.update_experiment_status(self.run_id, "failed")

        if hasattr(self, "_log_file") and self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass

"""履歴画面から手動起動する学習済みモデルの評価。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .database import ExperimentDB

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORD_SCRIPT = REPO_ROOT / "python_scripts" / "record_humanoid.py"

# 切り替え先地盤のバリエーション（エピソード開始時は常に硬地盤）
DEFAULT_EVAL_TERRAINS = ["soft", "random"]
# 0.0=hard, 1.0=soft, 1.1まで学習範囲, 1.2以上は外挿
DEFAULT_EVAL_SOFTNESS = [round(0.2 * i, 1) for i in range(8)]  # 0.0 .. 1.4


class EvalLauncher:
    def __init__(self, db: Optional[ExperimentDB] = None):
        self.db = db or ExperimentDB()

    def evaluate_run(
        self,
        run_id: str,
        run_dir: Path,
        terrains: Optional[list[str]] = None,
        softness_values: Optional[list[float]] = None,
        total_steps: int = 200,
    ) -> list[dict]:
        run_dir = Path(run_dir)
        checkpoint_dir = run_dir / "checkpoint"
        if not checkpoint_dir.exists():
            print(f"⚠️ チェックポイントがありません: {checkpoint_dir}")
            return []

        if terrains is None and softness_values is None:
            softness_values = DEFAULT_EVAL_SOFTNESS

        results = []
        for terrain in terrains or []:
            result = self._run_record(
                run_id, run_dir, checkpoint_dir, total_steps, terrain=terrain
            )
            if result:
                results.append(result)
        for softness in softness_values or []:
            result = self._run_record(
                run_id, run_dir, checkpoint_dir, total_steps,
                terrain_softness=softness,
            )
            if result:
                results.append(result)
        return results

    def _run_record(
        self,
        run_id: str,
        run_dir: Path,
        checkpoint_dir: Path,
        total_steps: int,
        terrain: Optional[str] = None,
        terrain_softness: Optional[float] = None,
    ) -> Optional[dict]:
        if terrain_softness is not None:
            label = f"softness_{terrain_softness:.2f}"
        else:
            label = terrain or "default"

        argv = [
            sys.executable,
            str(RECORD_SCRIPT),
            "--checkpoint-dir", str(checkpoint_dir),
            "--run-dir", str(run_dir),
            "--total-steps", str(total_steps),
        ]
        if terrain_softness is not None:
            argv.extend(["--terrain-softness", str(terrain_softness)])
        elif terrain:
            argv.extend(["--terrain", terrain])

        log_path = run_dir / f"eval_{label}.log"
        try:
            completed = subprocess.run(
                argv,
                cwd=str(REPO_ROOT / "python_scripts"),
                capture_output=True,
                text=True,
                env=os.environ.copy(),
                timeout=3600,
            )
            with open(log_path, "w") as f:
                f.write(completed.stdout)
                f.write(completed.stderr)

            result = None
            for line in completed.stdout.splitlines():
                if line.startswith("EVAL_RESULT_JSON:"):
                    result = json.loads(line[len("EVAL_RESULT_JSON:"):])
                    break

            if result is None:
                # フォールバック: 出力ファイルの存在だけ記録
                video = run_dir / f"{label}_demo.mp4"
                csv_path = run_dir / f"{label}_recording_log.csv"
                result = {
                    "terrain_mode": label,
                    "walk_distance": None,
                    "video_path": str(video) if video.exists() else None,
                    "log_csv_path": str(csv_path) if csv_path.exists() else None,
                }

            self.db.insert_evaluation(
                experiment_id=run_id,
                terrain_mode=result.get("terrain_mode", label),
                walk_distance=result.get("walk_distance"),
                video_path=result.get("video_path"),
                log_csv_path=result.get("log_csv_path"),
            )
            return result
        except Exception as e:
            print(f"❌ 評価失敗 ({label}): {e}")
            with open(log_path, "a") as f:
                f.write(f"\nERROR: {e}\n")
            return None

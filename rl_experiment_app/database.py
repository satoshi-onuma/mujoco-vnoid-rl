"""実験データ管理モジュール（SQLite）。

DBファイル: ~/vnoid-experiments/runs/experiments.db
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

DEFAULT_RUNS_ROOT = Path.home() / "vnoid-experiments" / "runs"
DEFAULT_DB_PATH = DEFAULT_RUNS_ROOT / "experiments.db"


def ensure_runs_root() -> Path:
    DEFAULT_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    return DEFAULT_RUNS_ROOT


class ExperimentDB:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if db_path is None:
            ensure_runs_root()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    run_dir TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    reward_weights_json TEXT NOT NULL,
                    terrain_mode TEXT NOT NULL,
                    hyperparams_json TEXT NOT NULL,
                    num_iterations INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    checkpoint_dir TEXT,
                    csv_path TEXT,
                    final_reward_mean REAL,
                    final_episode_len_mean REAL,
                    elapsed_time_s REAL,
                    note TEXT
                );

                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    terrain_mode TEXT NOT NULL,
                    walk_distance REAL,
                    video_path TEXT,
                    log_csv_path TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                );
                """
            )
            # 既存DB向け: note 列が無ければ追加
            cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(experiments)").fetchall()
            }
            if "note" not in cols:
                conn.execute("ALTER TABLE experiments ADD COLUMN note TEXT")

    def insert_experiment(
        self,
        run_id: str,
        run_dir: str,
        reward_weights: dict,
        terrain_mode: str,
        hyperparams: dict,
        num_iterations: int,
        status: str = "running",
        checkpoint_dir: Optional[str] = None,
        csv_path: Optional[str] = None,
        note: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                    id, run_dir, created_at,
                    reward_weights_json, terrain_mode, hyperparams_json,
                    num_iterations, status, checkpoint_dir, csv_path, note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    run_dir,
                    datetime.now().isoformat(timespec="seconds"),
                    json.dumps(reward_weights),
                    terrain_mode,
                    json.dumps(hyperparams),
                    num_iterations,
                    status,
                    checkpoint_dir,
                    csv_path,
                    note or "",
                ),
            )

    def update_experiment_note(self, run_id: str, note: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE experiments SET note = ? WHERE id = ?",
                (note or "", run_id),
            )

    def update_experiment_status(
        self,
        run_id: str,
        status: str,
        final_reward_mean: Optional[float] = None,
        final_episode_len_mean: Optional[float] = None,
        elapsed_time_s: Optional[float] = None,
        checkpoint_dir: Optional[str] = None,
        csv_path: Optional[str] = None,
    ) -> None:
        fields = ["status = ?"]
        values: list[Any] = [status]
        if final_reward_mean is not None:
            fields.append("final_reward_mean = ?")
            values.append(final_reward_mean)
        if final_episode_len_mean is not None:
            fields.append("final_episode_len_mean = ?")
            values.append(final_episode_len_mean)
        if elapsed_time_s is not None:
            fields.append("elapsed_time_s = ?")
            values.append(elapsed_time_s)
        if checkpoint_dir is not None:
            fields.append("checkpoint_dir = ?")
            values.append(checkpoint_dir)
        if csv_path is not None:
            fields.append("csv_path = ?")
            values.append(csv_path)
        values.append(run_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE experiments SET {', '.join(fields)} WHERE id = ?",
                values,
            )

    def insert_evaluation(
        self,
        experiment_id: str,
        terrain_mode: str,
        walk_distance: Optional[float],
        video_path: Optional[str],
        log_csv_path: Optional[str],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO evaluations (
                    experiment_id, terrain_mode, walk_distance,
                    video_path, log_csv_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    terrain_mode,
                    walk_distance,
                    video_path,
                    log_csv_path,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def get_experiment(self, run_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (run_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_experiments(self, limit: int = 200) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT e.*,
                       (SELECT walk_distance FROM evaluations ev
                        WHERE ev.experiment_id = e.id
                        ORDER BY ev.id LIMIT 1) AS walk_distance
                FROM experiments e
                ORDER BY e.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def list_evaluations(self, experiment_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM evaluations
                WHERE experiment_id = ?
                ORDER BY id
                """,
                (experiment_id,),
            ).fetchall()
            return [dict(r) for r in rows]

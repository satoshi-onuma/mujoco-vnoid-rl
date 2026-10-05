"""Plot 一覧用: eval/paper dir 名に学習 DB のメモを付ける。"""

from __future__ import annotations

from ..database import ExperimentDB
from ..paper_eval_link import parse_run_id


def experiment_note(db: ExperimentDB, run_id: str) -> str:
    row = db.get_experiment(run_id)
    if not row:
        return ""
    return (row.get("note") or "").strip()


def label_with_note(db: ExperimentDB, dir_name: str) -> str:
    """dir 名先頭 run_id からメモを取得し、`| note` を後ろに付ける。"""
    rid = parse_run_id(dir_name)
    if not rid:
        return dir_name
    note = experiment_note(db, rid)
    if note:
        return f"{dir_name}  | {note}"
    return dir_name


def run_id_with_note(db: ExperimentDB, run_id: str) -> str:
    note = experiment_note(db, run_id)
    if note:
        return f"{run_id}  | {note}"
    return run_id

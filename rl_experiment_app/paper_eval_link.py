"""学習 DB と paper_log 出力の紐付け（DB 記録 + ディレクトリ scan fallback）。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .database import DEFAULT_DB_PATH, ExperimentDB
from .softness_eval_launcher import predict_output_dir

DEFAULT_PAPER_ROOT = Path.home() / "vnoid-experiments" / "paper_logs"
DEFAULT_RUNS_ROOT = Path.home() / "vnoid-experiments" / "runs"

TARGET_SOFTNESS = 1.0
TRIAL_SOFT_RE = re.compile(
    r"^seed(?P<seed>\d+)_softness_(?P<softness>[0-9]+(?:\.[0-9]+)?)$"
)

PaperStatus = Literal["ready", "partial", "no_paper"]


@dataclass(frozen=True)
class PaperLink:
    experiment_id: str
    paper_dir: Path | None
    status: PaperStatus
    expected_seeds: int
    n_alpha_trials: int
    overall_trials: int
    overall_success: float | None
    source: str  # db | scan | none
    seeds_spec: str | None = None


def parse_run_id(dir_name: str) -> str | None:
    parts = dir_name.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
    return None


def _softness_matches(soft_str: str, target: float = TARGET_SOFTNESS) -> bool:
    return abs(float(soft_str) - target) < 1e-6


def list_alpha_one_trials(paper_run_dir: Path) -> list[tuple[int, Path]]:
    """α=1.00 trial → (seed, path)。"""
    out: list[tuple[int, Path]] = []
    if not paper_run_dir.is_dir():
        return out
    for d in sorted(paper_run_dir.iterdir()):
        if not d.is_dir():
            continue
        m = TRIAL_SOFT_RE.match(d.name)
        if not m or not _softness_matches(m.group("softness")):
            continue
        if (d / "control.csv").is_file():
            out.append((int(m.group("seed")), d))
    return out


def parse_expected_seed_count(meta_path: Path, seeds_spec: str | None = None) -> int:
    if meta_path.is_file():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("seeds="):
                continue
            val = line.split("=", 1)[1].strip()
            m = re.search(r"\((\d+)\)\s*$", val)
            if m:
                return int(m.group(1))
            if "-" in val and "(" not in val.split("-", 1)[0]:
                left, right = val.split("-", 1)
                left = left.strip().split()[0]
                right = re.split(r"\s|\(", right.strip())[0]
                return int(right) - int(left) + 1
    if seeds_spec:
        spec = seeds_spec.strip()
        if "-" in spec and "," not in spec:
            a, b = spec.split("-", 1)
            return int(b) - int(a) + 1
        return len([p for p in spec.split(",") if p.strip()])
    return 0


def load_overall(paper_dir: Path) -> tuple[float, int] | None:
    path = paper_dir / "summary.csv"
    if not path.is_file():
        return None
    import csv

    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("softness", "").strip() == "overall":
                return float(row["success_rate"]), int(row["trials"])
    return None


def load_softness_success(
    paper_dir: Path,
    target: float = TARGET_SOFTNESS,
) -> tuple[float, int] | None:
    """summary.csv の指定 softness 行 → (success_rate, trials)。overall は除外。"""
    path = paper_dir / "summary.csv"
    if not path.is_file():
        return None
    import csv

    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            soft = row.get("softness", "").strip()
            if soft == "overall" or not soft:
                continue
            try:
                if abs(float(soft) - target) < 1e-6:
                    return float(row["success_rate"]), int(row["trials"])
            except ValueError:
                continue
    return None


def intervention_dir_suffix(mode: str) -> str:
    """dir 名末尾の `_interv-{mode}`（softness_eval_launcher の命名と一致）。"""
    return f"_interv-{mode}"


def scan_paper_dirs(
    experiment_id: str,
    paper_root: Path,
    *,
    intervention_mode: str | None = None,
) -> list[Path]:
    if not paper_root.is_dir():
        return []
    prefix = experiment_id + "_"
    suffix = (
        intervention_dir_suffix(intervention_mode)
        if intervention_mode is not None
        else None
    )
    return sorted(
        d
        for d in paper_root.iterdir()
        if d.is_dir()
        and d.name.startswith(prefix)
        and (d / "summary.csv").is_file()
        and (suffix is None or d.name.endswith(suffix))
    )


def pick_best_paper_dir(candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    best: Path | None = None
    best_trials = -1
    for d in candidates:
        overall = load_overall(d)
        if overall is None:
            continue
        _rate, n = overall
        n_alpha = len(list_alpha_one_trials(d))
        if n_alpha == 0:
            continue
        score = n
        if score > best_trials:
            best_trials = score
            best = d
    return best


def resolve_paper_dir(
    experiment_id: str,
    db: ExperimentDB,
    paper_root: Path = DEFAULT_PAPER_ROOT,
    *,
    intervention_mode: str | None = None,
) -> tuple[Path | None, str, str | None]:
    """(paper_dir, source, seeds_spec from db row)。

    intervention_mode を渡すとそのモードの paper のみ採用。None なら従来どおり（フィルタなし）。
    """
    rows = db.list_softness_eval_runs(experiment_id=experiment_id, script="paper_log")
    for row in rows:
        if row.get("status") != "completed":
            continue
        if (
            intervention_mode is not None
            and row.get("intervention_mode") != intervention_mode
        ):
            continue
        p = Path(row["output_dir"])
        if p.is_dir() and (p / "summary.csv").is_file():
            return p, "db", row.get("seeds")
    scanned = scan_paper_dirs(
        experiment_id, paper_root, intervention_mode=intervention_mode
    )
    picked = pick_best_paper_dir(scanned)
    if picked is not None:
        return picked, "scan", None
    return None, "none", None


def assess_paper_link(
    experiment_id: str,
    paper_dir: Path | None,
    *,
    seeds_spec: str | None = None,
    min_eval_seeds: int = 10,
    min_overall_trials: int = 100,
) -> PaperLink:
    if paper_dir is None:
        return PaperLink(
            experiment_id=experiment_id,
            paper_dir=None,
            status="no_paper",
            expected_seeds=0,
            n_alpha_trials=0,
            overall_trials=0,
            overall_success=None,
            source="none",
            seeds_spec=seeds_spec,
        )

    meta = paper_dir / "meta.txt"
    expected = parse_expected_seed_count(meta, seeds_spec)
    if expected <= 0:
        expected = min_eval_seeds

    alpha = list_alpha_one_trials(paper_dir)
    n_alpha = len(alpha)
    overall = load_overall(paper_dir)
    if overall is None:
        return PaperLink(
            experiment_id=experiment_id,
            paper_dir=paper_dir,
            status="partial",
            expected_seeds=expected,
            n_alpha_trials=n_alpha,
            overall_trials=0,
            overall_success=None,
            source="",
            seeds_spec=seeds_spec,
        )
    success_rate, n_trials = overall

    ready = (
        n_alpha == expected
        and expected >= min_eval_seeds
        and n_trials >= min_overall_trials
    )
    status: PaperStatus = "ready" if ready else "partial"
    return PaperLink(
        experiment_id=experiment_id,
        paper_dir=paper_dir,
        status=status,
        expected_seeds=expected,
        n_alpha_trials=n_alpha,
        overall_trials=n_trials,
        overall_success=success_rate,
        source="",
        seeds_spec=seeds_spec,
    )


def list_paper_links_for_experiments(
    db: ExperimentDB,
    experiment_ids: list[str] | None = None,
    *,
    paper_root: Path = DEFAULT_PAPER_ROOT,
    min_eval_seeds: int = 10,
    min_overall_trials: int = 100,
    intervention_mode: str | None = None,
) -> list[PaperLink]:
    if experiment_ids is None:
        experiment_ids = [row["id"] for row in db.list_experiments(limit=500)]
    links: list[PaperLink] = []
    for eid in experiment_ids:
        paper_dir, source, seeds_spec = resolve_paper_dir(
            eid, db, paper_root, intervention_mode=intervention_mode
        )
        link = assess_paper_link(
            eid,
            paper_dir,
            seeds_spec=seeds_spec,
            min_eval_seeds=min_eval_seeds,
            min_overall_trials=min_overall_trials,
        )
        links.append(
            PaperLink(
                experiment_id=link.experiment_id,
                paper_dir=link.paper_dir,
                status=link.status,
                expected_seeds=link.expected_seeds,
                n_alpha_trials=link.n_alpha_trials,
                overall_trials=link.overall_trials,
                overall_success=link.overall_success,
                source=source,
                seeds_spec=seeds_spec,
            )
        )
    return links


def w_act_from_experiment(db: ExperimentDB, experiment_id: str) -> float | None:
    row = db.get_experiment(experiment_id)
    if row and row.get("reward_weights_json"):
        try:
            return float(json.loads(row["reward_weights_json"])["w_act"])
        except (KeyError, TypeError, json.JSONDecodeError):
            pass
    path = DEFAULT_RUNS_ROOT / experiment_id / "result.json"
    if path.is_file():
        try:
            return float(json.loads(path.read_text())["reward_weights"]["w_act"])
        except (KeyError, TypeError, json.JSONDecodeError):
            pass
    return None


def predict_paper_output_dir(
    run_id: str,
    softness_min: float,
    softness_max: float,
    softness_step: float,
    intervention_mode: str = "full",
) -> Path:
    return predict_output_dir(
        run_id,
        softness_min,
        softness_max,
        softness_step,
        "paper_log",
        intervention_mode,
    )

"""履歴タブ: Treeview一覧 / softness評価キュー投入 / 削除。"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

import tkinter as tk
from tkinter import ttk, messagebox

from ..database import DEFAULT_RUNS_ROOT, ExperimentDB
from ..paper_eval_link import list_paper_links_for_experiments


def parse_seeds_text(raw: str) -> str:
    """UI入力を検証し、CLI に渡す文字列を返す（空なら既定）。"""
    raw = raw.strip()
    if not raw:
        return "1001-1010"
    if "-" in raw and "," not in raw:
        left, right = raw.split("-", 1)
        start, end = int(left), int(right)
        if end < start:
            raise ValueError(f"seed 範囲が不正です: {raw}")
        return f"{start}-{end}"
    seeds: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise ValueError("seed が空です")
    return ",".join(str(s) for s in seeds)


class HistoryTab(ttk.Frame):
    def __init__(
        self,
        master,
        db: ExperimentDB,
        on_enqueue_eval: Callable[[dict], None],
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.db = db
        self.on_enqueue_eval = on_enqueue_eval
        self._build()
        self.refresh()

    def _build(self):
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", padx=8, pady=4)
        ttk.Button(toolbar, text="更新", command=self.refresh).pack(side="left", padx=2)
        ttk.Button(toolbar, text="削除", command=self.delete_selected).pack(side="left", padx=2)

        eval_frame = ttk.LabelFrame(self, text="Softness 評価（キューへ）")
        eval_frame.pack(fill="x", padx=8, pady=4)

        row1 = ttk.Frame(eval_frame)
        row1.pack(fill="x", padx=4, pady=2)
        ttk.Label(row1, text="seeds").pack(side="left")
        self.eval_seed_entry = ttk.Entry(row1, width=18)
        self.eval_seed_entry.insert(0, "1001-1010")
        self.eval_seed_entry.pack(side="left", padx=4)
        ttk.Label(row1, text="(1001-1010 またはカンマ区切り)").pack(side="left")

        ttk.Label(row1, text="script").pack(side="left", padx=(12, 2))
        self.script_var = tk.StringVar(value="grid")
        ttk.Combobox(
            row1,
            textvariable=self.script_var,
            values=["grid", "paper_log"],
            width=10,
            state="readonly",
        ).pack(side="left", padx=2)
        ttk.Label(
            row1,
            text="paper_log は control.csv が増え容量注意",
        ).pack(side="left", padx=4)

        ttk.Label(row1, text="interv").pack(side="left", padx=(12, 2))
        self.interv_var = tk.StringVar(value="full")
        ttk.Combobox(
            row1,
            textvariable=self.interv_var,
            values=["none", "full", "after_switch"],
            width=12,
            state="readonly",
        ).pack(side="left", padx=2)

        row2 = ttk.Frame(eval_frame)
        row2.pack(fill="x", padx=4, pady=2)
        ttk.Label(row2, text="softness min").pack(side="left")
        self.soft_min_entry = ttk.Entry(row2, width=8)
        self.soft_min_entry.insert(0, "0.0")
        self.soft_min_entry.pack(side="left", padx=4)
        ttk.Label(row2, text="max").pack(side="left")
        self.soft_max_entry = ttk.Entry(row2, width=8)
        self.soft_max_entry.insert(0, "1.2")
        self.soft_max_entry.pack(side="left", padx=4)
        ttk.Label(row2, text="step").pack(side="left")
        self.soft_step_entry = ttk.Entry(row2, width=8)
        self.soft_step_entry.insert(0, "0.1")
        self.soft_step_entry.pack(side="left", padx=4)
        ttk.Button(
            row2, text="評価をキューへ", command=self.enqueue_evaluation
        ).pack(side="left", padx=8)

        columns = (
            "id",
            "created_at",
            "terrain",
            "status",
            "paper",
            "reward",
            "distance",
            "note",
        )
        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=12)
        headings = {
            "id": "Run ID",
            "created_at": "日時",
            "terrain": "地盤",
            "status": "状態",
            "paper": "paper_log",
            "reward": "最終Reward",
            "distance": "歩行距離",
            "note": "メモ",
        }
        widths = {
            "id": 160,
            "created_at": 130,
            "terrain": 56,
            "status": 80,
            "paper": 64,
            "reward": 80,
            "distance": 72,
            "note": 160,
        }
        for col in columns:
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, width=widths[col], anchor="w")
        self.tree.pack(fill="both", expand=True, padx=8, pady=4)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        note_frame = ttk.Frame(self)
        note_frame.pack(fill="x", padx=8, pady=4)
        ttk.Label(note_frame, text="メモ").pack(side="left")
        self.note_entry = ttk.Entry(note_frame)
        self.note_entry.pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(note_frame, text="メモ保存", command=self.save_note).pack(
            side="left", padx=2
        )

        self.detail = ttk.Label(self, text="", wraplength=700, justify="left")
        self.detail.pack(anchor="w", padx=8, pady=2)
        self.eval_status = ttk.Label(self, text="eval: idle", wraplength=700)
        self.eval_status.pack(anchor="w", padx=8, pady=2)

    def set_eval_status(self, text: str):
        self.eval_status.config(text=text)

    def refresh(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        rows = self.db.list_experiments()
        ids = [row["id"] for row in rows]
        paper_by_id = {
            link.experiment_id: link.status
            for link in list_paper_links_for_experiments(self.db, ids)
        }
        paper_label = {"ready": "あり", "partial": "partial", "no_paper": "—"}
        for row in rows:
            reward = row.get("final_reward_mean")
            distance = row.get("walk_distance")
            pl = paper_label.get(paper_by_id.get(row["id"], "no_paper"), "—")
            self.tree.insert(
                "",
                "end",
                iid=row["id"],
                values=(
                    row["id"],
                    row.get("created_at", ""),
                    row.get("terrain_mode", ""),
                    row.get("status", ""),
                    pl,
                    f"{reward:.3f}" if reward is not None else "",
                    f"{distance:.3f}" if distance is not None else "",
                    row.get("note") or "",
                ),
            )

    def _selected_ids(self) -> list[str]:
        return list(self.tree.selection())

    def _on_select(self, _event=None):
        ids = self._selected_ids()
        if not ids:
            return
        experiment = self.db.get_experiment(ids[0])
        if not experiment:
            return
        note = experiment.get("note") or ""
        self.note_entry.delete(0, "end")
        self.note_entry.insert(0, note)
        from ..paper_eval_link import resolve_paper_dir

        paper_dir, source, _ = resolve_paper_dir(ids[0], self.db)
        extra = ""
        if paper_dir is not None:
            extra = f"\npaper_log ({source}): {paper_dir.name}"
        self.detail.config(text=f"選択中: {', '.join(ids)}{extra}")

    def save_note(self):
        ids = self._selected_ids()
        if not ids:
            messagebox.showinfo("メモ", "実験を選択してください")
            return
        if len(ids) > 1:
            messagebox.showinfo("メモ", "メモ編集は1件ずつ行ってください")
            return
        run_id = ids[0]
        note = self.note_entry.get().strip()
        self.db.update_experiment_note(run_id, note)
        self.refresh()
        if self.tree.exists(run_id):
            self.tree.selection_set(run_id)
            self.tree.focus(run_id)
        self.detail.config(text=f"メモを保存しました: {run_id}")

    def enqueue_evaluation(self):
        ids = self._selected_ids()
        if not ids:
            messagebox.showinfo("評価", "評価する実験を選択してください")
            return

        try:
            seeds = parse_seeds_text(self.eval_seed_entry.get())
            soft_min = float(self.soft_min_entry.get())
            soft_max = float(self.soft_max_entry.get())
            soft_step = float(self.soft_step_entry.get())
        except ValueError as exc:
            messagebox.showerror("評価", f"パラメータが不正です: {exc}")
            return
        if soft_step <= 0:
            messagebox.showerror("評価", "softness step は正の値にしてください")
            return
        if soft_max < soft_min:
            messagebox.showerror("評価", "softness max は min 以上にしてください")
            return

        script = self.script_var.get()
        intervention_mode = self.interv_var.get()
        queued = []
        skipped = []
        for run_id in ids:
            experiment = self.db.get_experiment(run_id)
            if not experiment:
                skipped.append(f"{run_id} (DBなし)")
                continue
            run_dir = Path(experiment["run_dir"])
            checkpoint_dir = run_dir / "checkpoint"
            if experiment.get("status") not in ("completed", "early_stopped"):
                skipped.append(f"{run_id} (status={experiment.get('status')})")
                continue
            if not checkpoint_dir.exists():
                skipped.append(f"{run_id} (checkpointなし)")
                continue
            item = {
                "run_id": run_id,
                "checkpoint_dir": str(checkpoint_dir),
                "seeds": seeds,
                "softness_min": soft_min,
                "softness_max": soft_max,
                "softness_step": soft_step,
                "script": script,
                "intervention_mode": intervention_mode,
                "label": (
                    f"eval {run_id} [{script}] interv={intervention_mode} "
                    f"s={soft_min:.2f}-{soft_max:.2f}/{soft_step:.2f}"
                ),
            }
            self.on_enqueue_eval(item)
            queued.append(run_id)

        if queued:
            self.detail.config(text=f"評価キューへ追加: {', '.join(queued)}")
        if skipped:
            messagebox.showwarning(
                "評価",
                "スキップしました:\n" + "\n".join(skipped),
            )
        if not queued and not skipped:
            messagebox.showinfo("評価", "追加できる実験がありません")

    @staticmethod
    def _is_safe_run_dir(run_dir: Path) -> bool:
        try:
            resolved = run_dir.resolve()
            root = DEFAULT_RUNS_ROOT.resolve()
            return resolved != root and root in resolved.parents
        except OSError:
            return False

    def delete_selected(self):
        ids = self._selected_ids()
        if not ids:
            messagebox.showinfo("削除", "削除する実験を選択してください")
            return

        experiments = []
        for run_id in ids:
            exp = self.db.get_experiment(run_id)
            if not exp:
                continue
            if exp.get("status") == "running":
                messagebox.showwarning(
                    "削除",
                    f"学習中の実験は削除できません: {run_id}",
                )
                return
            experiments.append(exp)

        if not experiments:
            messagebox.showinfo("削除", "削除対象が見つかりません")
            return

        labels = "\n".join(exp["id"] for exp in experiments)
        ok = messagebox.askyesno(
            "削除",
            f"以下の実験を DB 履歴と run フォルダから削除しますか？\n\n{labels}",
        )
        if not ok:
            return

        deleted = []
        for exp in experiments:
            run_id = exp["id"]
            run_dir = Path(exp["run_dir"]) if exp.get("run_dir") else None
            self.db.delete_experiment(run_id)
            if run_dir and self._is_safe_run_dir(run_dir) and run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)
            deleted.append(run_id)

        self.refresh()
        self.note_entry.delete(0, "end")
        self.detail.config(text=f"削除しました: {', '.join(deleted)}")

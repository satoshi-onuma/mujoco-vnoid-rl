"""履歴タブ: Treeview一覧 / 評価実行 / 動画再生。"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from tkinter import ttk, messagebox

from ..database import ExperimentDB
from ..eval_launcher import EvalLauncher


class HistoryTab(ttk.Frame):
    def __init__(
        self,
        master,
        db: ExperimentDB,
        eval_launcher: EvalLauncher,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.db = db
        self.eval_launcher = eval_launcher
        self._build()
        self.refresh()

    def _build(self):
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", padx=8, pady=4)
        ttk.Button(toolbar, text="更新", command=self.refresh).pack(side="left", padx=2)
        self.eval_button = ttk.Button(toolbar, text="評価を実行", command=self.run_evaluation)
        self.eval_button.pack(side="left", padx=2)
        ttk.Button(toolbar, text="動画を再生", command=self.play_video).pack(side="left", padx=2)

        columns = ("id", "created_at", "terrain", "status", "reward", "distance", "note")
        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=12)
        headings = {
            "id": "Run ID",
            "created_at": "日時",
            "terrain": "地盤",
            "status": "状態",
            "reward": "最終Reward",
            "distance": "歩行距離",
            "note": "メモ",
        }
        widths = {"id": 180, "created_at": 140, "terrain": 60, "status": 90,
                  "reward": 90, "distance": 80, "note": 200}
        for col in columns:
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, width=widths[col], anchor="w")
        self.tree.pack(fill="both", expand=True, padx=8, pady=4)
        self.tree.bind("<Double-1>", lambda e: self.play_video())
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        note_frame = ttk.Frame(self)
        note_frame.pack(fill="x", padx=8, pady=4)
        ttk.Label(note_frame, text="メモ").pack(side="left")
        self.note_entry = ttk.Entry(note_frame)
        self.note_entry.pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(note_frame, text="メモ保存", command=self.save_note).pack(side="left", padx=2)

        self.detail = ttk.Label(self, text="", wraplength=700, justify="left")
        self.detail.pack(anchor="w", padx=8, pady=4)

    def refresh(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        rows = self.db.list_experiments()
        for row in rows:
            reward = row.get("final_reward_mean")
            distance = row.get("walk_distance")
            self.tree.insert(
                "",
                "end",
                iid=row["id"],
                values=(
                    row["id"],
                    row.get("created_at", ""),
                    row.get("terrain_mode", ""),
                    row.get("status", ""),
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
        self.detail.config(text=f"選択中: {ids[0]}")

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
        # 選択が外れるので戻す
        if self.tree.exists(run_id):
            self.tree.selection_set(run_id)
            self.tree.focus(run_id)
        self.detail.config(text=f"メモを保存しました: {run_id}")

    def run_evaluation(self):
        ids = self._selected_ids()
        if not ids:
            messagebox.showinfo("評価", "評価する実験を選択してください")
            return
        if len(ids) > 1:
            messagebox.showinfo("評価", "評価は1件ずつ実行してください")
            return

        run_id = ids[0]
        experiment = self.db.get_experiment(run_id)
        if not experiment:
            messagebox.showerror("評価", f"実験情報が見つかりません: {run_id}")
            return

        run_dir = Path(experiment["run_dir"])
        checkpoint_dir = run_dir / "checkpoint"
        # Stopで途中終了した実験でも、途中保存されたcheckpointがあれば評価を許可する
        if experiment.get("status") not in ("completed", "early_stopped") or not checkpoint_dir.exists():
            messagebox.showwarning("評価", "評価可能なチェックポイントがありません")
            return

        self.eval_button.config(state="disabled")
        self.detail.config(text=f"評価中: {run_id}")

        # 録画・評価は時間がかかるため、tkinterのメインループ外で実行する
        threading.Thread(
            target=self._evaluate_in_background,
            args=(run_id, run_dir),
            daemon=True,
        ).start()

    def _evaluate_in_background(self, run_id: str, run_dir: Path):
        try:
            results = self.eval_launcher.evaluate_run(run_id, run_dir)
            self.after(0, self._evaluation_finished, run_id, results, None)
        except Exception as exc:
            self.after(0, self._evaluation_finished, run_id, None, str(exc))

    def _evaluation_finished(self, run_id: str, results, error):
        self.eval_button.config(state="normal")
        if error:
            self.detail.config(text=f"評価失敗: {run_id}")
            messagebox.showerror("評価", error)
            return

        self.refresh()
        self.detail.config(text=f"評価完了: {run_id} ({len(results)} 地盤)")

    def play_video(self):
        ids = self._selected_ids()
        if not ids:
            messagebox.showinfo("動画", "実験を選択してください")
            return
        run_id = ids[0]
        evals = self.db.list_evaluations(run_id)
        video_path = None
        if evals:
            for ev in evals:
                if ev.get("video_path") and Path(ev["video_path"]).exists():
                    video_path = ev["video_path"]
                    break
        if video_path is None:
            exp = self.db.get_experiment(run_id)
            if exp and exp.get("run_dir"):
                run_dir = Path(exp["run_dir"])
                for candidate in sorted(run_dir.glob("*_demo.mp4")):
                    video_path = str(candidate)
                    break

        if not video_path or not Path(video_path).exists():
            messagebox.showwarning("動画", f"動画ファイルが見つかりません: {run_id}")
            return

        try:
            subprocess.Popen(["xdg-open", video_path])
            self.detail.config(text=f"再生: {video_path}")
        except Exception as e:
            messagebox.showerror("動画", f"再生に失敗しました: {e}")

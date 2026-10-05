"""tkinter製 強化学習実験管理アプリのエントリポイント。"""
# train/eval 別キュー。eval は学習中 cap=1、idle 時 cap=K（既定8）。

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_experiment_app.database import ExperimentDB, ensure_runs_root
from rl_experiment_app.training_launcher import TrainingLauncher
from rl_experiment_app.eval_pool import EvalPool
from rl_experiment_app.softness_eval_launcher import SoftnessEvalLauncher
from rl_experiment_app.ui.run_tab import RunTab
from rl_experiment_app.ui.progress_tab import ProgressTab
from rl_experiment_app.ui.history_tab import HistoryTab
from rl_experiment_app.ui.plot_tab import PlotTab


def _train_label(params: dict) -> str:
    note = (params.get("note") or "").strip()
    note_part = f" note={note}" if note else ""
    return (
        f"{params.get('terrain', '?')} "
        f"seed={params.get('seed', '?')} "
        f"iter={params.get('num_iterations', '?')}"
        f"{note_part}"
    )


def _eval_label(params: dict) -> str:
    return str(
        params.get("label")
        or (
            f"eval {params.get('run_id', '?')} "
            f"[{params.get('script', 'grid')}] "
            f"interv={params.get('intervention_mode', 'full')}"
        )
    )


def _move_item(queue: list, index: int, direction: int) -> int | None:
    new_index = index + direction
    if index < 0 or index >= len(queue):
        return None
    if new_index < 0 or new_index >= len(queue):
        return None
    queue[index], queue[new_index] = queue[new_index], queue[index]
    return new_index


class ExperimentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vnoid RL Experiment Manager")
        self.geometry("1000x820")

        ensure_runs_root()
        self.db = ExperimentDB()
        self.launcher = TrainingLauncher(self.db)
        self.eval_pool = EvalPool()

        self._poll_job = None
        self._train_queue: list[dict] = []
        self._eval_queue: list[dict] = []
        self._train_running = False

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self.run_tab = RunTab(
            notebook,
            on_enqueue_train=self.enqueue_train,
            on_stop_train=self.stop_train,
            on_stop_eval=self.stop_eval,
            on_train_queue_remove=self.train_queue_remove,
            on_train_queue_clear=self.train_queue_clear,
            on_train_queue_move=self.train_queue_move,
            on_eval_queue_remove=self.eval_queue_remove,
            on_eval_queue_clear=self.eval_queue_clear,
            on_eval_queue_move=self.eval_queue_move,
        )
        self.progress_tab = ProgressTab(notebook)
        self.history_tab = HistoryTab(
            notebook, self.db, on_enqueue_eval=self.enqueue_eval
        )
        self.plot_tab = PlotTab(notebook, self.db)

        notebook.add(self.run_tab, text="実行")
        notebook.add(self.progress_tab, text="進捗")
        notebook.add(self.history_tab, text="履歴")
        notebook.add(self.plot_tab, text="Plot")

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _eval_max_parallel(self) -> int:
        if self._train_running:
            return 1
        return self.run_tab.get_eval_idle_parallel()

    # --- train queue ---

    def enqueue_train(self, params: dict):
        item = dict(params)
        item["label"] = _train_label(item)
        self._train_queue.append(item)
        self._refresh_queue_ui()
        if not self._train_running:
            self._start_next_train()
        self._update_status()
        self._ensure_poll()

    def train_queue_remove(self, index: int):
        if 0 <= index < len(self._train_queue):
            self._train_queue.pop(index)
            self._refresh_queue_ui()
            self._update_status()

    def train_queue_clear(self):
        self._train_queue.clear()
        self._refresh_queue_ui()
        self._update_status()

    def train_queue_move(self, index: int, direction: int):
        sel = _move_item(self._train_queue, index, direction)
        if sel is not None:
            self._refresh_queue_ui(train_select=sel)
            self._update_status()

    def _start_next_train(self):
        if not self._train_queue:
            self._train_running = False
            self.run_tab.set_train_running(False)
            self._try_start_evals()
            self._update_status()
            return

        params = self._train_queue.pop(0)
        self._refresh_queue_ui()
        try:
            run_id, _run_dir, csv_path = self.launcher.start(params)
        except Exception as e:
            messagebox.showerror("学習起動失敗", str(e))
            self._train_running = False
            self.run_tab.set_train_running(False)
            self._refresh_queue_ui()
            self._update_status()
            if self._train_queue:
                self._start_next_train()
            return

        self._train_running = True
        self.run_tab.set_train_running(True)
        self.progress_tab.set_csv_paths([(run_id, csv_path)])
        self._update_status()
        self._ensure_poll()

    def stop_train(self):
        self.launcher.stop()
        self._train_running = False
        self.run_tab.set_train_running(False)
        self.history_tab.refresh()
        self._try_start_evals()
        self._update_status()
        if self._train_queue:
            self._start_next_train()
        elif not self.eval_pool.any_running() and not self._eval_queue:
            self._cancel_poll()

    # --- eval queue ---

    def enqueue_eval(self, params: dict):
        item = dict(params)
        item["label"] = _eval_label(item)
        self._eval_queue.append(item)
        self._refresh_queue_ui()
        self._try_start_evals()
        self._update_status()
        self._ensure_poll()

    def eval_queue_remove(self, index: int):
        if 0 <= index < len(self._eval_queue):
            self._eval_queue.pop(index)
            self._refresh_queue_ui()
            self._update_status()

    def eval_queue_clear(self):
        self._eval_queue.clear()
        self._refresh_queue_ui()
        self._update_status()

    def eval_queue_move(self, index: int, direction: int):
        sel = _move_item(self._eval_queue, index, direction)
        if sel is not None:
            self._refresh_queue_ui(eval_select=sel)
            self._update_status()

    def _try_start_evals(self) -> None:
        cap = self._eval_max_parallel()
        started = True
        while self.eval_pool.count() < cap and self._eval_queue and started:
            started = False
            for index, params in enumerate(self._eval_queue):
                if self.eval_pool.is_output_dir_busy(params):
                    continue
                if self.eval_pool.count() >= cap:
                    break
                self._eval_queue.pop(index)
                self._refresh_queue_ui()
                try:
                    self.eval_pool.start_one(params)
                except Exception as e:
                    messagebox.showerror("評価起動失敗", str(e))
                    continue
                started = True
                break

        self.run_tab.set_eval_running(self.eval_pool.any_running())

    def stop_eval(self):
        stopped = self.eval_pool.stop_all()
        for launcher in stopped:
            self._record_softness_eval_run(launcher, 130)
        self._update_status()
        self._try_start_evals()
        if (
            not self._train_running
            and not self._train_queue
            and not self.eval_pool.any_running()
            and not self._eval_queue
        ):
            self._cancel_poll()

    # --- shared UI / poll ---

    def _refresh_queue_ui(
        self,
        train_select: int | None = None,
        eval_select: int | None = None,
    ):
        self.run_tab.set_train_queue_items(
            [item.get("label", "?") for item in self._train_queue],
            select=train_select,
        )
        self.run_tab.set_eval_queue_items(
            [item.get("label", "?") for item in self._eval_queue],
            select=eval_select,
        )

    def _update_status(self):
        tn = len(self._train_queue)
        en = len(self._eval_queue)
        cap = self._eval_max_parallel()
        n_run = self.eval_pool.count()

        if self._train_running:
            train_part = f"train: 実行中 {self.launcher.run_id or '?'} ／ 待ち {tn}"
        elif tn:
            train_part = f"train: 待ち {tn}"
        else:
            train_part = "train: idle"

        if n_run:
            labels = [
                (l.label or "?")[:40]
                for l in self.eval_pool.running_launchers()
            ]
            if len(labels) == 1:
                short = labels[0]
            else:
                short = f"{labels[0]} 他{len(labels) - 1}本"
            eval_part = f"eval: {n_run}/{cap} 実行中 ({short}) ／ 待ち {en}"
        elif en:
            eval_part = f"eval: 待ち {en} (cap={cap})"
        else:
            eval_part = "eval: idle"

        text = f"{train_part} | {eval_part}"
        self.run_tab.set_status(text)
        self.history_tab.set_eval_status(eval_part)

    def _record_softness_eval_run(
        self, launcher: SoftnessEvalLauncher, exit_code: int
    ) -> None:
        params = launcher.last_params
        output_dir = launcher.output_dir
        if not params or output_dir is None:
            return
        status = "completed" if exit_code == 0 else "failed"
        self.db.upsert_softness_eval_run(
            experiment_id=str(params["run_id"]),
            script=str(params["script"]),
            output_dir=str(output_dir),
            seeds=str(params["seeds"]),
            softness_min=float(params["softness_min"]),
            softness_max=float(params["softness_max"]),
            softness_step=float(params["softness_step"]),
            intervention_mode=str(params["intervention_mode"]),
            status=status,
        )

    def _ensure_poll(self):
        if self._poll_job is None and (
            self._train_running or self.eval_pool.any_running()
        ):
            self._schedule_poll()

    def _schedule_poll(self):
        self._cancel_poll()
        self._poll()

    def _cancel_poll(self):
        if self._poll_job is not None:
            self.after_cancel(self._poll_job)
            self._poll_job = None

    def _poll(self):
        self.progress_tab.refresh()

        train_code = self.launcher.poll()
        if train_code is not None:
            self.history_tab.refresh()
            self._train_running = False
            self.run_tab.set_train_running(False)
            if self._train_queue:
                self._start_next_train()
            else:
                self._try_start_evals()
                self._update_status()

        any_eval_done = False
        for launcher, eval_code in self.eval_pool.poll_all():
            any_eval_done = True
            self._record_softness_eval_run(launcher, eval_code)
            if eval_code != 0:
                messagebox.showwarning(
                    "評価",
                    f"評価が終了コード {eval_code} で終わりました。"
                    f"\nログ: {launcher.log_path}",
                )
            else:
                self.history_tab.refresh()

        if any_eval_done:
            self.run_tab.set_eval_running(self.eval_pool.any_running())
            self._try_start_evals()

        still = self._train_running or self.eval_pool.any_running()
        if still or self._train_queue or self._eval_queue:
            self._update_status()
            self._poll_job = self.after(1000, self._poll)
        else:
            self._poll_job = None
            self._update_status()

    def _on_close(self):
        train_run = self.launcher.is_running()
        eval_run = self.eval_pool.any_running()
        queued = bool(self._train_queue or self._eval_queue)
        if train_run or eval_run or queued:
            parts = []
            if train_run:
                parts.append("学習が実行中です")
            if eval_run:
                parts.append("評価が実行中です")
            if self._train_queue:
                parts.append(f"学習待ちが {len(self._train_queue)} 件あります")
            if self._eval_queue:
                parts.append(f"評価待ちが {len(self._eval_queue)} 件あります")
            msg = "。".join(parts) + "。終了しますか？"
            if not messagebox.askyesno("確認", msg):
                return
            self._cancel_poll()
            if train_run:
                self.launcher.stop()
            if eval_run:
                self.stop_eval()
        else:
            self._cancel_poll()
        self.destroy()


def main():
    app = ExperimentApp()
    app.mainloop()


if __name__ == "__main__":
    main()

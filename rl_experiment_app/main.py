"""tkinter製 強化学習実験管理アプリのエントリポイント。"""
# start_single() → _schedule_poll() → _poll()
#                                       ↓
#                          進捗グラフ更新 + プロセス終了チェック
#                                       ↓
#                          まだ動いてる → 1秒後にもう一度 _poll()
#                          終わった     → 停止してポーリング終了

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

# リポジトリルートを path に追加
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_experiment_app.database import ExperimentDB, ensure_runs_root
from rl_experiment_app.training_launcher import TrainingLauncher
from rl_experiment_app.eval_launcher import EvalLauncher
from rl_experiment_app.ui.run_tab import RunTab
from rl_experiment_app.ui.progress_tab import ProgressTab
from rl_experiment_app.ui.history_tab import HistoryTab


class ExperimentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vnoid RL Experiment Manager")
        self.geometry("900x700")

        ensure_runs_root()
        self.db = ExperimentDB()
        self.launcher = TrainingLauncher(self.db)
        self.eval_launcher = EvalLauncher(self.db)

        self._poll_job = None

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self.run_tab = RunTab(
            notebook,
            on_run=self.start_single,
            on_stop=self.stop_all,
        )
        self.progress_tab = ProgressTab(notebook)
        self.history_tab = HistoryTab(notebook, self.db, self.eval_launcher)

        notebook.add(self.run_tab, text="実行")
        notebook.add(self.progress_tab, text="進捗")
        notebook.add(self.history_tab, text="履歴")

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def start_single(self, params: dict):
        try:
            run_id, run_dir, csv_path = self.launcher.start(params)
        except Exception as e:
            messagebox.showerror("起動失敗", str(e))
            self.run_tab.set_running(False)
            return
        self.run_tab.set_status(f"実行中: {run_id}")
        self.progress_tab.set_csv_paths([(run_id, csv_path)])
        self._schedule_poll()

    def stop_all(self):
        self.launcher.stop()
        self._cancel_poll()
        self.run_tab.set_running(False)
        self.run_tab.set_status("停止しました")
        self.history_tab.refresh()

    def _schedule_poll(self):
        self._cancel_poll()
        self._poll()

    def _cancel_poll(self):
        if self._poll_job is not None:
            self.after_cancel(self._poll_job)
            self._poll_job = None

    def _poll(self):
        self.progress_tab.refresh()
        code = self.launcher.poll()
        done = code is not None

        if done:
            self.run_tab.set_running(False)
            self.run_tab.set_status("完了")
            self.history_tab.refresh()
            self._poll_job = None
            return

        self._poll_job = self.after(1000, self._poll)

    def _on_close(self):
        self._cancel_poll()
        if self.launcher.process and self.launcher.process.poll() is None:
            if messagebox.askyesno("確認", "学習が実行中です。停止して終了しますか？"):
                self.stop_all()
            else:
                return
        self.destroy()


def main():
    app = ExperimentApp()
    app.mainloop()


if __name__ == "__main__":
    main()

"""進捗タブ: matplotlib埋め込みでCSVを定期ポーリング。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class ProgressTab(ttk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.csv_paths: list[tuple[str, Path]] = []  # (label, path)
        self._build()

    def _build(self):
        self.info_label = ttk.Label(self, text="実行中の学習はありません")
        self.info_label.pack(anchor="w", padx=8, pady=4)

        self.fig = Figure(figsize=(7, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Mean Reward")
        self.ax.set_title("Training Progress")
        self.ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=4)

    def set_csv_paths(self, paths: list[tuple[str, Path]]):
        self.csv_paths = paths
        if paths:
            labels = ", ".join(label for label, _ in paths)
            self.info_label.config(text=f"監視中: {labels}")
        else:
            self.info_label.config(text="実行中の学習はありません")
        self.refresh()

    def clear(self):
        self.csv_paths = []
        self.info_label.config(text="実行中の学習はありません")
        self.ax.clear()
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Mean Reward")
        self.ax.set_title("Training Progress")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def refresh(self):
        self.ax.clear()
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Mean Reward")
        self.ax.set_title("Training Progress")
        self.ax.grid(True, alpha=0.3)

        legend_items = []
        for label, path in self.csv_paths:
            data = self._load_csv(path)
            if data is None:
                continue
            iters, rewards = data
            line, = self.ax.plot(iters, rewards, linewidth=1.5, label=label)
            last = rewards[-1] if rewards else float("nan")
            legend_items.append(f"{label}: {last:.3f}")

        if legend_items:
            self.ax.legend(loc="best", fontsize=8)
            self.info_label.config(text=" | ".join(legend_items))

        self.canvas.draw_idle()

    @staticmethod
    def _load_csv(path: Path) -> Optional[tuple[list[int], list[float]]]:
        path = Path(path)
        if not path.exists():
            return None
        try:
            iters, rewards = [], []
            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    iters.append(int(row["iteration"]))
                    rewards.append(float(row["reward_mean"]))
            if not iters:
                return None
            return iters, rewards
        except Exception:
            return None

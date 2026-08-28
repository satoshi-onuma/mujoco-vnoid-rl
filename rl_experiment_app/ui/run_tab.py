"""実行タブ: 単発学習のパラメータ入力 / Run・Stop。"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, Optional


class RunTab(ttk.Frame):
    def __init__(
        self,
        master,
        on_run: Callable[[dict], None],
        on_stop: Callable[[], None],
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.on_run = on_run
        self.on_stop = on_stop
        self._build()

    def _build(self):
        # 報酬重み
        reward_frame = ttk.LabelFrame(self, text="報酬重み")
        reward_frame.pack(fill="x", padx=8, pady=4)
        self.entries = {}

        reward_params = [
            ("w_track", "1.0"),
            ("w_act", "0.1"),
            ("w_healthy", "1.0"),
            ("tracking_sigma", "0.02"),
        ]
        for i, (name, default) in enumerate(reward_params):
            ttk.Label(reward_frame, text=name).grid(row=i, column=0, sticky="w", padx=4, pady=2)
            ent = ttk.Entry(reward_frame, width=10)
            ent.insert(0, default)
            ent.grid(row=i, column=1, padx=4, pady=2)
            self.entries[name] = ent

        # 地盤・HP
        hp_frame = ttk.LabelFrame(self, text="地盤・ハイパーパラメータ")
        hp_frame.pack(fill="x", padx=8, pady=4)

        # 開始時は常に硬地盤。ここで選ぶのは歩行途中で切り替わる先の地盤
        ttk.Label(hp_frame, text="切替先地盤 (random=softness[0,1.1])").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.terrain_var = tk.StringVar(value="soft")
        ttk.Combobox(
            hp_frame, textvariable=self.terrain_var,
            values=["hard", "soft", "debug", "random"], width=10, state="readonly",
        ).grid(row=0, column=1, padx=4, pady=2)

        hp_params = [
            ("lr", "0.0001"),
            ("gamma", "0.99"),
            ("num_workers", "8"),
            ("num_gpus", "1"),
            ("num_iterations", "100"),
            ("seed", "42"),
        ]
        for i, (name, default) in enumerate(hp_params):
            row = i + 1
            ttk.Label(hp_frame, text=name).grid(row=row, column=0, sticky="w", padx=4, pady=2)
            ent = ttk.Entry(hp_frame, width=12)
            ent.insert(0, default)
            ent.grid(row=row, column=1, padx=4, pady=2)
            self.entries[name] = ent

        # ひとことメモ（任意）
        note_frame = ttk.LabelFrame(self, text="メモ（任意）")
        note_frame.pack(fill="x", padx=8, pady=4)
        self.note_entry = ttk.Entry(note_frame)
        self.note_entry.pack(fill="x", padx=4, pady=4)

        # ボタン
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=8, pady=8)
        self.run_btn = ttk.Button(btn_frame, text="Run", command=self._on_run)
        self.run_btn.pack(side="left", padx=4)
        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self._on_stop, state="disabled")
        self.stop_btn.pack(side="left", padx=4)

        self.status_label = ttk.Label(self, text="待機中")
        self.status_label.pack(anchor="w", padx=8, pady=4)

    def _collect_params(self) -> Optional[dict]:
        try:
            params = {
                "w_track": float(self.entries["w_track"].get()),
                "w_act": float(self.entries["w_act"].get()),
                "w_healthy": float(self.entries["w_healthy"].get()),
                "tracking_sigma": float(self.entries["tracking_sigma"].get()),
                "terrain": self.terrain_var.get(),
                "lr": float(self.entries["lr"].get()),
                "gamma": float(self.entries["gamma"].get()),
                "num_workers": int(self.entries["num_workers"].get()),
                "num_gpus": int(self.entries["num_gpus"].get()),
                "num_iterations": int(self.entries["num_iterations"].get()),
                "seed": int(self.entries["seed"].get()),
                "note": self.note_entry.get().strip(),
            }
            return params
        except ValueError as e:
            messagebox.showerror("入力エラー", f"数値の変換に失敗しました: {e}")
            return None

    def _on_run(self):
        params = self._collect_params()
        if params is None:
            return
        self.on_run(params)
        self.set_running(True)

    def _on_stop(self):
        self.on_stop()
        self.set_running(False)
        self.status_label.config(text="停止要求済み")

    def set_running(self, running: bool):
        self.run_btn.config(state="disabled" if running else "normal")
        self.stop_btn.config(state="normal" if running else "disabled")

    def set_status(self, text: str):
        self.status_label.config(text=text)

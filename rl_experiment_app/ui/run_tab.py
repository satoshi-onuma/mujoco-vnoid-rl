"""実行タブ: 学習パラメータ / train・eval 待ちキュー / 各Stop。"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, Optional


class RunTab(ttk.Frame):
    def __init__(
        self,
        master,
        on_enqueue_train: Callable[[dict], None],
        on_stop_train: Callable[[], None],
        on_stop_eval: Callable[[], None],
        on_train_queue_remove: Callable[[int], None],
        on_train_queue_clear: Callable[[], None],
        on_train_queue_move: Callable[[int, int], None],
        on_eval_queue_remove: Callable[[int], None],
        on_eval_queue_clear: Callable[[], None],
        on_eval_queue_move: Callable[[int, int], None],
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.on_enqueue_train = on_enqueue_train
        self.on_stop_train = on_stop_train
        self.on_stop_eval = on_stop_eval
        self.on_train_queue_remove = on_train_queue_remove
        self.on_train_queue_clear = on_train_queue_clear
        self.on_train_queue_move = on_train_queue_move
        self.on_eval_queue_remove = on_eval_queue_remove
        self.on_eval_queue_clear = on_eval_queue_clear
        self.on_eval_queue_move = on_eval_queue_move
        self._build()

    def _build(self):
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
            ttk.Label(reward_frame, text=name).grid(
                row=i, column=0, sticky="w", padx=4, pady=2
            )
            ent = ttk.Entry(reward_frame, width=10)
            ent.insert(0, default)
            ent.grid(row=i, column=1, padx=4, pady=2)
            self.entries[name] = ent

        hp_frame = ttk.LabelFrame(self, text="地盤・ハイパーパラメータ")
        hp_frame.pack(fill="x", padx=8, pady=4)

        ttk.Label(hp_frame, text="切替先地盤 (random=softness[0,1.1])").grid(
            row=0, column=0, sticky="w", padx=4, pady=2
        )
        self.terrain_var = tk.StringVar(value="soft")
        ttk.Combobox(
            hp_frame,
            textvariable=self.terrain_var,
            values=["hard", "soft", "debug", "random"],
            width=10,
            state="readonly",
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
            ttk.Label(hp_frame, text=name).grid(
                row=row, column=0, sticky="w", padx=4, pady=2
            )
            ent = ttk.Entry(hp_frame, width=12)
            ent.insert(0, default)
            ent.grid(row=row, column=1, padx=4, pady=2)
            self.entries[name] = ent

        note_frame = ttk.LabelFrame(self, text="メモ（任意）")
        note_frame.pack(fill="x", padx=8, pady=4)
        self.note_entry = ttk.Entry(note_frame)
        self.note_entry.pack(fill="x", padx=4, pady=4)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=8, pady=8)
        ttk.Button(btn_frame, text="学習キューに追加", command=self._on_enqueue).pack(
            side="left", padx=4
        )
        self.train_stop_btn = ttk.Button(
            btn_frame, text="Train Stop", command=self._on_stop_train, state="disabled"
        )
        self.train_stop_btn.pack(side="left", padx=4)
        self.eval_stop_btn = ttk.Button(
            btn_frame, text="Eval Stop", command=self._on_stop_eval, state="disabled"
        )
        self.eval_stop_btn.pack(side="left", padx=4)

        eval_cap_frame = ttk.Frame(self)
        eval_cap_frame.pack(fill="x", padx=8, pady=2)
        ttk.Label(
            eval_cap_frame,
            text="学習 idle 時の eval 同時数（学習中は1）",
        ).pack(side="left")
        self.eval_idle_parallel = tk.IntVar(value=8)
        ttk.Spinbox(
            eval_cap_frame,
            from_=1,
            to=8,
            width=4,
            textvariable=self.eval_idle_parallel,
        ).pack(side="left", padx=4)

        self.status_label = ttk.Label(self, text="train: idle | eval: idle")
        self.status_label.pack(anchor="w", padx=8, pady=4)

        queues = ttk.Frame(self)
        queues.pack(fill="both", expand=True, padx=8, pady=4)

        train_q = ttk.LabelFrame(queues, text="学習待ちキュー（直列）")
        train_q.pack(side="left", fill="both", expand=True, padx=(0, 4))
        self.train_queue_list = self._make_queue_list(
            train_q,
            on_remove=self._on_train_remove,
            on_clear=self._on_train_clear,
            on_move=self._on_train_move,
        )

        eval_q = ttk.LabelFrame(queues, text="評価待ちキュー（直列）")
        eval_q.pack(side="left", fill="both", expand=True, padx=(4, 0))
        self.eval_queue_list = self._make_queue_list(
            eval_q,
            on_remove=self._on_eval_remove,
            on_clear=self._on_eval_clear,
            on_move=self._on_eval_move,
        )

    def _make_queue_list(self, parent, on_remove, on_clear, on_move) -> tk.Listbox:
        list_frame = ttk.Frame(parent)
        list_frame.pack(fill="both", expand=True, padx=4, pady=4)
        queue_list = tk.Listbox(list_frame, height=6, exportselection=False)
        queue_list.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(
            list_frame, orient="vertical", command=queue_list.yview
        )
        scrollbar.pack(side="right", fill="y")
        queue_list.config(yscrollcommand=scrollbar.set)

        qbtn = ttk.Frame(parent)
        qbtn.pack(fill="x", padx=4, pady=4)
        ttk.Button(qbtn, text="選択削除", command=on_remove).pack(side="left", padx=2)
        ttk.Button(qbtn, text="全クリア", command=on_clear).pack(side="left", padx=2)
        ttk.Button(qbtn, text="↑", width=3, command=lambda: on_move(-1)).pack(
            side="left", padx=2
        )
        ttk.Button(qbtn, text="↓", width=3, command=lambda: on_move(1)).pack(
            side="left", padx=2
        )
        return queue_list

    def _collect_params(self) -> Optional[dict]:
        try:
            return {
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
        except ValueError as e:
            messagebox.showerror("入力エラー", f"数値の変換に失敗しました: {e}")
            return None

    def _on_enqueue(self):
        params = self._collect_params()
        if params is None:
            return
        self.on_enqueue_train(params)

    def _on_stop_train(self):
        self.on_stop_train()

    def _on_stop_eval(self):
        self.on_stop_eval()

    def _on_train_remove(self):
        sel = self.train_queue_list.curselection()
        if sel:
            self.on_train_queue_remove(sel[0])

    def _on_train_clear(self):
        if self.train_queue_list.size() == 0:
            return
        if messagebox.askyesno("確認", "学習待ちキューをすべて削除しますか？"):
            self.on_train_queue_clear()

    def _on_train_move(self, direction: int):
        sel = self.train_queue_list.curselection()
        if sel:
            self.on_train_queue_move(sel[0], direction)

    def _on_eval_remove(self):
        sel = self.eval_queue_list.curselection()
        if sel:
            self.on_eval_queue_remove(sel[0])

    def _on_eval_clear(self):
        if self.eval_queue_list.size() == 0:
            return
        if messagebox.askyesno("確認", "評価待ちキューをすべて削除しますか？"):
            self.on_eval_queue_clear()

    def _on_eval_move(self, direction: int):
        sel = self.eval_queue_list.curselection()
        if sel:
            self.on_eval_queue_move(sel[0], direction)

    @staticmethod
    def _set_list(listbox: tk.Listbox, labels: list[str], select: int | None = None):
        listbox.delete(0, tk.END)
        for i, label in enumerate(labels):
            listbox.insert(tk.END, f"#{i + 1}  {label}")
        if select is not None and 0 <= select < len(labels):
            listbox.selection_set(select)
            listbox.activate(select)
            listbox.see(select)

    def set_train_queue_items(self, labels: list[str], select: int | None = None):
        self._set_list(self.train_queue_list, labels, select=select)

    def set_eval_queue_items(self, labels: list[str], select: int | None = None):
        self._set_list(self.eval_queue_list, labels, select=select)

    def set_train_running(self, running: bool):
        self.train_stop_btn.config(state="normal" if running else "disabled")

    def set_eval_running(self, running: bool):
        self.eval_stop_btn.config(state="normal" if running else "disabled")

    def set_status(self, text: str):
        self.status_label.config(text=text)

    def get_eval_idle_parallel(self) -> int:
        try:
            value = int(self.eval_idle_parallel.get())
        except (ValueError, tk.TclError):
            return 8
        return max(1, min(8, value))

"""Plotタブ: softness成功 / 学習曲線 / paper control / paper比較 / sink vs delta_x / Δx vs 成功。"""

from __future__ import annotations

import sys
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS = REPO_ROOT / "python_scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from plot_softness_success import (  # noqa: E402
    build_group_series as soft_build_series,
    draw_groups as soft_draw_groups,
    plot_groups as soft_plot_groups,
    write_aggregate_csv as soft_write_csv,
    write_seeds_csv as soft_write_seeds,
)
from plot_training_curves_seeds import (  # noqa: E402
    aggregate_group as train_aggregate,
    draw_groups as train_draw_groups,
    plot_groups as train_plot_groups,
    write_aggregate_csv as train_write_csv,
)
from plot_paper_control import (  # noqa: E402
    _safe_label_prefix,
    draw_compare_on_figure,
    infer_title,
    load_paper_control,
    plot_paper_trial,
    plot_paper_trials_compare,
    resolve_control_csv,
    switch_time,
    _mark_switch,
)
from plot_sink_vs_deltax import (  # noqa: E402
    build_group_points as sink_build_group,
    draw_groups as sink_draw_groups,
    plot_groups as sink_plot_groups,
    write_points_csv as sink_write_csv,
)
from plot_deltax_success import (  # noqa: E402
    CheckpointPoint,
    PointStatus,
    discover_points as deltax_discover,
    draw_scatter as deltax_draw_scatter,
    format_corr_lines as deltax_corr_lines,
    plot_scatter as deltax_plot_scatter,
    write_points_csv as deltax_write_csv,
)

from ..database import ExperimentDB
from .plot_labels import label_with_note, run_id_with_note

EVALS_ROOT = Path.home() / "vnoid-experiments" / "evals"
PAPER_ROOT = Path.home() / "vnoid-experiments" / "paper_logs"
PLOTS_ROOT = Path.home() / "vnoid-experiments" / "plots"


class PlotTab(ttk.Frame):
    def __init__(self, master, db: ExperimentDB, **kwargs):
        super().__init__(master, **kwargs)
        self.db = db
        # label -> list[Path]
        self._soft_groups: dict[str, list[Path]] = {}
        self._train_groups: dict[str, list[Path]] = {}
        self._sink_groups: dict[str, list[Path]] = {}
        self._deltax_points: list[CheckpointPoint] = []
        self._deltax_corrs = []
        self._paper_trial: Path | None = None
        # label -> trial dir（比較タブ）
        self._paper_compare: list[tuple[str, Path]] = []
        self._build()

    def _build(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self.soft_frame = ttk.Frame(self.notebook)
        self.train_frame = ttk.Frame(self.notebook)
        self.paper_frame = ttk.Frame(self.notebook)
        self.paper_cmp_frame = ttk.Frame(self.notebook)
        self.sink_frame = ttk.Frame(self.notebook)
        self.deltax_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.soft_frame, text="Softness成功")
        self.notebook.add(self.train_frame, text="学習曲線")
        self.notebook.add(self.paper_frame, text="Paper control")
        self.notebook.add(self.paper_cmp_frame, text="Paper比較")
        self.notebook.add(self.sink_frame, text="Sink vs Δx")
        self.notebook.add(self.deltax_frame, text="Δx vs 成功")

        self._build_softness()
        self._build_training()
        self._build_paper()
        self._build_paper_compare()
        self._build_sink()
        self._build_deltax()

    # ----- Softness success -----

    def _build_softness(self):
        top = ttk.Frame(self.soft_frame)
        top.pack(fill="x", padx=8, pady=4)
        ttk.Button(top, text="一覧更新", command=self._refresh_soft_list).pack(
            side="left", padx=2
        )
        ttk.Label(top, text="グループラベル").pack(side="left", padx=(8, 2))
        self.soft_label = ttk.Entry(top, width=16)
        self.soft_label.insert(0, "group")
        self.soft_label.pack(side="left", padx=2)
        ttk.Button(
            top, text="選択をグループへ", command=self._soft_add_group
        ).pack(side="left", padx=2)
        ttk.Button(top, text="選択グループ削除", command=self._soft_remove_groups).pack(
            side="left", padx=2
        )
        ttk.Label(top, text="スタイル").pack(side="left", padx=(8, 2))
        self.soft_style = ttk.Combobox(
            top,
            width=14,
            state="readonly",
            values=["mean±std", "シード別細線"],
        )
        self.soft_style.set("mean±std")
        self.soft_style.pack(side="left", padx=2)
        ttk.Button(top, text="描画", command=self._soft_draw).pack(side="left", padx=2)
        ttk.Button(top, text="PNG保存", command=self._soft_save).pack(side="left", padx=2)

        body = ttk.Frame(self.soft_frame)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=False)
        ttk.Label(left, text="evals + paper_logs (summary.csv)").pack(anchor="w")
        self.soft_list = tk.Listbox(
            left, width=48, height=12, selectmode=tk.EXTENDED, exportselection=False
        )
        self.soft_list.pack(fill="both", expand=True)
        self.soft_group_view = tk.Listbox(
            left, width=48, height=6, selectmode=tk.EXTENDED, exportselection=False
        )
        self.soft_group_view.pack(fill="x", pady=4)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=8)
        self.soft_fig = Figure(figsize=(6, 4), dpi=100)
        self.soft_ax = self.soft_fig.add_subplot(111)
        self.soft_canvas = FigureCanvasTkAgg(self.soft_fig, master=right)
        self.soft_canvas.get_tk_widget().pack(fill="both", expand=True)

        self._refresh_soft_list()

    @staticmethod
    def _soft_source_tag(d: Path) -> str:
        parent = d.parent.name
        if parent == PAPER_ROOT.name:
            return "paper"
        return "evals"

    def _refresh_soft_list(self):
        self.soft_list.delete(0, tk.END)
        self._soft_dirs: list[Path] = []
        for root, tag in ((EVALS_ROOT, "evals"), (PAPER_ROOT, "paper")):
            if not root.is_dir():
                continue
            for d in sorted(root.iterdir()):
                if d.name.startswith("_"):
                    continue
                if d.is_dir() and (d / "summary.csv").is_file():
                    self._soft_dirs.append(d)
                    self.soft_list.insert(
                        tk.END,
                        f"[{tag}] {label_with_note(self.db, d.name)}",
                    )

    def _soft_add_group(self):
        label = self.soft_label.get().strip()
        if not label:
            messagebox.showinfo("Plot", "グループラベルを入力してください")
            return
        sels = self.soft_list.curselection()
        if not sels:
            messagebox.showinfo("Plot", "ディレクトリを選択してください")
            return
        dirs = [self._soft_dirs[i] for i in sels]
        self._soft_groups.setdefault(label, [])
        for d in dirs:
            if d not in self._soft_groups[label]:
                self._soft_groups[label].append(d)
        self._refresh_soft_groups_view()

    def _soft_remove_groups(self):
        sels = self.soft_group_view.curselection()
        if not sels:
            messagebox.showinfo(
                "Plot", "削除するグループを下の一覧から選択してください"
            )
            return
        labels = list(self._soft_groups.keys())
        for i in sorted(sels, reverse=True):
            if 0 <= i < len(labels):
                self._soft_groups.pop(labels[i], None)
        self._refresh_soft_groups_view()

    def _refresh_soft_groups_view(self):
        self.soft_group_view.delete(0, tk.END)
        for label, dirs in self._soft_groups.items():
            names = ",".join(
                f"[{self._soft_source_tag(d)}]{d.name}" for d in dirs
            )
            self.soft_group_view.insert(tk.END, f"{label}: {names}")

    def _soft_style_key(self) -> str:
        return "per_seed" if self.soft_style.get() == "シード別細線" else "mean_std"

    def _soft_series(self):
        series = []
        for label, dirs in self._soft_groups.items():
            series.append(soft_build_series(label, dirs, verbose=False))
        return series

    def _soft_draw(self):
        if not self._soft_groups:
            messagebox.showinfo("Plot", "グループを追加してください")
            return
        try:
            series = self._soft_series()
        except (FileNotFoundError, ValueError) as e:
            messagebox.showerror("Plot", str(e))
            return
        self.soft_ax.clear()
        soft_draw_groups(self.soft_ax, series, style=self._soft_style_key())
        self.soft_fig.tight_layout()
        self.soft_canvas.draw_idle()

    def _soft_save(self):
        if not self._soft_groups:
            messagebox.showinfo("Plot", "グループを追加してください")
            return
        path = filedialog.asksaveasfilename(
            initialdir=str(EVALS_ROOT),
            initialfile="compare_softness_success.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
        )
        if not path:
            return
        try:
            series = self._soft_series()
            csv_rows = []
            for label, softs, means, stds, ns, _members in series:
                for soft, mean, std, n in zip(softs, means, stds, ns):
                    csv_rows.append((label, soft, mean, std, n))
            out = Path(path)
            soft_write_csv(out.with_name(out.stem + ".csv"), csv_rows)
            soft_write_seeds(out.with_name(out.stem + "_seeds.csv"), series)
            soft_plot_groups(series, out, style=self._soft_style_key())
            messagebox.showinfo(
                "Plot",
                f"保存しました:\n{out}\n(+ {out.stem}.csv, {out.stem}_seeds.csv)",
            )
        except Exception as e:
            messagebox.showerror("Plot", str(e))

    # ----- Training curves -----

    def _build_training(self):
        top = ttk.Frame(self.train_frame)
        top.pack(fill="x", padx=8, pady=4)
        ttk.Button(top, text="一覧更新", command=self._refresh_train_list).pack(
            side="left", padx=2
        )
        ttk.Label(top, text="グループラベル").pack(side="left", padx=(8, 2))
        self.train_label = ttk.Entry(top, width=16)
        self.train_label.insert(0, "group")
        self.train_label.pack(side="left", padx=2)
        ttk.Button(
            top, text="選択をグループへ", command=self._train_add_group
        ).pack(side="left", padx=2)
        ttk.Button(top, text="グループ削除", command=self._train_clear_groups).pack(
            side="left", padx=2
        )
        ttk.Button(top, text="描画", command=self._train_draw).pack(side="left", padx=2)
        ttk.Button(top, text="PNG保存", command=self._train_save).pack(
            side="left", padx=2
        )

        body = ttk.Frame(self.train_frame)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=False)
        ttk.Label(left, text="experiments (training_stats.csv あり)").pack(anchor="w")
        self.train_list = tk.Listbox(
            left, width=48, height=12, selectmode=tk.EXTENDED, exportselection=False
        )
        self.train_list.pack(fill="both", expand=True)
        self.train_group_view = tk.Listbox(left, width=48, height=6, exportselection=False)
        self.train_group_view.pack(fill="x", pady=4)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=8)
        self.train_fig = Figure(figsize=(6, 4), dpi=100)
        self.train_ax = self.train_fig.add_subplot(111)
        self.train_canvas = FigureCanvasTkAgg(self.train_fig, master=right)
        self.train_canvas.get_tk_widget().pack(fill="both", expand=True)

        self._refresh_train_list()

    def _refresh_train_list(self):
        self.train_list.delete(0, tk.END)
        self._train_dirs: list[Path] = []
        for row in self.db.list_experiments():
            run_dir = Path(row["run_dir"]) if row.get("run_dir") else None
            if run_dir is None:
                continue
            csv_path = run_dir / "training_stats.csv"
            if not csv_path.is_file():
                continue
            self._train_dirs.append(run_dir)
            note = (row.get("note") or "").strip()
            note_part = f" | {note}" if note else ""
            self.train_list.insert(
                tk.END,
                f"{row['id']}  terrain={row.get('terrain_mode', '')}{note_part}",
            )

    def _train_add_group(self):
        label = self.train_label.get().strip()
        if not label:
            messagebox.showinfo("Plot", "グループラベルを入力してください")
            return
        sels = self.train_list.curselection()
        if not sels:
            messagebox.showinfo("Plot", "run を選択してください")
            return
        dirs = [self._train_dirs[i] for i in sels]
        self._train_groups.setdefault(label, [])
        for d in dirs:
            if d not in self._train_groups[label]:
                self._train_groups[label].append(d)
        self._refresh_train_groups_view()

    def _train_clear_groups(self):
        self._train_groups.clear()
        self._refresh_train_groups_view()

    def _refresh_train_groups_view(self):
        self.train_group_view.delete(0, tk.END)
        for label, dirs in self._train_groups.items():
            names = ",".join(d.name for d in dirs)
            self.train_group_view.insert(tk.END, f"{label}: {names}")

    def _train_series(self):
        series = []
        for label, dirs in self._train_groups.items():
            iters, means, stds, ns = train_aggregate(label, dirs)
            series.append((label, iters, means, stds, ns))
        return series

    def _train_draw(self):
        if not self._train_groups:
            messagebox.showinfo("Plot", "グループを追加してください")
            return
        try:
            series = self._train_series()
        except (FileNotFoundError, ValueError) as e:
            messagebox.showerror("Plot", str(e))
            return
        self.train_ax.clear()
        train_draw_groups(self.train_ax, series)
        self.train_fig.tight_layout()
        self.train_canvas.draw_idle()

    def _train_save(self):
        if not self._train_groups:
            messagebox.showinfo("Plot", "グループを追加してください")
            return
        PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            initialdir=str(PLOTS_ROOT),
            initialfile="training_curves_episode_len.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
        )
        if not path:
            return
        try:
            series = self._train_series()
            csv_rows = []
            for label, iters, means, stds, ns in series:
                for it, mean, std, n in zip(iters, means, stds, ns):
                    csv_rows.append((label, it, mean, std, n))
            out = Path(path)
            train_write_csv(out.with_name(out.stem + ".csv"), csv_rows)
            train_plot_groups(series, out)
            messagebox.showinfo("Plot", f"保存しました:\n{out}")
        except Exception as e:
            messagebox.showerror("Plot", str(e))

    # ----- Paper control -----

    def _build_paper(self):
        top = ttk.Frame(self.paper_frame)
        top.pack(fill="x", padx=8, pady=4)
        ttk.Button(top, text="一覧更新", command=self._refresh_paper_runs).pack(
            side="left", padx=2
        )
        ttk.Button(top, text="描画", command=self._paper_draw).pack(side="left", padx=2)
        ttk.Button(top, text="PNG保存", command=self._paper_save).pack(
            side="left", padx=2
        )

        body = ttk.Frame(self.paper_frame)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=False)
        ttk.Label(left, text="paper_logs run").pack(anchor="w")
        self.paper_run_list = tk.Listbox(left, width=40, height=8, exportselection=False)
        self.paper_run_list.pack(fill="both", expand=True)
        self.paper_run_list.bind("<<ListboxSelect>>", self._on_paper_run_select)
        ttk.Label(left, text="trial").pack(anchor="w")
        self.paper_trial_list = tk.Listbox(left, width=40, height=8, exportselection=False)
        self.paper_trial_list.pack(fill="both", expand=True)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=8)
        self.paper_fig = Figure(figsize=(6, 7), dpi=90)
        self.paper_canvas = FigureCanvasTkAgg(self.paper_fig, master=right)
        self.paper_canvas.get_tk_widget().pack(fill="both", expand=True)

        self._refresh_paper_runs()

    def _refresh_paper_runs(self):
        self.paper_run_list.delete(0, tk.END)
        self.paper_trial_list.delete(0, tk.END)
        self._paper_runs: list[Path] = []
        self._paper_trials: list[Path] = []
        self._paper_trial = None
        if not PAPER_ROOT.is_dir():
            return
        for d in sorted(PAPER_ROOT.iterdir()):
            if d.is_dir():
                self._paper_runs.append(d)
                self.paper_run_list.insert(
                    tk.END, label_with_note(self.db, d.name)
                )

    def _on_paper_run_select(self, _event=None):
        sels = self.paper_run_list.curselection()
        # 選択解除イベントでは trial 一覧を消さない（他 Listbox クリック時の誤クリア防止）
        if not sels:
            return
        self.paper_trial_list.delete(0, tk.END)
        self._paper_trials = []
        self._paper_trial = None
        run_dir = self._paper_runs[sels[0]]
        for d in sorted(run_dir.iterdir()):
            if d.is_dir() and (d / "control.csv").is_file():
                self._paper_trials.append(d)
                self.paper_trial_list.insert(tk.END, d.name)

    def _selected_paper_trial(self) -> Path | None:
        sels = self.paper_trial_list.curselection()
        if not sels:
            return None
        return self._paper_trials[sels[0]]

    def _paper_draw(self):
        trial = self._selected_paper_trial()
        if trial is None:
            messagebox.showinfo("Plot", "trial を選択してください")
            return
        try:
            csv_path = resolve_control_csv(trial)
            data = load_paper_control(csv_path)
            title = infer_title(csv_path, data)
            self.paper_fig.clear()

            t = data["time"]
            t_sw = switch_time(data)
            axes = self.paper_fig.subplots(4, 1, sharex=True)
            self.paper_fig.suptitle(title, fontsize=12, fontweight="bold")
            axes[0].plot(t, data["dcm_error_norm"], color="purple", label="‖error‖")
            _mark_switch(axes[0], t_sw, label="terrain switch")
            axes[0].set_ylabel("DCM error norm")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(loc="upper right", fontsize=8)

            axes[1].plot(t, data["dcm_error_local_x"], "r-", label="error x")
            axes[1].plot(t, data["dcm_error_local_y"], "g-", label="error y")
            axes[1].axhline(0.0, color="k", linestyle=":", alpha=0.3)
            _mark_switch(axes[1], t_sw)
            axes[1].set_ylabel("DCM error local")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc="upper right", fontsize=8)

            axes[2].plot(t, data["obs_foot_sink_right"], "r-", label="sink R")
            axes[2].plot(t, data["obs_foot_sink_left"], "b-", label="sink L")
            axes[2].axhline(0.0, color="k", linestyle=":", alpha=0.3)
            _mark_switch(axes[2], t_sw)
            axes[2].set_ylabel("Foot sink")
            axes[2].grid(True, alpha=0.3)
            axes[2].legend(loc="upper right", fontsize=8)

            axes[3].plot(t, data["delta_x"], "r-", label="Δx")
            axes[3].plot(t, data["delta_y"], "g-", label="Δy")
            axes[3].axhline(0.0, color="k", linestyle=":", alpha=0.3)
            _mark_switch(axes[3], t_sw)
            axes[3].set_ylabel("RL delta")
            axes[3].set_xlabel("Time [s]")
            axes[3].grid(True, alpha=0.3)
            axes[3].legend(loc="upper right", fontsize=8)

            self.paper_fig.tight_layout(rect=(0, 0, 1, 0.96))
            self.paper_canvas.draw_idle()
            self._paper_trial = trial
        except Exception as e:
            messagebox.showerror("Plot", str(e))

    def _paper_save(self):
        trial = self._selected_paper_trial() or self._paper_trial
        if trial is None:
            messagebox.showinfo("Plot", "trial を選択してください")
            return
        path = filedialog.asksaveasfilename(
            initialdir=str(trial),
            initialfile="paper_control.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
        )
        if not path:
            return
        try:
            plot_paper_trial(trial, output_path=path)
            messagebox.showinfo("Plot", f"保存しました:\n{path}")
        except Exception as e:
            messagebox.showerror("Plot", str(e))

    # ----- Paper compare (7 panels) -----

    def _build_paper_compare(self):
        top = ttk.Frame(self.paper_cmp_frame)
        top.pack(fill="x", padx=8, pady=4)
        ttk.Button(top, text="一覧更新", command=self._refresh_paper_cmp_runs).pack(
            side="left", padx=2
        )
        ttk.Label(top, text="ラベル").pack(side="left", padx=(8, 2))
        self.paper_cmp_label = ttk.Entry(top, width=14)
        self.paper_cmp_label.insert(0, "trial")
        self.paper_cmp_label.pack(side="left", padx=2)
        ttk.Button(
            top, text="比較へ追加", command=self._paper_cmp_add
        ).pack(side="left", padx=2)
        ttk.Button(
            top, text="比較クリア", command=self._paper_cmp_clear
        ).pack(side="left", padx=2)
        ttk.Button(top, text="描画", command=self._paper_cmp_draw).pack(
            side="left", padx=2
        )
        ttk.Button(top, text="PNG保存", command=self._paper_cmp_save).pack(
            side="left", padx=2
        )

        body = ttk.Frame(self.paper_cmp_frame)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=False)
        ttk.Label(left, text="paper_logs run").pack(anchor="w")
        self.paper_cmp_run_list = tk.Listbox(
            left, width=40, height=7, exportselection=False
        )
        self.paper_cmp_run_list.pack(fill="both", expand=True)
        self.paper_cmp_run_list.bind("<<ListboxSelect>>", self._on_paper_cmp_run_select)
        ttk.Label(left, text="trial").pack(anchor="w")
        self.paper_cmp_trial_list = tk.Listbox(
            left, width=40, height=7, exportselection=False
        )
        self.paper_cmp_trial_list.pack(fill="both", expand=True)
        ttk.Label(left, text="比較リスト（7パネル・色分け）").pack(anchor="w")
        self.paper_cmp_view = tk.Listbox(
            left, width=40, height=6, exportselection=False
        )
        self.paper_cmp_view.pack(fill="x", pady=4)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=8)
        self.paper_cmp_fig = Figure(figsize=(6, 10), dpi=80)
        self.paper_cmp_canvas = FigureCanvasTkAgg(
            self.paper_cmp_fig, master=right
        )
        self.paper_cmp_canvas.get_tk_widget().pack(fill="both", expand=True)

        self._refresh_paper_cmp_runs()

    def _refresh_paper_cmp_runs(self):
        self.paper_cmp_run_list.delete(0, tk.END)
        self.paper_cmp_trial_list.delete(0, tk.END)
        self._paper_cmp_runs: list[Path] = []
        self._paper_cmp_trials: list[Path] = []
        if not PAPER_ROOT.is_dir():
            return
        for d in sorted(PAPER_ROOT.iterdir()):
            if d.is_dir():
                self._paper_cmp_runs.append(d)
                self.paper_cmp_run_list.insert(
                    tk.END, label_with_note(self.db, d.name)
                )

    def _on_paper_cmp_run_select(self, _event=None):
        sels = self.paper_cmp_run_list.curselection()
        if not sels:
            return
        self.paper_cmp_trial_list.delete(0, tk.END)
        self._paper_cmp_trials = []
        run_dir = self._paper_cmp_runs[sels[0]]
        for d in sorted(run_dir.iterdir()):
            if d.is_dir() and (d / "control.csv").is_file():
                self._paper_cmp_trials.append(d)
                self.paper_cmp_trial_list.insert(tk.END, d.name)

    def _selected_paper_cmp_trial(self) -> Path | None:
        sels = self.paper_cmp_trial_list.curselection()
        if not sels:
            return None
        return self._paper_cmp_trials[sels[0]]

    def _refresh_paper_cmp_view(self):
        self.paper_cmp_view.delete(0, tk.END)
        for label, trial in self._paper_compare:
            self.paper_cmp_view.insert(
                tk.END, f"{label}: {trial.parent.name}/{trial.name}"
            )

    def _paper_cmp_add(self):
        label = self.paper_cmp_label.get().strip()
        if not label:
            messagebox.showinfo("Plot", "ラベルを入力してください")
            return
        trial = self._selected_paper_cmp_trial()
        if trial is None:
            messagebox.showinfo("Plot", "trial を選択してください")
            return
        for _lab, path in self._paper_compare:
            if path == trial:
                messagebox.showinfo("Plot", "同じ trial は既に比較リストにあります")
                return
        self._paper_compare.append((label, trial))
        self._refresh_paper_cmp_view()

    def _paper_cmp_clear(self):
        self._paper_compare.clear()
        self._refresh_paper_cmp_view()

    def _paper_cmp_load_members(self) -> list[tuple[str, dict]]:
        members: list[tuple[str, dict]] = []
        for label, trial in self._paper_compare:
            csv_path = resolve_control_csv(trial)
            members.append((label, load_paper_control(csv_path)))
        return members

    def _paper_cmp_draw(self):
        if not self._paper_compare:
            messagebox.showinfo("Plot", "比較リストに trial を追加してください")
            return
        try:
            members = self._paper_cmp_load_members()
            draw_compare_on_figure(self.paper_cmp_fig, members)
            self.paper_cmp_canvas.draw_idle()
        except Exception as e:
            messagebox.showerror("Plot", str(e))

    def _paper_cmp_save(self):
        if not self._paper_compare:
            messagebox.showinfo("Plot", "比較リストに trial を追加してください")
            return
        PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            initialdir=str(PLOTS_ROOT),
            initialfile="paper_compare.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
        )
        if not path:
            return
        try:
            out = Path(path)
            plot_paper_trials_compare(self._paper_compare, output_path=out)
            csv_names = ", ".join(
                f"{_safe_label_prefix(lab)}_control.csv"
                for lab, _ in self._paper_compare
            )
            messagebox.showinfo(
                "Plot",
                f"保存しました:\n{out}\n(+ {csv_names})",
            )
        except Exception as e:
            messagebox.showerror("Plot", str(e))

    # ----- Sink vs delta_x -----

    def _build_sink(self):
        top = ttk.Frame(self.sink_frame)
        top.pack(fill="x", padx=8, pady=4)
        ttk.Button(top, text="一覧更新", command=self._refresh_sink_list).pack(
            side="left", padx=2
        )
        ttk.Label(top, text="グループラベル").pack(side="left", padx=(8, 2))
        self.sink_label = ttk.Entry(top, width=16)
        self.sink_label.insert(0, "group")
        self.sink_label.pack(side="left", padx=2)
        ttk.Button(
            top, text="選択をグループへ", command=self._sink_add_group
        ).pack(side="left", padx=2)
        ttk.Button(top, text="グループ削除", command=self._sink_clear_groups).pack(
            side="left", padx=2
        )
        ttk.Button(top, text="描画", command=self._sink_draw).pack(side="left", padx=2)
        ttk.Button(top, text="PNG保存", command=self._sink_save).pack(
            side="left", padx=2
        )

        body = ttk.Frame(self.sink_frame)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=False)
        ttk.Label(left, text=f"{PAPER_ROOT}").pack(anchor="w")
        ttk.Label(
            left,
            text="contact 列付き paper_log のみ（1 dir = 1 学習シード）",
            wraplength=320,
        ).pack(anchor="w")
        self.sink_list = tk.Listbox(
            left, width=48, height=12, selectmode=tk.EXTENDED, exportselection=False
        )
        self.sink_list.pack(fill="both", expand=True)
        self.sink_group_view = tk.Listbox(
            left, width=48, height=6, exportselection=False
        )
        self.sink_group_view.pack(fill="x", pady=4)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=8)
        self.sink_fig = Figure(figsize=(6, 4.5), dpi=100)
        self.sink_ax = self.sink_fig.add_subplot(111)
        self.sink_canvas = FigureCanvasTkAgg(self.sink_fig, master=right)
        self.sink_canvas.get_tk_widget().pack(fill="both", expand=True)

        self._refresh_sink_list()

    def _refresh_sink_list(self):
        self.sink_list.delete(0, tk.END)
        self._sink_dirs: list[Path] = []
        if not PAPER_ROOT.is_dir():
            return
        for d in sorted(PAPER_ROOT.iterdir()):
            if not d.is_dir():
                continue
            has_trial = any(
                c.is_dir() and (c / "control.csv").is_file() for c in d.iterdir()
            )
            if (d / "results.csv").is_file() or has_trial:
                self._sink_dirs.append(d)
                self.sink_list.insert(tk.END, label_with_note(self.db, d.name))

    def _sink_add_group(self):
        label = self.sink_label.get().strip()
        if not label:
            messagebox.showinfo("Plot", "グループラベルを入力してください")
            return
        sels = self.sink_list.curselection()
        if not sels:
            messagebox.showinfo("Plot", "paper_logs ディレクトリを選択してください")
            return
        dirs = [self._sink_dirs[i] for i in sels]
        self._sink_groups.setdefault(label, [])
        for d in dirs:
            if d not in self._sink_groups[label]:
                self._sink_groups[label].append(d)
        self._refresh_sink_groups_view()

    def _sink_clear_groups(self):
        self._sink_groups.clear()
        self._refresh_sink_groups_view()

    def _refresh_sink_groups_view(self):
        self.sink_group_view.delete(0, tk.END)
        for label, dirs in self._sink_groups.items():
            names = ",".join(d.name for d in dirs)
            self.sink_group_view.insert(tk.END, f"{label}: {names}")

    def _sink_series(self):
        series = []
        for label, dirs in self._sink_groups.items():
            series.append(sink_build_group(label, dirs, verbose=False))
        return series

    def _sink_draw(self):
        if not self._sink_groups:
            messagebox.showinfo("Plot", "グループを追加してください")
            return
        try:
            series = self._sink_series()
        except (FileNotFoundError, ValueError) as e:
            messagebox.showerror("Plot", str(e))
            return
        self.sink_ax.clear()
        sink_draw_groups(self.sink_ax, series)
        self.sink_fig.tight_layout()
        self.sink_canvas.draw_idle()

    def _sink_save(self):
        if not self._sink_groups:
            messagebox.showinfo("Plot", "グループを追加してください")
            return
        PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            initialdir=str(PLOTS_ROOT),
            initialfile="sink_vs_deltax.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
        )
        if not path:
            return
        try:
            series = self._sink_series()
            out = Path(path)
            sink_write_csv(out.with_name(out.stem + ".csv"), series)
            sink_plot_groups(series, out)
            messagebox.showinfo("Plot", f"保存しました:\n{out}")
        except Exception as e:
            messagebox.showerror("Plot", str(e))

    # ----- Δx mean vs overall success (checkpoint) -----

    def _build_deltax(self):
        top = ttk.Frame(self.deltax_frame)
        top.pack(fill="x", padx=8, pady=4)
        ttk.Button(top, text="一覧更新", command=self._refresh_deltax_list).pack(
            side="left", padx=2
        )
        ttk.Button(top, text="描画", command=self._deltax_draw).pack(side="left", padx=2)
        ttk.Button(top, text="PNG保存", command=self._deltax_save).pack(
            side="left", padx=2
        )
        ttk.Label(top, text="interv").pack(side="left", padx=(8, 2))
        self.deltax_interv = ttk.Combobox(
            top,
            width=12,
            state="readonly",
            values=["full", "after_switch", "none"],
        )
        self.deltax_interv.set("full")
        self.deltax_interv.pack(side="left", padx=2)
        self.deltax_interv.bind("<<ComboboxSelected>>", self._on_deltax_window)
        ttk.Label(top, text="Y").pack(side="left", padx=(8, 2))
        self.deltax_y = ttk.Combobox(
            top,
            width=10,
            state="readonly",
            values=["overall", "α=1"],
        )
        self.deltax_y.set("overall")
        self.deltax_y.pack(side="left", padx=2)
        self.deltax_y.bind("<<ComboboxSelected>>", self._on_deltax_window)
        ttk.Label(top, text="min seeds").pack(side="left", padx=(8, 2))
        self.deltax_min_seeds = ttk.Entry(top, width=4)
        self.deltax_min_seeds.insert(0, "10")
        self.deltax_min_seeds.pack(side="left", padx=2)
        ttk.Label(top, text="Δx窓").pack(side="left", padx=(8, 2))
        self.deltax_window = ttk.Combobox(
            top,
            width=14,
            state="readonly",
            values=["切替後すべて", "切替前すべて", "切替後N歩"],
        )
        self.deltax_window.set("切替後すべて")
        self.deltax_window.pack(side="left", padx=2)
        self.deltax_window.bind("<<ComboboxSelected>>", self._on_deltax_window)
        ttk.Label(top, text="N").pack(side="left", padx=(4, 2))
        self.deltax_post_n = ttk.Entry(top, width=4)
        self.deltax_post_n.insert(0, "5")
        self.deltax_post_n.pack(side="left", padx=2)
        self.deltax_post_n.bind("<Return>", self._on_deltax_window)
        self._sync_deltax_post_n_state()
        ttk.Label(top, text="w_act").pack(side="left", padx=(8, 2))
        self._deltax_wact_vars: dict[float, tk.BooleanVar] = {}
        for w in (0.1, 1.0, 10.0, 100.0):
            var = tk.BooleanVar(value=True)
            self._deltax_wact_vars[w] = var
            ttk.Checkbutton(top, text=str(w), variable=var).pack(side="left", padx=2)

        body = ttk.Frame(self.deltax_frame)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=False)
        ttk.Label(
            left,
            text="DB 実験 ↔ paper_log（選択中 interv のみ・ready のみ描画）",
            wraplength=360,
        ).pack(anchor="w")
        cols = ("run_id", "w_act", "status", "delta_x", "success", "n_seeds")
        self.deltax_tree = ttk.Treeview(
            left, columns=cols, show="headings", height=14
        )
        self.deltax_tree.heading("run_id", text="run_id")
        self.deltax_tree.heading("w_act", text="w_act")
        self.deltax_tree.heading("status", text="status")
        self.deltax_tree.heading("delta_x", text="Δx mean")
        self.deltax_tree.heading("success", text="overall")
        self.deltax_tree.heading("n_seeds", text="n")
        self.deltax_tree.column("run_id", width=180)
        self.deltax_tree.column("w_act", width=44)
        self.deltax_tree.column("status", width=56)
        self.deltax_tree.column("delta_x", width=68)
        self.deltax_tree.column("success", width=52)
        self.deltax_tree.column("n_seeds", width=28)
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.deltax_tree.yview)
        self.deltax_tree.configure(yscrollcommand=scroll.set)
        self.deltax_tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="left", fill="y")

        ttk.Label(left, text="相関 (ready・選択 w_act)").pack(anchor="w", pady=(6, 0))
        self.deltax_corr_view = tk.Listbox(
            left, width=48, height=5, exportselection=False
        )
        self.deltax_corr_view.pack(fill="x", pady=2)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=8)
        self.deltax_fig = Figure(figsize=(6, 4.5), dpi=100)
        self.deltax_ax = self.deltax_fig.add_subplot(111)
        self.deltax_canvas = FigureCanvasTkAgg(self.deltax_fig, master=right)
        self.deltax_canvas.get_tk_widget().pack(fill="both", expand=True)

        self._refresh_deltax_list()

    def _deltax_w_act_filter(self) -> list[float]:
        return [w for w, var in self._deltax_wact_vars.items() if var.get()]

    def _deltax_window_key(self) -> str:
        label = self.deltax_window.get()
        if label == "切替前すべて":
            return "pre_all"
        if label == "切替後N歩":
            return "post_n"
        return "post_all"

    def _deltax_y_metric(self) -> str:
        return "alpha_1" if self.deltax_y.get() == "α=1" else "overall"

    def _deltax_post_n_value(self) -> int:
        try:
            n = int(self.deltax_post_n.get().strip())
        except ValueError:
            n = 5
        return n if n >= 1 else 5

    def _sync_deltax_post_n_state(self):
        state = "normal" if self._deltax_window_key() == "post_n" else "disabled"
        self.deltax_post_n.configure(state=state)

    def _on_deltax_window(self, _event=None):
        self._sync_deltax_post_n_state()
        self._refresh_deltax_list()

    def _deltax_min_params(self) -> tuple[int, int]:
        try:
            n = int(self.deltax_min_seeds.get().strip())
        except ValueError:
            n = 10
        return n, n * 10

    def _refresh_deltax_list(self):
        for item in self.deltax_tree.get_children():
            self.deltax_tree.delete(item)
        min_seeds, min_trials = self._deltax_min_params()
        window = self._deltax_window_key()
        post_n = self._deltax_post_n_value()
        y_metric = self._deltax_y_metric()
        y_heading = "α=1" if y_metric == "alpha_1" else "overall"
        self.deltax_tree.heading("success", text=y_heading)
        if window == "post_n" and post_n < 1:
            messagebox.showerror("Plot", "N は 1 以上にしてください")
            return
        try:
            self._deltax_points = deltax_discover(
                self.db,
                paper_root=PAPER_ROOT,
                min_eval_seeds=min_seeds,
                min_overall_trials=min_trials,
                window=window,
                post_n=post_n,
                intervention_mode=self.deltax_interv.get().strip() or "full",
                y_metric=y_metric,
            )
        except Exception as e:
            messagebox.showerror("Plot", str(e))
            self._deltax_points = []
            return
        present_w = {p.w_act for p in self._deltax_points}
        for w, var in self._deltax_wact_vars.items():
            if any(abs(pw - w) < 1e-6 for pw in present_w):
                var.set(True)
        for p in sorted(self._deltax_points, key=lambda x: (x.w_act, x.run_id)):
            y_str = f"{p.success_rate:.4f}" if p.status != PointStatus.NO_PAPER else "—"
            dx_str = (
                f"{p.delta_x_mean:.5f}"
                if p.status != PointStatus.NO_PAPER
                else "—"
            )
            self.deltax_tree.insert(
                "",
                tk.END,
                values=(
                    run_id_with_note(self.db, p.run_id),
                    f"{p.w_act:g}",
                    p.status.value,
                    dx_str,
                    y_str,
                    p.n_eval_seeds,
                ),
            )

    def _deltax_draw(self):
        self._refresh_deltax_list()
        if not self._deltax_points:
            messagebox.showinfo("Plot", "一覧更新してください（データがありません）")
            return
        w_filter = self._deltax_w_act_filter()
        if not w_filter:
            messagebox.showinfo("Plot", "w_act を1つ以上選択してください")
            return
        window = self._deltax_window_key()
        post_n = self._deltax_post_n_value()
        y_metric = self._deltax_y_metric()
        self.deltax_ax.clear()
        self._deltax_corrs = deltax_draw_scatter(
            self.deltax_ax,
            self._deltax_points,
            w_act_filter=w_filter,
            ready_only=True,
            window=window,
            post_n=post_n,
            y_metric=y_metric,
            show_corr=False,
        )
        self.deltax_corr_view.delete(0, tk.END)
        for line in deltax_corr_lines(self._deltax_corrs):
            self.deltax_corr_view.insert(tk.END, line)
        self.deltax_fig.tight_layout()
        self.deltax_canvas.draw_idle()

    def _deltax_save(self):
        self._refresh_deltax_list()
        if not self._deltax_points:
            messagebox.showinfo("Plot", "一覧更新してください")
            return
        w_filter = self._deltax_w_act_filter()
        if not w_filter:
            messagebox.showinfo("Plot", "w_act を1つ以上選択してください")
            return
        window = self._deltax_window_key()
        post_n = self._deltax_post_n_value()
        y_metric = self._deltax_y_metric()
        PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            initialdir=str(PLOTS_ROOT),
            initialfile="deltax_vs_success.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
        )
        if not path:
            return
        try:
            out = Path(path)
            deltax_write_csv(out.with_name(out.stem + ".csv"), self._deltax_points)
            deltax_plot_scatter(
                self._deltax_points,
                out,
                w_act_filter=w_filter,
                ready_only=True,
                window=window,
                post_n=post_n,
                y_metric=y_metric,
                show_corr=False,
                write_corr=True,
            )
            corr_path = out.with_name(out.stem + "_corr.csv")
            messagebox.showinfo(
                "Plot",
                f"保存しました:\n{out}\n(+ {out.stem}.csv, {corr_path.name})",
            )
        except Exception as e:
            messagebox.showerror("Plot", str(e))

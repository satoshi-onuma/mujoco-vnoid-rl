"""複数 softness 評価 subprocess のプール。"""

from __future__ import annotations

from pathlib import Path

from .softness_eval_launcher import SoftnessEvalLauncher, predict_output_dir


def output_dir_for_params(params: dict) -> Path:
    run_id = str(params.get("run_id", "?"))
    return predict_output_dir(
        run_id,
        float(params["softness_min"]),
        float(params["softness_max"]),
        float(params["softness_step"]),
        str(params.get("script", "grid")),
        str(params.get("intervention_mode", "full")),
    )


class EvalPool:
    def __init__(self) -> None:
        self._running: list[SoftnessEvalLauncher] = []

    def count(self) -> int:
        return sum(1 for launcher in self._running if launcher.is_running())

    def any_running(self) -> bool:
        return self.count() > 0

    def active_output_dirs(self) -> set[str]:
        dirs: set[str] = set()
        for launcher in self._running:
            if not launcher.is_running() or launcher.output_dir is None:
                continue
            dirs.add(str(launcher.output_dir.resolve()))
        return dirs

    def is_output_dir_busy(self, params: dict) -> bool:
        target = str(output_dir_for_params(params).resolve())
        return target in self.active_output_dirs()

    def start_one(self, params: dict) -> SoftnessEvalLauncher:
        launcher = SoftnessEvalLauncher()
        launcher.start(params)
        self._running.append(launcher)
        return launcher

    def poll_all(self) -> list[tuple[SoftnessEvalLauncher, int]]:
        finished: list[tuple[SoftnessEvalLauncher, int]] = []
        still: list[SoftnessEvalLauncher] = []
        for launcher in self._running:
            code = launcher.poll()
            if code is not None:
                finished.append((launcher, code))
            else:
                still.append(launcher)
        self._running = still
        return finished

    def stop_all(self) -> list[SoftnessEvalLauncher]:
        """走行中を止め、プールから外す。呼び出し側で DB 記録すること。"""
        stopped = [l for l in self._running if l.is_running()]
        for launcher in stopped:
            launcher.stop()
        self._running = []
        return stopped

    def running_launchers(self) -> list[SoftnessEvalLauncher]:
        return [l for l in self._running if l.is_running()]

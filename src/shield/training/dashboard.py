from __future__ import annotations

import os
import sys
import time
from typing import Any


def hms(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:d}h {minutes:02d}m {secs:02d}s" if hours else f"{minutes:d}m {secs:02d}s"


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None and "IPKernelApp" in get_ipython().config
    except Exception:
        return False


def _clear() -> None:
    if _in_notebook():
        from IPython.display import clear_output

        clear_output(wait=True)
    elif sys.stdout.isatty():
        print("\033[H\033[J", end="")
    else:
        print()


def _cell(key: str, value: Any) -> str:
    if not isinstance(value, float):
        return str(value)
    if key == "epoch":
        return f"{value:.2f}"
    if key.endswith(("_s", "seconds")):
        return hms(value)
    return f"{value:.4f}"


def bar(done: int, total: int, width: int = 28) -> str:
    filled = int(width * done / total) if total else 0
    return "#" * filled + "." * (width - filled)


class LiveDashboard:
    def __init__(self, title: str, total_steps: int | None = None, tail: int = 25):
        self.title = title
        self.total_steps = total_steps
        self.tail = tail
        self.train_rows: list[dict[str, Any]] = []
        self.val_rows: list[dict[str, Any]] = []
        self.t0: float | None = None
        self.t_end: float | None = None
        self.phase: tuple[str, int, int, float] | None = None
        self.status = "in attesa…"
        self.quiet = bool(os.getenv("SHIELD_DASHBOARD_QUIET"))


    def start(self) -> None:
        self.t0 = time.time()

    def stop(self) -> None:
        self.t_end = time.time()

    def elapsed(self) -> float:
        if self.t0 is None:
            return 0.0
        return (self.t_end or time.time()) - self.t0

    def eta(self, step: int) -> float | None:
        if not self.total_steps or step <= 0:
            return None
        rate = self.elapsed() / step
        return max(0.0, (self.total_steps - step) * rate)


    def log_step(self, step: int, epoch: float, logs: dict[str, Any]) -> None:
        if "loss" not in logs:
            return
        self.train_rows.append(
            {
                "step": step,
                "epoch": round(float(epoch), 4),
                "loss": float(logs["loss"]),
                "learning_rate": (
                    float(logs["learning_rate"]) if "learning_rate" in logs else None
                ),
                "elapsed_s": round(self.elapsed(), 2),
            }
        )
        self.render()

    def log_val(self, row: dict[str, Any]) -> None:
        self.phase = None
        self.val_rows.append(row)
        self.render()

    def log_progress(self, label: str, done: int, total: int, t_start: float) -> None:
        self.phase = (label, done, total, t_start)
        self.render()


    def render(self) -> None:
        if self.quiet:
            return
        _clear()
        out: list[str] = []
        out.append(f"═══ {self.title}")
        out.append(f"stato: {self.status}")

        last_step = self.train_rows[-1]["step"] if self.train_rows else 0
        line = f"tempo dall'inizio: {hms(self.elapsed())}"
        if self.total_steps:
            eta = self.eta(last_step)
            line += f"   |   step {last_step}/{self.total_steps}"
            if eta is not None:
                line += f"   |   ETA training {hms(eta)}"
        out.append(line)

        if self.phase is not None:
            label, done, total, t_start = self.phase
            spent = time.time() - t_start
            rate = done / spent if spent > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            out.append(
                f"{label}: {done:>4}/{total}  [{bar(done, total)}]  "
                f"{spent:.0f}s trascorsi  ETA {hms(eta)}"
            )
        out.append("=" * 78)

        if self.val_rows:
            keys = ["epoch", "step", "val_loss"] + [
                k for k in self.val_rows[-1] if k not in ("epoch", "step", "val_loss")
            ]
            widths = {k: max(11, len(k) + 2) for k in keys}
            out.append("VALIDATION")
            out.append("  " + "".join(f"{k:>{widths[k]}}" for k in keys))
            for row in self.val_rows:
                cells = []
                for key in keys:
                    cells.append(f"{_cell(key, row.get(key)):>{widths[key]}}")
                out.append("  " + "".join(cells))
            out.append("-" * 78)

        if self.train_rows:
            last = self.train_rows[-1]
            last_lr = last["learning_rate"]
            out.append(
                f"TRAIN LIVE   step {last['step']}   epoca {last['epoch']:.2f}   "
                f"loss {last['loss']:.4f}   "
                f"lr {f'{last_lr:.2e}' if last_lr is not None else '—'}"
            )
            out.append(f"  {'step':>8}{'epoca':>9}{'loss':>12}{'lr':>12}{'trascorso':>12}")
            for row in self.train_rows[-self.tail :]:
                lr = row["learning_rate"]
                out.append(
                    f"  {row['step']:>8}{row['epoch']:>9.2f}{row['loss']:>12.4f}"
                    f"{(f'{lr:.2e}' if lr is not None else '—'):>12}"
                    f"{hms(row['elapsed_s']):>12}"
                )
        print("\n".join(out), flush=True)

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any


def ensure_directories(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def append_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(content)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def init_project_documents(journal_path: Path, run_summary_path: Path, config_summary: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not journal_path.exists():
        write_text(
            journal_path,
            "# Project Journal\n\n"
            f"Initialized: {timestamp}\n\n"
            "This journal is updated after each pipeline phase.\n\n",
        )
    if not run_summary_path.exists():
        write_text(
            run_summary_path,
            "# Run Summary\n\n"
            f"Initialized: {timestamp}\n\n"
            "## Configuration\n\n"
            f"{config_summary}\n",
        )


def append_phase_journal(
    journal_path: Path,
    phase_name: str,
    objective: str,
    inputs_used: list[str],
    methods_applied: list[str],
    metrics_plots: list[str],
    decisions: list[str],
    challenges: list[str],
    observations: list[str],
    outputs_saved: list[str],
    next_step: str,
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = [
        f"## {phase_name} ({timestamp})",
        "",
        f"**Objective**: {objective}",
        "",
        "**Inputs used**",
        *[f"- {item}" for item in inputs_used],
        "",
        "**Methods applied**",
        *[f"- {item}" for item in methods_applied],
        "",
        "**Metrics/plots generated**",
        *[f"- {item}" for item in metrics_plots],
        "",
        "**Decisions and justification**",
        *[f"- {item}" for item in decisions],
        "",
        "**Challenges and resolution**",
        *[f"- {item}" for item in (challenges or ['None'])],
        "",
        "**Observations and insights**",
        *[f"- {item}" for item in observations],
        "",
        "**Outputs saved**",
        *[f"- {item}" for item in outputs_saved],
        "",
        f"**Next step**: {next_step}",
        "",
    ]
    append_text(journal_path, "\n".join(content))


def append_run_summary(run_summary_path: Path, heading: str, body: str) -> None:
    append_text(run_summary_path, f"\n## {heading}\n\n{body}\n")


class RuntimeTracker:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    @contextmanager
    def track(self, stage_name: str):
        print(f"[START] {stage_name}", flush=True)
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.timings[stage_name] = elapsed
            print(f"[END] {stage_name}: {elapsed:.2f}s", flush=True)


def log_progress(message: str) -> None:
    print(f"[INFO] {message}", flush=True)

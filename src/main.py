from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

COMMANDS = {
    "clean": ["src/data_cleaning.py"],
    "eda": ["src/eda.py"],
    "train": ["src/run_experiments.py"],
    "robustness": ["src/run_robustness.py"],
}

PIPELINES = {
    "stage1": ["clean", "eda"],
    "all": ["clean", "eda", "train", "robustness"],
}


def run_script(script: str) -> None:
    print(f"\n==> python3 {script}")
    subprocess.run([PYTHON, script], cwd=PROJECT_ROOT, check=True)


def run_command(name: str) -> None:
    if name in COMMANDS:
        for script in COMMANDS[name]:
            run_script(script)
        return
    if name in PIPELINES:
        for child in PIPELINES[name]:
            run_command(child)
        return
    raise ValueError(f"Unknown command: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified command-line entry for the Dry Bean classification project."
    )
    parser.add_argument(
        "command",
        choices=[*COMMANDS.keys(), *PIPELINES.keys()],
        help="Task to run.",
    )
    args = parser.parse_args()
    run_command(args.command)


if __name__ == "__main__":
    main()

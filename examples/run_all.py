"""
Run the full project pipeline in logical order.

Each step lives in its own script so you can still run or debug one stage
at a time. This file chains them for a one-command demo or GitHub preview.

Run from the repo root:

    python examples/run_all.py

Steps:
    1. Stress test pipeline (FDIC fetch, model, backtest, capital projection)
    2. Out-of-sample forecast evaluation
    3. Model validation and diagnostics
    4. Phillips-Perron unit root tests
    5. Zivot-Andrews unit root tests
    6. PDF README preview
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = Path(__file__).resolve().parent

STEPS = [
    ("Stress test pipeline", "run_stress_test.py"),
    ("Out-of-sample forecast", "run_forecast_sample.py"),
    ("Model validation", "run_model_diagnostics.py"),
    ("Phillips-Perron unit root tests", "run_phillips_perron.py"),
    ("Zivot-Andrews unit root tests", "run_zivot_andrews.py"),
    ("PDF README preview", "generate_readme_pdf.py"),
]


def run_step(label: str, script: str) -> None:
    path = EXAMPLES / script
    if not path.exists():
        raise FileNotFoundError(f"Missing script: {path}")

    print()
    print("=" * 72)
    print(f"STEP: {label}")
    print(f"RUN:  python examples/{script}")
    print("=" * 72)
    started = time.perf_counter()

    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(ROOT),
        check=False,
    )
    elapsed = time.perf_counter() - started

    if result.returncode != 0:
        print()
        print(f"FAILED: {script} (exit code {result.returncode}, {elapsed:.1f}s)")
        sys.exit(result.returncode)

    print()
    print(f"DONE: {script} ({elapsed:.1f}s)")


def main() -> None:
    print("US Bank Credit Stress Test - full pipeline")
    print(f"Project root: {ROOT}")
    total_start = time.perf_counter()

    for label, script in STEPS:
        run_step(label, script)

    total = time.perf_counter() - total_start
    print()
    print("=" * 72)
    print("ALL STEPS COMPLETED")
    print("=" * 72)
    print(f"Total time: {total:.1f}s")
    print()
    print("Key outputs:")
    print(f"  Panel:      {ROOT / 'data' / 'regression_panel.csv'}")
    print(f"  Charts:     {ROOT / 'docs'}")
    print(f"  PDF report: {ROOT / 'docs' / 'US_Bank_Credit_Stress_Test.pdf'}")


if __name__ == "__main__":
    main()

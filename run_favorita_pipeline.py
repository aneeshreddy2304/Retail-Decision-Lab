"""Run the complete Favorita forecasting and decision-science pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from favorita_analytics import main as build_analytics
from favorita_benchmark import build_weekly_panel, main as run_benchmark


def main() -> None:
    print("[1/3] Building weekly Favorita panel")
    panel = build_weekly_panel()
    panel.to_parquet(ROOT / "data" / "processed" / "favorita_weekly.parquet", index=False)
    print("[2/3] Running leakage-safe forecasting benchmark")
    run_benchmark()
    print("[3/3] Building promotion, classification, and SQL artifacts")
    build_analytics()
    print("Favorita pipeline completed successfully.")


if __name__ == "__main__":
    main()

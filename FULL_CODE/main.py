"""Command-line entry point: ``python main.py validate`` or ``python main.py submit``."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from r2sorter.config import load_config, resolve_path
from r2sorter.data import load_all
from r2sorter.features import build_feature_table
from r2sorter.pipeline import validate, write_submission


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R2 Returns Fraud vs Genuine Grievance Sorter")
    parser.add_argument(
        "command",
        choices=["validate", "submit"],
        help="holdout validation, or write submission files",
    )
    parser.add_argument("--config", default=None, help="path to an alternative config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "validate":
        data = load_all(cfg)
        frame = build_feature_table(data["train"], data["test"], data["products"], cfg)
        report = json.dumps(validate(cfg, frame), indent=2, default=float)
        reports = resolve_path(cfg, "reports_dir")
        reports.mkdir(parents=True, exist_ok=True)
        (reports / "validation.json").write_text(report)
        print(report)
    else:
        print(json.dumps(write_submission(cfg), indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())

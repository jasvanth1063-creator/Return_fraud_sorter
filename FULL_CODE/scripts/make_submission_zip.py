"""Build ``<teamID>_R2.zip`` from a fresh git clone of this repository (keeps .git, as the rules require).

Usage: python FULL_CODE/scripts/make_submission_zip.py TEAM_ID [--out DIR]
Commit everything first; only committed files end up in the zip.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBLEM_ID = "R2"
REQUIRED_DATA = (
    "return_requests_train.csv",
    "return_requests_test.csv",
    "products.csv",
    "size_charts.csv",
    "customers.csv",
)
FORBIDDEN_PARTS = {
    ".venv",
    "node_modules",
    "__pycache__",
    ".ipynb_checkpoints",
    ".DS_Store",
    ".env",
}


def validate_team_id(team_id: str) -> str:
    """Team IDs become folder and file names, so allow only safe characters."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", team_id):
        raise ValueError("team ID may only contain letters, digits, '_' and '-'")
    return team_id


def clone_repository(destination: Path) -> None:
    subprocess.run(["git", "clone", "--quiet", str(REPO_ROOT), str(destination)], check=True, timeout=300)


def check_layout(folder: Path) -> None:
    """Fail if the clone would break the packaging rules."""
    allowed_top = {"README.md", ".gitignore", "LICENSE", "submission", "FULL_CODE", ".git"}
    extra = {item.name for item in folder.iterdir()} - allowed_top
    if extra:
        raise RuntimeError(f"unexpected top-level entries: {sorted(extra)}")
    for name in ("submission/R2_predictions.csv", "submission/R2_size_chart_top5.csv", "README.md"):
        if not (folder / name).is_file():
            raise RuntimeError(f"missing {name}")
    for path in folder.rglob("*"):
        if ".git" in path.relative_to(folder).parts:
            continue
        if FORBIDDEN_PARTS & set(path.relative_to(folder).parts) or path.suffix in {
            ".db",
            ".log",
            ".zip",
        }:
            raise RuntimeError(f"forbidden file in package: {path.relative_to(folder)}")
    for name in REQUIRED_DATA:
        if not (folder / "FULL_CODE" / "data" / name).is_file():
            raise RuntimeError(f"FULL_CODE/data/{name} is missing (the dataset is shipped inside the package)")


def zip_folder(folder: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(folder.rglob("*")):
            handle.write(path, Path(folder.name) / path.relative_to(folder))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team_id")
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT.parent),
        help="folder for the zip (default: next to the repo)",
    )
    args = parser.parse_args(argv)
    name = f"{validate_team_id(args.team_id)}_{PROBLEM_ID}"
    out_dir = Path(args.out).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / name
        clone_repository(clone)
        check_layout(clone)
        archive = out_dir / f"{name}.zip"
        zip_folder(clone, archive)
    print(f"wrote {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

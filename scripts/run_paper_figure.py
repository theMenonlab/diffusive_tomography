#!/usr/bin/env python3
"""Run an exact Figure 2-5 CWDT reconstruction from the data-package layout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


FIGURES = {
    "2": "figure_2_transmission",
    "3": "figure_3_phantom",
    "4": "figure_4_fungus",
    "5": "figure_5_insertion",
}


def default_data_root(repo_root: Path) -> Path | None:
    env_value = os.environ.get("CWDT_DATA_ROOT")
    if env_value:
        return Path(env_value).expanduser()

    sibling = repo_root.parent / "data_kaggle_package"
    if sibling.exists():
        return sibling
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the exact saved paper-figure reconstruction script with the "
            "current working directory set to the matching Kaggle figure folder."
        )
    )
    parser.add_argument(
        "--figure",
        required=True,
        choices=sorted(FIGURES),
        help="Paper figure to reproduce.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Path to the Kaggle-style data package root. Defaults to "
            "$CWDT_DATA_ROOT or ../data_kaggle_package when available."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to this GitHub repository checkout.",
    )
    parser.add_argument(
        "--script-source",
        choices=("repo", "data"),
        default="repo",
        help=(
            "Use the exact script copied into this repository, or the copy inside "
            "the data package. Both should have the same SHA-256 hash."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run the reconstruction.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command and checks without launching the long run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve() if args.data_root else default_data_root(repo_root)
    if data_root is None:
        raise SystemExit(
            "Data package not found. Pass --data-root /path/to/data_kaggle_package "
            "or set CWDT_DATA_ROOT."
        )
    data_root = data_root.resolve()

    figure_dir_name = FIGURES[args.figure]
    figure_dir = data_root / "figures" / figure_dir_name
    config_path = figure_dir / "config_package.json"
    if args.script_source == "repo":
        script_path = repo_root / "paper" / figure_dir_name / "forward_model_exact.py"
    else:
        script_path = figure_dir / "code" / "forward_model_exact.py"

    required_paths = [
        data_root,
        figure_dir,
        figure_dir / "input_images",
        config_path,
        script_path,
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        for path in missing:
            print(f"Missing: {path}")
        return 2

    config = json.loads(config_path.read_text())
    output_base = figure_dir / config["data_paths"]["output_base_path"]
    image_count = sum(
        1
        for path in (figure_dir / "input_images").iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )

    command = [args.python, str(script_path), str(config_path)]
    print(f"Figure: {args.figure} ({figure_dir_name})")
    print(f"Working directory: {figure_dir}")
    print(f"Input images: {image_count}")
    print(f"Output directory: {output_base}")
    print("Command:")
    print(" ".join(command))
    if output_base.exists():
        print("Warning: output directory already exists and may be overwritten by the reconstruction.")

    if args.dry_run:
        return 0

    completed = subprocess.run(command, cwd=figure_dir)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

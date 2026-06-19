#!/usr/bin/env python3
"""Verify the local GitHub release folder and optional Figure 2-5 data package."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MAX_GITHUB_FILE_BYTES = 100 * 1024 * 1024

FIGURES = {
    "2": {
        "slug": "figure_2_transmission",
        "label": "tartrazine-cleared poplar transmission",
        "input_images": 53,
        "result_files_min": 193,
        "script_sha256": "c51fba093b2b96f731d34f72cd7205ae7c5497a780b853ae6872d9d3b4c00f35",
        "config_package_sha256": "573f4587c9af110aeef721380e94c46eb02246591b3a2c4512fd0fd0e1f95c12",
        "config_original_sha256": "4691cf7b3c5997178dbef194d49b49bb82f6d7670d904a9dafcb3a146c64165b",
    },
    "3": {
        "slug": "figure_3_phantom",
        "label": "scattering phantom",
        "input_images": 24,
        "result_files_min": 82,
        "script_sha256": "612695c2f59224155e81f6aa7e0f8d962842f850699cc450a12ff242d0b8ab12",
        "config_package_sha256": "88cc7ac41a7869ae0a32fe63a68a353d290faab9c7a65b39af536b99b340051c",
        "config_original_sha256": "cfe4480f3714bd43bdc5154eedbd483a4c60bf78e78473e3eba8cd6af4937312",
    },
    "4": {
        "slug": "figure_4_fungus",
        "label": "fungus/root sample",
        "input_images": 21,
        "result_files_min": 135,
        "script_sha256": "ea76390752c5b26ac942d578529dedec3204c2ab0322756015f6b1d6570c8d9a",
        "config_package_sha256": "5a3e4b4b179d73b4d7ba81f71a061482a970235941efc6ea939fdb7a1d8aeebc",
        "config_original_sha256": "0e88dec0bdaf99e1d328bca95a09b35a1b03b2570326e8ba228f4a688418ee9c",
    },
    "5": {
        "slug": "figure_5_insertion",
        "label": "inserted side-emitting fiber",
        "input_images": 27,
        "result_files_min": 183,
        "script_sha256": "2256b423f14faead09d7e65beac886715b0ee38e3f0908a8998653234095322b",
        "config_package_sha256": "5403ace40372d1d7b3e2d33fc5332dd3599d0e0410455b670eae78fe905ef1b1",
        "config_original_sha256": "d97fe0c3a226ed37b530472c6e93e340b92fae8df7fcf9022ddc47b6d2cb76a3",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_files(path: Path) -> list[Path]:
    return sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
    )


def check(condition: bool, message: str, failures: list[str]) -> None:
    if condition:
        print(f"OK: {message}")
    else:
        print(f"FAIL: {message}")
        failures.append(message)


def validate_repo(repo_root: Path, failures: list[str]) -> None:
    print("\nRepository checks")
    for rel in ["README.md", "LICENSE", "requirements.txt", "src/forward_model.py"]:
        check((repo_root / rel).exists(), f"{rel} exists", failures)

    oversized = [
        path.relative_to(repo_root)
        for path in repo_root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and path.stat().st_size > MAX_GITHUB_FILE_BYTES
    ]
    check(not oversized, f"no tracked release file is larger than {MAX_GITHUB_FILE_BYTES} bytes", failures)
    if oversized:
        for path in oversized:
            print(f"  oversized: {path}")

    for figure, meta in FIGURES.items():
        folder = repo_root / "paper" / meta["slug"]
        print(f"\nFigure {figure}: {meta['slug']}")
        check(folder.exists(), f"{folder.relative_to(repo_root)} exists", failures)
        script = folder / "forward_model_exact.py"
        config_package = folder / "config_package.json"
        config_original = folder / "config_original.json"
        for path in [script, folder / "fourier_resolution.py", config_package, config_original]:
            check(path.exists(), f"{path.relative_to(repo_root)} exists", failures)
        if script.exists():
            check(sha256(script) == meta["script_sha256"], "exact script SHA-256 matches provenance", failures)
        if config_package.exists():
            check(
                sha256(config_package) == meta["config_package_sha256"],
                "portable config SHA-256 matches expected package config",
                failures,
            )
            config = json.loads(config_package.read_text())
            check(
                config["data_paths"]["folder_path"] == "input_images",
                "portable config uses input_images relative path",
                failures,
            )
            check(
                config["data_paths"]["output_base_path"] == "reproduction_output",
                "portable config writes to reproduction_output",
                failures,
            )
        if config_original.exists():
            check(
                sha256(config_original) == meta["config_original_sha256"],
                "original config SHA-256 matches saved result",
                failures,
            )


def validate_data(data_root: Path, repo_root: Path, sample_images: int, failures: list[str]) -> None:
    print("\nData-package checks")
    check(data_root.exists(), f"data root exists: {data_root}", failures)
    for rel in ["README.md", "figure_manifest.csv", "checksums_sha256.txt", "figures"]:
        check((data_root / rel).exists(), f"data package has {rel}", failures)

    if sample_images:
        try:
            from PIL import Image
        except ImportError:
            Image = None
            check(False, "Pillow is installed for sample image checks", failures)
    else:
        Image = None

    for figure, meta in FIGURES.items():
        folder = data_root / "figures" / meta["slug"]
        print(f"\nFigure {figure} data: {meta['slug']}")
        for rel in [
            "input_images",
            "results",
            "paper_figure.png",
            "config_package.json",
            "config_original.json",
            "code/forward_model_exact.py",
            "code/fourier_resolution.py",
        ]:
            check((folder / rel).exists(), f"{meta['slug']} has {rel}", failures)

        input_dir = folder / "input_images"
        if input_dir.exists():
            inputs = image_files(input_dir)
            check(
                len(inputs) == meta["input_images"],
                f"{meta['slug']} has {meta['input_images']} input images",
                failures,
            )
            if Image is not None:
                for path in inputs[:sample_images]:
                    with Image.open(path) as image:
                        print(f"  sample: {path.name} size={image.size} mode={image.mode}")

        result_dir = folder / "results"
        if result_dir.exists():
            result_count = sum(1 for path in result_dir.rglob("*") if path.is_file())
            check(
                result_count >= meta["result_files_min"],
                f"{meta['slug']} has at least {meta['result_files_min']} saved result files",
                failures,
            )

        data_script = folder / "code" / "forward_model_exact.py"
        repo_script = repo_root / "paper" / meta["slug"] / "forward_model_exact.py"
        if data_script.exists() and repo_script.exists():
            check(
                sha256(data_script) == sha256(repo_script) == meta["script_sha256"],
                "data-package and repository exact scripts match",
                failures,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the GitHub release folder.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional path to data_kaggle_package for full data checks.",
    )
    parser.add_argument(
        "--sample-images",
        type=int,
        default=0,
        help="Open this many input images per figure with Pillow during data checks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures: list[str] = []
    repo_root = args.repo_root.expanduser().resolve()
    validate_repo(repo_root, failures)
    if args.data_root is not None:
        validate_data(args.data_root.expanduser().resolve(), repo_root, args.sample_images, failures)

    print("\nSummary")
    if failures:
        print(f"{len(failures)} check(s) failed.")
        return 1
    print("All requested checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

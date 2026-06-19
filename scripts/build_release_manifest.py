#!/usr/bin/env python3
"""Build checksum manifests for the GitHub release folder."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints"}
SKIP_NAMES = {"RELEASE_MANIFEST.csv", "checksums_sha256.txt"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def should_skip(path: Path) -> bool:
    rel_parts = path.relative_to(ROOT).parts
    return path.name in SKIP_NAMES or any(part in SKIP_DIRS for part in rel_parts)


def main() -> None:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or should_skip(path):
            continue
        rel = path.relative_to(ROOT).as_posix()
        rows.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)})

    manifest_path = ROOT / "RELEASE_MANIFEST.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    checksums_path = ROOT / "checksums_sha256.txt"
    with checksums_path.open("w") as handle:
        for row in rows:
            handle.write(f"{row['sha256']}  {row['path']}\n")

    print(f"Wrote {manifest_path}")
    print(f"Wrote {checksums_path}")
    print(f"Files: {len(rows)}")
    print(f"Bytes: {sum(row['bytes'] for row in rows)}")


if __name__ == "__main__":
    main()

"""Download and integrity-check the selected public Kaggle dataset archive."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from .utils import load_config, resolve_project_path, sha256_file


def _verify(path: Path, expected_sha256: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, observed {actual}"
        )


def download(config_path: str | Path, force: bool = False) -> dict[str, str]:
    """Download once, extract only the US files, and verify every artifact."""
    config = load_config(config_path)["dataset"]
    archive = resolve_project_path(config["raw_archive"])
    raw_file = resolve_project_path(config["raw_file"])
    category_file = resolve_project_path(config["category_file"])
    archive.parent.mkdir(parents=True, exist_ok=True)
    raw_file.parent.mkdir(parents=True, exist_ok=True)

    archive_valid = archive.exists() and sha256_file(archive) == config["raw_archive_sha256"]
    if force or not archive_valid:
        partial = archive.with_suffix(archive.suffix + ".part")
        request = urllib.request.Request(
            config["download_url"], headers={"User-Agent": "Beyond-Engagement-Research/1.0"}
        )
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as target:
            shutil.copyfileobj(response, target, length=1024 * 1024)
        _verify(partial, config["raw_archive_sha256"], "Downloaded archive")
        partial.replace(archive)
    _verify(archive, config["raw_archive_sha256"], "Archive")

    member_to_destination = {
        "USvideos.csv": raw_file,
        "US_category_id.json": category_file,
    }
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        missing = sorted(set(member_to_destination) - names)
        if missing:
            raise RuntimeError(f"Archive is missing expected members: {missing}")
        for member, destination in member_to_destination.items():
            with bundle.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)

    _verify(raw_file, config["raw_file_sha256"], "USvideos.csv")
    _verify(category_file, config["category_file_sha256"], "US_category_id.json")
    return {
        "archive": str(archive),
        "archive_sha256": config["raw_archive_sha256"],
        "raw_file": str(raw_file),
        "raw_file_sha256": config["raw_file_sha256"],
        "category_file": str(category_file),
        "category_file_sha256": config["category_file_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = download(args.config, force=args.force)
    print(f"Verified raw data: {result['raw_file_sha256']}")


if __name__ == "__main__":
    main()

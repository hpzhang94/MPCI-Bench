"""Public helpers for the MPCI-Bench package."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_DATA_PATH = PACKAGE_ROOT / "dataset" / "mpci_bench.json"
DEFAULT_IMAGE_IDS_PATH = PACKAGE_ROOT / "dataset" / "required_image_ids.json"

__all__ = [
    "DEFAULT_DATA_PATH",
    "DEFAULT_IMAGE_IDS_PATH",
    "PACKAGE_ROOT",
    "REPO_ROOT",
]

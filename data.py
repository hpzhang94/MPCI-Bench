"""Dataset loading and schema helpers for MPCI-Bench.

The benchmark JSON uses the refactored NeurIPS submission schema:
``name``, ``seed``, ``story``, ``trace``, and ``img_metadata``. These helpers
also tolerate older field names used by the original research scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import DEFAULT_DATA_PATH, REPO_ROOT


def load_benchmark(path: str | Path = DEFAULT_DATA_PATH) -> list[dict[str, Any]]:
    """Load the benchmark JSON file."""
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of benchmark entries in {path}")
    return data


def is_inappropriate(entry_or_name: dict[str, Any] | str) -> bool:
    """Return True for negative/inappropriate benchmark cases."""
    name = entry_or_name.get("name", "") if isinstance(entry_or_name, dict) else entry_or_name
    lower = str(name).lower()
    return lower.endswith("_neg") or lower.endswith("_inappropriate")


def is_appropriate(entry_or_name: dict[str, Any] | str) -> bool:
    """Return True for positive/appropriate benchmark cases."""
    name = entry_or_name.get("name", "") if isinstance(entry_or_name, dict) else entry_or_name
    lower = str(name).lower()
    return lower.endswith("_pos") or lower.endswith("_appropriate")


def get_pair_id(entry_or_name: dict[str, Any] | str) -> str:
    """Return the shared pair/image identifier without the polarity suffix."""
    name = entry_or_name.get("name", "") if isinstance(entry_or_name, dict) else entry_or_name
    for suffix in ("_neg", "_pos", "_inappropriate", "_appropriate"):
        if str(name).lower().endswith(suffix):
            return str(name)[: -len(suffix)]
    return str(name)


def get_seed(entry: dict[str, Any]) -> dict[str, Any]:
    """Return the CI seed object."""
    return entry.get("seed", {}) if isinstance(entry.get("seed"), dict) else {}


def get_story(entry: dict[str, Any]) -> dict[str, Any]:
    """Return the refactored story object, falling back to legacy vignette."""
    story = entry.get("story")
    if isinstance(story, dict):
        return story
    vignette = entry.get("vignette")
    return vignette if isinstance(vignette, dict) else {}


def get_story_content(entry: dict[str, Any]) -> str:
    """Return the story narrative text."""
    story = get_story(entry)
    return str(story.get("content") or story.get("story") or "")


def get_trace(entry: dict[str, Any]) -> dict[str, Any]:
    """Return the refactored trace object, falling back to legacy trajectory."""
    trace = entry.get("trace")
    if isinstance(trace, dict):
        return trace
    trajectory = entry.get("trajectory")
    return trajectory if isinstance(trajectory, dict) else {}


def get_image_metadata(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Return image metadata records."""
    metadata = entry.get("img_metadata")
    if isinstance(metadata, list):
        return [item for item in metadata if isinstance(item, dict)]
    return []


def get_image_path(entry: dict[str, Any], resolve: bool = False) -> str | None:
    """Return the first referenced image path."""
    for item in get_image_metadata(entry):
        path = item.get("path") or item.get("image_path") or item.get("actual_path")
        if path:
            return str((REPO_ROOT / path).resolve()) if resolve else str(path)

    for source_name in ("trace", "trajectory", "story", "vignette"):
        source = entry.get(source_name)
        if not isinstance(source, dict):
            continue
        for image in source.get("images", []) or []:
            if not isinstance(image, dict):
                continue
            for key in ("actual_path", "image_path", "positive", "negative", "path", "filename"):
                path = image.get(key)
                if path:
                    return str((REPO_ROOT / path).resolve()) if resolve else str(path)
    return None


def get_image_labels(entry: dict[str, Any]) -> list[str]:
    """Return VISPR labels associated with the entry image."""
    labels: list[str] = []
    for item in get_image_metadata(entry):
        label = item.get("label")
        if label:
            labels.append(str(label))
        for label in item.get("labels", []) or []:
            labels.append(str(label))

    seed = get_seed(entry)
    for key in ("label", "labels", "all_labels"):
        value = seed.get(key)
        if isinstance(value, str):
            labels.append(value)
        elif isinstance(value, list):
            labels.extend(str(label) for label in value if label)

    return list(dict.fromkeys(label for label in labels if label and label != "a0_safe"))


def get_image_sensitive_description(entry: dict[str, Any]) -> str:
    """Return a compact description of the sensitive visual signal."""
    for item in get_image_metadata(entry):
        description = item.get("description")
        if description:
            return str(description)
    seed = get_seed(entry)
    return str(seed.get("data_type") or ", ".join(get_image_labels(entry)) or "")

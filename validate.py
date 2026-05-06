"""Validate the released MPCI-Bench artifact for reviewer inspection."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .data import get_image_path, get_pair_id, is_appropriate, is_inappropriate, load_benchmark


REQUIRED_TOP_LEVEL_KEYS = {"name", "seed", "story", "trace", "img_metadata"}
REQUIRED_SEED_KEYS = {
    "data_type",
    "data_subject",
    "data_sender",
    "data_recipient",
    "transmission_principle",
    "contextual_domain",
}
REQUIRED_STORY_KEYS = {"content"}
REQUIRED_TRACE_KEYS = {"user_instruction", "toolkits", "executable_trajectory", "final_action"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(path: Path) -> list[str]:
    errors: list[str] = []
    data = load_benchmark(path)
    names = [entry.get("name") for entry in data]

    if len(names) != len(set(names)):
        errors.append("Duplicate scenario names found")

    polarity = Counter(
        "neg" if is_inappropriate(entry) else "pos" if is_appropriate(entry) else "unknown"
        for entry in data
    )
    if polarity["unknown"]:
        errors.append(f"{polarity['unknown']} entries lack _neg/_pos or legacy polarity suffixes")

    pairs: dict[str, set[str]] = defaultdict(set)
    for index, entry in enumerate(data):
        name = entry.get("name", f"entry[{index}]")
        missing = REQUIRED_TOP_LEVEL_KEYS - set(entry)
        if missing:
            errors.append(f"{name}: missing top-level keys {sorted(missing)}")

        seed = entry.get("seed", {})
        story = entry.get("story", {})
        trace = entry.get("trace", {})
        if not isinstance(seed, dict):
            errors.append(f"{name}: seed is not an object")
            seed = {}
        if not isinstance(story, dict):
            errors.append(f"{name}: story is not an object")
            story = {}
        if not isinstance(trace, dict):
            errors.append(f"{name}: trace is not an object")
            trace = {}

        for field_group, required, value in (
            ("seed", REQUIRED_SEED_KEYS, seed),
            ("story", REQUIRED_STORY_KEYS, story),
            ("trace", REQUIRED_TRACE_KEYS, trace),
        ):
            missing_fields = [field for field in required if not value.get(field)]
            if missing_fields:
                errors.append(f"{name}: missing {field_group} fields {missing_fields}")

        if not get_image_path(entry):
            errors.append(f"{name}: missing image path")

        pairs[get_pair_id(entry)].add("neg" if is_inappropriate(entry) else "pos")

    incomplete_pairs = [pair_id for pair_id, values in pairs.items() if values != {"neg", "pos"}]
    if incomplete_pairs:
        errors.append(f"{len(incomplete_pairs)} pair IDs do not have exactly one positive and one negative case")

    print(f"Dataset: {path}")
    print(f"Entries: {len(data)}")
    print(f"Pairs: {len(pairs)}")
    print(f"Polarity: {dict(polarity)}")
    print(f"SHA256: {sha256(path)}")
    return errors


def validate_croissant(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"Croissant file not found: {path}"]
    with path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    for key in ("@context", "@type", "name", "description", "license", "url", "distribution", "recordSet"):
        if key not in metadata:
            errors.append(f"Croissant metadata missing {key}")

    distributions = metadata.get("distribution", [])
    if not isinstance(distributions, list) or not distributions:
        errors.append("Croissant metadata has no distribution entries")
    else:
        first = distributions[0]
        if first.get("contentUrl") != "mpci_bench/dataset/mpci_bench.json":
            errors.append("Croissant distribution contentUrl does not match the released dataset path")
        if first.get("sha256") in (None, "", "to_be_computed"):
            errors.append("Croissant distribution sha256 is not populated")

    rai_keys = [key for key in metadata if key.startswith("rai:")]
    if not rai_keys:
        errors.append("Croissant metadata does not include Responsible AI fields")

    print(f"Croissant: {path}")
    print(f"RAI fields: {len(rai_keys)}")
    print(f"SHA256: {sha256(path)}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MPCI-Bench release artifacts")
    parser.add_argument("--data", type=Path, default=Path("mpci_bench/dataset/mpci_bench.json"))
    parser.add_argument("--croissant", type=Path, default=Path("croissant_metadata.json"))
    args = parser.parse_args()

    errors = validate_dataset(args.data)
    errors.extend(validate_croissant(args.croissant))
    if errors:
        print("\nValidation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("\nValidation passed.")


if __name__ == "__main__":
    main()

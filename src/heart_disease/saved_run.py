"""Read and verify a completed experiment without fitting or scoring the holdout."""

import hashlib
import json
from pathlib import Path

from .dataset import load_snapshot
from .evaluation import metrics
from .experiment import frozen_split
from .modeling import select_model


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_saved_run(directory: Path):
    if not (directory / "holdout.started").is_file():
        raise ValueError("The saved run has no holdout marker")
    return tuple(json.loads((directory / name).read_text(encoding="utf-8")) for name in (
        "selection.json", "validation.json", "heldout.json",
    ))


def verify_run(directory: Path, data: Path, manifest: Path):
    selection, validation, heldout = read_saved_run(directory)
    records, checksum = load_snapshot(data)
    train, test, split_checksum = frozen_split(manifest, records, checksum)
    if (selection["dataset_sha256"], selection["split_sha256"], selection["training_row_ids"],
        heldout["row_ids"]) != (checksum, split_checksum, train, test):
        raise ValueError("Saved experiment does not match the frozen partitions")
    name = select_model(validation["outer_comparison"])
    tuned = validation["full_training_configurations"][name]
    if (selection["algorithm"], selection["threshold"], selection["parameters"]) != (
        name, tuned["threshold"], tuned["parameters"],
    ):
        raise ValueError("Saved selection disagrees with training-only validation")
    labels = [int(records[row - 1][-1] != 0) for row in test]
    if labels != heldout["labels"] or len(labels) != len(heldout["scores"]):
        raise ValueError("Saved labels or scores do not match the holdout")
    if metrics(labels, heldout["scores"], selection["threshold"]) != heldout["metrics"]:
        raise ValueError("Saved metrics disagree with the recorded scores")
    return selection, validation, heldout, records

"""Run the source audit or generate the split and verify training preprocessing."""

import argparse
import json
from pathlib import Path

from .audit import audit_records
from .dataset import binary_target, load_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "split", "prepare"])
    parser.add_argument("--data", type=Path, default=Path("data/processed.cleveland.data"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records, checksum = load_snapshot(args.data)
    if args.command == "audit":
        result = audit_records(records, checksum)
    else:
        from collections import Counter

        from .split import SEED, TEST_FRACTION, split_rows

        train_ids, test_ids = split_rows(records)
        result = {
            "sha256": checksum,
            "seed": SEED,
            "test_fraction": TEST_FRACTION,
            "train_row_ids": train_ids,
            "test_row_ids": test_ids,
            "train_class_counts": Counter(binary_target(records[i - 1][-1]) for i in train_ids),
            "test_class_counts": Counter(binary_target(records[i - 1][-1]) for i in test_ids),
        }

        if args.command == "prepare":
            from .preprocessing import build_preprocessor, training_features

            train = training_features(records, train_ids)
            transformed_train = build_preprocessor().fit_transform(train)
            result["transformed_training_shape"] = list(transformed_train.shape)

    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()

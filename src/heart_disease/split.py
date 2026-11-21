"""Deterministically stratify predictor groups on the binary diagnosis."""

from hashlib import sha256

from .dataset import Record, binary_target, predictor_groups

TEST_FRACTION = 0.20
SEED = 42


def split_rows(records: list[Record]) -> tuple[list[int], list[int]]:
    """Return sorted source row IDs. Matching predictors always stay together."""
    by_class: dict[int, list[list[int]]] = {0: [], 1: []}
    for row_ids in predictor_groups(records).values():
        labels = {binary_target(records[row_id - 1][-1]) for row_id in row_ids}
        if len(labels) != 1:
            raise ValueError(f"Conflicting binary labels for source rows {row_ids}")
        by_class[labels.pop()].append(row_ids)

    absent, present = by_class[0], by_class[1]
    if min(len(absent), len(present)) < 2:
        raise ValueError("Each binary class needs at least two distinct predictor groups")
    total = len(absent) + len(present)
    test_size = (total + 4) // 5  # Round 20% up to a whole group.
    if test_size < 2:
        raise ValueError("The 20% holdout is too small to contain both classes")
    ideal_absent = round(test_size * len(absent) / total)
    low = max(1, test_size - len(present) + 1)
    high = min(len(absent) - 1, test_size - 1)
    test_absent = min(max(ideal_absent, low), high)

    def rank(row_ids: list[int]) -> bytes:
        return sha256(f"{SEED}:{row_ids[0]}".encode("ascii")).digest()

    absent = sorted(absent, key=rank)
    present = sorted(present, key=rank)
    test_present = test_size - test_absent
    train = absent[test_absent:] + present[test_present:]
    test = absent[:test_absent] + present[:test_present]
    return (
        sorted(row_id for group in train for row_id in group),
        sorted(row_id for group in test for row_id in group),
    )

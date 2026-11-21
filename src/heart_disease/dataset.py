"""Parse and validate the published Cleveland file without changing row order."""

import csv
import hashlib
import io
import math
from pathlib import Path

NUMERIC = ("age", "trestbps", "chol", "thalach", "oldpeak")
CATEGORIES = {
    "sex": (0, 1),
    "cp": (1, 2, 3, 4),
    "fbs": (0, 1),
    "restecg": (0, 1, 2),
    "exang": (0, 1),
    "slope": (1, 2, 3),
    "ca": (0, 1, 2, 3),
    "thal": (3, 6, 7),
}
FEATURES = (
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal",
)
TARGET = "num"
COLUMNS = (*FEATURES, TARGET)
NULLABLE = frozenset({"ca", "thal"})
SHA256 = "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8"
SOURCE = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
ROW_COUNT = 303

Record = tuple[float | None, ...]


def binary_target(value: float | None) -> int:
    if value not in (0, 1, 2, 3, 4):
        raise ValueError("num must be in 0–4")
    return int(value != 0)


def parse_value(column: str, token: str) -> float | None:
    if token == "?" and column in NULLABLE:
        return None

    value = float(token)
    if not math.isfinite(value):
        raise ValueError("expected finite value")
    if column == TARGET:
        binary_target(value)
    elif column in CATEGORIES:
        if value not in CATEGORIES[column]:
            raise ValueError(f"expected one of {CATEGORIES[column]}")
    elif column == "oldpeak":
        if value < 0:
            raise ValueError("expected nonnegative ST depression")
    elif value <= 0 or not value.is_integer():
        raise ValueError("expected positive integer")
    return value


def parse_records(payload: bytes) -> list[Record]:
    records = []
    for row_id, fields in enumerate(csv.reader(io.StringIO(payload.decode("ascii")), strict=True), 1):
        if len(fields) != len(COLUMNS):
            raise ValueError(f"Row {row_id}: expected {len(COLUMNS)} fields, found {len(fields)}")
        record = []
        for column, token in zip(COLUMNS, fields, strict=True):
            try:
                record.append(parse_value(column, token))
            except ValueError as exc:
                raise ValueError(f"Row {row_id}, {column}: invalid value {token!r} ({exc})") from exc
        records.append(tuple(record))
    if not records:
        raise ValueError("Dataset contains no records")
    return records


def load_snapshot(path: Path) -> tuple[list[Record], str]:
    payload = path.read_bytes()
    checksum = hashlib.sha256(payload).hexdigest()
    if checksum != SHA256:
        raise ValueError(f"Cleveland snapshot SHA-256 mismatch: {checksum}")
    records = parse_records(payload)
    if len(records) != ROW_COUNT:
        raise ValueError(f"Expected {ROW_COUNT} records, found {len(records)}")
    return records, checksum


def predictor_groups(records: list[Record]) -> dict[Record, list[int]]:
    """Group identical predictors, including matching missing positions."""
    groups: dict[Record, list[int]] = {}
    for row_id, record in enumerate(records, 1):
        groups.setdefault(record[:-1], []).append(row_id)
    return groups
